#!/usr/bin/env python3
import datetime
import functools
import getpass
import os
import pickle
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import megfile
import timm
import torch
import torch.distributed as dist
import torchvision.models as models
from accelerate import Accelerator
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from torch.optim import SGD, Adam, AdamW, RMSprop

import settings
import orjson


def dump_orjson(obj, json_path: str):
    obj_data = orjson.dumps(obj)
    with megfile.smart_open(json_path, 'wb') as w_file:
        w_file.write(obj_data)
    return


def load_orjson(json_path: str):
    with megfile.smart_open(json_path, 'rb') as r_file:
        res = orjson.loads(r_file.read())
    return res


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_optim(identifier):
    """Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).

    Args:
        identifier (str or Callable): the optimizer identifier.

    Returns:
        :class:`torch.optim.Optimizer` or None
    """
    if isinstance(identifier, torch.optim.Optimizer):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(
                f"Could not interpret optimizer : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret optimizer : {str(identifier)}")


def contain_parameters(m: torch.nn.Module):

    if len(list(m.parameters())) > 0:
        return True


def contain_statedict(m):

    if hasattr(m, 'state_dict'):
        if len(getattr(m, 'state_dict')()) > 0:
            return True
    else:
        return False


def dump_pkl(file, obj):
    with open(file, 'wb') as wf:
        pickle.dump(obj, wf)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(log_model_dir, save_info, name):
    ckp_path = os.path.join(log_model_dir, name)
    while True:
        try:
            with megfile.smart_open(ckp_path, 'wb') as fobj:
                torch.save(save_info, fobj)
            break
        except Exception as e:
            print(e)


def load_checkpoint(log_model_dir, name):
    ckp_path = os.path.join(log_model_dir, name)
    return torch.load(megfile.smart_open(ckp_path, 'rb'), map_location=torch.device('cpu'))


def resume_checkpoint(ckp_path):

    return torch.load(megfile.smart_open(ckp_path, 'rb'), map_location=torch.device('cpu'))


def load_imagenet_pretrain(net, model_name):
    pretrain_dict = models.__dict__[model_name](pretrained=True).state_dict()
    pretrain_dict = {k: v if not 'fc' in k else net.state_dict()[
        k] for k, v in pretrain_dict.items()}
    net.load_state_dict(pretrain_dict)


# TODO:
def multiply_decay_scheduler(optimizer, n_iter, n_epoch, lr=0.01, decay_factor=0.98, decay_step=1):

    cur_lr = lr * decay_factor ** (n_epoch//decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def warmup_cos_scheduler(optimizer, n_iter, n_epoch, total_steps=5005*300, lr=3e-3, warmup=1e4):
    """
    All models are trained with a batch size of 4096 and learning rate warmup of 10k steps.
    cosine lr decay
    adam
    0.3 weight decay
    dropout=0.1
    epochs 300
    """

    if n_iter+1 < warmup:
        cur_lr = lr * np.minimum(1., n_iter / warmup)

    else:
        progress = (n_iter - warmup) / float(total_steps - warmup)
        cur_lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    # print(n_iter, warmup, cur_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


SCHEDULER_MAP = {
    'multiply_decay': lambda args: functools.partial(multiply_decay_scheduler, lr=args.lr, decay_factor=args.decay_factor, decay_step=args.decay_step),
    'warmup_cos': lambda args: functools.partial(warmup_cos_scheduler, lr=args.lr, total_steps=args.total_steps, warmup=1e4, ),
}


def get_training_setting(net: torch.nn.Module, args):

    if not hasattr(args, 'optim'):
        print('Trying to infer the proper optimizer&scheduler setting for training')
        if args.arch.lower() in ('mbnv2', 'basis-mbnet-v2') or args.dataset.lower() in ['imagenet', 'inat19', 'inat21']:
            args.optim = 'RMSprop'

    # get optimizer
    optimizer = get_optim(args.optim)(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # get the scheduler function with n_iter&n_epoch as inputs
    scheduler = SCHEDULER_MAP[args.scheduler](args)
    return optimizer, scheduler


def adjust_learning_rate(args, optimizer, epoch, base_lr, max_epoch):
    """
    select the training settings  based on the dataset and model architecture 
    """
    lr = base_lr
    if args.dataset.lower() in ['cifar10', 'cifar100', 'tinyimagenet']:
        """decrease the learning rate at 100 and 150 epoch"""
        if epoch >= 0.5 * max_epoch:
            lr /= 10
        if epoch >= 0.75 * max_epoch:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset.lower() in ['imagenet', 'inat19', 'inat21']:
        lr = 0.1 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset.lower() in ['inat19_bbn']:
        """decrease the learning rate at 120 and 160 epoch"""
        if epoch >= 120:
            lr /= 10
        if epoch >= 160:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset.lower() == 'cub':
        lr = base_lr * (0.1 ** (epoch // 80))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset.lower() in ['imagenetlt']:
        # cosine decay
        decay_rate = 0.5 * (1 + np.cos(epoch * np.pi / max_epoch))
        lr = base_lr * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def get_task_id(task_name):
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    username = getpass.getuser()
    task_file = os.path.join(
        BASE_DIR, 'train_log', '{}_{}_taskid.txt'.format(username, task_name))
    print(task_file)
    task_id = None
    if os.path.isfile(task_file):
        with open(task_file, 'r') as fin:
            lines = fin.readlines()
            if len(lines) > 0:
                task_id = lines[0].strip()

    return task_id


def write_task_id(task_id, task_name):
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    username = getpass.getuser()
    task_file = os.path.join(
        BASE_DIR, 'train_log', '{}_{}_taskid.txt'.format(username, task_name))
    with open(task_file, 'w') as fout:
        fout.write(task_id+'\n')


def parse_args_as_dict(parser, return_plain_args=False, args=None):
    """Get a dict of dicts out of process `parser.parse_args()`

    Top-level keys corresponding to groups and bottom-level keys corresponding
    to arguments. Under `'main_args'`, the arguments which don't belong to a
    argparse group (i.e main arguments defined before parsing from a dict) can
    be found.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser instance containing
            groups. Output of `prepare_parser_from_dict`.
        return_plain_args (bool): Whether to return the output or
            `parser.parse_args()`.
        args (list): List of arguments as read from the command line.
            Used for unit testing.

    Returns:
        dict:
            Dictionary of dictionaries containing the arguments. Optionally the
            direct output `parser.parse_args()`.
    """
    args = parser.parse_args(args=args)
    args_dic = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None)
                      for a in group._group_actions}
        args_dic[group.title] = group_dict
    args_dic["main_args"] = args_dic["optional arguments"]
    del args_dic["optional arguments"]
    if return_plain_args:
        return args_dic, args
    return args_dic


def forward_att(self, x):
    B, N, C = x.shape
    # print(B, N, C)
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)
    # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # print(attn.size())
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn


def forward_block(self, x):
    x_ori = x
    x, attn = self.attn(self.norm1(x))
    x = x_ori + self.drop_path1(x)
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x, attn


def calculate_token_structure_sim(att1: torch.Tensor, att2: torch.Tensor, avg_head=True, linear_assign=False):
    # the current attentions and the last attentions
    # NOTE: att1 should not contain the cls token
    # TODO: linear assigment for batch is slow
    h, n = att1.shape[1], att1.shape[1]
    sim = 1 - torch.abs(att1-att2).sum(dim=3) * 0.5
    if avg_head:
        sim = sim.mean(dim=1, keepdim=False)
        sim = sim.unsqueeze(-1)
        sim = torch.cat([1-sim, sim], dim=-1)
    else:
        assert 'not implemented'
    return sim


def forward_token_attn_sims(self, x, prune_loc=[], avg_head=True, linear_assign=False):

    with torch.no_grad():
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        token_attn_sim_list = []
        max_layer_idx = max(prune_loc)
        for i, blk in enumerate(self.blocks):
            if i > max_layer_idx:
                break
            x, attn = blk(x)
            attn = attn[:, :, 1:, 1:]
            if i in prune_loc:
                token_attn_sim_list.append(calculate_token_structure_sim(
                    attn, last_attn, avg_head=avg_head, linear_assign=linear_assign
                ))
            last_attn = attn
    return token_attn_sim_list


def forward_token_attn_layer_var(self, x):

    with torch.no_grad():
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_square = 0.0
        x_sum = 0.0
        depths = len(self.blocks)
        for blk in self.blocks:
            x, attn = blk(x)
            attn = attn[:, :, 1:, 1:]
            x_sum += attn
            x_square += attn ** 2
            del attn

        layer_var = x_square/depths - (x_sum/depths)**2
        layer_var = layer_var.mean(dim=1).mean(dim=1).unsqueeze(-1)
        layer_var = torch.cat([layer_var, 1-layer_var], dim=-1)

    return [layer_var for _ in range(depths)]


def forward_dynamicViT_distill(self, x):

    with torch.no_grad():
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_square = 0.0
        x_sum = 0.0
        depths = len(self.blocks)

        for blk in self.blocks:
            x, _ = blk(x)

        feature = self.norm(x)
        cls = feature[:, 0]
        tokens = feature[:, 1:]
        cls = self.head(cls)

    return cls, tokens


def forward_dynamicViT_distill_attn(self, x):

    with torch.no_grad():
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_square = 0.0
        x_sum = 0.0
        depths = len(self.blocks)
        for blk in self.blocks:
            x, attn = blk(x)
            attn = attn[:, :, 1:, 1:]
            x_sum += attn
            x_square += attn ** 2
            del attn

        layer_var = x_square/depths - (x_sum/depths)**2
        layer_var = layer_var.mean(dim=1).mean(dim=1).unsqueeze(-1)
        layer_var = torch.cat([layer_var, 1-layer_var], dim=-1)

        feature = self.norm(x)
        cls = feature[:, 0]
        tokens = feature[:, 1:]
        cls = self.head(cls)

    return cls, tokens, [layer_var for _ in range(depths)]


def get_modified_teacher_model(args):

    model = timm.create_model(
        args.tea,  pretrained=True)

    # lvvit and t2t teacher is directly fixed in folder 'models'
    if 'lvvit' in args.tea or 't2t' in args.tea:
        return model
    elif not args.prune_metric:
        return model

    print("Modifying the teacher model forward ")

    def partial_forward(self, x):
        if args.prune_metric == 'token_att_layer_sim':
            return forward_token_attn_sims(self, x, avg_head=True if args.avg_head else False,
                                           prune_loc=args.prune_loc,
                                           linear_assign=True if args.linear_assign else False)
        elif args.prune_metric == "token_attn_layer_var":
            return forward_token_attn_layer_var(self, x)
        elif args.prune_metric == "distill_kl_loss":
            return forward_dynamicViT_distill(self, x)
        elif args.prune_metric == 'distill_attn_kl_loss':
            return forward_dynamicViT_distill_attn(self, x)
        else:
            raise ValueError(
                f"Not supported token prune metric : {args.prune_metric}")
    for m in model.modules():
        if isinstance(m, Attention):
            setattr(m, 'forward', forward_att.__get__(
                m, m.__class__))
            # m.forward = new.instancemethod(forward_att, m, None)

        elif isinstance(m, Block):
            # m.forward = new.instancemethod(forward_block, m, None)
            setattr(m, 'forward', forward_block.__get__(
                m, m.__class__))
    # model.forward = new.instancemethod(forward_with_attns, model, None)
    setattr(model, 'forward', partial_forward.__get__(
        model, model.__class__))
    return model


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def sum_cfgs(*cfgs):

    result = dict()
    for c in cfgs:
        result.update(c)
    return result


def get_param_group_fn(cfgs):

    return functools.partial(get_param_by_cfgs, cfgs=cfgs)


def get_param_by_cfgs(model, cfgs=None):

    if cfgs.token_prune and cfgs.backbone_freeze_epochs > 0:
        prune_branch_params = model.score_predictor.parameters()
        prune_branch_params_ids = list(
            map(id, prune_branch_params))
        base_params = filter(lambda p: id(
            p) not in prune_branch_params_ids, model.parameters())
        params = [
            {'params': base_params, 'lr': cfgs.lr * cfgs.backbone_lr_scale,
                'backbone': True},
            {'params': prune_branch_params, 'lr': cfgs.lr, 'backbone': False}
        ]
    else:
        params = model.parameters()
    return params


def freeze_backbone_if_needed(model, args, exclude='score_predictor'):
    if args.token_prune and args.backbone_freeze_epochs > 0:
        for name_p, p in model.named_parameters():
            if f'{exclude}' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False


def unfreeze_model_if_needed(model: torch.nn.Module, epoch, args):
    if args.token_prune and args.backbone_freeze_epochs > 0 and epoch >= args.backbone_freeze_epochs:
        for p in model.parameters():
            p.requires_grad = True


def check_head(checkpoint_model, state_dict):
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    return checkpoint_model

def interpolate_position_embedding(checkpoint_model, model):
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    return checkpoint_model
