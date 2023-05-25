import argparse
import os
import random
import time
from copy import deepcopy

import cv2
import megengine
import megengine.autodiff as autodiff
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import numpy as np
from basecls.utils import registers
from megengine import amp
from megengine.jit import DTRConfig, SublinearMemoryConfig, trace
from megengine.utils.profiler import Profiler, profile
from tensorboardX import SummaryWriter
from tqdm import tqdm

import models.loss as loss
import utils
from dataset import load_dataset

create_model = registers.models.get
logging = megengine.logger.get_logger()


def route_and_calculate_loss(criterion_dict, outs, inp, target, model, args, tea_model=None):

    if args.arch.startswith('dtps'):
        logit, spatial_tokens, hard_decision_list = outs
    else:
        logit = outs[0]

    loss_dict = {}
    for name, (loss_fn, loss_w) in criterion_dict.items():

        if isinstance(loss_fn, loss.SoftTargetCrossEntropy):
            loss_dict[name] = loss_fn(logit, target) * loss_w
        elif isinstance(loss_fn, loss.PruningLoss):
            loss_dict[name] = loss_fn(hard_decision_list) * loss_w
        elif isinstance(loss_fn, loss.DistillKLLoss):
            assert tea_model is not None
            t_logit, t_spatial_tokens = tea_model(inp)[:2]
            loss_dict[name] = loss_w * loss_fn(
                logit, t_logit, spatial_tokens, t_spatial_tokens, hard_decision_list[-1])

    total_loss = sum(loss_dict.values())
    return total_loss, loss_dict


# @trace(symbolic=False,dtr_config=DTRConfig(eviction_threshold=8*1024**3))
def train_iter(gm, optimizer, scaler, model, images, labels, criterion_dict, tea_model, args):
    with gm:
        outs = model(images)
        total_loss, loss_dict = route_and_calculate_loss(
            criterion_dict, outs, images, labels, model, args, tea_model)
        if scaler is None:
            gm.backward(total_loss)
        else:
            scaler.backward(gm, total_loss)
        optimizer.step().clear_grad()
    return total_loss, loss_dict


def train(args, criterion_dict, optimizer, train_queue, minibatch_count, gm, scaler, model, train_writer, epoch, nums_epoch, tea_model=None, mixup_fn=None):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, nums_epoch, args)

    profiler = Profiler()
    for i in range(minibatch_count):
        step = epoch * minibatch_count + i

        data = next(train_queue)
        images, labels = data[:2]
        images = megengine.Tensor(images, dtype='float32')
        labels = megengine.Tensor(labels, dtype='int32')

        if mixup_fn is not None:

            images, labels = mixup_fn(images, labels)

        total_loss, loss_dict = train_iter(
            gm, optimizer, scaler, model, images, labels, criterion_dict, tea_model, args)
        if dist.get_world_size() > 1:
            total_loss = F.distributed.all_reduce_sum(
                total_loss.detach()) / dist.get_world_size()
        total_loss = total_loss.item()

        # logs

        if step % 100 == 0 and dist.get_rank() == 0:
            train_writer.add_scalar(
                'train/total_loss', total_loss, step)
            train_writer.flush()
            info = f'Epoch = {epoch}, ' + f'process = {i} / {minibatch_count}, ' + \
                f'total_loss = {total_loss:.3f}, '
            for k, v in loss_dict.items():
                info += f'{k}:{v.item():.3f}, '
            for i in range(len(optimizer.param_groups)):
                train_writer.add_scalar(
                    f'train/param_group{i}_lr', optimizer.param_groups[i]["lr"], step)
            logging.info(info)


def evaluate(model, test_dataloader, epoch, val_writer, best_acc=0):
    model.eval()
    acc1_sum = 0
    acc5_sum = 0
    total_loss = 0
    valdation_num = 0
    for data in tqdm(test_dataloader, desc='Testing'):
        image, labels = data[:2]
        image = megengine.Tensor(image, dtype='float32')
        labels = megengine.Tensor(labels, dtype='int32')
        outs = model(image)
        logits = outs[0] if isinstance(outs, tuple) else outs
        num = labels.shape[0]
        acc1, acc5 = F.metric.topk_accuracy(logits, labels, topk=(1, 5))
        loss = F.nn.cross_entropy(logits, labels)
        # calculate mean values
        if dist.get_world_size() > 1:
            loss = F.distributed.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = F.distributed.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = F.distributed.all_reduce_sum(acc5) / dist.get_world_size()
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num

    total_loss = total_loss / len(test_dataloader)
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num

    if best_acc < acc1:
        best_acc = acc1

    if dist.get_rank() == 0:
        val_writer.add_scalar('val/acc1', acc1 * 100., epoch)
        val_writer.add_scalar('val/acc5', acc5 * 100., epoch)
        val_writer.add_scalar('val/loss', total_loss, epoch)
        val_writer.flush()
        logging.info(f'Epoch = {epoch}, '
                     f'val_acc = {acc1:.4f}, '
                     f'best_acc = {best_acc:.4f}, '
                     )
    return best_acc


def worker(args):

    # enable DTR for less GPU memory , see Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, and Zachary Tatlock. Dynamic tensor rematerialization. In International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=Vfs_2RnOD0H.
    megengine.dtr.enable()
    
    # set seed
    seed = args.seed + dist.get_rank()
    megengine.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    logging.info(f'Task : {args.desc}')
    log_dir = os.path.join(args.log_dir, args.dataset, args.arch, args.desc)
    train_writer, val_writer = None, None
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
        megengine.logger.set_log_file(os.path.join(log_dir, "log.txt"))

        train_writer = SummaryWriter(os.path.join(log_dir, 'train.events'))
        val_writer = SummaryWriter(os.path.join(log_dir, 'val.events'))

    train_dataloader, test_dataloader, num_classes, dataset_size = load_dataset(
        args.dataset, args.data_dir, args.batch_size, args.workers, args)

    args.arch_kwargs.update(num_classes=num_classes)
    train_queue = iter(train_dataloader)
    logging.info(
        f'Dataset size : {dataset_size}, World size : {dist.get_world_size()}, Batch size : {args.batch_size}')
    steps_per_epoch = dataset_size // (dist.get_world_size() * args.batch_size)

    model = create_model(args.arch)(pretrained=True, **args.arch_kwargs)

    if args.test_only:
        logging.info("Test Mode".center(50, '-'))
        model.eval()
        evaluate(model, test_dataloader, 0, val_writer)
        exit(0)

    tea_model = None
    if args.tea:
        logging.info(f"Using teacher model : {args.tea}")
        tea_model = create_model(args.tea)(pretrained=True)
        tea_model.eval()  # megengine disables any grads in default

    # Sync parameters and buffers
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())
        dist.bcast_list_(model.buffers())

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb(
            "mean") if dist.get_world_size() > 1 else None,
    )

    scaler = None if not args.fp16 else amp.GradScaler()
    train_func = amp.autocast(enabled=args.fp16)(train)

    # generate loss modules with given args
    criterion_dict = dict()
    for loss_name, loss_w in zip(args.loss_fns, args.loss_ws):
        logging.info(f"Create training loss {loss_name}, weight : {loss_w}")
        loss_fn = create_model(loss_name)(model, deepcopy(args))
        criterion_dict[loss_name] = (loss_fn, loss_w)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    args.mixup_active = mixup_active
    if mixup_active:
        from mixup import Mixup
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)

    # get params groups with different learnable rate and weight decay
    param_groups = utils.get_param_groups(model, args)
    optimizer = optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.decay)

    # if dist.get_rank() == 0:
    #     project_name = 'megengine/{}/{}'.format(args.dataset, args.arch)
    #     task = Task.init(project_name=project_name, task_name=args.desc)

    # Training and validation
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_func(args, criterion_dict, optimizer, train_queue, steps_per_epoch, gm, scaler, model, train_writer,
                   epoch, args.epochs, tea_model, mixup_fn)

        best_acc = evaluate(model, test_dataloader,
                            epoch, val_writer, best_acc)

        # save checkpoint
        if dist.get_rank() == 0:
            megengine.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                },
                os.path.join(log_dir, f'epoch{epoch}_ckpt.pkl'),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='mge.baseline.1434',
                        help='descript your exp')
    parser.add_argument('--seed', default=1234, type=int,
                        help='the random seed')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--dataset', '-d', default='imagenet', type=str,
                        help='dataset name')
    parser.add_argument('--resolution', default=32, type=int,
                        help='input size')
    parser.add_argument('--batch_size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=None,
                        type=int, help='validation batch size')
    parser.add_argument('--epochs', '-e', default=200,
                        type=int, help='stop epoch')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.05,
                        type=float, help='weight decay')
    parser.add_argument('--arch', '-a', default='dtps_deit_small_patch16_224', type=str,
                        help='model type registered')
    parser.add_argument('--arch_kwargs', default="None", type=eval,
                        help='the key word args for the specific architecture model and it will be evaled as the python dict')
    # simple random crop
    parser.add_argument('--test-only', action='store_true')

    # * Hyper-parameters for training TPS-ViT，
    # * We follow most of training strategies from DynamicViT and EViT without adjustment that benefits our methods
    # * The learning rate and its scheduler are slightly different for the training stability under aggressive compression hyper-parameters
    parser.add_argument('--loss_fns', type=str, nargs='*',
                        help='the loss modules')
    parser.add_argument('--loss_ws',  type=float, nargs='*',
                        help='the weight for reweighting the loss  ')
    parser.add_argument('--only_distill_logits', action='store_true')
    parser.add_argument('--mse_token_loss', action='store_true')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--backbone_freeze_epochs', type=int, default=-1,
                        help='the epochs to fix the weights of the backbone model')
    parser.add_argument('--backbone_lr_scale', type=float, default=1e-2,
                        help='the learning rate scale for finetuning backbone')
    parser.add_argument('--tea', default="", type=str,
                        help='the teacher model architecture name')
    # * augmentations from DeiT codebase (https://github.com/facebookresearch/deit)
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    parser.add_argument('--fp16',
                        action='store_true', help='enable fp16 training', default=False)
    parser.add_argument('--log_dir', type=str, default='train_log')
    parser.add_argument('--data_dir', type=str,
                        default='/data/datasets/imagenet1k')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('-n', '--ngpus', default=None, type=int,
                        help='number of GPUs per node (default: None, use all available GPUs)',)
    parser.add_argument('--dist-addr', default='localhost')
    parser.add_argument('--dist-port', default=23456, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)

    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork('gpu')
    args.arch_kwargs = args.arch_kwargs if args.arch_kwargs else dict()

    if args.world_size * args.ngpus > 1:
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.world_size * args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
