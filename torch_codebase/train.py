#!/usr/bin/env python3
import random
import argparse
import collections
import datetime
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy
import brainpp
import numpy as np
import pandas as pd
import megfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer, create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, accuracy, get_state_dict
from tqdm import tqdm

import datasets
import samplers
import settings
import utils
from augmentations import new_data_aug_generator
from datasets import build_dataset
from losses import *
from models.dynamic_vit import DynamicVit
from samplers import RASampler
from utils import get_modified_teacher_model, nullcontext, check_head, interpolate_position_embedding
import pickle


CE_LOSS_FAMILY = (
    LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, torch.nn.CrossEntropyLoss,
    torch.nn.BCEWithLogitsLoss
)

DYNAMIC_VIT_MODULEs = (DynamicVit,)
NON_BLOCKING = True


if utils.is_main_process():
    from settings import get_logger
    logger = get_logger(os.path.join(settings.base_dir,
                                     'logs', 'worklog.txt'))


def check_model_nan(model: torch.nn.Module):
    flag = False
    pname = None
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            flag = True
            pname = name
            break
    return flag, pname


def dump_pkl(file, obj):
    with open(file, 'wb') as wf:
        pickle.dump(obj, wf)


def train(device, train_loader, args, net, criterion, optimizer, epoch, train_writer, log_output, scheduler, tea_net=None, mixup_fn=None, loss_scaler=None, model_ema=None):
    # switch to train mode
    net.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    minibatch_count = len(train_loader)

    amp_context_if_needed = torch.cuda.amp.autocast if args.fp16 else nullcontext
    for i_iter, (images, labels) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        images = images.to(device, non_blocking=NON_BLOCKING)
        labels = labels.to(device, non_blocking=NON_BLOCKING)
        labels_org = labels
        # mixup
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        if args.bce_loss:
            labels = labels.gt(0.0).type(labels.dtype)
        eval_dict = dict()
        loss_dict = dict()
        # NOTE: define loss calculation here
        with amp_context_if_needed():
            outs = net(images)

            # parse outs
            if isinstance(net, DYNAMIC_VIT_MODULEs) \
                    or (hasattr(net, 'module') and isinstance(net.module, DYNAMIC_VIT_MODULEs)):
                pred, spatial_features, last_decision, hard_decision_list, token_prune_logit_list = outs
                for i, r in enumerate(hard_decision_list):
                    eval_dict[f'L{i}KeepRatio'] = r.mean().cpu().item()
            else:
                pred = outs

            # calculate the loss
            for k, fn in criterion.items():
                if isinstance(fn, CE_LOSS_FAMILY):
                    loss_dict[k] = fn(pred, labels)
                elif isinstance(fn, (PruneSoftTargetKDLoss, PruneHardTargetKDLoss)):
                    with torch.no_grad():
                        token_metics = tea_net(images)
                    loss_dict[k] = fn(token_prune_logit_list,
                                      token_metics)
                elif isinstance(fn, AttnDistillKLLoss):
                    with torch.no_grad():
                        pred_t, spatial_features_t, token_attn_sim_list = tea_net(
                            images)
                    loss_dict[k] = fn(pred, pred_t, spatial_features, last_decision,
                                      spatial_features_t, hard_decision_list, token_attn_sim_list)
                elif isinstance(fn, DistillKLLoss):
                    with torch.no_grad():
                        pred_t, spatial_features_t = tea_net(images)[:2]
                    if isinstance(net.module, DYNAMIC_VIT_MODULEs) \
                            or isinstance(net, DYNAMIC_VIT_MODULEs):
                        loss_dict[k] = fn(
                            pred, pred_t, spatial_features, last_decision, spatial_features_t)
                    else:
                        loss_dict[k] = fn(pred, pred_t)
                elif isinstance(fn, (KeepRatioLoss, GlobalKeepRatioLoss, TokenActLayerRatio)):
                    loss_dict[k] = fn(hard_decision_list)
                elif isinstance(fn, TopkmaskLoss):
                    with torch.no_grad():
                        token_metics = tea_net(images)
                    loss_dict[k] = fn(hard_decision_list, token_metics)
                else:
                    raise ValueError(f"un-supported loss : {k}, {fn}")

            loss = sum(
                [loss_dict[name]
                    for name in criterion]
            )
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                    parameters=net.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(net)
        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{k: v for k, v in eval_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        step = epoch * minibatch_count + i_iter
        if utils.is_main_process():
            if step % 200 == 0:
                acc1, acc5 = accuracy(pred, labels_org, topk=(1, 5))
                for k, v in loss_dict.items():
                    train_writer.add_scalar(f'train/{k}', v.item(), step)
                for k, v in eval_dict.items():
                    train_writer.add_scalar(f'train/{k}', v, step)
                train_writer.add_scalar('train/acc1', acc1.item(), step)
                train_writer.add_scalar('train/acc5', acc5.item(), step)

                for i in range(len(optimizer.param_groups)):
                    train_writer.add_scalar(
                        f'train/param_group{i}_lr', optimizer.param_groups[i]["lr"], step)
                train_writer.flush()
        # break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, val_loader, net, device, criterion, epoch, train_writer, log_output):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    net.eval()

    metrics_dict = collections.defaultdict(list)
    amp_context_if_needed = torch.cuda.amp.autocast if args.fp16 else nullcontext

    results = dict()
    for i, (images, labels) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        images = images.to(device, non_blocking=NON_BLOCKING)
        labels = labels.to(device, non_blocking=NON_BLOCKING)

        eval_dict = dict()
        with amp_context_if_needed():

            outs = net(images)
            # parse outs
            if isinstance(net, DYNAMIC_VIT_MODULEs) or (hasattr(net, 'module') and isinstance(net.module, DYNAMIC_VIT_MODULEs)):
                pred, spatial_features, last_decision, hard_decision_list, token_prune_logit_list = outs

                for i, r in enumerate(hard_decision_list):
                    eval_dict[f'L{i}KeepRatio'] = r.mean().cpu().item()
            else:
                pred = outs

        acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if eval_dict:
            metric_logger.update(**{k: v for k, v in eval_dict.items()})

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, results


def hash_opt(args):
    print(args)
    args = deepcopy(args)
    args.run_num = -1
    key = hashlib.sha256(json.dumps(
        vars(args), sort_keys=True).encode('utf-8')).hexdigest()
    return key[:6]


def main():

    from config import get_args
    args = get_args()
    print(args)
    utils.init_distributed_mode(args)

    if utils.is_main_process():
        print("*"*100)
        print("Cuda Version : ", torch.version.cuda)
        print("Cudnn Version : ", torch.backends.cudnn.version())
        print("*"*100)

    device = torch.device(args.device)
    # fix the seed for reproducibility
    torch.backends.cudnn.benchmark = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    stop_epoch = args.epochs
    base_lr = args.lr
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size / 1024
        args.lr = linear_scaled_lr

    task_dataset = args.dataset
    task_name = args.task_name
    print(f"Task name : {task_name}")

    project_name = f'dynamicnets/{task_dataset}'
    log_dir = settings.get_log_dir(task_name)
    print(f"log_dir : {log_dir}".center(100, '-'))
    log_model_dir = settings.get_log_model_dir(project_name, task_name)
    print("log model dir", log_model_dir)
    os.makedirs(os.path.join(settings.base_dir,
                             'logs'), exist_ok=True)
    os.makedirs(os.path.join(settings.base_dir,
                             'logs', task_name), exist_ok=True)
    # logger
    train_writer = None
    log_output = None

    if utils.is_main_process():
        from settings import get_logger
        worklog = get_logger(os.path.join(settings.base_dir,
                             'logs', 'worklog.txt'))
        log_output = worklog.info
        train_writer = SummaryWriter(
            os.path.join(log_dir, 'train.events'))

    if args.workers < 0:
        args.workers = len(os.sched_getaffinity(
            0)) // utils.get_world_size()
    print(f"Num of workers : {args.workers}")

    dataset_train, args.ncls = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print(
        f"Train Size : {len(dataset_train)},Validation Size : {len(dataset_val)}")

    if True:  # args.distributed:

        num_tasks = utils.get_world_size()
        print(f"World Size : {num_tasks}")
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=bool(args.repeat_eval))
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=int(1. * args.batch_size // utils.get_world_size()),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=1,
    )
    if args.ThreeAugment:
        train_loader.dataset.transform = new_data_aug_generator(args)

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1. * args.batch_size // utils.get_world_size()),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.persistent_workers if args.workers else False,
        prefetch_factor=1,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    args.mixup_active = mixup_active
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.ncls)

    args.total_steps = len(train_loader) * args.epochs
    print(f'Total Steps : {args.total_steps}')
    print(f"Num of classes : {args.ncls}")

    from model import get
    model, criterion_dict = get(args)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(megfile.smart_open(
                args.finetune, 'rb'), map_location='cpu')
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        elif 'net' in checkpoint.keys():
            checkpoint_model = checkpoint['net']
        state_dict = model.state_dict()

        checkpoint_model = check_head(checkpoint_model, state_dict)
        checkpoint_model = interpolate_position_embedding(
            checkpoint_model, model)
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model

    if args.grad_checkpoint and hasattr(model, "grad_checkpointing"):
        model.grad_checkpointing = True
        if args.grad_checkpoint_layers:
            model.grad_checkpoint_layers = args.grad_checkpoint_layers

    if args.distributed:
        print(f"distributed:{args.distributed}:{args.gpu}".center(100, '-'))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer_v2(
        model_without_ddp,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=True,
        param_group_fn=utils.get_param_group_fn(args)
    )

    loss_scaler = NativeScaler()
    scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0

    if args.resume:
        print(f"Load checkpoint from resuming path : {args.resume}")
        if megfile.smart_exists(args.resume):
            print(f"load checkpoint for {args.resume}")
            model_info = utils.resume_checkpoint(args.resume)
            start_epoch = model_info['epoch'] + 1
            model_without_ddp.load_state_dict(model_info['net'])
            optimizer.load_state_dict(model_info['optimizer'])
            if scheduler is not None and 'scheduler' in model_info:
                scheduler.load_state_dict(model_info['scheduler'])
                # scheduler.step(start_epoch)
            if loss_scaler is not None and 'scaler' in model_info:
                loss_scaler.load_state_dict(model_info['scaler'])

    if args.continue_train:
        checkpoint = os.path.join(log_model_dir, 'latest')

        if megfile.smart_exists(checkpoint):
            print(f"load checkpoint for {checkpoint}")
            model_info = utils.load_checkpoint(log_model_dir, 'latest')
            start_epoch = model_info['epoch'] + 1
            model_without_ddp.load_state_dict(model_info['net'])
            optimizer.load_state_dict(model_info['optimizer'])
            if scheduler is not None and 'scheduler' in model_info:
                scheduler.load_state_dict(model_info['scheduler'])
                # scheduler.step(start_epoch)
            if loss_scaler is not None and 'scaler' in model_info:
                loss_scaler.load_state_dict(model_info['scaler'])

    if start_epoch >= stop_epoch:
        exit(0)

    if args.eval_only:
        print("Eval-only".center(200, "-"))

        eval_times = 1 if not args.repeat_eval else args.repeat_eval

        repeat_stats = defaultdict(list)
        for i in range(eval_times):
            test_stats, results = validate(args, val_loader, model, device, criterion_dict,
                                           start_epoch, train_writer, log_output)
            if utils.is_main_process():
                for k, v in test_stats.items():
                    train_writer.add_scalar(f'val/{k}', v, i)
                    repeat_stats[k].append(v)
                train_writer.flush()
        if args.repeat_eval and utils.is_main_process():
            for k, v in repeat_stats.items():
                v = np.array(v)
                print(
                    f'{k}, {v.mean():.3f} +- {v.std():.3f}, {v.min():.3f}->{v.max():.3f}')

        if args.save_results:
            from utils import dump_orjson
            assert args.resume or args.outpath
            args.outpath = args.resume + ".json" if not args.outpath else args.outpath
            dump_orjson(results, args.outpath)
            logger.info(f"Dump the eval results to {args.outpath}")
        exit(0)

    tea_net = None
    if args.tea:
        print(f"Use the teacher model : {args.tea}")
        args.prune_loc = model_without_ddp.prune_loc
        tea_net = get_modified_teacher_model(args)
        tea_net.to(device)
        tea_net = tea_net.eval()

    table_file = os.path.join(log_dir, "table.pkl")
    table = defaultdict(list)
    if os.path.exists(table_file):
        table = pickle.load(table_file)

    for epoch in range(start_epoch, stop_epoch):

        # utils.unfreeze_model_if_needed(model_without_ddp, epoch, args)
        if args.token_prune and args.backbone_freeze_epochs > 0:
            for param_group in optimizer.param_groups:
                if param_group['backbone']:
                    if epoch < args.backbone_freeze_epochs:
                        param_group['ori_lr'] = param_group["lr"] if 'ori_lr' not in param_group else param_group['ori_lr']
                        param_group["lr"] = 0.
                    else:
                        param_group["lr"] = param_group["ori_lr"]
                    break

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(device, train_loader, args, model, criterion_dict, optimizer,  epoch,
              train_writer, log_output, scheduler, tea_net=tea_net, mixup_fn=mixup_fn, loss_scaler=loss_scaler, model_ema=model_ema)

        # evaluate on validation set
        scheduler.step(epoch)

        test_stats, _ = validate(args, val_loader, model, device, criterion_dict,
                                 epoch, train_writer, log_output)
        torch.cuda.empty_cache()

        if utils.is_main_process():
            table['epoch'].append(epoch)
            for k, v in test_stats.items():
                table[k].append(v)
                train_writer.add_scalar(f'val/{k}', v, epoch)
            train_writer.flush()
        if utils.is_main_process():

            save_info = {
                'net': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                "metrics": test_stats,
                "scheduler": scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
            }
            utils.save_checkpoint(log_model_dir, save_info, 'latest')
            utils.save_checkpoint(log_model_dir, save_info,
                                  'epoch_{}'.format(epoch+1))
    if utils.is_main_process():

        try:

            table = pd.DataFrame(table)
            table = table.sort_values(
                by=['acc1'], ascending=False)[:5]
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(table)
            pickle.dump(table, table_file)
        except:
            pass
    exit(0)


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
