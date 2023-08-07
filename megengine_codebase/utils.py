import megengine as mge
import megengine.module as M
import numpy as np
from argparse import Namespace
from collections import defaultdict
import math
import pickle
from functools import reduce


def load_torch_weights(net: M.Module, f):
    with open(f, 'rb') as f:
        state_dict = pickle.load(f)
    for k, newv in state_dict.items():
        if k in net.state_dict():
            new_shape = newv.shape
            old_shape = net.state_dict()[k].shape
            new_psize = reduce(lambda a, b: a*b, new_shape)
            old_psize = reduce(lambda a, b: a*b, old_shape)
            if new_psize == old_psize and new_shape != old_shape:
                state_dict[k] = newv.reshape(*old_shape)
                print(f'reshape {k} shape to {old_shape}')
    net.load_state_dict(state_dict, strict=False)


def contain_bias_and_norm(name, param):
    if name.endswith('.bias'):  # bias
        return True
    elif param.ndim <= 1:
        return True
    else:
        return False


def get_param_groups(model: M.Module, args: Namespace, no_decay_bias_norm=False):

    score_head_params_ids = []
    backbone_lr_scale = float(args.backbone_lr_scale) if hasattr(
        model, "score_predictor") else 1
    if backbone_lr_scale != 1:
        score_head_params = []
        for m in model.score_predictor:
            score_head_params += m.parameters()
        score_head_params_ids = list(
            map(id, score_head_params))

    lr_wd_params = defaultdict(list)
    for name, param in model.named_parameters():

        lr = args.lr if id(
            param) in score_head_params_ids else args.lr * backbone_lr_scale
        wd = 0 if no_decay_bias_norm and contain_bias_and_norm(
            name, param) else args.decay
        lr_wd_params[(lr, wd)].append(param)

    param_groups = []
    for (lr, wd), params in lr_wd_params.items():
        print(f'{len(params)}, lr:{lr},weight decay : {wd}')
        param_groups.append(
            {
                'params': params, 'lr': lr, 'weight_decay': wd, 'baselr': lr,
            }
        )
    return param_groups


def adjust_learning_rate(optimizer: mge.optimizer.Optimizer, epoch: int, max_epoch: int, args: Namespace):

    # warmup + cosine learning rate, and rescale the backbone part learning rate if needed
    # the implement of cosine learning rate scheduler follows timm (https://github.com/huggingface/pytorch-image-models)
    warmup_epochs = int(args.warmup_epochs)
    lr_min = float(args.min_lr)
    warmup_lr_init = float(args.warmup_lr)
    cycle_decay = 0.1
    cycle_limit = 1

    k_decay = 1
    if warmup_epochs:
        warmup_steps = [
            (params['baselr'] - warmup_lr_init)/warmup_epochs
            for params in optimizer.param_groups
        ]
    # In dynamicViT and DeiT codebase: https://github.com/raoyongming/DynamicViT
    # they step the learning rate scheduler after per epoch training
    # and this leads to the learning rate being warmup_lr_init in the first 2 epochs
    if epoch < warmup_epochs:
        lrs = [warmup_lr_init + epoch * s for s in warmup_steps]
    else:
        i = epoch//max_epoch
        t_i = max_epoch
        t_curr = epoch - (max_epoch * i)

        gamma = cycle_decay ** i
        lr_max_values = [params['baselr'] *
                         gamma for params in optimizer.param_groups]
        k = k_decay

        if i < cycle_limit:
            lrs = [
                lr_min + 0.5 * (lr_max - lr_min) *
                (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                for lr_max in lr_max_values
            ]
        else:
            lrs = [lr_min for _ in optimizer.param_groups]

    for params, lr in zip(optimizer.param_groups, lrs):
        params['lr'] = lr
