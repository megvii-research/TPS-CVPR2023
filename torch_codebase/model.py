from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from models import *
import megfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import models
from timm.models import vision_transformer
from timm.models.helpers import load_pretrained
from losses import KeepRatioLoss
from utils import sum_cfgs
from copy import deepcopy
from losses.loss import get_loss_cls

def get(args):

    arch_kwargs = args.arch_kwargs
    arch_kwargs = arch_kwargs if arch_kwargs else {}

    ce_fn = LabelSmoothingCrossEntropy()
    if args.mixup_active:
        ce_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        ce_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        ce_fn = torch.nn.CrossEntropyLoss()
    if args.bce_loss:
        ce_fn = torch.nn.BCEWithLogitsLoss()
        
    net = timm.create_model(
        args.arch, pretrained=args.pretrain, **arch_kwargs)
    criterion_dict = {
        'ce': ce_fn,
    }

    if args.axu_loss:
        for axu_loss, axu_w in zip(args.axu_loss, args.axu_w):
            print(f"Axu loss : {axu_loss}, Weight : {axu_w}")
            loss_args = deepcopy(args)
            loss_args.axu_w = axu_w
            loss_cls = get_loss_cls(axu_loss)
            criterion_dict[axu_loss] = loss_cls(net, loss_args)
    return net, criterion_dict
