import argparse


def get_args():

    # NOTE training configs
    parser = argparse.ArgumentParser()
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--task_name', default='',
                        help='the clearml task name will be used if the task name is not empty, otherwise the taskname will be automatically generated based on args')
    parser.add_argument('--grad_checkpoint', action='store_true',
                        help="use grad_checkpoint to decrease the GPU mem, https://pytorch.org/docs/stable/checkpoint.html?highlight=checkpoint")
    parser.add_argument('--grad_checkpoint_layers',  type=int, nargs='*',
                        help='grad_checkpoint layers ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--persistent_workers', action='store_true',
                        default=False, help='If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once')
    parser.add_argument('--eval-only', action='store_true',
                        default=False, help='Only eval the model')

    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--repeat-eval', default=None, type=int,
                        help='times to repeat eval for robustness experiments')
    parser.add_argument('--eval-benchmarks',
                        action='store_true', default=False)
    parser.add_argument('--ddp', default=True, type=bool, help='DDP training')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='whether to use the pretrained model to finetune the model')
    parser.add_argument('--dataset', '-d', default="ImageNet", type=str,
                        help='dataset name')
    parser.add_argument('--data_path',required=True, type=str,
                        help='dataset path for torch ImageFolder dataset')
    parser.add_argument('--save_results', action='store_true',
                        help='whether to save the validation results')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--batch-size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--epochs', '-e', default=200,
                        type=int, help='stop epoch')
    parser.add_argument('--workers', '-w', default=-1,
                        type=int, help='num of data loader workers')
    parser.add_argument('--resume', default="", type=str,
                        help='the checkpoint path for resuming training or evaluating')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # NOTE: hyperparams for loss functions
    parser.add_argument('--bce-loss', action='store_true')

    # NOTE: for model EMA
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # for KD prune training
    parser.add_argument('--token_prune', action='store_true', default=False,
                        help='whether to use token_prune training')
    parser.add_argument('--prune_metric', default="", choices=["token_att_layer_sim", "token_attn_layer_var", 'distill_kl_loss', 'distill_attn_kl_loss'],
                        help='the metric from the pretrained teacher model to evaluate the token prune probability')
    parser.add_argument('--axu_loss', type=str, nargs='*',
                        help='the axuillary loss for token pruning')
    parser.add_argument('--axu_w',  type=float, nargs='*',
                        help='the weight for controling the axuillary loss for token pruning ')
    parser.add_argument('--keep_ratio',  default=0.7, type=float,
                        help='the token keep ratio comparing with the current layer token num')
    parser.add_argument('--backbone_freeze_epochs', type=int, default=-1,
                        help='the epochs to fix the weights of the backbone model')
    parser.add_argument('--backbone_lr_scale', type=float, default=1e-2,
                        help='the learning rate scale for finetuning backbone')
    parser.add_argument('--tea', default="", type=str,
                        help='the teacher model architecture name')
    parser.add_argument('--linear_assign', action='store_true', default=False,
                        help='whether to use linear assigment to match heads between different layers')
    parser.add_argument('--distill_w',  default=0.5, type=float,
                        help='distill weight of dynamicViT')
    parser.add_argument('--keep_ratio_w',  default=2.0, type=float,
                        help='keep raito loss weight for dynamicViT')
    parser.add_argument('--mse_token_loss', action='store_true', default=False,
                        help='whether to use mse_token_loss in distill process of dynamicViT')
    parser.add_argument('--attn_distill_w',  default=0.5, type=float,
                        help='attn distill weight of dynamicViT')
    parser.add_argument('--attn_distill_layers', type=int, default=-1,
                        help='number of layers to add attn distill loss')

    # Optimizer parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate for a batch size of 4096 (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--arch', '-a', default="", type=str,
                        help='model type')
    parser.add_argument('--arch_kwargs', default="None", type=eval,
                        help='the key word args for the specific architecture model and it will be evaled as the python dict')

    parser.add_argument('--sampler', '-s', default="None", type=str,
                        help='sample method')
    parser.add_argument('--run_num', '-r', default='0',
                        type=str, help='running number')
    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    parser.add_argument('--sync_bn',
                        action='store_true', default=False)
    parser.add_argument('--cudnn_bmk',
                        action='store_true', default=True)

    # NOTE: data preprocess&augmentation  hyper-params
    # Augmentation parameters
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

    # NOTE: others
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # NOTE: for eval and analysis
    parser.add_argument('--outpath', '-o', default="", type=str,
                        help='the output path for preserving the results')
    args = parser.parse_args()

    # NOTE: 辅助loss与其对应的权重
    assert (not args.axu_loss and not args.axu_w) or (
        args.axu_loss and args.axu_w and len(args.axu_loss) == len(args.axu_w))

    return args


if __name__ == "__main__":
    print(get_args())
