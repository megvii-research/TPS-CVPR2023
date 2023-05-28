
keep_ratio=0.5
prune_loc=[3,6,9]

data_cfg="--input-size 224 --reprob 0.25 --smoothing 0.1 --mixup .8 --cutmix 1.0 --repeated-aug --color-jitter 0.3  --aa rand-m9-mstd0.5-inc1"
learning_cfg="--batch_size 128  --lr 2.5e-4  --epochs 30 --decay 0.05  --warmup-lr 1e-6 --warmup-epochs 5 --backbone_lr_scale 1e-2"
loss_cfg=' --loss_fns SoftTargetCrossEntropy PruningLoss DistillKLLoss  --loss_ws 1. 2 0.5 --tea deit_small_patch16_224'


prune_loc_str="prune${prune_loc//\[/}"
prune_loc_str="${prune_loc_str//\]/}"
prune_loc_str="${prune_loc_str//,/}"
task_name=dtps_deit_s_${keep_ratio}_${prune_loc_str}

export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_PATH=/data/.cuda_cache


export NCCL_IB_DISABLE=1
python3 train.py --desc ${task_name} \
    --arch dtps_deit_small_patch16_224 --fp16 --workers 4 \
    ${data_cfg} ${learning_cfg} ${loss_cfg} \
    --arch_kwargs "dict( prune_loc=${prune_loc},keep_ratio=${keep_ratio})" \
    