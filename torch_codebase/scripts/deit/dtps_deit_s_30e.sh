#!/bin/bash

n_proc=8

deit_aug_settings="--input-size 224 --reprob 0.25 --smoothing 0.1 --mixup .8 --cutmix 1.0 --repeated-aug --color-jitter 0.3  --aa rand-m9-mstd0.5-inc1"
dynamic_prune_lr_sch_opt_loss_settings="--batch-size 1024  --lr 2.5e-4 --unscale-lr  --epochs 30 --weight-decay 0.05 --sched cosine   --opt adamw  --warmup-lr 1e-6 --warmup-epochs 5 --backbone_freeze_epochs -1 --backbone_lr_scale 1e-2"

train_setting="${deit_aug_settings} ${dynamic_prune_lr_sch_opt_loss_settings} --token_prune --tea deit_small_patch16_224 --prune_metric distill_kl_loss   --pretrain --axu_loss KeepRatioLoss DistillKLLoss  --axu_w 2 0.5"
prefix=$(basename $0) # 当前shell脚本的文件名
rerun_num=1
keep_ratio_list=(0.7)
prune_loc_list=([3,6,9])


arch_list=(dynamic_deit_small_patch16_224_token_merge_softmax_reweight)

for prune_loc in ${prune_loc_list[@]};
do 
    for keep_ratio in ${keep_ratio_list[@]};
    do  
        for arch in ${arch_list[@]};
        do 
            for rerun_i in $(seq 1 $rerun_num);
            do     
                prune_loc_str="prune${prune_loc//\[/}"
                prune_loc_str="${prune_loc_str//\]/}"
                prune_loc_str="${prune_loc_str//,/}"
                task_name="${USER}-${prefix}-${arch}-${keep_ratio}-${prune_loc_str}-rerun${rerun_i}"
                echo ${task_name} ;

                while : ; do
                    python3 -m torch.distributed.launch --nproc_per_node=${n_proc} --use_env train.py --data_path /data/datasets/imagenet1k \
                    --grad_checkpoint \
                    --fp16  -c  -dist-eval --pin-mem  --dataset ImageNet \
                    --arch $arch ${train_setting} \
                    --arch_kwargs "dict( prune_loc=${prune_loc},keep_ratio=${keep_ratio} )" \
                    --task_name ${task_name} --seed ${rerun_i};
                # exit while exit code == 0
                [[ $? != 0 ]] || break
                done
            

            done
        done
        

    done
done
