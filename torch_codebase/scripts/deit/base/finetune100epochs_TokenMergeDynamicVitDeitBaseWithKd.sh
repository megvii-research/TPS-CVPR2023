#!/bin/bash

while1(){
        command=$@
        if [[ $command == make\ * ]]; then
                command=`$@ -n`
                echo $command
        fi
        while true;do
                        eval $command
        if [ $? = 233 ];then
                        break
        fi
                        sleep 10
        done
}

charged_group=is_biometric
n_proc=8

preemptible=best-effort
envs="OMP_NUM_THREADS=1"



# --arch_kwargs "dict( prune_loc=[3,6,9],keep_ratio=${keep_ratio} )"

deit_aug_settings="--input-size 224 --reprob 0.25 --smoothing 0.1 --mixup .8 --cutmix 1.0 --repeated-aug --color-jitter 0.3  --aa rand-m9-mstd0.5-inc1"
deit_lr_sch_opt_loss_settings="--batch-size 1024 --lr 1e-3 --unscale-lr --epochs 300 --weight-decay 0.05 --sched cosine --model-ema   --opt adamw  --warmup-lr 1e-6 --warmup-epochs 5"
dynamic_prune_lr_sch_opt_loss_settings="--batch-size 512  --lr 1.25e-4 --unscale-lr  --epochs 100 --weight-decay 0.05 --sched cosine   --opt adamw  --warmup-lr 1e-6 --warmup-epochs 5 --backbone_freeze_epochs -1 --backbone_lr_scale 1e-2"
# 不确定dynamicViT用了没
deit_net_settings="--arch_kwargs 'dict( drop_rate=0, attn_drop_rate=0., drop_path_rate=0.1, )'"


# 无蒸馏的训练settings
deit1_dynamicViT_args="${deit_aug_settings} ${dynamic_prune_lr_sch_opt_loss_settings} --token_prune   --pretrain --axu_loss KeepRatioLoss  --axu_w 2"
# 有蒸馏的训练settings , 但是在deit tiny上效果不佳
deit1_dynamicViT_distill_args="${deit_aug_settings} ${dynamic_prune_lr_sch_opt_loss_settings} --token_prune --tea deit_small_patch16_224 --prune_metric distill_kl_loss   --pretrain --axu_loss KeepRatioLoss DistillKLLoss  --axu_w 2 0.5"

deit1_dynamicViT_base_distill_args="${deit_aug_settings} ${dynamic_prune_lr_sch_opt_loss_settings} --token_prune --tea deit_base_patch16_224 --prune_metric distill_kl_loss   --pretrain --axu_loss KeepRatioLoss DistillKLLoss  --axu_w 2 0.5"



# grid-experiments' hyper-parameters
prefix=$(basename $0) 
rerun_num=1
keep_ratio_list=(0.7 0.5)
prune_loc_list=([3,6,9] [2,4,6,8] [3,5,7,9])


arch_list_small=(dynamic_deit_base_patch16_224_token_merge_softmax_reweight)
# arch_list_small=(dynamic_deit_small_patch16_224_token_merge_use_dots_softmax_reweight)
for prune_loc in ${prune_loc_list[@]};
do 
    for keep_ratio in ${keep_ratio_list[@]};
    do  
        for arch in ${arch_list_small[@]};
        do 
            for rerun_i in $(seq 1 $rerun_num);
            do     
                prune_loc_str="prune${prune_loc//\[/}"
                prune_loc_str="${prune_loc_str//\]/}"
                prune_loc_str="${prune_loc_str//,/}"
                task_name="${USER}-${prefix}-${arch}-${keep_ratio}-${prune_loc_str}-rerun${rerun_i}"
                echo ${task_name} ;

                while : ; do
                    CUDA_HOME=/data/cuda/cuda-10.2/cuda/ rlaunch --charged-group=${charged_group} --preemptible=${preemptible}  --positive-tags 2080ti --cpu=48 --gpu=8 --memory=80000 \
                    -- python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
                    --fp16  -c  -dist-eval --pin-mem  --dataset ImageNet \
                    --grad_checkpoint --grad_checkpoint_layers 0 1 2 3 4 \
                    --arch $arch ${deit1_dynamicViT_base_distill_args} \
                    --arch_kwargs "dict( prune_loc=${prune_loc},keep_ratio=${keep_ratio} )" \
                    --task_name ${task_name};
                # exit while exit code == 0
                [[ $? != 0 ]] || break
                done
            

            done
        done
        

    done
done

