This repo is the official implementation of the CVPR2023 paper: Joint Token Pruning and Squeezing Towards More Aggressive Compression of
Vision Transformers. 

# [Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers]
## Framework & Comparison
<div style="text-align:center"><img src="pics/main.png" width="100%" ></div>

## Requirements

```
conda env create -f environment.yml

```

## Training & Evaluation


train dTPS-DeiT-small on a 8-gpu machine, you can modify hyperparams, including the location index of pruned layers and token keep ratio in the .sh scripts.

'''
bash scripts/finetune_dtps_deit_s.sh
'''


## Liscense
TPS-CVPR2023 is released under the Apache 2.0 license. See [LICENSE](LICENSE) for details.