#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  \
#     train.py \
#     -c "configs/config_imagenet_lt_resnext50_ride_ea.json" \
#     -r "ckpt/ImageNet_LT_ResNeXt50_RIDE/1002_203019/checkpoint-epoch100.pth" \
#     --reduce_dimension 1 \
#     --num_experts 3

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  \
    train.py \
    -c "configs/config_imagenet_lt_resnext50_ride_ea.json" \
    -r "saved/models/ImageNet_LT_ResNeXt50_RIDE/1003_082230/model_best.pth" \
    --reduce_dimension 1 \
    --num_experts 3