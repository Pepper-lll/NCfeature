#!/bin/bash

# python train.py \
#     -c "configs/config_imagenet_lt_resnext50_ride.json" \
#     --reduce_dimension 1 \
#     --num_experts 3


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  \
    train.py \
    -c "configs/config_imagenet_lt_resnext50_ride.json" \
    --reduce_dimension 1 \
    --num_experts 3