#!/bin/bash

python train.py \
    -c "configs/config_imagenet_lt_resnext50_ride.json" \
    --reduce_dimension 1 \
    --num_experts 3