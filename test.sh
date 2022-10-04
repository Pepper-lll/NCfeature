#!/bin/bash

path_to_ckpt='saved/models/ImageNet_LT_ResNeXt50_RIDE/1003_082230/model_best.pth'
python test.py -r $path_to_ckpt -c '/comp_robot/mm_generative/code/ride/configs/config_imagenet_lt_resnext50_ride.json'

path_to_ckpt='saved/models/ImageNet_LT_ResNeXt50_RIDE/1003_132602/model_best.pth'
python test.py -r $path_to_ckpt -c 'configs/config_imagenet_lt_resnext50_ride_ea.json'
