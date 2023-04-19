#!/bin/bash

config_name="prw"
config_path="../../configs/cgps/${config_name}.py" 

python -u ../../tools/train.py "../../configs/cgps/prw.py"  >train_log.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py "../../configs/dnfnet/prw.py" --resume-from  "/home/linhuadong/DNFNet/jobs/prw_dnfnet2_max_iou_update_cluster_sample/work_dirs/prw/epoch_28.pth"