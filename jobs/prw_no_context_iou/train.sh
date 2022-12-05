#!/bin/bash

config_name="prw"
config_path="../../configs/cgps/${config_name}.py" 

python -u ../../tools/train.py "../../configs/cgps/prw.py"  >train_log.txt 2>&1

CUDA_VISIBLE_DEVICES=7 python ../../tools/train.py "../../configs/cgps/prw.py" --resume-from  "/home/linhuadong/CGPS/jobs/prw_part_IoU_cls3_0p5/work_dirs/prw/epoch_34.pth"