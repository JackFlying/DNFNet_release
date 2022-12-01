#!/bin/bash

config_name="prw"
config_path="../../configs/cgps/${config_name}.py" 

python -u ../../tools/train.py "../../configs/cgps/prw.py"  >train_log.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py "../../configs/cgps/prw.py" --resume-from  "./work_dirs/prw/epoch_15.pth"