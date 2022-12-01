#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python ../../tools/train.py "../../configs/cgps/cuhk.py"  --resume-from  "./work_dirs/cuhk/latest.pth"