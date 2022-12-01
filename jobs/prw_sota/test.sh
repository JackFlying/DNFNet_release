#!/bin/bash
../../tools/dist_test.sh ./work_dirs/prw/prw.py ./work_dirs/prw/latest.pth 1 --out results_1000.pkl >log_tmp.txt 2>&1 
python ../../tools/test_results_prw.py >result.txt 2>&1