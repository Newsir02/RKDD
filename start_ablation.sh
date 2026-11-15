#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/ablation-trd
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python run_trd_ablation.py \
  > results/ablation-trd.log 2>&1 &
echo $! > results/ablation-trd.pid
echo "ablation pid=$(cat results/ablation-trd.pid)"
