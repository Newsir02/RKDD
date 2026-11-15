#!/bin/bash
set -u
cd /data/ljh/RKD
PID=$(cat results/ablation-trd.pid 2>/dev/null || echo 0)
while ps -p "$PID" > /dev/null 2>&1; do
  sleep 120
done
mkdir -p results/ablation-capacity
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python run_capacity_ablation.py \
  > results/ablation-capacity.log 2>&1 &
echo $! > results/ablation-capacity.pid
echo "capacity started pid=$(cat results/ablation-capacity.pid)"
