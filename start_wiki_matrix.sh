#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
mkdir -p results
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.matrix \
  --config configs/comparison.json --output-root results/main --execute --max-runs 141 \
  > results/main-wiki.log 2>&1 &
echo $! > results/main-wiki.pid
echo "started pid=$(cat results/main-wiki.pid)"
sleep 5
ps -p "$(cat results/main-wiki.pid)" -o pid,etime,cmd
head -3 results/main-wiki.log
