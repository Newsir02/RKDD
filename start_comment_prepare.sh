#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-comment --data-root datasets \
  --output results/comment-data.json > results/comment-prepare.log 2>&1 &
echo $! > results/comment-prepare.pid
echo "comment prepare pid=$(cat results/comment-prepare.pid)"
