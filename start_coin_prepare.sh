#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-coin --data-root datasets \
  --output results/coin-data.json > results/coin-prepare.log 2>&1 &
echo $! > results/coin-prepare.pid
echo "coin prepare pid=$(cat results/coin-prepare.pid)"
