#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-flight --data-root datasets \
  --output results/flight-data.json > results/flight-prepare.log 2>&1 &
echo $! > results/flight-prepare.pid
echo "flight prepare pid=$(cat results/flight-prepare.pid)"
