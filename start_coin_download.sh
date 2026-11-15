#!/bin/bash
set -u
cd /data/ljh/RKD
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-coin-v2.zip datasets/tgbl_coin --jobs 16 \
  > results/download-coin.log 2>&1 &
echo $! > results/download-coin.pid
echo "coin download pid=$(cat results/download-coin.pid)"
