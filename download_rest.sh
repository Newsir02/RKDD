#!/bin/bash
set -u
cd /data/ljh/RKD
COIN_PID=$(cat results/download-coin.pid 2>/dev/null || echo 0)
echo "waiting for coin pid=$COIN_PID at $(date -Is)" >> results/download-rest.log
while ps -p "$COIN_PID" > /dev/null 2>&1; do sleep 120; done
echo "coin download ended at $(date -Is)" >> results/download-rest.log
/home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-comment.zip datasets/tgbl_comment --jobs 16 >> results/download-rest.log 2>&1
echo "comment exit=$?" >> results/download-rest.log
/home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-flight-v2.zip datasets/tgbl_flight --jobs 16 >> results/download-rest.log 2>&1
echo "flight exit=$?" >> results/download-rest.log
echo "ALL_REST_DONE $(date -Is)" >> results/download-rest.log
