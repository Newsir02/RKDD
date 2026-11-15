#!/bin/bash
set -u
cd /data/ljh/RKD
nohup bash start_capacity_after_ablation.sh \
  > results/capacity-watcher.log 2>&1 &
echo $! > results/capacity-watcher.pid
echo "capacity watcher pid=$(cat results/capacity-watcher.pid)"
