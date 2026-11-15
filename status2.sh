#!/bin/bash
set -u
cd /data/ljh/RKD
echo "=== processes ==="
pgrep -af 'experiments.matrix' || true
pgrep -af 'download_tgb.py' || true
echo "=== wiki matrix ==="
tail -3 results/main-wiki.log
/home/ubuntu/anaconda3/envs/Graph/bin/python list_results.py
echo "=== coin download ==="
tail -5 results/download-coin.log 2>/dev/null || true
echo "=== data ==="
du -sh datasets/* 2>/dev/null
df -h /data | tail -1
