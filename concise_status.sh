#!/bin/bash
set -u
cd /data/ljh/RKD
echo "=== matrix ==="
grep 'Run ' results/main-wiki.log | tail -n 1
echo "=== completed ==="
/home/ubuntu/anaconda3/envs/Graph/bin/python list_results.py | tail -n 12
echo "=== downloads ==="
grep -E '^(tgbl-|coin exit|comment exit|flight exit|ALL_DONE)' results/download-all-fixed.log | tail -n 8
echo "=== data ==="
du -sh datasets/* 2>/dev/null
df -h /data | tail -1
