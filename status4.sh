#!/bin/bash
set -u
cd /data/ljh/RKD
echo "=== main ==="
grep 'Run ' results/main-wiki.log | tail -n 1
echo "completed=$(find results/main -name result.json | wc -l)"
echo "=== ablation ==="
grep -E '^(RUN|DONE|ABLATION)' results/ablation-trd.log | tail -n 4
echo "completed=$(find results/ablation-trd -name result.json | wc -l)"
tail -n 2 results/ablation-trd.log
echo "=== capacity watcher ==="
ps -p "$(cat results/capacity-watcher.pid)" -o pid,etime,cmd 2>/dev/null || true
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
