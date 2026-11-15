#!/bin/bash
set -u
cd /data/ljh/RKD
echo "=== processes ==="
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'experiments.matrix|download_remaining|fetch_tgb|fetch_part' | grep -v grep
echo "=== wiki matrix ==="
tail -4 results/main-wiki.log
echo "completed_results=$(find results/main -name result.json 2>/dev/null | wc -l) / 141"
echo "current=$(grep -o 'Run [0-9]*/705:.*' results/main-wiki.log | tail -1)"
echo "=== downloads ==="
tail -8 results/download-remaining.log
echo "=== dataset sizes ==="
du -sh datasets/* 2>/dev/null
echo "=== disk ==="
df -h /data | tail -1
