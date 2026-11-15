#!/bin/bash
set -u
echo "=== relevant processes ==="
ps -eo pid,ppid,etime,cmd | grep -E 'finalize_wiki|download_rest|experiments.matrix|download_tgb.py' | grep -v grep || true
echo "=== finalize log ==="
cat /data/ljh/RKD/results/finalize-wiki.log
echo "=== download rest log ==="
cat /data/ljh/RKD/results/download-rest.log
echo "=== markers ==="
ls -l /data/ljh/RKD/results/WIKI_DONE.txt 2>&1 || true
