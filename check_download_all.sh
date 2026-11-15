#!/bin/bash
set -u
cd /data/ljh/RKD
ps -eo pid,ppid,etime,cmd | grep -E 'download_all_fixed|download_tgb.py' | grep -v grep || true
echo "--- log ---"
tail -n 10 results/download-all-fixed.log 2>&1 || true
echo "--- dirs ---"
du -sh datasets/* 2>/dev/null
