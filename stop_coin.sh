#!/bin/bash
set -u
kill 1351850 1359434 2>/dev/null || true
sleep 2
pkill -f 'download_tgb.py tgbl-coin' 2>/dev/null || true
pkill -f 'download_rest.sh' 2>/dev/null || true
sleep 2
ps -eo pid,ppid,cmd | grep -E 'download_tgb.py|download_rest.sh' | grep -v grep || true
echo "coin/watcher stopped"
