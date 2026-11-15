#!/bin/bash
set -u
pkill -f '/data/ljh/RKD/download_remaining.sh' 2>/dev/null || true
pkill -f '/data/ljh/RKD/fetch_tgb.sh' 2>/dev/null || true
pkill -f '/data/ljh/RKD/fetch_part.sh' 2>/dev/null || true
pkill -f 'object-arbutus.alliancecan.ca' 2>/dev/null || true
sleep 2
ps -eo pid,cmd | grep -E 'download_remaining|fetch_tgb|fetch_part' | grep -v grep || true
echo "stopped"
