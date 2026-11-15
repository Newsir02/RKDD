#!/bin/bash
set -u
pkill -f 'download_remaining.sh' 2>/dev/null || true
pkill -f 'fetch_tgb.sh' 2>/dev/null || true
pkill -f 'fetch_part.sh' 2>/dev/null || true
pkill -f 'object-arbutus.alliancecan.ca' 2>/dev/null || true
sleep 3
echo "remaining download processes:"
ps -eo pid,ppid,cmd | grep -E 'download_remaining|fetch_tgb|fetch_part|object-arbutus' | grep -v grep || true
