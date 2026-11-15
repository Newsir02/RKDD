#!/bin/bash
set -u
cd /data/ljh/RKD
echo "--- files >5MB (excluding data/deps) ---"
find . \
  -path ./datasets -prune -o \
  -path ./.benchmark-deps -prune -o \
  -path ./.test-deps -prune -o \
  -path ./.download-test -prune -o \
  -path ./.pip-cache -prune -o \
  -path ./.pytest-tmp -prune -o \
  -path ./.download-parts -prune -o \
  -type f -size +5M -printf '%s %p\n' | sort -nr | head -20
echo "--- metadata size ---"
find results -type f ! -name '*.pt' -printf '%s\n' | awk '{s+=$1} END {printf "%.1f MB\n", s/1024/1024}'
echo "--- metadata files ---"
find results -type f ! -name '*.pt' | wc -l
echo "--- total non-data size ---"
du -sh --exclude=datasets --exclude=.benchmark-deps --exclude=.test-deps --exclude=.download-test --exclude=.pip-cache --exclude=.pytest-tmp . 2>/dev/null
