#!/bin/bash
set -u
cd /data/ljh/RKD
mkdir -p .pip-tmp .pytest-tmp .test-deps results
export TMPDIR=/data/ljh/RKD/.pip-tmp
export PIP_CACHE_DIR=/data/ljh/RKD/.pip-cache
/home/ubuntu/anaconda3/envs/Graph/bin/python -m pip install \
  --target /data/ljh/RKD/.test-deps --no-cache-dir "pytest>=7,<10" \
  > results/pytest-install.log 2>&1
echo "PIP_EXIT=$?"
tail -3 results/pytest-install.log
