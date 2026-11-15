#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps:/data/ljh/RKD/.test-deps
export TMPDIR=/data/ljh/RKD/.pytest-tmp
/home/ubuntu/anaconda3/envs/Graph/bin/python -m pytest -q \
  --basetemp=/data/ljh/RKD/.pytest-tmp/run \
  > results/pytest-graph.log 2>&1
echo "PYTEST_EXIT=$?"
tail -15 results/pytest-graph.log
