#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
/home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-review --data-root datasets \
  --output results/review-data.json > results/review-prepare.log 2>&1
echo "PREPARE_EXIT=$?"
tail -20 results/review-prepare.log
cat results/review-data.json 2>/dev/null
