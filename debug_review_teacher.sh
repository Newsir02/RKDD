#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/review-debug
/home/ubuntu/anaconda3/envs/Graph/bin/python benchmark.py \
  --stage teacher --dataset tgbl-review --data-root datasets --no-tgb-download \
  --teacher-arch tgn --device cuda --teacher-epochs 1 --student-epochs 1 \
  --patience 0 --seed 42 --output-dir results/review-debug/teacher-tgn \
  > results/review-debug/teacher-tgn.log 2>&1
echo "REVIEW_DEBUG_EXIT=$?"
tail -20 results/review-debug/teacher-tgn.log
