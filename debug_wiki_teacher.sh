#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
mkdir -p results/wiki-debug
/home/ubuntu/anaconda3/envs/Graph/bin/python benchmark.py \
  --stage teacher --dataset tgbl-wiki --data-root datasets --no-tgb-download \
  --teacher-arch tgn --device cuda --teacher-epochs 1 --student-epochs 1 \
  --patience 0 --seed 42 --output-dir results/wiki-debug/teacher-tgn \
  > results/wiki-debug/teacher-tgn.log 2>&1
echo "DEBUG_EXIT=$?"
tail -30 results/wiki-debug/teacher-tgn.log
echo "--- result ---"
cat results/wiki-debug/teacher-tgn/result.json 2>/dev/null | head -c 3000
