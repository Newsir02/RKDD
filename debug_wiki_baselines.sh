#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/ubuntu/anaconda3/envs/Graph/bin/python
for method in dkd rkd_distance rkd_angle logit_standard nosmog_rsd; do
  echo "=== $method ==="
  $PY benchmark.py --stage distill --dataset tgbl-wiki --data-root datasets --no-tgb-download \
    --teacher-arch tgn --student-model gcn --method "$method" --device cuda \
    --teacher-epochs 1 --student-epochs 1 --patience 0 --seed 42 \
    --teacher-checkpoint results/wiki-debug/teacher-tgn/teacher.pt \
    --output-dir "results/wiki-debug/${method}-gcn" \
    > "results/wiki-debug/${method}-gcn.log" 2>&1
  echo "EXIT=$?"
  tail -2 "results/wiki-debug/${method}-gcn.log"
done
