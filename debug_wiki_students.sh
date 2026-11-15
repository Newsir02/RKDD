#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
PY=/home/ubuntu/anaconda3/envs/Graph/bin/python
common="--dataset tgbl-wiki --data-root datasets --no-tgb-download --teacher-arch tgn --device cuda --teacher-epochs 1 --student-epochs 1 --patience 0 --seed 42"

run_one () {
  name="$1"; shift
  echo "=== $name ==="
  $PY benchmark.py "$@" > "results/wiki-debug/${name}.log" 2>&1
  echo "EXIT=$?"
  tail -4 "results/wiki-debug/${name}.log"
}

run_one supervised-gcn --stage student --method supervised --student-model gcn $common \
  --output-dir results/wiki-debug/supervised-gcn
run_one logits-gcn --stage distill --method logits --student-model gcn $common \
  --teacher-checkpoint results/wiki-debug/teacher-tgn/teacher.pt \
  --output-dir results/wiki-debug/logits-gcn
run_one trd-gcn --stage distill --method trd --student-model gcn $common \
  --teacher-checkpoint results/wiki-debug/teacher-tgn/teacher.pt \
  --output-dir results/wiki-debug/trd-gcn
run_one trd-sage --stage distill --method trd --student-model sage $common \
  --teacher-checkpoint results/wiki-debug/teacher-tgn/teacher.pt \
  --output-dir results/wiki-debug/trd-sage
