#!/bin/bash
set -u
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
mkdir -p results datasets
/home/ubuntu/anaconda3/envs/Graph/bin/python - <<'PY'
from dataset.tgb_temporal import _resolve_tgb_root_argument
from tgb.utils.info import PROJ_DIR
print("PROJ_DIR:", PROJ_DIR)
print("resolved data root:", _resolve_tgb_root_argument("datasets"))
PY
/home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-wiki --data-root datasets --download \
  --output results/wiki-data.json > results/wiki-prepare.log 2>&1
echo "PREPARE_EXIT=$?"
tail -20 results/wiki-prepare.log
