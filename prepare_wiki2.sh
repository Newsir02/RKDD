#!/bin/bash
set -u
cd /data/ljh/RKD
rm -f datasets/tgbl_wiki/tgbl-wiki.zip
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export PYTHONUNBUFFERED=1
/home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.prepare_data \
  --datasets tgbl-wiki --data-root datasets \
  --output results/wiki-data.json > results/wiki-prepare.log 2>&1
echo "PREPARE_EXIT=$?"
tail -20 results/wiki-prepare.log
echo "--- manifest ---"
cat results/wiki-data.json 2>/dev/null
echo "--- data dir ---"
ls -lh datasets/tgbl_wiki | head -20
