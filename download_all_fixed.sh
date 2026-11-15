#!/bin/bash
set -u
cd /data/ljh/RKD
rm -rf datasets/tgbl_coin .download-parts/tgbl-coin-v2.zip
/home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-coin-v2.zip datasets/tgbl_coin --jobs 32
rc1=$?
echo "coin exit=$rc1"
/home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-comment.zip datasets/tgbl_comment --jobs 32
rc2=$?
echo "comment exit=$rc2"
/home/ubuntu/anaconda3/envs/Graph/bin/python download_tgb.py \
  tgbl-flight-v2.zip datasets/tgbl_flight --jobs 32
rc3=$?
echo "flight exit=$rc3"
echo "ALL_DONE coin=$rc1 comment=$rc2 flight=$rc3"
