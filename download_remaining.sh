#!/bin/bash
set -u
cd /data/ljh/RKD
LOG=/data/ljh/RKD/results/download-remaining.log
: > "$LOG"
for spec in \
  "tgbl-review-v2.zip /data/ljh/RKD/datasets/tgbl_review" \
  "tgbl-coin-v2.zip /data/ljh/RKD/datasets/tgbl_coin" \
  "tgbl-comment.zip /data/ljh/RKD/datasets/tgbl_comment" \
  "tgbl-flight-v2.zip /data/ljh/RKD/datasets/tgbl_flight"; do
  set -- $spec
  echo "=== START $(date -Is) $1 -> $2 ===" >> "$LOG"
  bash /data/ljh/RKD/fetch_tgb.sh "$1" "$2" >> "$LOG" 2>&1
  echo "=== END $(date -Is) $1 exit=$? ===" >> "$LOG"
done
echo "ALL_DOWNLOADS_DONE $(date -Is)" >> "$LOG"
