#!/bin/bash
set -u
cd /data/ljh/RKD
echo "=== current matrix job ==="
grep 'Run ' results/main-wiki.log | tail -n 1
echo "=== current epoch logs ==="
for f in $(find results/main -name epochs.jsonl -newermt '-4 hours' 2>/dev/null); do
  d=$(dirname "$f")
  echo "-- $d --"
  /home/ubuntu/anaconda3/envs/Graph/bin/python - "$f" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if rows:
    best = max(rows, key=lambda r: r["validation"].get("mrr", -1))
    print("epochs", len(rows), "best_epoch", best["epoch"], "best_val", best["validation"].get("mrr"), "last", rows[-1]["epoch"], rows[-1]["validation"].get("mrr"))
PY
done
echo "=== completed ==="
/home/ubuntu/anaconda3/envs/Graph/bin/python list_results.py | tail -n 1
echo "=== coin ==="
tail -n 3 results/download-coin.log
echo "=== review debug ==="
tail -n 3 results/review-debug/teacher-tgn.log
