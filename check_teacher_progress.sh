#!/bin/bash
set -u
cd /data/ljh/RKD
/home/ubuntu/anaconda3/envs/Graph/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/main/tgbl-wiki/seed-42/teacher-tgn/epochs.jsonl")
rows = [json.loads(line) for line in p.open()]
best = max(rows, key=lambda r: r["validation"]["mrr"])
print("epochs_logged", len(rows))
print("best_epoch", best["epoch"], "best_val_mrr", best["validation"]["mrr"])
print("last_epoch", rows[-1]["epoch"], "last_val_mrr", rows[-1]["validation"]["mrr"])
print("train_seconds_total", sum(r["train_seconds"] for r in rows))
PY
