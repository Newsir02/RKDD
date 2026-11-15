import json
from pathlib import Path

rows = []
for path in sorted(Path("results/main").rglob("result.json")):
    r = json.loads(path.read_text(encoding="utf-8"))
    rows.append((r["dataset"], r["teacher"], r["student"], r["method"], r["seed"],
                 r["test"]["mrr"], r["best_epoch"], r["epochs_run"],
                 round(r["train_seconds"], 1), r["test"]["official"],
                 r["cuda_peak_allocated_bytes"]))
print("dataset teacher student method seed test_mrr best_ep epochs train_s official peak_bytes")
for row in rows:
    print(*row)
print("completed", len(rows))
