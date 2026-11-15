#!/usr/bin/env python3
"""Student-capacity ablation for TRD on Wiki, sharing the seed-42 TGN teacher."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import benchmark

BASE = Path("results/main/tgbl-wiki/seed-42/gcn/tgn/trd/job.json")
ROOT = Path("results/ablation-capacity/tgbl-wiki/seed-42/gcn")
CONFIG_DIR = Path("results/ablation-capacity/configs")
SUMMARY = Path("results/ablation-capacity/capacity-summary.csv")

VARIANTS = [
    ("hidden-16", {"student_hidden": 16}),
    ("hidden-64", {"student_hidden": 64}),
    ("layers-1", {"student_blocks": 1}),
    ("layers-3", {"student_blocks": 3}),
    ("neighbors-4", {"student_neighbors": 4}),
    ("neighbors-16", {"student_neighbors": 16}),
]


def write_summary() -> None:
    rows = []
    for path in sorted(ROOT.rglob("result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "variant": path.parent.name, "seed": r["seed"],
            "test_mrr": r["test"]["mrr"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_seconds": round(r["train_seconds"], 1),
            "parameters": r["parameters"]["trainable"],
            "student_hidden": r["config"]["student_hidden"],
            "student_blocks": r["config"]["student_blocks"],
            "student_neighbors": r["config"]["student_neighbors"],
            "source_hash": r["source_hash"], "data_signature": r["data_signature"],
        })
    if not rows:
        return
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS:
        output = ROOT / name
        result_file = output / "result.json"
        if result_file.exists():
            print(f"SKIP complete {output}", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"incomplete run directory exists: {output}")
        config = dict(base)
        config.update(overrides)
        config["output_dir"] = str(output)
        config_path = CONFIG_DIR / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        args = benchmark.parse_args(["--config", str(config_path)])
        print(f"RUN {name}", flush=True)
        result = benchmark.run(args)
        print(f"DONE {name} test_mrr={result['test']['mrr']:.6f} "
              f"epochs={result['epochs_run']} best={result['best_epoch']}", flush=True)
        write_summary()
    write_summary()
    print("CAPACITY_SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
