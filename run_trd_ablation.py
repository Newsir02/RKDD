#!/usr/bin/env python3
"""Run the TRD ablation suite on Wiki using the shared seed-42 TGN teacher."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import benchmark

BASE = Path("results/main/tgbl-wiki/seed-42/gcn/tgn/trd/job.json")
ROOT = Path("results/ablation-trd/tgbl-wiki/seed-42")
CONFIG_DIR = Path("results/ablation-trd/configs")
SUMMARY = Path("results/ablation-trd/ablation-summary.csv")

VARIANTS = [
    ("gcn", "trd-static", {"trd_mode": "static"}),
    ("gcn", "trd-shuffled", {"trd_mode": "shuffled"}),
    ("gcn", "trd-reverse", {"trd_mode": "reverse"}),
    ("gcn", "trd-zero", {"trd_mode": "zero"}),
    ("gcn", "trd-kd0", {"trd_mode": "temporal", "kd_weight": 0.0}),
    ("gcn", "trd-rel0", {"trd_mode": "temporal", "relation_weight": 0.0}),
    ("sage", "trd-temporal", {"trd_mode": "temporal"}),
    ("sage", "trd-static", {"trd_mode": "static"}),
    ("sage", "trd-shuffled", {"trd_mode": "shuffled"}),
    ("sage", "trd-reverse", {"trd_mode": "reverse"}),
    ("sage", "trd-zero", {"trd_mode": "zero"}),
    ("sage", "trd-kd0", {"trd_mode": "temporal", "kd_weight": 0.0}),
    ("sage", "trd-rel0", {"trd_mode": "temporal", "relation_weight": 0.0}),
]


def write_summary() -> None:
    rows = []
    for path in sorted(ROOT.rglob("result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "student": r["student"], "method": r["method"], "seed": r["seed"],
            "test_mrr": r["test"]["mrr"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_seconds": round(r["train_seconds"], 1),
            "parameters": r["parameters"]["trainable"],
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
    for student, name, overrides in VARIANTS:
        output = ROOT / student / name
        result_file = output / "result.json"
        if result_file.exists():
            print(f"SKIP complete {output}", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"incomplete run directory exists: {output}")
        config = dict(base)
        config.update(overrides)
        config["student_model"] = student
        config["output_dir"] = str(output)
        config_path = CONFIG_DIR / f"{student}-{name}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        args = benchmark.parse_args(["--config", str(config_path)])
        print(f"RUN {student} {name}", flush=True)
        result = benchmark.run(args)
        print(f"DONE {student} {name} test_mrr={result['test']['mrr']:.6f} "
              f"epochs={result['epochs_run']} best={result['best_epoch']}", flush=True)
        write_summary()
    write_summary()
    print("ABLATION_SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
