from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

from benchmark import TEACHER_FIELDS, fingerprint


def summarize(root, expected_seeds=(42, 43, 44)):
    groups, teacher_hashes = defaultdict(list), {}
    for path in sorted(Path(root).rglob("result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete":
            continue
        cfg = dict(row["config"])
        for field in ("seed", "student_seed", "output_dir", "teacher_checkpoint", "config"):
            cfg.pop(field, None)
        key = (row["dataset"], row["teacher"] if row["method"] != "supervised" else "none",
               row["student"], row["method"], row["test"]["metric"], row["test"]["official"],
               fingerprint(cfg), row["source_hash"], fingerprint(row["environment"]),
               row["data_signature"] if row["dataset"] != "synthetic" else "seed-generated-synthetic",
               row.get("dataset_version"))
        if row["teacher_checkpoint_sha256"]:
            # Include teacher contract fields so separate legitimate budgets do not collide.
            teacher_cfg = {k: row["config"][k] for k in TEACHER_FIELDS}
            teacher_key = (fingerprint(teacher_cfg), row["data_signature"], row["protocol"])
            previous = teacher_hashes.setdefault(teacher_key, row["teacher_checkpoint_sha256"])
            if previous != row["teacher_checkpoint_sha256"]:
                raise ValueError("Different teacher checkpoints used for comparable student runs")
        groups[key].append(row)
    summary = []
    for key, rows in groups.items():
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed results for {key[:4]}")
        values = [r["test"][key[4]] for r in rows]
        summary.append(dict(dataset=key[0], teacher=key[1], student=key[2], method=key[3], metric=key[4],
                            structure_mode=(rows[0]["config"].get("structure_mode", "temporal")
                                            if key[3] == "structure_interaction" else None),
                            official=key[5], configuration=key[6], source_hash=key[7],
                            data_signature=key[9], dataset_version=key[10],
                            n=len(rows), seeds=",".join(map(str, sorted(seeds))),
                            status="complete" if set(seeds) == set(expected_seeds) else "incomplete",
                            mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None,
                            parameters=statistics.mean(r["parameters"]["trainable"] for r in rows),
                            train_seconds=statistics.mean(r["train_seconds"] for r in rows),
                            query_ms=statistics.mean(r["latency"]["query_ms"] for r in rows),
                            cuda_peak_bytes=(statistics.mean(r["cuda_peak_allocated_bytes"] for r in rows)
                                             if all(r["cuda_peak_allocated_bytes"] is not None for r in rows) else None)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output", default="summary.csv")
    args = parser.parse_args(argv)
    rows = summarize(args.root, tuple(map(int, args.seeds.split(","))))
    if not rows:
        raise ValueError("No completed run results found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} groups; {sum(r['status'] != 'complete' for r in rows)} incomplete")


if __name__ == "__main__":
    main()
