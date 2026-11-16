"""Measure teacher/student four-view interaction signals on a fixed checkpoint."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

import benchmark
import train
from dataset.benchmark_loader import replace_candidate_loaders
from loss.structure_interaction import interaction
from model.snapshot_gnn import SnapshotGNN
from util.structure_interaction_training import (
    pair_query, score_views, select_evidence, select_random_evidence,
)
from util.training import move_to_device


def summary(values):
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "energy": float(tensor.square().mean()),
        "nonzero_fraction": float((tensor.abs() > 1e-12).float().mean()),
        "positive_fraction": float((tensor > 0).float().mean()),
    }


def correlation(left, right):
    if len(left) < 2:
        return None
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    if a.std().item() == 0 or b.std().item() == 0:
        return None
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="A benchmark job.json")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--evidence", choices=("path", "random"), default="path")
    parser.add_argument("--reference", choices=("next", "random", "hard"), default="next")
    parsed = parser.parse_args()

    args = benchmark.parse_args(["--config", parsed.config])
    args.device = parsed.device
    device = train.resolve_device(args.device)
    bundle, training, _, _ = train.make_loaders(args)
    training, _, _ = replace_candidate_loaders(bundle, args)
    benchmark.normalize_train_time(bundle)
    signature = benchmark.data_fingerprint(args, bundle)
    teacher = train.load_teacher_from_checkpoint(args.teacher_checkpoint, bundle, signature, device)
    train.freeze_teacher(teacher)

    payload = train.load_model_checkpoint(parsed.student_checkpoint)
    config = payload["model_config"]
    student = SnapshotGNN(
        bundle.node_features, config["architecture"], config["hidden_dim"],
        config["num_layers"], config["neighbors"], config["use_node_ids"],
    ).to(device)
    student.load_state_dict(payload["model"], strict=True)
    student.eval()
    student.reset_state()
    teacher.reset_state()

    query_rng = random.Random(args.seed)
    evidence_rng = random.Random(args.seed + 1009)
    reference_rng = random.Random(args.seed + 917)
    records = []
    queries = eligible = selected = batches = 0
    for batch_index, raw in enumerate(training):
        if batch_index >= parsed.batches:
            break
        batch = move_to_device(raw, device)
        if parsed.evidence == "random":
            candidate_rows = list(select_random_evidence(
                student.neighbor_loader, teacher.neighbor_loader, batch,
                args.structure_group_events, evidence_rng))
        else:
            candidate_rows = list(select_evidence(
                student.neighbor_loader, teacher.neighbor_loader, batch,
                args.structure_group_events))
        chosen = query_rng.sample(
            candidate_rows, min(args.structure_queries_per_batch, len(candidate_rows)))
        teacher_scores = None
        if chosen and parsed.reference == "hard":
            with torch.no_grad():
                teacher_scores = teacher.score_candidates(batch, return_relations=False)["logits"]
        with torch.no_grad():
            for row, a, b in chosen:
                query = pair_query(batch, row, parsed.reference, reference_rng, teacher_scores)
                student_views = score_views(student, query, a, b)
                teacher_views = score_views(teacher.score_candidates, query, a, b)
                teacher_delta = interaction(teacher_views)
                student_delta = interaction(student_views)
                records.append({
                    "batch": batch_index,
                    "row": row,
                    "teacher_interaction": float(teacher_delta),
                    "student_interaction": float(student_delta),
                    "a_events": int(a.numel()),
                    "b_events": int(b.numel()),
                })
        batches += 1
        queries += int(batch["src"].numel())
        eligible += len(candidate_rows)
        selected += len(chosen)
        student.update_state(batch)
        teacher.update_state(batch)

    teacher_values = [row["teacher_interaction"] for row in records]
    student_values = [row["student_interaction"] for row in records]
    sign_agreement = None
    if records:
        left = torch.tensor(teacher_values)
        right = torch.tensor(student_values)
        sign_agreement = float((left.sign() == right.sign()).float().mean())
    result = {
        "config": parsed.config,
        "student_checkpoint": parsed.student_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "evidence": parsed.evidence,
        "reference": parsed.reference,
        "batches": batches,
        "queries": queries,
        "eligible": eligible,
        "selected": selected,
        "coverage": {
            "eligible_fraction": eligible / queries if queries else None,
            "distilled_fraction": selected / queries if queries else None,
        },
        "teacher_interaction": summary(teacher_values),
        "student_interaction": summary(student_values),
        "correlation": correlation(teacher_values, student_values),
        "sign_agreement": sign_agreement,
        "mean_absolute_error": (
            float(torch.tensor(teacher_values).sub(torch.tensor(student_values)).abs().mean())
            if records else None
        ),
        "pairs": records,
    }
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "batches", "queries", "eligible", "selected", "coverage",
        "teacher_interaction", "student_interaction", "correlation",
        "sign_agreement", "mean_absolute_error")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
