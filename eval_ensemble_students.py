from __future__ import annotations

import argparse
from pathlib import Path

from model import EnsembleStudent
from train import (
    compute_data_signature,
    evaluate_teacher_from_history,
    load_student_from_checkpoint as load_student,
    make_loaders,
    parse_args,
    resolve_device,
)
from util import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--aggregate",
        choices=("prob", "logit", "zlogit", "rank", "median"),
        default="logit",
        help="Ensemble aggregation: probability, logit, z-score logit, rank, median.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Evaluate validation or test split. Validation must not replay validation labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--edge-feature-dim", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-max-events", type=int, default=0)
    cli = parser.parse_args()

    args = parse_args(
        [
            "--stage",
            "student",
            "--dataset",
            cli.dataset,
            "--data-root",
            cli.data_root,
            "--seed",
            str(cli.seed),
            "--batch-size",
            str(cli.batch_size),
            "--num-candidates",
            str(cli.num_candidates),
            "--history-size",
            str(cli.history_size),
            "--feature-dim",
            str(cli.feature_dim),
            "--edge-feature-dim",
            str(cli.edge_feature_dim),
            "--num-workers",
            str(cli.num_workers),
            "--eval-max-events",
            str(cli.eval_max_events),
        ]
    )
    device = resolve_device(cli.device)
    seed_everything(cli.seed)
    bundle, train_loader, validation_loader, test_loader = make_loaders(args)
    paths = [item.strip() for item in cli.checkpoints.split(",") if item.strip()]
    if len(paths) < 2:
        raise ValueError("--checkpoints must contain at least two checkpoints.")
    signature = compute_data_signature(args, bundle)
    students = [load_student(path, bundle, device, data_signature=signature) for path in paths]
    ensemble = EnsembleStudent(
        students,
        aggregate=cli.aggregate,
    ).to(device).eval()
    history = [train_loader]
    target_loader = validation_loader
    if cli.split == "test":
        history.append(validation_loader)
        target_loader = test_loader
    result = evaluate_teacher_from_history(
        ensemble, bundle, history, target_loader, cli.split, args, device,
    )
    print("Ensemble students:", ", ".join(paths))
    print(f"Ensemble {cli.split} MRR: {result}")


if __name__ == "__main__":
    main()
