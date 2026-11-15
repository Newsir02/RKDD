"""Fair continuous-event KD benchmark, isolated from legacy entrypoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

import train
from loss.paper_baselines import METHODS, make_objective
from model.snapshot_gnn import SnapshotGNN
from dataset.benchmark_loader import replace_candidate_loaders
from util import save_model_checkpoint, model_parameter_summary
from util.seed import seed_everything
from util.temporal_delta_training import train_temporal_delta_epoch
from util.training import move_to_device


PROTOCOL = "batch-prefix-v2-uniform-sample-train-time-scale"
TEACHER_FIELDS = ("dataset", "seed", "batch_size", "num_candidates", "teacher_arch",
                  "teacher_hidden", "teacher_layers", "temporal_neighbors", "time_dim",
                  "relation_dim", "dropout", "teacher_epochs", "patience", "teacher_lr",
                  "weight_decay", "grad_clip", "teacher_node_features")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parent
    paths = [p for directory in ("model", "loss", "dataset", "util", "experiments")
             for p in (root / directory).glob("*.py")]
    paths += [root / "train.py", root / "benchmark.py"]
    return fingerprint({str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)})


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--config")
    known, remaining = initial.parse_known_args(argv)
    config = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[initial])
    parser.add_argument("--stage", choices=("teacher", "student", "distill"), default="distill")
    parser.add_argument("--method", choices=METHODS, default="logits")
    parser.add_argument("--student-model", choices=("gcn", "sage"), default="sage")
    parser.add_argument("--teacher-arch", choices=("tgn", "tgat", "graphmixer"), default="tgn")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", default="results/single")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trd-mode", choices=tuple(train.TemporalDeltaDistillation.MODES), default="temporal")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-tgb-download", dest="tgb_download", action="store_false", default=True)
    for name, default in {"seed":42, "batch_size":128, "num_candidates":16,
                          "teacher_hidden":64, "teacher_layers":1, "student_hidden":32,
                          "student_blocks":2, "student_neighbors":8, "temporal_neighbors":24,
                          "time_dim":16, "relation_dim":64, "teacher_epochs":100,
                          "student_epochs":100, "patience":20, "latency_batches":20,
                          "feature_dim":16, "edge_feature_dim":16, "num_nodes":256,
                          "num_events":3000, "max_real_events":0, "dkd_warmup":20}.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in {"teacher_lr":.001, "student_lr":.001, "weight_decay":0.,
                          "grad_clip":5., "temperature":2., "kd_weight":1., "relation_weight":1.,
                          "dkd_alpha":1., "dkd_beta":8.}.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    valid = {action.dest for action in parser._actions}
    if set(config) - valid:
        parser.error("Unknown config fields: " + str(sorted(set(config) - valid)))
    parser.set_defaults(**config)
    chosen = parser.parse_args(remaining)
    # argparse does not check JSON defaults against choices/type.
    for action in parser._actions:
        value = getattr(chosen, action.dest, None)
        if action.choices and value not in action.choices:
            parser.error(f"Invalid {action.dest}: {value}")
        if action.type is not None:
            try:
                setattr(chosen, action.dest, action.type(value))
            except (ValueError, TypeError):
                parser.error(f"Invalid {action.dest}: {value}")
    if chosen.stage == "student" and chosen.method != "supervised":
        parser.error("student stage requires --method supervised")
    if chosen.stage == "distill" and (chosen.method == "supervised" or not chosen.teacher_checkpoint):
        parser.error("distill requires a non-supervised method and --teacher-checkpoint")
    base = train.parse_args([])
    base.__dict__.update(vars(chosen))
    base.benchmark_method = chosen.method
    base.method = "trd"  # Shared teacher builder uses train-prefix time scaling.
    base.dropout = 0.0
    base.structure_features = False
    base.teacher_relation_weight = 0.0
    base.student_seed = base.seed
    for name in ("teacher_epochs", "student_epochs", "batch_size", "num_candidates", "student_blocks",
                 "student_neighbors", "teacher_hidden", "student_hidden", "latency_batches"):
        if getattr(base, name) <= 0:
            parser.error(f"{name} must be positive")
    if base.num_candidates < 2 or base.patience < 0 or base.max_real_events < 0 or base.dkd_warmup < 0:
        parser.error("Invalid candidate count/patience/max_real_events")
    for name in ("temperature", "grad_clip", "teacher_lr", "student_lr"):
        if not math.isfinite(getattr(base, name)) or getattr(base, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in ("kd_weight", "relation_weight", "weight_decay", "dkd_alpha", "dkd_beta"):
        if not math.isfinite(getattr(base, name)) or getattr(base, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if base.quick:
        train.apply_quick_configuration(base)
    return base


def normalize_train_time(bundle):
    origin = bundle.train_data.t[0].double()
    span = (bundle.train_data.t[-1].double() - origin).clamp_min(1)
    for data in (bundle.data, bundle.train_data, bundle.val_data, bundle.test_data):
        data.timestamp = ((data.t.double() - origin) / span).float()
    return {"origin": float(origin), "span": float(span)}


def data_fingerprint(args, bundle):
    # Model/sampling seed is not part of dataset identity. Synthetic streams
    # still have different identities because their actual tensors differ.
    identity_options = argparse.Namespace(**{**vars(args), "seed": 0})
    return train.compute_data_signature(identity_options, bundle)


def teacher_contract(args, signature):
    return {"protocol": PROTOCOL, "data_signature": signature,
            **{key: getattr(args, key) for key in TEACHER_FIELDS}}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def measure_latency(model, loader, device, count):
    """Fixed pre-test snapshot; excludes data transfer and history updates."""
    scorer = getattr(model, "score_candidates", model)
    durations, queries = [], 0
    for i, raw in enumerate(loader):
        batch = move_to_device(raw, device)
        if i == 0:
            scorer(batch, return_relations=False)
        synchronize(device)
        start = time.perf_counter()
        scorer(batch, return_relations=False)
        synchronize(device)
        durations.append(time.perf_counter() - start)
        queries += batch["src"].numel()
        if len(durations) >= count:
            break
    return {"query_ms": 1000 * sum(durations) / queries,
            "batch_ms_mean": 1000 * float(np.mean(durations)),
            "batches": len(durations), "candidates": int(batch["candidates"].size(1)),
            "protocol": "fixed-pre-test-snapshot-sampled-candidates"}


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("result.json", "epochs.jsonl", "teacher.pt", "student.pt")):
        raise FileExistsError(f"Run directory is not empty: {output}; use a new directory")
    seed_everything(args.seed)
    np.random.seed(args.seed)
    device = train.resolve_device(args.device)
    if device.type == "cuda":
        # Matrix jobs share a process; remove unused allocator blocks from
        # earlier jobs before measuring this run's reserved-memory peak.
        gc.collect()
        torch.cuda.empty_cache()
    effective = dict(vars(args))
    write_json(output / "config.json", effective)
    bundle, training, validation, testing = train.make_loaders(args)
    training, validation, testing = replace_candidate_loaders(bundle, args)
    time_scale = normalize_train_time(bundle)
    signature = data_fingerprint(args, bundle)
    contract = teacher_contract(args, signature)
    teacher = objective = None
    checkpoint_hash = None
    if args.stage == "teacher":
        model = train.build_teacher(bundle, args).to(device)
    else:
        if args.stage == "distill":
            payload = train.load_model_checkpoint(args.teacher_checkpoint)
            if payload["args"].get("benchmark_contract") != contract:
                raise ValueError("Teacher checkpoint protocol/config mismatch; train it with the same benchmark configuration")
            checkpoint_hash = sha256_file(args.teacher_checkpoint)
            teacher = train.load_teacher_from_checkpoint(args.teacher_checkpoint, bundle, signature, device)
            train.freeze_teacher(teacher)
        # Initialization is independent of teacher construction/loading RNG use.
        seed_everything(args.seed)
        np.random.seed(args.seed)
        model = SnapshotGNN(bundle.node_features, args.student_model, args.student_hidden,
                            args.student_blocks, args.student_neighbors).to(device)
        if teacher is not None:
            objective = make_objective(args.benchmark_method, args.student_hidden, args.teacher_hidden,
                                       args.temperature, args.kd_weight, args.relation_weight,
                                       args.trd_mode).to(device)
            if args.benchmark_method == "dkd":
                objective.dkd_alpha, objective.dkd_beta = args.dkd_alpha, args.dkd_beta
                objective.dkd_warmup = args.dkd_warmup
    parameters = list(model.parameters()) + (list(objective.parameters()) if objective is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.teacher_lr if args.stage == "teacher" else args.student_lr,
                                 weight_decay=args.weight_decay)
    stopper = train.EarlyStopping(args.patience)
    epochs = args.teacher_epochs if args.stage == "teacher" else args.student_epochs
    best = best_objective = None
    best_epoch, train_seconds, validation_seconds, peak_allocated, peak_reserved = 0, 0., 0., None, None
    for epoch in range(1, epochs + 1):
        model.reset_state()
        if teacher is not None:
            teacher.reset_state()
        if objective is not None and hasattr(objective, "epoch"):
            objective.epoch = epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        if args.stage == "teacher":
            losses = {"task": train.train_teacher_epoch(model, training, optimizer, device, args.grad_clip)}
        elif teacher is None:
            losses = {"task": train.train_supervised_student_epoch(model, training, optimizer, device, args.grad_clip)}
        else:
            losses = train_temporal_delta_epoch(model, teacher, objective, training, optimizer, device, args.grad_clip)
        synchronize(device)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError("Non-finite training objective")
        if device.type == "cuda":
            peak_allocated = max(peak_allocated or 0, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved or 0, torch.cuda.max_memory_reserved(device))
        start = time.perf_counter()
        metrics = train.evaluate_teacher_from_history(model, bundle, [training], validation, "val", args, device)
        synchronize(device)
        validation_seconds += time.perf_counter() - start
        value = float(metrics[str(metrics["metric"])])
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite validation metric")
        improved, stop = stopper.update(value, epoch)
        if improved:
            best, best_epoch = snapshot(model), epoch
            best_objective = snapshot(objective) if objective is not None else {}
        row = {"epoch": epoch, "losses": losses, "validation": metrics, "train_seconds": elapsed}
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if stop:
            break
    model.load_state_dict(best, strict=True)
    checkpoint = output / ("teacher.pt" if args.stage == "teacher" else "student.pt")
    saved_args = {**effective, "benchmark_contract": contract}
    save_model_checkpoint(checkpoint, model, type(model).__name__,
                          {**model.model_config, "data_signature": signature}, saved_args)
    if objective is not None:
        torch.save({"method": args.benchmark_method, "state_dict": best_objective}, output / "objective.pt")
    test = train.evaluate_teacher_from_history(model, bundle, [training, validation], testing, "test", args, device)
    # Rebuild pre-test state for comparable inference measurements.
    model.reset_state()
    train.replay_teacher(model, [training, validation], device)
    latency = measure_latency(model, testing, device, args.latency_batches)
    if checkpoint_hash is not None and sha256_file(args.teacher_checkpoint) != checkpoint_hash:
        raise RuntimeError("Shared teacher checkpoint changed during the run")
    packages = {}
    for name in ("torch", "torch-geometric", "numpy", "py-tgb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_hash = source_fingerprint()
    dataset_version = None
    if bundle.tgb_dataset is not None:
        from tgb.utils.info import DATA_VERSION_DICT
        dataset_version = DATA_VERSION_DICT.get(bundle.dataset_name)
    result = {"status": "complete", "dataset": args.dataset, "teacher": args.teacher_arch,
              "student": None if args.stage == "teacher" else args.student_model,
              "method": "teacher" if args.stage == "teacher" else args.benchmark_method,
              "seed": args.seed, "protocol": PROTOCOL, "config": effective,
              "config_hash": fingerprint(effective), "source_hash": source_hash,
              "data_signature": signature, "dataset_version": dataset_version,
              "teacher_checkpoint_sha256": checkpoint_hash,
              "checkpoint_sha256": sha256_file(checkpoint), "time_scale": time_scale,
              "dataset_stats": {"nodes": int(bundle.node_features.size(0)), "events": int(bundle.data.num_events),
                  "train": int(bundle.train_data.num_events), "val": int(bundle.val_data.num_events),
                  "test": int(bundle.test_data.num_events)},
              "best_epoch": best_epoch, "epochs_run": epoch, "test": test,
              "train_seconds": train_seconds, "validation_seconds": validation_seconds,
              "parameters": model_parameter_summary(model),
              "training_auxiliary_parameters": sum(p.numel() for p in objective.parameters()) if objective is not None else 0,
              "cuda_peak_allocated_bytes": peak_allocated, "cuda_peak_reserved_bytes": peak_reserved,
              "latency": latency, "environment": {"python": platform.python_version(),
                  "platform": platform.platform(), "device": str(device), "packages": packages,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
