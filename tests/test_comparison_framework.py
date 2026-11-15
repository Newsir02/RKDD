import json
from pathlib import Path

import pytest
import torch

import benchmark
from experiments.matrix import build_jobs
from experiments.summarize import summarize
from util.checkpoint import load_model_checkpoint
from model.snapshot_gnn import SnapshotGNN


def test_full_matrix_deduplicates_teacher_and_supervised_training():
    config = json.loads(Path("configs/comparison.json").read_text())
    jobs = build_jobs(config, "results/test-plan")
    assert len(jobs) == 705
    assert sum(j["stage"] == "teacher" for j in jobs) == 45
    assert sum(j["stage"] == "student" for j in jobs) == 30
    assert len({j["output_dir"] for j in jobs}) == 705
    references = {j["teacher_checkpoint"] for j in jobs if j["stage"] == "distill"}
    assert len(references) == 45


def test_json_config_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"stage": "teacher", "teacher_arch": "not-real"}))
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--config", str(path)])


def test_shared_checkpoint_roundtrip_contract_and_three_seed_summary(tmp_path, capsys):
    common = ["--dataset", "synthetic", "--quick", "--device", "cpu", "--num-nodes", "12",
              "--num-events", "20", "--num-candidates", "3", "--batch-size", "4",
              "--teacher-hidden", "8", "--student-hidden", "6", "--time-dim", "4"]
    teacher_dir = tmp_path / "teacher"
    benchmark.main([*common, "--stage", "teacher", "--output-dir", str(teacher_dir)])
    teacher_path = teacher_dir / "teacher.pt"
    original_hash = benchmark.sha256_file(teacher_path)
    student_dir = tmp_path / "student"
    result = benchmark.main([*common, "--stage", "distill", "--method", "nosmog_rsd",
                             "--teacher-checkpoint", str(teacher_path), "--output-dir", str(student_dir)])
    assert result["teacher_checkpoint_sha256"] == original_hash == benchmark.sha256_file(teacher_path)
    assert result["test"]["official"] is False
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["training_auxiliary_parameters"] > 0
    payload = load_model_checkpoint(student_dir / "student.pt")
    model_config = {k: payload["model_config"][k]
                    for k in ("architecture", "hidden_dim", "num_layers", "neighbors")}
    restored = SnapshotGNN(payload["model"]["node_features"], **model_config)
    restored.load_state_dict(payload["model"], strict=True)
    assert restored.last_event == -1
    rows = summarize(tmp_path)
    assert all(row["status"] == "incomplete" and row["std"] is None for row in rows)
    with pytest.raises(ValueError, match="mismatch"):
        benchmark.main([*common, "--stage", "distill", "--method", "logits", "--batch-size", "2",
                        "--teacher-checkpoint", str(teacher_path), "--output-dir", str(tmp_path / "bad")])
    capsys.readouterr()


def test_data_fingerprint_does_not_depend_on_model_seed():
    args = benchmark.parse_args(["--stage", "teacher", "--dataset", "synthetic", "--quick",
                                 "--num-nodes", "12", "--num-events", "20", "--num-candidates", "3"])
    bundle, *_ = benchmark.train.make_loaders(args)
    benchmark.normalize_train_time(bundle)
    before = benchmark.data_fingerprint(args, bundle)
    args.seed += 1
    assert benchmark.data_fingerprint(args, bundle) == before
    bundle.data.msg[0, 0] += 1
    assert benchmark.data_fingerprint(args, bundle) != before
