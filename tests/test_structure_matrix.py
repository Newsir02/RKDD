import json
from pathlib import Path

import pytest

from experiments.matrix import build_jobs


TEMPORAL_CONTROLS = {
    "dyngkd_structure", "distill2vec_latent", "lightst_adaptive", "lkd4dytag_edge"
}


def load_config(name):
    return json.loads(Path("configs", name).read_text(encoding="utf-8"))


def test_main_uses_tgn_and_gcn_on_exactly_four_datasets():
    config = load_config("structure_comparison.json")
    jobs = build_jobs(config, "planned")
    main = [j for j in jobs if Path(j["output_dir"]).parts[1] == "main"]
    assert len(config["datasets"]) == 4
    assert config["seeds"] == [42, 43, 44]
    assert {j["teacher_arch"] for j in main} == {"tgn"}
    assert {j["student_model"] for j in main if j["stage"] != "teacher"} == {"gcn"}
    assert {j["method"] for j in main if j["stage"] == "distill"} == (
        TEMPORAL_CONTROLS | {"logits", "trd", "structure_interaction"}
    )
    assert len(main) == 108


def test_supplemental_runs_reuse_main_and_compare_new_method_only():
    jobs = build_jobs(load_config("structure_comparison.json"), "planned")
    extra = [j for j in jobs if Path(j["output_dir"]).parts[1] == "multi-teacher"]
    ablation = [j for j in jobs if Path(j["output_dir"]).parts[1] == "ablation"]
    assert len(extra) == 48
    assert {j["teacher_arch"] for j in extra} == {"tgat", "graphmixer"}
    assert {j["method"] for j in extra if j["stage"] == "distill"} == {"structure_interaction"}
    assert len(ablation) == 3
    assert {j["dataset"] for j in ablation} == {"tgbl-wiki"}
    assert {j["method"] for j in ablation} == {"structure_interaction"}
    assert {j["structure_mode"] for j in ablation} == {"zero"}
    assert len(jobs) == len({j["output_dir"] for j in jobs}) == 159
    available_before_job = set()
    for job in jobs:
        if job["stage"] == "teacher":
            available_before_job.add(Path(job["output_dir"]) / "teacher.pt")
        if job["stage"] == "distill":
            assert Path(job["teacher_checkpoint"]) in available_before_job
    for control in ablation:
        full = [
            j for j in jobs
            if j["dataset"] == control["dataset"] and j["seed"] == control["seed"]
            and j["teacher_arch"] == "tgn" and j["method"] == "structure_interaction"
            and j["structure_mode"] == "temporal"
        ]
        assert len(full) == 1
        ignored = {"structure_mode", "output_dir"}
        assert {k: v for k, v in full[0].items() if k not in ignored} == {
            k: v for k, v in control.items() if k not in ignored
        }


def test_structure_logits_control_is_a_distinct_structure_method():
    from benchmark import parse_args

    args = parse_args([
        "--stage", "distill", "--method", "structure_logits", "--dataset", "synthetic",
        "--student-model", "gcn", "--teacher-checkpoint", "unused.pt",
        "--structure-evidence", "random", "--structure-reference", "hard",
    ])
    assert args.benchmark_method == "structure_logits"
    assert args.structure_evidence == "random"
    assert args.structure_reference == "hard"
    with pytest.raises(SystemExit):
        parse_args([
            "--stage", "distill", "--method", "structure_logits", "--dataset", "synthetic",
            "--student-model", "gcn", "--teacher-checkpoint", "unused.pt",
            "--structure-mode", "zero",
        ])


def test_structure_design_rejects_scope_drift():
    config = load_config("structure_smoke.json")
    with pytest.raises(ValueError, match="subset"):
        build_jobs({**config, "multi_teacher_datasets": ["tgbl-wiki"]}, "planned")
    with pytest.raises(ValueError, match="GCN"):
        build_jobs({**config, "students": ["sage"]}, "planned")
    jobs = build_jobs(config, "planned")
    assert len(jobs) == 14
    assert {j["dataset"] for j in jobs} == {"synthetic"}
