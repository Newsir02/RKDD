import pytest
import torch
from torch.nn import functional as F

from loss.paper_baselines import make_objective
from model.snapshot_gnn import SnapshotGNN


def make_output(values):
    return {"logits": torch.tensor(values, dtype=torch.float32, requires_grad=True)}


def test_score_delta_matches_candidate_log_odds_and_detaches_teacher():
    student = make_output([[2., 5., 1.]])
    before = make_output([[0., 1., 1.]])
    teacher = make_output([[6., 3., 3.]])
    teacher_before = make_output([[3., 4., 3.]])
    criterion = make_objective("trd_score", 4, 8, temperature=2, relation_weight=.25)
    total, parts = criterion(student, teacher, torch.tensor([0]), before, teacher_before)
    s_delta = torch.tensor([[0., 2., -2.]]) / 2
    t_delta = torch.tensor([[7., -5., -2.]]) / 6
    expected = F.smooth_l1_loss(s_delta, t_delta)
    assert torch.allclose(parts["score_delta"], expected)
    assert torch.allclose(total, parts["task"] + parts["logit"] + .25 * expected)
    total.backward()
    assert before["logits"].grad.abs().sum() > 0
    assert teacher["logits"].grad is None and teacher_before["logits"].grad is None


def test_score_delta_ignores_query_offsets_and_zero_weight_is_logits_kd():
    student, before = make_output([[4., 2., 1.]]), make_output([[2., 0., 5.]])
    teacher, teacher_before = make_output([[2., 3., 1.]]), make_output([[5., 0., 2.]])
    args = (student, teacher, torch.tensor([0]), before, teacher_before)
    criterion = make_objective("trd_score", 4, 8)
    _, parts = criterion(*args)
    _, shifted = criterion({"logits": student["logits"] + 10}, teacher, args[2],
                           before, {"logits": teacher_before["logits"] - 9})
    assert torch.allclose(parts["score_delta"], shifted["score_delta"])
    no_relation = make_objective("trd_score", 4, 8, relation_weight=0)
    assert not no_relation.needs_previous
    assert torch.allclose(no_relation(*args[:3])[0], make_objective("logits", 4, 8)(*args[:3])[0])
    assert criterion(*args[:3])[1]["score_delta"] == 0
    with pytest.raises(ValueError, match="mode"):
        make_objective("trd_score", 4, 8, trd_mode="static")


def test_student_identity_can_be_disabled_for_capacity_ablation():
    with_id = SnapshotGNN(torch.zeros(10, 2), "sage", 8)
    no_id = SnapshotGNN(torch.zeros(10, 2), "sage", 8, use_node_ids=False)
    assert sum(p.numel() for p in with_id.parameters()) - sum(p.numel() for p in no_id.parameters()) == 80
    assert no_id.model_config["use_node_ids"] is False
    assert "node_embeddings.weight" not in no_id.state_dict()


def test_score_delta_gate_only_applies_teacher_signal_to_better_supported_queries():
    student = make_output([[4., 0.], [0., 3.]])
    teacher = make_output([[0., 3.], [4., 0.]])
    before, teacher_before = make_output([[2., 0.], [0., 2.]]), make_output([[0., 1.], [2., 0.]])
    criterion = make_objective("trd_score", 4, 8, score_delta_gate=True)
    target = torch.zeros(2, dtype=torch.long)
    total, parts = criterion(student, teacher, target, before, teacher_before)
    assert parts["teacher_gate_fraction"] == .5
    task = F.cross_entropy(student["logits"], target)
    single_target = {"logits": teacher["logits"][1:]}
    single_student = {"logits": student["logits"][1:]}
    ungated = make_objective("trd_score", 4, 8)
    _, active = ungated(single_student, single_target, target[1:],
                        {"logits": before["logits"][1:]}, {"logits": teacher_before["logits"][1:]})
    assert torch.allclose(total, task + .5 * (active["logit"] + active["score_delta"]))
    total.backward()
    assert torch.equal(before["logits"].grad[0], torch.zeros(2))
    assert teacher["logits"].grad is None
