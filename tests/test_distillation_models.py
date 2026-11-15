from __future__ import annotations

import torch

from loss import DistillationObjective
from model import (
    LightGNNStudent,
    PyGTGNTeacher,
)


def make_event_batch() -> dict[str, torch.Tensor]:
    return {
        "event_index": torch.tensor([0, 1, 2]),
        "src": torch.tensor([0, 0, 0]),
        "positive_dst": torch.tensor([1, 2, 3]),
        "candidates": torch.tensor(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            dtype=torch.long,
        ),
        "target": torch.tensor([0, 1, 2]),
        "timestamp": torch.tensor([0.0, 0.5, 1.0]),
        "event_time": torch.tensor([10, 20, 30]),
        "edge_features": torch.zeros(3, 2),
    }


def test_pyg_tgn_and_gnn_backpropagate_through_distillation() -> None:
    node_features = torch.zeros(5, 3)
    teacher = PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30]),
        event_messages=torch.zeros(3, 2),
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    ).eval()
    student = LightGNNStudent(
        node_features,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        relation_slots=3,
        dropout=0.0,
        num_blocks=2,
        neighbors=2,
    )
    objective = DistillationObjective()
    batch = make_event_batch()

    with torch.no_grad():
        teacher_output = teacher(batch, return_relations=True, update_memory=True)
    student_output = student(batch, return_relations=True)
    total, parts = objective(student_output, teacher_output, batch["target"])
    total.backward()

    assert torch.isfinite(total)
    assert student.node_embeddings.weight.grad is not None
    assert torch.isfinite(student.node_embeddings.weight.grad).all()
    assert set(parts) == {
        "total",
        "task",
        "logit",
        "rank",
        "anchor",
        "diversity",
        "kernel",
        "rkd",
        "dkd",
    }


def test_default_objective_only_needs_logits() -> None:
    student_logits = torch.randn(3, 5, requires_grad=True)
    teacher_logits = torch.randn(3, 5, requires_grad=True)
    objective = DistillationObjective()

    total, parts = objective(
        {"logits": student_logits},
        {"logits": teacher_logits},
        torch.tensor([0, 1, 2]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert teacher_logits.grad is None
    assert torch.allclose(total.detach(), parts["task"] + parts["logit"] + 0.5 * parts["rank"])


def test_decoupled_kd_backpropagates() -> None:
    student_logits = torch.randn(3, 5, requires_grad=True)
    teacher_logits = torch.randn(3, 5)
    objective = DistillationObjective(
        task_weight=0.0,
        logit_weight=0.0,
        rank_weight=0.0,
        diversity_weight=0.0,
        anchor_weight=0.0,
        dkd_weight=1.0,
    )

    total, parts = objective(
        {"logits": student_logits},
        {"logits": teacher_logits},
        torch.tensor([0, 1, 2]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert float(parts["dkd"]) > 0.0


def test_confidence_gated_logit_distillation_backpropagates() -> None:
    student_logits = torch.randn(3, 5, requires_grad=True)
    teacher_logits = torch.randn(3, 5)
    objective = DistillationObjective(
        task_weight=0.0,
        logit_weight=1.0,
        rank_weight=0.0,
        diversity_weight=0.0,
        anchor_weight=0.0,
        confidence_weight=1.0,
    )

    total, parts = objective(
        {"logits": student_logits},
        {"logits": teacher_logits},
        torch.tensor([0, 1, 2]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_logits.grad is not None
    assert torch.isfinite(parts["logit"])


def test_rkd_distance_and_angle_backpropagate() -> None:
    student_embeddings = torch.randn(2, 4, 5, requires_grad=True)
    teacher_embeddings = torch.randn(2, 4, 7)
    student_output = {
        "logits": torch.randn(2, 4, requires_grad=True),
        "candidate_embeddings": student_embeddings,
    }
    teacher_output = {
        "logits": torch.randn(2, 4),
        "candidate_embedding": teacher_embeddings,
    }
    objective = DistillationObjective(
        task_weight=0.0,
        logit_weight=0.0,
        rank_weight=0.0,
        diversity_weight=0.0,
        anchor_weight=0.0,
        rkd_weight=1.0,
        rkd_mode="both",
    )

    total, parts = objective(
        student_output, teacher_output, torch.tensor([0, 1])
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_embeddings.grad is not None
    assert torch.isfinite(student_embeddings.grad).all()
    assert float(parts["rkd"]) > 0.0
