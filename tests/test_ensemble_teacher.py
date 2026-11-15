from __future__ import annotations

import torch
from torch import nn

from model import EnsembleTeacher


class DummyTeacher(nn.Module):
    hidden_dim = 4
    relation_dim = 4

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.updates = 0
        self.state_is_empty = True

    def score_candidates(self, batch, return_relations: bool = False):
        if return_relations:
            raise AssertionError("ensemble should not request relations")
        return {"logits": self.logits.expand(batch["src"].numel(), -1)}

    def update_state(self, batch) -> None:
        self.updates += 1
        self.state_is_empty = False

    def reset_state(self) -> None:
        self.updates = 0
        self.state_is_empty = True

    def runtime_state_elements(self) -> int:
        return 3


def test_ensemble_averages_teacher_probabilities() -> None:
    teacher_a = DummyTeacher(torch.tensor([[2.0, 0.0, -1.0]]))
    teacher_b = DummyTeacher(torch.tensor([[0.0, 1.0, 2.0]]))
    ensemble = EnsembleTeacher([teacher_a, teacher_b]).eval()
    batch = {"src": torch.tensor([0, 1])}

    output = ensemble.score_candidates(batch)
    expected = torch.stack(
        [
            torch.softmax(teacher_a.logits, dim=-1),
            torch.softmax(teacher_b.logits, dim=-1),
        ]
    ).mean(dim=0)
    assert torch.allclose(output["logits"].exp(), expected.expand(2, -1), atol=1e-6)


def test_ensemble_updates_and_resets_all_teachers() -> None:
    teacher_a = DummyTeacher(torch.zeros(1, 3))
    teacher_b = DummyTeacher(torch.zeros(1, 3))
    ensemble = EnsembleTeacher([teacher_a, teacher_b])

    ensemble.update_state({"src": torch.tensor([0])})
    assert teacher_a.updates == 1 and teacher_b.updates == 1
    assert not ensemble.state_is_empty

    ensemble.reset_state()
    assert teacher_a.updates == 0 and teacher_b.updates == 0
    assert ensemble.state_is_empty
    assert ensemble.runtime_state_elements() == 6
