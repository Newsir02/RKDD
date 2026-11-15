from __future__ import annotations

import torch
from torch import nn

from model import EnsembleStudent


class DummyStudent(nn.Module):
    hidden_dim = 4
    relation_dim = 4

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.updates = 0
        self.state_is_empty = True

    def forward(self, batch, return_relations: bool = False):
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
        return 5


def test_ensemble_student_averages_probabilities() -> None:
    first = DummyStudent(torch.tensor([[2.0, 0.0, -1.0]]))
    second = DummyStudent(torch.tensor([[0.0, 1.0, 2.0]]))
    ensemble = EnsembleStudent(
        [first, second],
        aggregate="prob",
    ).eval()
    batch = {"src": torch.tensor([0, 1])}

    output = ensemble.score_candidates(batch)
    expected = torch.stack(
        [
            torch.softmax(first.logits, dim=-1),
            torch.softmax(second.logits, dim=-1),
        ]
    ).mean(dim=0)
    assert torch.allclose(output["logits"].exp(), expected.expand(2, -1), atol=1e-6)


def test_ensemble_student_averages_logits_by_default() -> None:
    first = DummyStudent(torch.tensor([[2.0, 0.0, -1.0]]))
    second = DummyStudent(torch.tensor([[0.0, 1.0, 2.0]]))
    ensemble = EnsembleStudent([first, second]).eval()
    batch = {"src": torch.tensor([0, 1])}

    output = ensemble.score_candidates(batch)
    expected = (first.logits + second.logits) / 2.0
    assert torch.allclose(output["logits"], expected.expand(2, -1), atol=1e-6)


def test_ensemble_student_alternative_aggregations() -> None:
    first = DummyStudent(torch.tensor([[2.0, 0.0, -1.0]]))
    second = DummyStudent(torch.tensor([[0.0, 1.0, 2.0]]))
    batch = {"src": torch.tensor([0, 1])}

    for mode in ("zlogit", "rank", "median"):
        ensemble = EnsembleStudent([first, second], aggregate=mode).eval()
        output = ensemble.score_candidates(batch)
        assert output["logits"].shape == (2, 3)
        assert torch.isfinite(output["logits"]).all()


def test_ensemble_student_updates_and_resets_all_students() -> None:
    first = DummyStudent(torch.zeros(1, 3))
    second = DummyStudent(torch.zeros(1, 3))
    ensemble = EnsembleStudent([first, second])

    ensemble.update_state({"src": torch.tensor([0])})
    assert first.updates == 1 and second.updates == 1
    assert not ensemble.state_is_empty

    ensemble.reset_state()
    assert first.updates == 0 and second.updates == 0
    assert ensemble.state_is_empty
    assert ensemble.runtime_state_elements() == 10
