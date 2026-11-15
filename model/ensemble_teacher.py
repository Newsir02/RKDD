from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EnsembleTeacher(nn.Module):
    """Average the candidate distributions of multiple temporal GNN teachers."""

    def __init__(self, teachers: Iterable[nn.Module]) -> None:
        super().__init__()
        self.teachers = nn.ModuleList(list(teachers))
        if len(self.teachers) == 0:
            raise ValueError("EnsembleTeacher requires at least one teacher.")
        first = self.teachers[0]
        self.hidden_dim = int(getattr(first, "hidden_dim"))
        self.relation_dim = int(getattr(first, "relation_dim"))
        self.model_config = {
            "teacher_arch": "ensemble",
            "teacher_count": len(self.teachers),
            "hidden_dim": self.hidden_dim,
            "relation_dim": self.relation_dim,
        }

    def runtime_state_elements(self) -> int:
        return sum(
            int(teacher.runtime_state_elements())
            for teacher in self.teachers
            if hasattr(teacher, "runtime_state_elements")
        )

    @property
    def state_is_empty(self) -> bool:
        return all(bool(getattr(t, "state_is_empty", True)) for t in self.teachers)

    @torch.no_grad()
    def reset_state(self) -> None:
        for teacher in self.teachers:
            reset = getattr(teacher, "reset_state", None)
            if callable(reset):
                reset()

    def detach_state(self) -> None:
        for teacher in self.teachers:
            detach = getattr(teacher, "detach_state", None)
            if callable(detach):
                detach()

    def score_candidates(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = False,
    ) -> Dict[str, Tensor]:
        if return_relations:
            raise ValueError("EnsembleTeacher supports logit distillation only.")
        probabilities = []
        for teacher in self.teachers:
            scorer = getattr(teacher, "score_candidates", None)
            output = (
                scorer(batch, return_relations=False)
                if callable(scorer)
                else teacher(batch, return_relations=False)
            )
            probabilities.append(
                F.softmax(output["logits"].detach(), dim=-1)
            )
        mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
        return {
            "logits": torch.log(mean_probability.clamp_min(1.0e-8)),
        }

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        for teacher in self.teachers:
            update = getattr(teacher, "update_state", None)
            if callable(update):
                update(batch)

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = False,
        update_memory: bool = True,
    ) -> Dict[str, Tensor]:
        output = self.score_candidates(batch, return_relations=return_relations)
        if update_memory:
            self.update_state(batch)
        return output
