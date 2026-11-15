from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EnsembleStudent(nn.Module):
    """Average candidate distributions of multiple distilled students."""

    def __init__(
        self,
        students: Iterable[nn.Module],
        aggregate: str = "logit",
    ) -> None:
        super().__init__()
        self.students = nn.ModuleList(list(students))
        if len(self.students) == 0:
            raise ValueError("EnsembleStudent requires at least one student.")
        valid = {"prob", "logit", "zlogit", "rank", "median"}
        if aggregate not in valid:
            raise ValueError(
                "EnsembleStudent aggregate must be one of: "
                + ", ".join(sorted(valid))
            )
        self.aggregate = aggregate
        first = self.students[0]
        self.hidden_dim = int(getattr(first, "hidden_dim"))
        self.relation_dim = int(getattr(first, "relation_dim", 0))
        self.model_config = {
            "student_arch": "ensemble",
            "student_count": len(self.students),
            "hidden_dim": self.hidden_dim,
            "relation_dim": self.relation_dim,
        }
        self.last_candidate_embeddings: list[Tensor] = []

    def runtime_state_elements(self) -> int:
        return sum(
            int(student.runtime_state_elements())
            for student in self.students
            if hasattr(student, "runtime_state_elements")
        )

    @property
    def state_is_empty(self) -> bool:
        return all(bool(getattr(s, "state_is_empty", True)) for s in self.students)

    @torch.no_grad()
    def reset_state(self) -> None:
        for student in self.students:
            reset = getattr(student, "reset_state", None)
            if callable(reset):
                reset()

    def detach_state(self) -> None:
        for student in self.students:
            detach = getattr(student, "detach_state", None)
            if callable(detach):
                detach()

    def score_candidates(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = False,
    ) -> Dict[str, Tensor]:
        if return_relations:
            raise ValueError("EnsembleStudent supports logit evaluation only.")
        probabilities = []
        logits = []
        candidate_embeddings = []
        for student in self.students:
            output = student(batch, return_relations=False)
            logit = output["logits"].detach()
            logits.append(logit)
            probabilities.append(F.softmax(logit, dim=-1))
            embedding = output.get("candidate_embeddings", output.get("candidate_embedding"))
            if embedding is not None:
                candidate_embeddings.append(embedding.detach())
        self.last_candidate_embeddings = candidate_embeddings
        stacked = torch.stack(logits, dim=0)
        if self.aggregate == "logit":
            return {"logits": stacked.mean(dim=0)}
        if self.aggregate == "zlogit":
            centered = stacked - stacked.mean(dim=-1, keepdim=True)
            scaled = centered / stacked.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1.0e-6)
            return {"logits": scaled.mean(dim=0)}
        if self.aggregate == "median":
            return {"logits": stacked.median(dim=0).values}
        if self.aggregate == "rank":
            order = torch.argsort(stacked, dim=-1, descending=True)
            ranks = torch.empty_like(order)
            positions = torch.arange(
                stacked.size(-1),
                device=stacked.device,
                dtype=order.dtype,
            ).expand_as(order)
            ranks.scatter_(-1, order, positions)
            return {
                "logits": -ranks.to(stacked.dtype).mean(dim=0)
            }
        mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
        return {"logits": torch.log(mean_probability.clamp_min(1.0e-8))}

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        for student in self.students:
            update = getattr(student, "update_state", None)
            if callable(update):
                update(batch)

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = False,
    ) -> Dict[str, Tensor]:
        return self.score_candidates(batch, return_relations=return_relations)
