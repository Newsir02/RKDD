"""Configuration and four-view target for candidate-link distillation."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .kd import logit_kd_loss


def interaction(view_logits):
    """Return R11 - R10 - R01 + R00 for fixed (positive, reference) columns."""
    if view_logits.ndim != 3 or view_logits.size(0) != 4 or view_logits.size(-1) != 2:
        raise ValueError("Expected [4, queries, 2] logits in 00, 10, 01, 11 order")
    relation = view_logits[..., 0] - view_logits[..., 1]
    return (relation[3] - relation[2]) - (relation[1] - relation[0])


class StructureInteractionObjective(nn.Module):
    kind = "structure_interaction"

    def __init__(self, weight, mode="temporal", beta=1.0, *, kd_weight=1.0, temperature=2.0):
        super().__init__()
        if mode not in ("temporal", "zero"):
            raise ValueError("Unknown structural interaction mode")
        if not math.isfinite(kd_weight) or kd_weight < 0:
            raise ValueError("Logits KD weight must be finite and nonnegative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Logits KD temperature must be finite and positive")
        self.weight, self.mode, self.beta = weight, mode, beta
        self.kd_weight, self.temperature = kd_weight, temperature

    def forward(self, student_views, teacher_views):
        student_delta = interaction(student_views)
        teacher_delta = interaction(teacher_views.detach())
        target = teacher_delta if self.mode == "temporal" else torch.zeros_like(teacher_delta)
        loss = F.huber_loss(student_delta, target, delta=self.beta)
        return loss, student_delta.detach(), teacher_delta


class StructureLogitObjective(nn.Module):
    """Matched control: four-view candidate KD without the interaction target.

    The same evidence groups, candidate pair, number of views and forward
    budget are used. The auxiliary loss is the mean per-view logits KD; no
    full-candidate KD is added by this objective.
    """

    kind = "structure_logits"
    kd_weight = 0.0

    def __init__(self, weight, temperature=2.0):
        super().__init__()
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Structure logits weight must be finite and positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Logits KD temperature must be finite and positive")
        self.weight, self.temperature = weight, temperature

    def forward(self, student_views, teacher_views):
        per_view = torch.stack([
            logit_kd_loss(student_views[index], teacher_views[index].detach(), self.temperature)
            for index in range(4)
        ])
        return (per_view.mean(), interaction(student_views).detach(),
                interaction(teacher_views.detach()))
