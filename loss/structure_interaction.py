"""Candidate-link relation and its conditional two-group interaction."""

import torch
from torch import nn
from torch.nn import functional as F


def interaction(view_logits):
    """Return R11 - R10 - R01 + R00 for fixed (positive, reference) columns."""
    if view_logits.ndim != 3 or view_logits.size(0) != 4 or view_logits.size(-1) != 2:
        raise ValueError("Expected [4, queries, 2] logits in 00, 10, 01, 11 order")
    relation = view_logits[..., 0] - view_logits[..., 1]
    return (relation[3] - relation[2]) - (relation[1] - relation[0])


class StructureInteractionObjective(nn.Module):
    def __init__(self, weight, mode="temporal", beta=1.0):
        super().__init__()
        if mode not in ("temporal", "zero"):
            raise ValueError("Unknown structural interaction mode")
        self.weight, self.mode, self.beta = weight, mode, beta

    def forward(self, student_views, teacher_views):
        student_delta = interaction(student_views)
        teacher_delta = interaction(teacher_views.detach())
        target = teacher_delta if self.mode == "temporal" else torch.zeros_like(teacher_delta)
        loss = F.huber_loss(student_delta, target, delta=self.beta)
        return loss, student_delta.detach(), teacher_delta
