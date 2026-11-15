"""Paper-derived loss components adapted to a frozen TGN and snapshot GCN.

These methods share the candidate classification task and the causal prefix.
See docs/research/temporal_baselines.md for source, formula and scope.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .kd import logit_kd_loss


TEMPORAL_PAPER_METHODS = (
    "dyngkd_structure", "distill2vec_latent", "lightst_adaptive", "lkd4dytag_edge",
)


def latent_kl(student, teacher, temperature):
    """Compute KL(student || teacher) over node embedding channels."""
    ls = F.log_softmax(student / temperature, dim=-1)
    lt = F.log_softmax(teacher.detach() / temperature, dim=-1)
    return (ls.exp() * (ls - lt)).sum(dim=-1).mean()


def lightst_spatial_loss(student, teacher, temperature=.05):
    """Single-time query counterpart of the author's `loss_cls` objective.

    Each row is a contemporaneous set of source/candidate occurrences. Positive
    similarities are cosine squared. The two negative normalizers are the
    within-student and within-teacher similarity sums, including diagonals.
    """
    s = F.normalize(student, p=2, dim=-1)
    t = F.normalize(teacher.detach(), p=2, dim=-1)
    log_positive = (s * t).sum(dim=-1).square() / temperature
    log_student_neg = ((s @ s.transpose(-1, -2)) / temperature).logsumexp(dim=-1)
    log_teacher_neg = ((t @ t.transpose(-1, -2)) / temperature).logsumexp(dim=-1)
    return (F.softplus(log_student_neg - log_positive).mean()
            + F.softplus(log_teacher_neg - log_positive).mean()) * .5


def dy_tag_edge_loss(student_edge, teacher_edge):
    """LKD4DyTAG Eq. 18 on the candidate-edge representation."""
    similarity = F.cosine_similarity(student_edge, teacher_edge.detach(), dim=-1)
    return torch.exp(-similarity).mean()


class TemporalPaperObjective(nn.Module):
    """Common trainer contract for methods requiring only the intact prefix."""

    needs_previous = False

    def __init__(self, method, student_dim, teacher_dim, temperature=2., kd_weight=1., relation_weight=1.):
        super().__init__()
        if method not in TEMPORAL_PAPER_METHODS:
            raise ValueError(f"Unknown temporal paper adaptation: {method}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if not all(math.isfinite(w) and w >= 0 for w in (kd_weight, relation_weight)):
            raise ValueError("Loss weights must be finite and non-negative")
        self.method, self.temperature = method, temperature
        self.kd_weight, self.relation_weight = kd_weight, relation_weight
        self.projector = nn.Linear(student_dim, teacher_dim, bias=False)

    @staticmethod
    def _nodes(output, suffix):
        return torch.cat([output[f"source_{suffix}"][:, None], output[f"candidate_{suffix}"]], dim=1)

    def forward(self, student, teacher, target, student_before=None, teacher_before=None):
        task = F.cross_entropy(student["logits"], target)
        response = task.new_zeros(())
        student_suffix = "structure" if self.method == "dyngkd_structure" else "embedding"
        s = self._nodes(student, student_suffix)
        t = self._nodes(teacher, "embedding").detach()
        if self.method in ("dyngkd_structure", "distill2vec_latent"):
            feature = latent_kl(self.projector(s), t, self.temperature)
        elif self.method == "lightst_adaptive":
            feature = lightst_spatial_loss(self.projector(s), t)
            response = logit_kd_loss(student["logits"], teacher["logits"], self.temperature)
        else:
            # Eq. 7 of LKD4DyTAG forms edge features by Hadamard product.
            student_edge = self.projector(s[:, :1] * s[:, 1:])
            teacher_edge = t[:, :1] * t[:, 1:]
            feature = dy_tag_edge_loss(student_edge, teacher_edge)
        total = task + self.kd_weight * response + self.relation_weight * feature
        return total, {"total": total.detach(), "task": task.detach(),
                       "kd": response.detach(), "relation": feature.detach()}
