"""One relational target: signed change of node-pair affinity across history prefixes."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kd import logit_kd_loss, rkd_distance_loss, rkd_angle_loss


def query_relations(output: dict[str, Tensor]) -> Tensor:
    """Source-candidate cosine relations, independent of representation width."""
    source = F.normalize(output["source_embedding"], dim=-1, eps=1e-8)
    candidates = F.normalize(output["candidate_embedding"], dim=-1, eps=1e-8)
    return (source.unsqueeze(1) * candidates).sum(-1)


class TemporalDeltaDistillation(nn.Module):
    MODES = ("temporal", "static", "rkd_distance", "rkd_angle", "shuffled", "reverse", "zero")

    def __init__(self, logit_weight=1.0, rkd_weight=1.0, temperature=2.0, mode="temporal"):
        super().__init__()
        if not all(math.isfinite(x) and x >= 0 for x in (logit_weight, rkd_weight)):
            raise ValueError("Distillation weights must be finite and non-negative.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if mode not in self.MODES:
            raise ValueError(f"Unknown TRD mode: {mode}")
        self.logit_weight, self.rkd_weight = logit_weight, rkd_weight
        self.temperature, self.mode = temperature, mode

    @property
    def needs_previous(self):
        return self.rkd_weight > 0 and self.mode not in ("static", "rkd_distance", "rkd_angle")

    def forward(self, student, teacher, target, student_before=None, teacher_before=None):
        task = F.cross_entropy(student["logits"], target)
        zero = student["logits"].sum() * 0.0
        logits = (logit_kd_loss(student["logits"], teacher["logits"], self.temperature)
                  if self.logit_weight else zero)
        rkd = zero
        if self.rkd_weight:
            if self.mode in ("rkd_distance", "rkd_angle"):
                fn = rkd_distance_loss if self.mode == "rkd_distance" else rkd_angle_loss
                # Sampled node set for each query: source plus the same candidates.
                s = torch.cat([student["source_embedding"].unsqueeze(1), student["candidate_embedding"]], 1)
                t = torch.cat([teacher["source_embedding"].unsqueeze(1), teacher["candidate_embedding"]], 1)
                rkd = fn(s, t.detach())
            elif self.mode == "static":
                rkd = F.mse_loss(query_relations(student), query_relations(teacher).detach())
            elif student_before is not None and teacher_before is not None:
                delta_s = query_relations(student) - query_relations(student_before)
                delta_t = (query_relations(teacher) - query_relations(teacher_before)).detach()
                if self.mode == "shuffled":
                    # Destroy pair correspondence while preserving the target multiset.
                    flat = delta_t.flatten()
                    delta_t = flat[torch.randperm(flat.numel(), device=flat.device)].reshape_as(delta_t)
                elif self.mode == "reverse":
                    delta_t = -delta_t
                elif self.mode == "zero":
                    delta_t = torch.zeros_like(delta_t)
                rkd = F.mse_loss(delta_s, delta_t)
        total = task + self.logit_weight * logits + self.rkd_weight * rkd
        return total, {"total": total.detach(), "task": task.detach(),
                       "logit": logits.detach(), "rkd": rkd.detach()}
