"""Distill relative link score changes after a causal history update."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .kd import logit_kd_loss
from .temporal_delta import query_relations


class ScoreDeltaDistillation(nn.Module):
    MODES = ("temporal", "reverse", "zero", "shuffled")

    def __init__(self, kd_weight, relation_weight, temperature, mode, warmup=0, teacher_gate=False):
        super().__init__()
        if not all(math.isfinite(w) and w >= 0 for w in (kd_weight, relation_weight)):
            raise ValueError("Score delta weights must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0 or warmup < 0:
            raise ValueError("Score delta requires positive temperature and non-negative warmup")
        if mode not in self.MODES:
            raise ValueError(f"Unsupported score delta mode: {mode}")
        self.kd_weight, self.relation_weight = kd_weight, relation_weight
        self.temperature, self.mode, self.warmup = temperature, mode, warmup
        self.teacher_gate = teacher_gate
        self.epoch = 1

    @property
    def needs_previous(self):
        return self.relation_weight > 0

    def relative_delta(self, after, before):
        change = after - before
        return (change - change.mean(-1, keepdim=True)) / self.temperature

    @torch.no_grad()
    def diagnostics(self, student_delta, teacher_delta, teacher, teacher_before, target):
        parts = {
            "teacher_score_delta_energy": teacher_delta.square().mean(),
            "student_score_delta_energy": student_delta.square().mean(),
            "teacher_positive_relative_delta": teacher_delta.gather(1, target[:, None]).mean(),
        }
        if "source_embedding" in teacher and "source_embedding" in teacher_before:
            cosine_delta = query_relations(teacher) - query_relations(teacher_before)
            cosine_delta = cosine_delta - cosine_delta.mean(-1, keepdim=True)
            parts["teacher_cosine_score_alignment"] = F.cosine_similarity(cosine_delta, teacher_delta, dim=-1).mean()
        return parts

    def forward(self, student, teacher, target, student_before=None, teacher_before=None):
        logits = student["logits"]
        task_per_query = F.cross_entropy(logits, target, reduction="none")
        task = task_per_query.mean()
        zero = logits.sum() * 0
        relation, diagnostics = zero, {}
        gate = None
        if self.teacher_gate:
            with torch.no_grad():
                teacher_task = F.cross_entropy(teacher["logits"], target, reduction="none")
                gate = (teacher_task < task_per_query).to(logits.dtype)
            diagnostics["teacher_gate_fraction"] = gate.mean()
        kd = logit_kd_loss(logits, teacher["logits"], self.temperature, sample_weight=gate) if self.kd_weight else zero
        if gate is not None:
            # Convert the active-query average to a full-batch average.
            kd = kd * diagnostics["teacher_gate_fraction"]
        if self.needs_previous and student_before is not None and teacher_before is not None:
            sd = self.relative_delta(logits, student_before["logits"])
            td = self.relative_delta(teacher["logits"].detach(), teacher_before["logits"].detach())
            diagnostics.update(self.diagnostics(sd, td, teacher, teacher_before, target))
            if self.mode == "reverse":
                td = -td
            elif self.mode == "zero":
                td = torch.zeros_like(td)
            elif self.mode == "shuffled":
                # Keep the batch multiset and remove each original query pairing.
                td = td.roll(1, dims=0 if td.size(0) > 1 else 1)
            per_query_relation = F.smooth_l1_loss(sd, td, reduction="none").mean(-1)
            relation = per_query_relation.mean() if gate is None else (per_query_relation * gate).mean()
        weight = self.relation_weight * (min(self.epoch / self.warmup, 1.) if self.warmup else 1.)
        total = task + self.kd_weight * kd + weight * relation
        return total, {"total": total.detach(), "task": task.detach(), "logit": kd.detach(),
                       "score_delta": relation.detach(), "weighted_score_delta": (weight * relation).detach(),
                       **diagnostics}
