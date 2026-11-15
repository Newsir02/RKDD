"""Independent paper objectives; provenance and adaptation limits in distillation_methods.md."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .kd import logit_kd_loss
from .temporal_delta import TemporalDeltaDistillation
from .score_delta import ScoreDeltaDistillation
from .structure_interaction import StructureInteractionObjective
from .temporal_paper_adaptations import TEMPORAL_PAPER_METHODS, TemporalPaperObjective


class StandardizedLogitKD(nn.Module):
    """CVPR 2024: official implementation uses sample std (correction=1), eps=1e-7."""
    def __init__(self, temperature=2.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, student, teacher):
        if student.shape != teacher.shape or student.size(-1) < 2:
            raise ValueError("Matching logits with at least two candidates are required")
        def standardize(value):
            return (value - value.mean(-1, keepdim=True)) / (value.std(-1, keepdim=True) + 1e-7)
        s, t = standardize(student), standardize(teacher.detach())
        return self.temperature ** 2 * F.kl_div(
            F.log_softmax(s / self.temperature, -1),
            F.softmax(t / self.temperature, -1), reduction="batchmean")


class NOSMOGRelationKD(nn.Module):
    """NOSMOG's projected raw Gram MSE component, not the complete NOSMOG algorithm.

    The caller supplies node embeddings from a common historical prefix.
    No cosine/row normalization: the official code uses unnormalized dot products.
    """
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(student_dim, teacher_dim), nn.ReLU())

    def forward(self, student, teacher):
        s = self.projector(student)
        t = teacher.detach()
        return F.mse_loss(s @ s.T, t @ t.T)


def paper_dkd(student, teacher, target, temperature, alpha=1., beta=8.):
    """DKD with separately normalized non-target probabilities (CVPR 2022).

    Exact removal of the target column replaces the official -1000 mask;
    logsumexp computes the binary KL without cancellation at extreme logits.
    """
    if student.ndim != 2 or student.shape != teacher.shape or student.size(1) < 2:
        raise ValueError("DKD requires matching [batch, candidates>=2] logits")
    mask = torch.zeros_like(student, dtype=torch.bool).scatter_(1, target[:, None], True)
    s, t = student / temperature, teacher.detach() / temperature
    sn, tn = s[~mask].reshape(s.size(0), -1), t[~mask].reshape(t.size(0), -1)
    sb = torch.stack([s[mask], torch.logsumexp(sn, -1)], -1)
    tb = torch.stack([t[mask], torch.logsumexp(tn, -1)], -1)
    tckd = F.kl_div(F.log_softmax(sb, -1), F.softmax(tb, -1), reduction="batchmean")
    nckd = F.kl_div(F.log_softmax(sn, -1), F.softmax(tn, -1), reduction="batchmean")
    return temperature ** 2 * (alpha * tckd + beta * nckd)


def paper_rkd(student, teacher, mode):
    """Official RKD Huber objective, averaged over query-local node sets.

    Unlike the retained legacy MSE controls, diagonal/degenerate entries are
    included in the final mean, as in the official CVPR 2019 implementation.
    """
    losses = []
    for s, t in zip(student, teacher.detach()):
        if mode == "distance":
            def distance(x):
                squared = (x[:, None] - x[None, :]).square().sum(-1)
                d = squared.clamp_min(1e-12).sqrt()
                d = d * (1 - torch.eye(x.size(0), device=x.device, dtype=x.dtype))
                nonzero = d[d > 0]
                scale = nonzero.mean() if nonzero.numel() else d.new_tensor(1.)
                return d / scale.clamp_min(1e-12)
            left, right = distance(s), distance(t)
        elif mode == "angle":
            def angle(x):
                unit = F.normalize(x[None, :] - x[:, None], dim=-1)
                return unit @ unit.transpose(-1, -2)
            left, right = angle(s), angle(t)
        else:
            raise ValueError("RKD mode must be distance or angle")
        losses.append(F.smooth_l1_loss(left, right))
    return torch.stack(losses).mean()


class PaperObjective(nn.Module):
    needs_previous = False

    def __init__(self, method, student_dim, teacher_dim, temperature=2.0,
                 kd_weight=1.0, relation_weight=1.0):
        super().__init__()
        if method not in ("logits", "dkd", "rkd_distance", "rkd_angle", "logit_standard", "nosmog_rsd"):
            raise ValueError(f"Unknown paper objective: {method}")
        if not all(math.isfinite(w) and w >= 0 for w in (kd_weight, relation_weight)):
            raise ValueError("Weights must be finite and nonnegative")
        self.method, self.temperature = method, temperature
        self.kd_weight, self.relation_weight = kd_weight, relation_weight
        self.standardized = StandardizedLogitKD(temperature)
        self.relation = NOSMOGRelationKD(student_dim, teacher_dim) if method == "nosmog_rsd" else None
        self.dkd_alpha, self.dkd_beta, self.dkd_warmup, self.epoch = 1., 8., 20, 1

    def forward(self, student, teacher, target, student_before=None, teacher_before=None):
        task = F.cross_entropy(student["logits"], target)
        s, t = student["logits"], teacher["logits"].detach()
        if self.method == "logit_standard":
            kd = self.standardized(s, t)
        elif self.method == "dkd":
            kd = paper_dkd(s, t, target, self.temperature, self.dkd_alpha, self.dkd_beta)
            kd = kd * (min(self.epoch / self.dkd_warmup, 1.) if self.dkd_warmup else 1.)
        else:
            kd = logit_kd_loss(s, t, self.temperature)
        relation = s.sum() * 0
        if self.method in ("rkd_distance", "rkd_angle", "nosmog_rsd"):
            sn = torch.cat([student["source_embedding"][:, None], student["candidate_embedding"]], 1)
            tn = torch.cat([teacher["source_embedding"][:, None], teacher["candidate_embedding"]], 1).detach()
            if self.method == "nosmog_rsd":
                # Sampled occurrences, including repeats, exactly shared by teacher/student.
                relation = self.relation(sn.flatten(0, 1), tn.flatten(0, 1))
            else:
                relation = paper_rkd(sn, tn, self.method.removeprefix("rkd_"))
        total = task + self.kd_weight * kd + self.relation_weight * relation
        return total, {"total": total.detach(), "task": task.detach(),
                       "kd": kd.detach(), "relation": relation.detach()}


METHODS = ("supervised", "logits", "dkd", "rkd_distance", "rkd_angle",
           "logit_standard", "nosmog_rsd", "trd", "trd_score", "structure_interaction") + TEMPORAL_PAPER_METHODS


def make_objective(method, student_dim, teacher_dim, temperature=2.0,
                   kd_weight=1.0, relation_weight=1.0, trd_mode="temporal", score_delta_warmup=0,
                   score_delta_gate=False, *, structure_mode="temporal", structure_huber_beta=1.):
    if method == "structure_interaction":
        return StructureInteractionObjective(relation_weight, structure_mode, structure_huber_beta)
    if method in TEMPORAL_PAPER_METHODS:
        return TemporalPaperObjective(method, student_dim, teacher_dim, temperature, kd_weight, relation_weight)
    if method == "trd":
        return TemporalDeltaDistillation(kd_weight, relation_weight, temperature, trd_mode)
    if method == "trd_score":
        return ScoreDeltaDistillation(kd_weight, relation_weight, temperature, trd_mode,
                                      score_delta_warmup, score_delta_gate)
    return PaperObjective(method, student_dim, teacher_dim, temperature, kd_weight, relation_weight)
