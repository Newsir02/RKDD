from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kd import (
    anchor_relation_loss,
    decoupled_kd_loss,
    logit_kd_loss,
    rank_wasserstein_loss,
    relation_slot_diversity_loss,
    rkd_angle_loss,
    rkd_distance_loss,
    temporal_kernel_distillation_loss,
)


class DistillationObjective(nn.Module):
    """Supervised and prediction losses with optional relation objectives."""

    def __init__(
        self,
        task_weight: float = 1.0,
        logit_weight: float = 1.0,
        rank_weight: float = 0.5,
        diversity_weight: float = 0.0,
        anchor_weight: float = 0.0,
        kernel_weight: float = 0.0,
        kernel_divergence: str = "kl",
        kernel_teacher_mode: str = "teacher",
        rkd_weight: float = 0.0,
        rkd_mode: str = "both",
        confidence_weight: float = 0.0,
        dkd_weight: float = 0.0,
        dkd_tckd_weight: float = 1.0,
        dkd_nckd_weight: float = 1.0,
        logit_temperature: float = 2.0,
        rank_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.task_weight = task_weight
        self.logit_weight = logit_weight
        self.rank_weight = rank_weight
        self.diversity_weight = diversity_weight
        self.anchor_weight = anchor_weight
        self.kernel_weight = kernel_weight
        self.kernel_divergence = kernel_divergence
        if kernel_teacher_mode not in {"teacher", "uniform", "recency", "shuffled"}:
            raise ValueError(
                "kernel_teacher_mode must be teacher, uniform, recency, or shuffled."
            )
        self.kernel_teacher_mode = kernel_teacher_mode
        self.rkd_weight = rkd_weight
        self.rkd_mode = rkd_mode
        self.confidence_weight = confidence_weight
        self.dkd_weight = dkd_weight
        self.dkd_tckd_weight = dkd_tckd_weight
        self.dkd_nckd_weight = dkd_nckd_weight
        self.logit_temperature = logit_temperature
        self.rank_temperature = rank_temperature

    def _controlled_teacher_mass(
        self,
        mass: Tensor,
        ages: Tensor | None,
        mask: Tensor,
    ) -> Tensor:
        """Build uniform/recency/shuffled controls for the TRID target."""

        mask = mask.bool()
        if self.kernel_teacher_mode == "uniform":
            raw = mask.to(mass.dtype)
        elif self.kernel_teacher_mode == "recency":
            if ages is None:
                raw = mask.to(mass.dtype)
            else:
                age = ages.to(mass.dtype).clamp(0.0, 1.0)
                raw = torch.exp(-age) * mask.to(mass.dtype)
        elif self.kernel_teacher_mode == "shuffled":
            # Permute the teacher mass within each row, keeping its multiset
            # but breaking the correspondence to historical event ids.
            random = torch.rand_like(mass)
            random = random.masked_fill(~mask, -1.0)
            order = random.argsort(dim=-1, descending=True)
            raw = torch.gather(mass, -1, order) * mask.to(mass.dtype)
        else:
            return mass
        return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def forward(
        self,
        student_output: Dict[str, Tensor],
        teacher_output: Dict[str, Tensor],
        target: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # task 是真实标签监督项，保证 student 仍直接优化下游排序目标。
        task = F.cross_entropy(student_output["logits"], target)
        # 以 student logits 为基准构造零标量：设备/dtype 一致，且权重为 0 时真正跳过计算。
        zero = student_output["logits"].sum() * 0.0

        # 教师置信度门控：用 top-1/top-2 概率 margin 重新加权每个 query 的 KD 信号。
        confidence = None
        if self.confidence_weight != 0.0:
            teacher_probability = torch.softmax(
                teacher_output["logits"].detach() / self.logit_temperature,
                dim=-1,
            )
            top_two = teacher_probability.topk(2, dim=-1).values
            margin = top_two[..., 0] - top_two[..., 1]
            scale = margin.mean().clamp_min(1.0e-8)
            confidence = (margin / scale).clamp(0.25, 4.0).pow(
                self.confidence_weight
            )

        # Prediction Distillation：对齐 teacher 与 student 的候选预测分布。
        if self.logit_weight != 0.0:
            logit = logit_kd_loss(
                student_output["logits"],
                teacher_output["logits"],
                self.logit_temperature,
                sample_weight=confidence,
            )
        else:
            logit = zero

        # DKD 对照：目标类与非目标类的输出蒸馏。
        if self.dkd_weight != 0.0:
            dkd = decoupled_kd_loss(
                student_output["logits"],
                teacher_output["logits"],
                target,
                self.logit_temperature,
                self.dkd_tckd_weight,
                self.dkd_nckd_weight,
            )
        else:
            dkd = zero

        # TARD 的候选排序关系：对齐 teacher 的排序结构。
        if self.rank_weight != 0.0:
            rank = rank_wasserstein_loss(
                student_output["logits"],
                teacher_output["logits"],
                self.rank_temperature,
                sample_weight=confidence,
            )
        else:
            rank = zero

        # Anchor Relation Distillation：以年龄为锚点对齐关系表示。
        if self.anchor_weight != 0.0:
            anchor = anchor_relation_loss(
                student_output["relation_tokens"],
                student_output["relation_ages"],
                teacher_output["relation_tokens"],
                teacher_output["relation_ages"],
                teacher_output["relation_mask"],
            )
        else:
            anchor = zero

        # 多样性正则避免所有关系槽学成同一个向量。
        if self.diversity_weight != 0.0:
            diversity = relation_slot_diversity_loss(
                student_output["relation_tokens"]
            )
        else:
            diversity = zero

        # Temporal Relational Influence Distillation：在共享事件支撑上对齐教师
        # 对历史交互的注意力质量，即对齐其条件时序影响强度（Hawkes 式核）。
        if self.kernel_weight != 0.0:
            if "kernel_mass" not in student_output:
                raise ValueError(
                    "kernel distillation requires a student with kernel aggregation."
                )
            if "relation_event_ids" not in teacher_output:
                raise ValueError(
                    "kernel distillation requires a teacher exposing relation_event_ids."
                )
            teacher_mass = self._controlled_teacher_mass(
                teacher_output["relation_mass"],
                teacher_output.get("relation_ages"),
                teacher_output["relation_mask"],
            )
            kernel = temporal_kernel_distillation_loss(
                student_output["kernel_mass"],
                student_output["kernel_event_ids"],
                student_output["kernel_mask"],
                teacher_mass,
                teacher_output["relation_event_ids"],
                teacher_output["relation_mask"],
                divergence=self.kernel_divergence,
            )
        else:
            kernel = zero

        # RKD：候选嵌入的距离/角度关系（与师生隐藏维度无关）。
        if self.rkd_weight != 0.0:
            if self.rkd_mode not in {"distance", "angle", "both"}:
                raise ValueError("rkd_mode must be distance, angle, or both.")
            if "candidate_embeddings" not in student_output:
                raise ValueError(
                    "RKD requires student candidate_embeddings."
                )
            if "candidate_embedding" in teacher_output:
                teacher_candidates = teacher_output["candidate_embedding"]
            elif "candidate_embeddings" in teacher_output:
                teacher_candidates = teacher_output["candidate_embeddings"]
            else:
                raise ValueError(
                    "RKD requires teacher candidate_embedding."
                )
            if self.rkd_mode == "distance":
                rkd = rkd_distance_loss(
                    student_output["candidate_embeddings"],
                    teacher_candidates,
                )
            elif self.rkd_mode == "angle":
                rkd = rkd_angle_loss(
                    student_output["candidate_embeddings"],
                    teacher_candidates,
                )
            else:
                rkd = 0.5 * (
                    rkd_distance_loss(
                        student_output["candidate_embeddings"],
                        teacher_candidates,
                    )
                    + rkd_angle_loss(
                        student_output["candidate_embeddings"],
                        teacher_candidates,
                    )
                )
        else:
            rkd = zero

        total = (
            self.task_weight * task
            + self.logit_weight * logit
            + self.rank_weight * rank
            + self.anchor_weight * anchor
            + self.diversity_weight * diversity
            + self.kernel_weight * kernel
            + self.rkd_weight * rkd
            + self.dkd_weight * dkd
        )
        # components 用于日志统计；detach 后避免日志路径保留计算图。
        components = {
            "total": total.detach(),
            "task": task.detach(),
            "logit": logit.detach(),
            "rank": rank.detach(),
            "anchor": anchor.detach(),
            "diversity": diversity.detach(),
            "kernel": kernel.detach(),
            "rkd": rkd.detach(),
            "dkd": dkd.detach(),
        }
        return total, components
