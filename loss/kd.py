from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def logit_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    """Listwise KL distillation over candidates belonging to the same query."""

    # temperature 软化 teacher 分布，让 student 学到非目标候选之间的相对偏好。
    teacher_probability = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probability = torch.log_softmax(student_logits / temperature, dim=-1)
    per_sample = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)
    if sample_weight is None:
        return per_sample.mean() * (temperature**2)
    weights = sample_weight.to(per_sample.dtype).flatten()
    return (
        (per_sample * weights).sum() / weights.sum().clamp_min(1.0e-8)
    ) * (temperature**2)


def decoupled_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target: Tensor,
    temperature: float = 2.0,
    tckd_weight: float = 1.0,
    nckd_weight: float = 1.0,
) -> Tensor:
    """Decoupled knowledge distillation (DKD, CVPR 2022).

    The usual KD loss couples target-class knowledge (TCKD) and non-target
    class knowledge (NCKD).  DKD splits them so the informative non-target
    relations can be weighted independently.
    """

    if student_logits.dim() != 2 or teacher_logits.shape != student_logits.shape:
        raise ValueError("DKD expects matching [batch, candidates] logits.")
    if temperature <= 0.0:
        raise ValueError("DKD temperature must be positive.")
    if tckd_weight < 0.0 or nckd_weight < 0.0:
        raise ValueError("DKD weights must be non-negative.")
    target = target.long().flatten()
    if target.numel() != student_logits.size(0):
        raise ValueError("DKD target must contain one index per row.")

    teacher_probability = torch.softmax(
        teacher_logits.detach() / temperature, dim=-1
    )
    student_probability = torch.softmax(student_logits / temperature, dim=-1)
    index = target.unsqueeze(-1)
    teacher_target = teacher_probability.gather(-1, index).squeeze(-1)
    student_target = student_probability.gather(-1, index).squeeze(-1)

    teacher_binary = torch.stack(
        [teacher_target, 1.0 - teacher_target], dim=-1
    ).clamp_min(1.0e-8)
    student_binary = torch.stack(
        [student_target, 1.0 - student_target], dim=-1
    ).clamp_min(1.0e-8)
    tckd = F.kl_div(
        student_binary.log(),
        teacher_binary,
        reduction="batchmean",
    ) * (temperature**2)

    teacher_other = teacher_probability.clone()
    student_other = student_probability.clone()
    teacher_other.scatter_(-1, index, 0.0)
    student_other.scatter_(-1, index, 0.0)
    nckd = F.kl_div(
        student_other.clamp_min(1.0e-8).log(),
        teacher_other,
        reduction="batchmean",
    ) * (temperature**2)
    return tckd_weight * tckd + nckd_weight * nckd


def rank_wasserstein_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    """One-dimensional W1 distance on teacher-ordered candidate identities.

    Both distributions are gathered with the teacher permutation, so transport
    cannot obtain a small loss by matching two different candidate identities.
    """

    if student_logits.size(-1) < 2:
        return student_logits.sum() * 0.0
    # 以 teacher 的排序作为一维支撑，保证同一个候选身份之间做 transport。
    teacher_order = torch.argsort(teacher_logits.detach(), dim=-1, descending=True)
    teacher_probability = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_probability = torch.softmax(student_logits / temperature, dim=-1)
    teacher_ordered = torch.gather(teacher_probability, -1, teacher_order)
    student_ordered = torch.gather(student_probability, -1, teacher_order)

    # 对一维有序支撑上的分布，W1 等价于 CDF 差值的 L1 距离；最后一个 CDF 恒为 1。
    cdf_difference = torch.cumsum(teacher_ordered - student_ordered, dim=-1)[..., :-1].abs()
    # 排名靠前的候选对推荐质量更重要，因此给前部 CDF 更高权重。
    positions = torch.arange(
        2,
        student_logits.size(-1) + 1,
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    weights = 1.0 / torch.log2(positions + 1.0)
    weights = weights / weights.mean().clamp_min(1.0e-8)
    cdf_difference = cdf_difference * weights
    per_sample = cdf_difference.mean(dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    sample_weight = sample_weight.to(per_sample.dtype).flatten()
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0e-8)


def masked_relation_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    """Masked mean over the variable-length relation token axis."""

    weights = mask.unsqueeze(-1).to(tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


def relation_alignment_loss(
    student_repr: Tensor,
    teacher_repr: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Align anchor relation representations with cosine plus MSE."""

    target = teacher_repr.detach()
    if mask is not None:
        valid = mask.bool()
        if not bool(valid.any()):
            return student_repr.sum() * 0.0
        student_repr = student_repr[valid]
        target = target[valid]
    # cosine 对齐方向，MSE 对齐尺度，两者互补避免只学到方向或只学到幅度。
    cosine = 1.0 - F.cosine_similarity(student_repr, target, dim=-1, eps=1.0e-8)
    squared = F.mse_loss(student_repr, target, reduction="none").mean(dim=-1)
    return (cosine + squared).mean()


def anchor_relation_loss(
    student_tokens: Tensor,
    student_ages: Tensor,
    teacher_tokens: Tensor,
    teacher_ages: Tensor,
    teacher_mask: Tensor,
) -> Tensor:
    """Age-anchored relation representation distillation.

    Teacher relation tokens and student relation slots are both ordered by age.
    We use the normalized age as an anchor to build a stable correspondence:
    every student slot is matched to the valid teacher token with the closest
    age and then aligned in representation space. This transfers the teacher's
    relation content without requiring equal token counts.
    """

    teacher_mask = teacher_mask.bool()
    valid_rows = teacher_mask.any(dim=-1)
    if not bool(valid_rows.any()):
        return student_tokens.sum() * 0.0
    # 教师年龄按全局时间跨度归一化后可能非常小（近期历史都接近 0），学生槽年龄却
    # 遍布 [0,1]。必须逐样本把两侧年龄各自重标定到 [0,1]，否则所有学生槽会匹配到
    # 同一个（最近的）教师 token，锚点匹配退化为常数。
    positive_inf = torch.full(
        (),
        float("inf"),
        device=teacher_ages.device,
        dtype=teacher_ages.dtype,
    )
    negative_inf = torch.full(
        (),
        float("-inf"),
        device=teacher_ages.device,
        dtype=teacher_ages.dtype,
    )
    teacher_min = torch.where(
        teacher_mask, teacher_ages, positive_inf
    ).amin(dim=-1, keepdim=True)
    teacher_max = torch.where(
        teacher_mask, teacher_ages, negative_inf
    ).amax(dim=-1, keepdim=True)
    teacher_norm = (teacher_ages - teacher_min) / (
        teacher_max - teacher_min
    ).clamp_min(1.0e-6)
    teacher_norm = torch.where(
        teacher_mask,
        teacher_norm,
        torch.zeros_like(teacher_norm),
    )
    student_min = student_ages.amin(dim=-1, keepdim=True)
    student_max = student_ages.amax(dim=-1, keepdim=True)
    student_norm = (student_ages - student_min) / (
        student_max - student_min
    ).clamp_min(1.0e-6)
    # 年龄差的绝对值作为匹配代价，无效 teacher token 置为极大值后取最近邻。
    distance = (student_norm.unsqueeze(-1) - teacher_norm.unsqueeze(1)).abs()
    negative_large = torch.full(
        (),
        1.0e9,
        device=distance.device,
        dtype=distance.dtype,
    )
    distance = torch.where(teacher_mask.unsqueeze(1), distance, negative_large)
    nearest = distance.argmin(dim=-1)
    gather_index = nearest.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1))
    target = torch.gather(teacher_tokens, 1, gather_index).detach()
    slot_valid = valid_rows.unsqueeze(1).expand_as(student_ages)
    cosine = 1.0 - F.cosine_similarity(student_tokens, target, dim=-1, eps=1.0e-8)
    squared = F.mse_loss(student_tokens, target, reduction="none").mean(dim=-1)
    return (cosine + squared)[slot_valid].mean()


def relation_slot_diversity_loss(slots: Tensor) -> Tensor:
    """Prevent train-only relation slots from collapsing to one vector."""

    if slots.size(1) < 2:
        return slots.sum() * 0.0
    # 惩罚不同槽之间的余弦相似度，促使槽学习互补的关系模式。
    normalized = F.normalize(slots, p=2, dim=-1, eps=1.0e-8)
    similarities = torch.bmm(normalized, normalized.transpose(1, 2))
    identity = torch.eye(slots.size(1), device=slots.device, dtype=slots.dtype).unsqueeze(0)
    off_diagonal = 1.0 - identity
    denominator = slots.size(0) * off_diagonal.sum().clamp_min(1.0)
    return ((similarities * off_diagonal) ** 2).sum() / denominator


def _off_diagonal_mask(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return 1.0 - torch.eye(count, device=device, dtype=dtype).unsqueeze(0)


def rkd_distance_loss(
    student_embeddings: Tensor,
    teacher_embeddings: Tensor,
    eps: float = 1.0e-8,
) -> Tensor:
    """Distance-wise RKD on candidate embeddings (scale-normalised)."""

    if student_embeddings.size(1) < 2:
        return student_embeddings.sum() * 0.0
    student_distance = torch.cdist(student_embeddings, student_embeddings, p=2)
    teacher_distance = torch.cdist(
        teacher_embeddings.detach(), teacher_embeddings.detach(), p=2
    )
    mask = _off_diagonal_mask(
        student_embeddings.size(1),
        student_embeddings.device,
        student_embeddings.dtype,
    )
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    student_scale = (student_distance * mask).sum(dim=(1, 2)) / denominator
    teacher_scale = (teacher_distance * mask).sum(dim=(1, 2)) / denominator
    student_normalized = student_distance / student_scale.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    teacher_normalized = teacher_distance / teacher_scale.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    difference = (student_normalized - teacher_normalized) * mask
    return difference.square().sum(dim=(1, 2)).div(denominator).mean()


def rkd_angle_loss(
    student_embeddings: Tensor,
    teacher_embeddings: Tensor,
    eps: float = 1.0e-8,
) -> Tensor:
    """Angle-wise RKD on candidate triples (dimension-agnostic)."""

    count = student_embeddings.size(1)
    if count < 3:
        return student_embeddings.sum() * 0.0
    student_difference = (
        student_embeddings.unsqueeze(2) - student_embeddings.unsqueeze(1)
    )
    teacher_difference = (
        teacher_embeddings.detach().unsqueeze(2)
        - teacher_embeddings.detach().unsqueeze(1)
    )
    student_unit = F.normalize(student_difference, p=2, dim=-1, eps=eps)
    teacher_unit = F.normalize(teacher_difference, p=2, dim=-1, eps=eps)
    student_cosine = torch.einsum("bijh,bkjh->bijk", student_unit, student_unit)
    teacher_cosine = torch.einsum("bijh,bkjh->bijk", teacher_unit, teacher_unit)
    index = torch.arange(count, device=student_embeddings.device)
    anchor = index.view(1, -1, 1, 1)
    first = index.view(1, 1, -1, 1)
    second = index.view(1, 1, 1, -1)
    valid = (anchor != first) & (anchor != second) & (first != second)
    difference = (student_cosine - teacher_cosine) * valid.to(student_cosine.dtype)
    return difference.square().sum(dim=(1, 2, 3)).div(
        valid.sum(dim=(1, 2, 3)).clamp_min(1)
    ).mean()


def temporal_kernel_distillation_loss(
    student_mass: Tensor,
    student_event_ids: Tensor,
    student_mask: Tensor,
    teacher_mass: Tensor,
    teacher_event_ids: Tensor,
    teacher_mask: Tensor,
    divergence: str = "kl",
) -> Tensor:
    """Match the teacher's temporal influence distribution on shared support.

    Teacher and student observe the same event stream, so the historical
    interaction support is *known*: instead of estimating a transport plan
    between two unrelated token sets, we look up the teacher's mass for the
    exact event ids used by the student and compare the two conditional
    distributions.  This turns relation distillation into a direct
    distribution-matching problem over the teacher's temporal influence
    operator (a discrete conditional intensity / Hawkes-style kernel).

    Args:
        student_mass: ``[B, K]`` normalised student influence weights.
        student_event_ids: ``[B, K]`` event ids of the student support (-1 pad).
        student_mask: ``[B, K]`` valid student slots.
        teacher_mass: ``[B, N]`` normalised teacher attention mass.
        teacher_event_ids: ``[B, N]`` event ids of the teacher support.
        teacher_mask: ``[B, N]`` valid teacher slots.
        divergence: ``kl``, ``js`` or ``mse`` (attention-transfer style control).
    """

    if divergence not in {"kl", "js", "mse"}:
        raise ValueError("divergence must be one of: kl, js, mse.")
    student_mask = student_mask.bool()
    teacher_mask = teacher_mask.bool()
    if not bool(student_mask.any()) or not bool(teacher_mask.any()):
        return student_mass.sum() * 0.0
    if student_event_ids.shape != student_mass.shape:
        raise ValueError("student_event_ids must match student_mass shape.")
    if teacher_event_ids.shape != teacher_mass.shape:
        raise ValueError("teacher_event_ids must match teacher_mass shape.")

    # [B, K, N] exact-support match. Teacher ids are unique within a row, so
    # summing matched masses is equivalent to a lookup.
    matched = (
        student_event_ids.unsqueeze(-1) == teacher_event_ids.unsqueeze(-2)
    )
    matched = (
        matched
        & teacher_mask.unsqueeze(-2)
        & student_mask.unsqueeze(-1)
    )
    teacher_target = (
        teacher_mass.unsqueeze(-2)
        * matched.to(teacher_mass.dtype)
    ).sum(dim=-1)
    student_target = student_mass * matched.any(dim=-1).to(student_mass.dtype)

    teacher_total = teacher_target.sum(dim=-1, keepdim=True)
    student_total = student_target.sum(dim=-1, keepdim=True)
    valid = (teacher_total.squeeze(-1) > 1.0e-8) & (
        student_total.squeeze(-1) > 1.0e-8
    )
    if not bool(valid.any()):
        return student_mass.sum() * 0.0
    teacher_probability = teacher_target[valid] / teacher_total[valid]
    student_probability = student_target[valid] / student_total[valid]
    log_teacher = teacher_probability.clamp_min(1.0e-12).log()
    log_student = student_probability.clamp_min(1.0e-12).log()
    if divergence == "kl":
        loss = (teacher_probability * (log_teacher - log_student)).sum(dim=-1)
    elif divergence == "js":
        mixture = 0.5 * (teacher_probability + student_probability)
        log_mixture = mixture.clamp_min(1.0e-12).log()
        loss = 0.5 * (
            (teacher_probability * (log_teacher - log_mixture)).sum(dim=-1)
            + (student_probability * (log_student - log_mixture)).sum(dim=-1)
        )
    else:
        loss = (teacher_probability - student_probability).square().sum(dim=-1)
    return loss.mean()

