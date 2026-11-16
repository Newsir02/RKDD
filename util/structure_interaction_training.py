"""Build matched historical views without mutating live stream state."""

import random

import torch
from torch.nn import functional as F

from loss.kd import logit_kd_loss
from .training import AverageMeter, move_to_device


def select_evidence(student_store, teacher_store, batch, max_events):
    """Find u-a-b-v paths with common arm event IDs in both readouts.

    The bridge is an incoming message at a or b in the student's two-hop
    computation. Its event is held fixed. Exact event ID intersections matter
    for the TGN ring buffer and repeated interactions.
    """
    if max_events < 1:
        raise ValueError("max_events must be positive")
    sources = batch["src"]
    positives = batch["candidates"].gather(1, batch["target"][:, None]).squeeze(1)
    left_nodes = student_store.neighbors[sources]
    right_nodes = student_store.neighbors[positives]
    left_ids = student_store.e_id[sources]
    right_ids = student_store.e_id[positives]
    left_support = student_store.neighbors[left_nodes.masked_fill(left_ids < 0, 0)]
    right_support = student_store.neighbors[right_nodes.masked_fill(right_ids < 0, 0)]
    left_support_ids = student_store.e_id[left_nodes.masked_fill(left_ids < 0, 0)]
    right_support_ids = student_store.e_id[right_nodes.masked_fill(right_ids < 0, 0)]
    teacher_left = teacher_store.e_id[sources]
    teacher_right = teacher_store.e_id[positives]
    fields = (
        sources, positives, left_nodes, left_ids, right_nodes, right_ids,
        left_support, left_support_ids, right_support, right_support_ids,
        teacher_left, teacher_right,
    )
    rows = zip(*(field.detach().cpu().tolist() for field in fields))
    for row, (u, v, ln, le, rn, re, ls, lse, rs, rse, tl, tr) in enumerate(rows):
        if u == v:
            continue
        allowed_l, allowed_r = set(tl), set(tr)
        left_groups, right_groups = {}, {}
        for a, eid in zip(ln, le):
            if eid >= 0 and eid in allowed_l and a not in (u, v):
                left_groups.setdefault(a, []).append(eid)
        for b, eid in zip(rn, re):
            if eid >= 0 and eid in allowed_r and b not in (u, v):
                right_groups.setdefault(b, []).append(eid)
        bridges_at_left = {a: {n for n, e in zip(nodes, ids) if e >= 0}
                           for a, arm, nodes, ids in zip(ln, le, ls, lse) if arm >= 0 and a in left_groups}
        bridges_at_right = {b: {n for n, e in zip(nodes, ids) if e >= 0}
                            for b, arm, nodes, ids in zip(rn, re, rs, rse) if arm >= 0 and b in right_groups}
        path = next(((a, b) for a in left_groups for b in right_groups
                     if a != b and (b in bridges_at_left[a] or a in bridges_at_right[b])), None)
        if path is not None:
            a, b = path
            yield (row, sources.new_tensor(sorted(left_groups[a], reverse=True)[:max_events]),
                   sources.new_tensor(sorted(right_groups[b], reverse=True)[:max_events]))


def _shared_pool(student_store, teacher_store, batch, row):
    """Candidate historical events visible to both readouts for one query."""
    source = int(batch["src"][row])
    positive = int(batch["candidates"][row, batch["target"][row]])
    if source == positive:
        return set()
    allowed = set(teacher_store.e_id[source].detach().cpu().tolist())
    allowed.update(teacher_store.e_id[positive].detach().cpu().tolist())
    allowed.discard(-1)
    pool = set()
    for node in (source, positive):
        for event in student_store.e_id[node].detach().cpu().tolist():
            if event >= 0 and event in allowed:
                pool.add(event)
        for neighbor in student_store.neighbors[node].detach().cpu().tolist():
            if neighbor < 0:
                continue
            for event in student_store.e_id[neighbor].detach().cpu().tolist():
                if event >= 0 and event in allowed:
                    pool.add(event)
    return pool


def select_random_evidence(student_store, teacher_store, batch, max_events, rng):
    """Matched control: same query and group sizes, random shared events.

    The real u-a-b-v path is not required. Events remain inside the student's
    two-hop support and the teacher's explicit readout, so the masking budget
    and visibility match the real-path condition as closely as this control
    permits. When enough alternatives exist, the real groups are excluded.
    """
    for row, real_a, real_b in select_evidence(student_store, teacher_store, batch, max_events):
        pool = _shared_pool(student_store, teacher_store, batch, row)
        needed = real_a.numel() + real_b.numel()
        if len(pool) < needed:
            continue
        real = set(real_a.tolist()) | set(real_b.tolist())
        alternatives = pool - real
        if len(alternatives) >= needed:
            pool = alternatives
        picked = rng.sample(sorted(pool), needed)
        random_a = batch["src"].new_tensor(picked[:real_a.numel()])
        random_b = batch["src"].new_tensor(picked[real_a.numel():])
        yield row, random_a, random_b


def pair_query(batch, row, reference="next", rng=None, teacher_scores=None):
    """Keep the positive and one reference from the already sampled list."""
    if reference not in ("next", "random", "hard"):
        raise ValueError("Unknown reference candidate mode")
    point = {key: value[row:row + 1] for key, value in batch.items()}
    positive = point["target"]
    count = point["candidates"].size(1)
    if reference == "next":
        chosen = (positive + 1) % count
    elif reference == "random":
        if rng is None:
            raise ValueError("Random reference requires a random generator")
        choices = [index for index in range(count) if index != int(positive)]
        chosen = torch.full_like(positive, rng.choice(choices))
    else:
        if teacher_scores is None:
            raise ValueError("Hard reference requires teacher scores")
        scores = teacher_scores[row:row + 1]
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, positive[:, None], True)
        chosen = scores.masked_fill(mask, float("-inf")).argmax(1)
    point["candidates"] = point["candidates"].gather(1, torch.stack((positive, chosen), 1))
    point["target"] = torch.zeros_like(positive)
    return point


def score_views(scorer, batch, a, b):
    views = []
    for excluded in (torch.cat((a, b)), b, a, None):
        view = batch if excluded is None else {**batch, "history_mask_event_ids": excluded}
        views.append(scorer(view, return_relations=False)["logits"])
    return torch.stack(views)


def train_structure_epoch(student, teacher, objective, loader, optimizer, device, args, epoch):
    student.train()
    teacher.eval()
    sampler = random.Random(args.seed + epoch)
    evidence_mode = getattr(args, "structure_evidence", "path")
    reference_mode = getattr(args, "structure_reference", "next")
    evidence_rng = random.Random(args.seed + epoch * 1009 + 1)
    reference_rng = random.Random(args.seed + epoch * 917 + 2)
    totals = {key: AverageMeter() for key in (
        "total", "task", "kd", "rel", "eligible_fraction", "distilled_fraction",
        "teacher_interaction_energy", "student_interaction_energy", "mean_group_events",
    )}
    eligible_queries = 0
    for raw in loader:
        batch = move_to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        logits = student(batch, return_relations=False)["logits"]
        task = F.cross_entropy(logits, batch["target"])
        kd = logits.new_zeros(())
        if objective.kd_weight:
            with torch.no_grad():
                teacher_logits = teacher.score_candidates(batch, return_relations=False)["logits"]
            kd = logit_kd_loss(logits, teacher_logits, objective.temperature)
        count = int(logits.size(0))
        if evidence_mode == "random":
            eligible = list(select_random_evidence(
                student.neighbor_loader, teacher.neighbor_loader, batch,
                args.structure_group_events, evidence_rng))
        else:
            eligible = list(select_evidence(student.neighbor_loader, teacher.neighbor_loader,
                                            batch, args.structure_group_events))
        eligible_queries += len(eligible)
        selected = sampler.sample(eligible, min(args.structure_queries_per_batch, len(eligible)))
        rel = logits.sum() * 0
        teacher_scores = None
        if selected and reference_mode == "hard":
            with torch.no_grad():
                teacher_scores = teacher.score_candidates(batch, return_relations=False)["logits"]
        if selected:
            student_views, teacher_views = [], []
            for row, a, b in selected:
                query = pair_query(batch, row, reference_mode, reference_rng, teacher_scores)
                student_views.append(score_views(student, query, a, b))
                with torch.no_grad():
                    teacher_views.append(score_views(teacher.score_candidates, query, a, b))
            rel, sd, td = objective(torch.cat(student_views, 1), torch.cat(teacher_views, 1))
            totals["teacher_interaction_energy"].update(float(td.square().mean()), len(selected))
            totals["student_interaction_energy"].update(float(sd.square().mean()), len(selected))
            mean_size = sum(a.numel() + b.numel() for _, a, b in selected) / (2 * len(selected))
            totals["mean_group_events"].update(mean_size, len(selected))
        total = task + objective.kd_weight * kd + objective.weight * rel
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        for name, value in (("total", float(total.detach())), ("task", float(task.detach())),
                            ("kd", float(kd.detach())), ("rel", float(rel.detach())),
                            ("eligible_fraction", len(eligible) / count),
                            ("distilled_fraction", len(selected) / count)):
            totals[name].update(value, count)
        student.update_state(batch)
        teacher.update_state(batch)
    if not eligible_queries:
        raise ValueError("No shared three-hop history motifs in the training split; inspect neighbor budgets and data")
    return {name: meter.average for name, meter in totals.items()}
