"""Build matched historical views without mutating live stream state."""

import random

import torch
from torch.nn import functional as F

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


def pair_query(batch, row):
    """Keep the positive and a fixed reference from the already sampled list."""
    point = {key: value[row:row + 1] for key, value in batch.items()}
    positive = point["target"]
    reference = (positive + 1) % point["candidates"].size(1)
    point["candidates"] = point["candidates"].gather(1, torch.stack((positive, reference), 1))
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
    totals = {key: AverageMeter() for key in (
        "total", "task", "rel", "eligible_fraction", "distilled_fraction",
        "teacher_interaction_energy", "student_interaction_energy", "mean_group_events",
    )}
    eligible_queries = 0
    for raw in loader:
        batch = move_to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        logits = student(batch, return_relations=False)["logits"]
        task = F.cross_entropy(logits, batch["target"])
        count = int(logits.size(0))
        eligible = list(select_evidence(student.neighbor_loader, teacher.neighbor_loader,
                                        batch, args.structure_group_events))
        eligible_queries += len(eligible)
        selected = sampler.sample(eligible, min(args.structure_queries_per_batch, len(eligible)))
        rel = logits.sum() * 0
        if selected:
            student_views, teacher_views = [], []
            for row, a, b in selected:
                query = pair_query(batch, row)
                student_views.append(score_views(student, query, a, b))
                with torch.no_grad():
                    teacher_views.append(score_views(teacher.score_candidates, query, a, b))
            rel, sd, td = objective(torch.cat(student_views, 1), torch.cat(teacher_views, 1))
            totals["teacher_interaction_energy"].update(float(td.square().mean()), len(selected))
            totals["student_interaction_energy"].update(float(sd.square().mean()), len(selected))
            mean_size = sum(a.numel() + b.numel() for _, a, b in selected) / (2 * len(selected))
            totals["mean_group_events"].update(mean_size, len(selected))
        total = task + objective.weight * rel
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        for name, value in (("total", float(total.detach())), ("task", float(task.detach())),
                            ("rel", float(rel.detach())), ("eligible_fraction", len(eligible) / count),
                            ("distilled_fraction", len(selected) / count)):
            totals[name].update(value, count)
        student.update_state(batch)
        teacher.update_state(batch)
    if not eligible_queries:
        raise ValueError("No shared three-hop history motifs in the training split; inspect neighbor budgets and data")
    return {name: meter.average for name, meter in totals.items()}
