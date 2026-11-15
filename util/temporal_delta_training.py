"""Paired-prefix training without stale feature caches or teacher-state cloning."""
from __future__ import annotations

import torch

from loss.temporal_delta import query_relations
from .training import AverageMeter, move_to_device


def train_temporal_delta_epoch(student, teacher, objective, loader, optimizer, device,
                               grad_clip, ema_callback=None):
    student.train()
    teacher.eval()
    # At iteration k, state ends before batch k-1. Only the previous batch is
    # revealed between the two forwards of the *same* query k, with fixed weights.
    previous = None
    meters = {}
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        student_before = teacher_before = None
        if previous is not None and objective.needs_previous:
            student_before = student(batch, return_relations=False)
            with torch.no_grad():
                teacher_before = teacher.score_candidates(batch, return_relations=False)
        if previous is not None:
            student.update_state(previous)
            with torch.no_grad():
                teacher.update_state(previous)
        with torch.no_grad():
            teacher_after = teacher.score_candidates(batch, return_relations=False)
        student_after = student(batch, return_relations=False)
        total, parts = objective(student_after, teacher_after, batch["target"],
                                 student_before, teacher_before)
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if ema_callback is not None:
            ema_callback()
        count = batch["target"].numel()
        for name, value in parts.items():
            meters.setdefault(name, AverageMeter()).update(float(value), count)
        if teacher_before is not None:
            with torch.no_grad():
                dt = query_relations(teacher_after) - query_relations(teacher_before)
                ds = query_relations(student_after) - query_relations(student_before)
                for name, value in (("teacher_delta_energy", dt.square().mean()),
                                    ("student_delta_energy", ds.square().mean())):
                    meters.setdefault(name, AverageMeter()).update(float(value), count)
        previous = batch
    if previous is not None:
        student.update_state(previous)
        with torch.no_grad():
            teacher.update_state(previous)
    return {name: meter.average for name, meter in meters.items()}
