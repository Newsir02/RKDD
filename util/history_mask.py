"""Read-only masking of already-observed historical event IDs."""

import torch


def history_mask(batch):
    mask = batch.get("history_mask_event_ids")
    if mask is not None:
        if mask.ndim != 1 or mask.dtype != torch.long:
            raise ValueError("history_mask_event_ids must be a one-dimensional long tensor")
        if torch.any(mask < 0) or torch.any(mask >= batch["event_index"].min()):
            raise ValueError("Cannot mask an unobserved or invalid history event")
    return mask


def visible_history(event_ids, excluded):
    visible = event_ids >= 0
    if excluded is not None:
        visible = visible & ~torch.isin(event_ids, excluded)
    return visible
