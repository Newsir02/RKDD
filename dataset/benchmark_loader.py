"""Uniform candidate sampling without a full node-domain permutation per event.

The original loader and its random sequence remain unchanged for legacy runs.
"""
import random

import torch

from .temporal import CandidateBatchLoader


def sample_candidate_batch(batch, domain, domain_members, num_candidates, generator):
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least two")
    src, positive_dst = batch.src.long().cpu(), batch.dst.long().cpu()
    candidates = torch.empty(src.numel(), num_candidates, dtype=torch.long)
    target = torch.empty(src.numel(), dtype=torch.long)
    required = num_candidates - 1
    for row, (source, positive) in enumerate(zip(src.tolist(), positive_dst.tolist())):
        forbidden = {source, positive}
        available = len(domain) - sum(node in domain_members for node in forbidden)
        if available < required:
            raise ValueError("Not enough destination nodes for negative sampling")
        # A random ordered sample, with at most two forbidden entries removed,
        # remains a uniform ordered sample of the allowed domain. Taking K+2
        # guarantees K survivors, without rejection loops or O(|V|) allocation.
        indices = generator.sample(range(len(domain)), min(len(domain), required + 2))
        negatives = [domain[i] for i in indices if domain[i] not in forbidden][:required]
        position = generator.randrange(num_candidates)
        candidates[row] = torch.tensor(negatives[:position] + [positive] + negatives[position:])
        target[row] = position
    return {"event_index": batch.event_index.long().cpu(), "src": src,
            "positive_dst": positive_dst, "candidates": candidates, "target": target,
            "timestamp": batch.timestamp.float().cpu(), "event_time": batch.t.long().cpu(),
            "edge_features": batch.msg.float().cpu()}


class BenchmarkCandidateLoader(CandidateBatchLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = self.bundle.destination_nodes.long().cpu().tolist()
        self.domain_members = set(self.domain)
        if len(self.domain) != len(self.domain_members):
            raise ValueError("Destination domain must contain unique node IDs")

    def __iter__(self):
        generator = random.Random(self.seed)
        for batch in self.temporal_loader:
            yield sample_candidate_batch(batch, self.domain, self.domain_members,
                                         self.num_candidates, generator)


def replace_candidate_loaders(bundle, args):
    return tuple(BenchmarkCandidateLoader(bundle=bundle, split=split, seed=args.seed + offset,
                                         batch_size=args.batch_size, num_candidates=args.num_candidates,
                                         num_workers=args.num_workers)
                 for offset, split in enumerate(("train", "val", "test")))
