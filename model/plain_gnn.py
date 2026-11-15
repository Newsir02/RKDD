"""Ordinary, memoryless GNN evaluated on a causal, bounded graph snapshot."""
from __future__ import annotations

import torch
from torch import nn

from .layers import MLP, ResidualMLPBlock
from .temporal_neighbor_store import ExactTemporalNeighborStore


class PlainGNNStudent(nn.Module):
    def __init__(self, node_features, hidden_dim=32, neighbors=8, num_blocks=2):
        super().__init__()
        if node_features.ndim != 2 or node_features.size(0) == 0:
            raise ValueError("node_features must be a nonempty matrix.")
        if min(hidden_dim, neighbors, num_blocks) <= 0:
            raise ValueError("Student dimensions must be positive.")
        self.register_buffer("node_features", node_features.detach().clone().float())
        self.num_nodes, self.hidden_dim, self.neighbors = node_features.size(0), hidden_dim, neighbors
        self.model_config = dict(hidden_dim=hidden_dim, student_neighbors=neighbors,
                                 num_blocks=num_blocks, student_blocks=num_blocks,
                                 student_layers=num_blocks, student_arch="plain_gnn",
                                 num_nodes=self.num_nodes, node_feature_dim=node_features.size(1))
        # Deterministic paired views: dropout would create a false temporal change.
        self.node_encoder = MLP(node_features.size(1), [hidden_dim], hidden_dim, 0.0)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        self.neighbor_mlp = MLP(hidden_dim, [hidden_dim], hidden_dim, 0.0)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, 0.0) for _ in range(num_blocks)])
        self.score_head = MLP(4 * hidden_dim, [2 * hidden_dim, hidden_dim], 1, 0.0)
        self.neighbor_loader = self._make_store()

    def _make_store(self):
        return ExactTemporalNeighborStore(self.num_nodes, self.neighbors, self.node_features.device)

    def _apply(self, fn):
        result = super()._apply(fn)
        self.neighbor_loader = self._make_store()
        return result

    def _base_encode(self, ids):
        return self.node_encoder(self.node_features[ids]) + self.node_embeddings(ids)

    def _encode(self, ids):
        neighbors, events = self.neighbor_loader.query(ids)
        mask = (events >= 0).unsqueeze(-1)
        messages = self._base_encode(neighbors.clamp_min(0))
        context = (messages * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        state = self._base_encode(ids) + self.neighbor_mlp(context)
        for block in self.blocks:
            state = block(state)
        return state

    def forward(self, batch, return_relations=False):
        source = self._encode(batch["src"].long())
        candidates = self._encode(batch["candidates"].long())
        expanded = source.unsqueeze(1).expand_as(candidates)
        features = torch.cat([expanded, candidates, expanded * candidates,
                              (expanded - candidates).abs()], -1)
        return {"logits": self.score_head(features).squeeze(-1),
                "source_embedding": source, "candidate_embedding": candidates}

    @torch.no_grad()
    def reset_state(self):
        self.neighbor_loader.reset_state()

    @torch.no_grad()
    def update_state(self, batch):
        self.neighbor_loader.insert(batch["src"].long(), batch["positive_dst"].long(),
                                    event_ids=batch["event_index"])

    def runtime_state_elements(self):
        return self.neighbor_loader.neighbors.numel() + self.neighbor_loader.e_id.numel()
