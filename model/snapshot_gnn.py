"""Ordinary GCN / mean GraphSAGE on the observed bounded graph snapshot.

No time encoder, recurrent memory, edge features or learned temporal attention.
The external history store selects available edges, not a temporal neural model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from .temporal_neighbor_store import ExactTemporalNeighborStore
from .layers import MLP


class SnapshotGNN(nn.Module):
    def __init__(self, node_features, architecture="sage", hidden_dim=32, num_layers=2, neighbors=8):
        super().__init__()
        if architecture not in ("gcn", "sage"):
            raise ValueError("architecture must be gcn or sage")
        if min(hidden_dim, num_layers, neighbors) <= 0:
            raise ValueError("dimensions and neighbor budget must be positive")
        self.register_buffer("node_features", node_features.detach().float().clone())
        self.num_nodes = node_features.size(0)
        self.hidden_dim, self.num_layers, self.neighbors = hidden_dim, num_layers, neighbors
        self.architecture = architecture
        self.model_config = dict(architecture=architecture, hidden_dim=hidden_dim,
                                 num_layers=num_layers, neighbors=neighbors)
        self.encoder = nn.Linear(node_features.size(1), hidden_dim)
        # Ordinary transductive node features prevent featureless GraphSAGE
        # from collapsing every node to the same constant representation.
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, normalize=False, add_self_loops=False)
            if architecture == "gcn" else SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)])
        self.score_head = MLP(4 * hidden_dim, [hidden_dim], 1, 0.0)
        self._make_store()

    def _make_store(self):
        self.neighbor_loader = ExactTemporalNeighborStore(self.num_nodes, self.neighbors, self.node_features.device)
        self.last_event = -1

    def _apply(self, fn):
        result = super()._apply(fn)
        self._make_store()
        return result

    def _sample(self, query_nodes):
        # Expand each node once. Each retained interaction is a directed message
        # into its owner; repeated historical interactions remain parallel edges.
        nodes = torch.unique(query_nodes)
        frontier, visited = nodes, nodes.new_empty(0)
        sources, destinations = [], []
        for _ in range(self.num_layers):
            nbr, eid = self.neighbor_loader.query(frontier)
            mask = eid >= 0
            source = nbr[mask]
            destination = frontier[:, None].expand_as(nbr)[mask]
            sources.append(source)
            destinations.append(destination)
            visited = torch.unique(torch.cat([visited, frontier]))
            nodes = torch.unique(torch.cat([nodes, source]))
            frontier = nodes[~torch.isin(nodes, visited)]
        edge_src, edge_dst = torch.cat(sources), torch.cat(destinations)
        edges = torch.stack([torch.searchsorted(nodes, edge_src), torch.searchsorted(nodes, edge_dst)])
        # Snapshot degrees, not query-union degrees, prevent unrelated candidates
        # or evaluation chunking from changing GCN normalization.
        degree = (self.neighbor_loader.e_id[nodes] >= 0).sum(-1).to(self.node_features.dtype) + 1
        if self.architecture == "gcn":
            loops = torch.arange(nodes.numel(), device=nodes.device).repeat(2, 1)
            edges = torch.cat([edges, loops], 1)
        return nodes, edges, degree

    def forward(self, batch, return_relations=False):
        if "event_index" in batch and self.last_event >= int(batch["event_index"].min()):
            raise ValueError("Student history must end before the query batch")
        ids = torch.cat([batch["src"], batch["candidates"].reshape(-1)])
        nodes, edges, degree = self._sample(ids)
        h = self.encoder(self.node_features[nodes]) + self.node_embeddings(nodes)
        weights = (degree[edges[0]] * degree[edges[1]]).rsqrt()
        for layer in self.convs:
            h = layer(h, edges, weights) if self.architecture == "gcn" else layer(h, edges)
            h = F.relu(h)
        embeddings = h[torch.searchsorted(nodes, ids)]
        count = batch["src"].numel()
        source = embeddings[:count]
        candidates = embeddings[count:].reshape(count, batch["candidates"].size(1), self.hidden_dim)
        expanded = source[:, None].expand_as(candidates)
        features = torch.cat([expanded, candidates, expanded * candidates, (expanded - candidates).abs()], -1)
        return {"logits": self.score_head(features).squeeze(-1),
                "source_embedding": source, "candidate_embedding": candidates}

    @torch.no_grad()
    def reset_state(self):
        self.neighbor_loader.reset_state()
        self.last_event = -1

    @torch.no_grad()
    def update_state(self, batch):
        if int(batch["event_index"].min()) <= self.last_event:
            raise ValueError("History updates must be chronological and non-overlapping")
        self.neighbor_loader.insert(batch["src"], batch["positive_dst"], batch["event_index"])
        self.last_event = int(batch["event_index"].max())

    def runtime_state_elements(self):
        return self.neighbor_loader.neighbors.numel() + self.neighbor_loader.e_id.numel()
