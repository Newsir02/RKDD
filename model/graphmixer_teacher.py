from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .layers import MLP, TimeEncoding
from .temporal_neighbor_store import ExactTemporalNeighborStore


class GraphMixerTeacher(nn.Module):
    """Memory-free temporal GNN teacher with fixed time encoding and neighbor mixing."""

    def __init__(
        self,
        event_times: Tensor,
        event_messages: Tensor,
        num_nodes: int,
        hidden_dim: int = 64,
        time_dim: int = 16,
        relation_dim: int = 64,
        temporal_neighbors: int = 24,
        dropout: float = 0.1,
        node_features: Optional[Tensor] = None,
        time_span: float | None = None,
    ) -> None:
        super().__init__()
        event_times = torch.as_tensor(event_times, dtype=torch.long).flatten()
        event_messages = torch.as_tensor(event_messages, dtype=torch.float)
        if event_times.numel() == 0:
            raise ValueError("GraphMixer requires at least one temporal event.")
        if event_messages.dim() != 2 or event_messages.size(0) != event_times.numel():
            raise ValueError("event_messages must have shape [events, features].")
        if event_messages.size(1) == 0:
            raise ValueError("GraphMixer event messages must have at least one feature.")
        if not bool(torch.isfinite(event_messages).all()):
            raise ValueError("GraphMixer event messages must be finite.")
        if event_times.numel() > 1 and bool(
            (event_times[1:] < event_times[:-1]).any()
        ):
            raise ValueError("GraphMixer event times must be chronological.")
        if num_nodes <= 0 or hidden_dim <= 0 or time_dim <= 0:
            raise ValueError("GraphMixer node and hidden dimensions must be positive.")
        if relation_dim <= 0 or temporal_neighbors <= 0:
            raise ValueError("GraphMixer relation dimensions must be positive.")

        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.relation_dim = int(relation_dim)
        self.temporal_neighbors = int(temporal_neighbors)
        self.msg_dim = int(event_messages.size(1))
        feature_dim = 0
        if node_features is not None:
            node_features = torch.as_tensor(node_features, dtype=torch.float)
            if node_features.dim() != 2 or node_features.size(0) != num_nodes:
                raise ValueError(
                    "GraphMixer node_features must have shape [num_nodes, features]."
                )
            if node_features.size(1) == 0:
                raise ValueError("GraphMixer node_features must not be empty.")
            feature_dim = int(node_features.size(1))
        self.model_config = {
            "teacher_arch": "graphmixer",
            "num_nodes": self.num_nodes,
            "num_events": int(event_times.numel()),
            "msg_dim": self.msg_dim,
            "hidden_dim": self.hidden_dim,
            "time_dim": self.time_dim,
            "relation_dim": self.relation_dim,
            "temporal_neighbors": self.temporal_neighbors,
            "dropout": dropout,
            "teacher_node_features": node_features is not None,
            "node_feature_dim": feature_dim,
        }

        self.register_buffer("event_times", event_times.detach().clone(), persistent=False)
        if time_span is not None:
            self.model_config["time_span"] = float(time_span)
        span = (event_times[-1].double() - event_times[0].double() if time_span is None
                else event_times.new_tensor(time_span, dtype=torch.float64))
        self.register_buffer("time_span", span.clamp_min(1.0), persistent=False)
        self.register_buffer(
            "event_messages", event_messages.detach().clone(), persistent=False
        )
        self.register_buffer(
            "stream_event_index",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "stream_timestamp",
            torch.tensor(torch.iinfo(torch.long).min, dtype=torch.long),
            persistent=False,
        )
        self.time_encoder = TimeEncoding(self.time_dim)
        if node_features is not None:
            self.register_buffer(
                "node_features", node_features.detach().clone(), persistent=False
            )
            self.node_encoder: Optional[nn.Module] = MLP(
                feature_dim, [self.hidden_dim], self.hidden_dim, dropout
            )
            self.node_embeddings = None
        else:
            self.node_encoder = None
            self.node_embeddings = nn.Embedding(self.num_nodes, self.hidden_dim)
            nn.init.normal_(self.node_embeddings.weight, std=0.02)
        self.edge_encoder = MLP(
            self.msg_dim + self.time_dim,
            [self.hidden_dim],
            self.hidden_dim,
            dropout,
        )
        self.node_mixer = MLP(
            3 * self.hidden_dim + self.time_dim + 1,
            [self.hidden_dim],
            self.hidden_dim,
            dropout,
        )
        self.link_classifier = MLP(
            4 * self.hidden_dim,
            [2 * self.hidden_dim, self.hidden_dim],
            1,
            dropout,
        )
        self.relation_projector = MLP(
            self.hidden_dim + self.time_dim + self.msg_dim,
            [self.relation_dim],
            self.relation_dim,
            dropout,
        )
        self.relation_align_head = nn.Linear(self.relation_dim, self.hidden_dim)
        self.neighbor_loader = self._make_neighbor_store()

    def _make_neighbor_store(self) -> ExactTemporalNeighborStore:
        return ExactTemporalNeighborStore(
            self.num_nodes,
            size=self.temporal_neighbors,
            device=self.event_times.device,
        )

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "neighbor_loader"):
            self.neighbor_loader = self._make_neighbor_store()
        if hasattr(self, "stream_event_index"):
            self.stream_event_index.fill_(-1)
            self.stream_timestamp.fill_(torch.iinfo(torch.long).min)
        return result

    def runtime_state_elements(self) -> int:
        return int(self.neighbor_loader.neighbors.numel()) + int(
            self.neighbor_loader.e_id.numel()
        )

    @property
    def state_is_empty(self) -> bool:
        return int(self.neighbor_loader.cur_e_id) == 0

    @torch.no_grad()
    def reset_state(self) -> None:
        self.neighbor_loader.reset_state()
        self.stream_event_index.fill_(-1)
        self.stream_timestamp.fill_(torch.iinfo(torch.long).min)

    def detach_state(self) -> None:
        return None

    def _validate_score_batch(self, batch: Dict[str, Tensor]) -> None:
        required = {
            "event_index",
            "src",
            "candidates",
            "timestamp",
            "event_time",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "GraphMixer batch is missing fields: " + ", ".join(missing)
            )
        event_index = batch["event_index"]
        source = batch["src"]
        candidates = batch["candidates"]
        event_time = batch["event_time"]
        batch_size = int(source.numel())
        if batch_size == 0:
            raise ValueError("GraphMixer cannot score an empty batch.")
        if event_index.shape != (batch_size,) or event_time.shape != (batch_size,):
            raise ValueError("event_index and event_time must have shape [batch].")
        if candidates.dim() != 2 or candidates.size(0) != batch_size:
            raise ValueError("candidates must have shape [batch, candidates].")
        if batch_size > 1:
            if bool((event_index[1:] <= event_index[:-1]).any()) or bool(
                (event_time[1:] < event_time[:-1]).any()
            ):
                raise ValueError("GraphMixer batches must be chronological.")
        node_ids = torch.cat([source.flatten(), candidates.flatten()])
        if bool((node_ids < 0).any()) or bool((node_ids >= self.num_nodes).any()):
            raise ValueError("GraphMixer batch contains an out-of-range node id.")
        if int(self.stream_event_index) >= 0:
            if int(event_index[0]) <= int(self.stream_event_index):
                raise ValueError(
                    "Event was already observed; call reset_state() before replaying."
                )
            if int(event_time[0]) < int(self.stream_timestamp):
                raise ValueError("GraphMixer batches must remain chronological.")

    def _node_base(self, node_ids: Tensor) -> Tensor:
        if self.node_encoder is not None and self.node_features is not None:
            return self.node_encoder(self.node_features[node_ids])
        assert self.node_embeddings is not None
        return self.node_embeddings(node_ids)

    def _aggregate(
        self,
        query_ids: Tensor,
        query_time: Tensor,
        return_relations: bool = False,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        neighbor_ids, event_ids = self.neighbor_loader.query(query_ids)
        mask = event_ids >= 0
        safe_ids = event_ids.clamp_min(0)
        event_time = self.event_times[safe_ids]
        span = self.time_span
        age = (
            (query_time.unsqueeze(-1).double() - event_time.double()) / span
        ).clamp_min(0.0).float()
        age = age * mask.to(age.dtype)
        time_features = self.time_encoder(age)
        edge_messages = self.event_messages[safe_ids]
        edge_hidden = self.edge_encoder(
            torch.cat([time_features, edge_messages], dim=-1)
        )
        count = mask.to(edge_hidden.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_node = self._node_base(neighbor_ids.clamp_min(0))
        neighbor_node_summary = (
            neighbor_node * mask.unsqueeze(-1).to(neighbor_node.dtype)
        ).sum(dim=-2) / count
        link_summary = (
            edge_hidden * mask.unsqueeze(-1).to(edge_hidden.dtype)
        ).sum(dim=-2) / count
        count_feature = torch.log1p(count)
        query_norm = (
            (query_time.double() - self.event_times[0].double()) / span
        ).to(torch.float32)
        query_time_features = self.time_encoder(query_norm)
        state = self.node_mixer(
            torch.cat(
                [
                    self._node_base(query_ids),
                    neighbor_node_summary,
                    link_summary,
                    count_feature,
                    query_time_features,
                ],
                dim=-1,
            )
        )
        if not return_relations:
            return state, {}
        relation_tokens = self.relation_projector(
            torch.cat([edge_hidden, time_features, edge_messages], dim=-1)
        )
        relation_tokens = relation_tokens * mask.unsqueeze(-1).to(
            relation_tokens.dtype
        )
        relation_mass = mask.to(relation_tokens.dtype) / count
        return state, {
            "relation_tokens": relation_tokens,
            "relation_ages": age,
            "relation_mask": mask,
            "relation_mass": relation_mass,
            "relation_event_ids": event_ids,
        }

    def score_candidates(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        self._validate_score_batch(batch)
        source = batch["src"].long()
        candidates = batch["candidates"].long()
        query_time = batch["event_time"]
        source_state, source_relations = self._aggregate(source, query_time, return_relations)
        candidate_state, _ = self._aggregate(
            candidates, query_time.unsqueeze(-1).expand_as(candidates)
        )
        expanded_source = source_state.unsqueeze(1).expand_as(candidate_state)
        score_inputs = torch.cat(
            [
                expanded_source,
                candidate_state,
                expanded_source * candidate_state,
                torch.abs(expanded_source - candidate_state),
            ],
            dim=-1,
        )
        logits = self.link_classifier(score_inputs).squeeze(-1)
        output: Dict[str, Tensor] = {
            "logits": logits,
            "candidate_embedding": candidate_state,
            "source_embedding": source_state,
        }
        if return_relations:
            output.update(source_relations)
        return output

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        required = {
            "event_index",
            "src",
            "positive_dst",
            "event_time",
            "edge_features",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "GraphMixer update batch is missing fields: " + ", ".join(missing)
            )
        event_index = batch["event_index"].long()
        expected = torch.arange(
            self.neighbor_loader.cur_e_id,
            self.neighbor_loader.cur_e_id + event_index.numel(),
            device=event_index.device,
        )
        if not torch.equal(event_index, expected):
            raise ValueError(
                "GraphMixer updates must be contiguous; "
                "call reset_state() before replaying."
            )
        source = batch["src"].long()
        destination = batch["positive_dst"].long()
        event_time = batch["event_time"].long()
        edge_features = batch["edge_features"].float()
        if edge_features.shape != (source.numel(), self.msg_dim):
            raise ValueError(
                f"edge_features must have shape [batch, {self.msg_dim}]."
            )
        self.neighbor_loader.insert(source, destination, event_ids=event_index)
        self.stream_event_index.copy_(event_index[-1])
        self.stream_timestamp.copy_(event_time[-1])

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
        update_memory: bool = True,
    ) -> Dict[str, Tensor]:
        output = self.score_candidates(batch, return_relations=return_relations)
        if update_memory:
            self.update_state(batch)
        return output
