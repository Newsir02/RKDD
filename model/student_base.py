from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import MLP, ResidualMLPBlock, TimeEncoding


class LightStudentBase(nn.Module):
    """Shared components for the lightweight temporal GNN student.

    This base class only provides node encoding, residual temporal blocks,
    candidate scoring and the fixed relation-slot branch.  It has no forward
    path of its own: the student must aggregate temporal neighbours, which is
    implemented by :class:`LightGNNStudent`.
    """

    def __init__(
        self,
        node_features: Tensor,
        hidden_dim: int = 48,
        time_dim: int = 16,
        relation_dim: int = 64,
        relation_slots: int = 8,
        dropout: float = 0.1,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if node_features.dim() != 2 or node_features.size(0) == 0:
            raise ValueError("node_features must have shape [nodes, features].")
        if hidden_dim <= 0 or relation_dim <= 0 or relation_slots <= 0:
            raise ValueError("Student dimensions and relation_slots must be positive.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        self.register_buffer("node_features", node_features.detach().clone().float())
        self.num_nodes = int(node_features.size(0))
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.relation_dim = relation_dim
        self.relation_slots = relation_slots
        self.model_config = {
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
            "relation_dim": relation_dim,
            "relation_slots": relation_slots,
            "dropout": dropout,
            "num_blocks": num_blocks,
            "student_blocks": num_blocks,
            "student_layers": num_blocks,
            "num_nodes": self.num_nodes,
            "node_feature_dim": int(node_features.size(1)),
        }

        feature_dim = int(node_features.size(1))
        self.node_encoder = MLP(feature_dim, [hidden_dim], hidden_dim, dropout)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        self.time_encoder = TimeEncoding(time_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.score_head = MLP(
            4 * hidden_dim,
            [2 * hidden_dim, hidden_dim],
            1,
            dropout,
        )
        self.slot_embeddings = nn.Parameter(
            torch.randn(relation_slots, hidden_dim) * 0.02
        )
        self.age_gaps = nn.Parameter(torch.zeros(relation_slots))
        self.slot_time_projection = nn.Linear(time_dim, hidden_dim)
        self.relation_decoder = MLP(
            hidden_dim,
            [relation_dim],
            relation_dim,
            dropout,
        )

    def _base_encode(self, node_ids: Tensor) -> Tensor:
        return self.node_encoder(self.node_features[node_ids]) + self.node_embeddings(
            node_ids
        )

    def _apply_temporal_blocks(self, states: Tensor) -> Tensor:
        for block in self.temporal_blocks:
            states = block(states)
        return states

    def _ordered_slot_ages(self, batch_size: int) -> Tensor:
        # 正间隔保证槽年龄从旧到新单调递减，与教师邻居队列顺序一致。
        gaps = F.softplus(self.age_gaps) + 1.0e-4
        centers = torch.cumsum(gaps, dim=0) - 0.5 * gaps
        normalized_ages = 1.0 - centers / gaps.sum()
        return normalized_ages.unsqueeze(0).expand(batch_size, -1)

    def _run_heads(
        self,
        source_state: Tensor,
        candidate_state: Tensor,
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        source_expanded = source_state.unsqueeze(1).expand_as(candidate_state)
        score_inputs = torch.cat(
            [
                source_expanded,
                candidate_state,
                source_expanded * candidate_state,
                torch.abs(source_expanded - candidate_state),
            ],
            dim=-1,
        )
        outputs = {"logits": self.score_head(score_inputs).squeeze(-1)}
        outputs["candidate_embeddings"] = candidate_state

        if return_relations:
            slot_ages = self._ordered_slot_ages(source_state.size(0))
            slot_time = self.slot_time_projection(self.time_encoder(slot_ages))
            slot_inputs = (
                source_state.unsqueeze(1)
                + self.slot_embeddings.unsqueeze(0)
                + slot_time
            )
            relation_tokens = self.relation_decoder(slot_inputs)
            outputs.update(
                {
                    "relation_tokens": relation_tokens,
                    "relation_ages": slot_ages,
                    "relation_mask": torch.ones(
                        source_state.size(0),
                        self.relation_slots,
                        dtype=torch.bool,
                        device=source_state.device,
                    ),
                }
            )
        return outputs
