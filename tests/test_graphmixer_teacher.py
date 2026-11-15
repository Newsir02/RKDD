from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dataset import build_synthetic_temporal_graph
from model import GraphMixerTeacher
from train import load_teacher_from_checkpoint
from util.checkpoint import save_model_checkpoint


def make_teacher() -> GraphMixerTeacher:
    return GraphMixerTeacher(
        event_times=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        event_messages=torch.arange(8, dtype=torch.float).reshape(4, 2) / 10.0,
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    )


def make_batch(
    event_indices: tuple[int, ...] = (0,),
    sources: tuple[int, ...] = (0,),
    positives: tuple[int, ...] = (1,),
    candidates: tuple[tuple[int, ...], ...] = ((1, 2),),
    targets: tuple[int, ...] = (0,),
) -> dict[str, torch.Tensor]:
    event_index = torch.tensor(event_indices, dtype=torch.long)
    event_times = torch.tensor([10, 20, 30, 40], dtype=torch.long)[event_index]
    event_messages = (
        torch.arange(8, dtype=torch.float).reshape(4, 2)[event_index] / 10.0
    )
    return {
        "event_index": event_index,
        "src": torch.tensor(sources, dtype=torch.long),
        "positive_dst": torch.tensor(positives, dtype=torch.long),
        "candidates": torch.tensor(candidates, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "timestamp": event_index.float() / 3.0,
        "event_time": event_times,
        "edge_features": event_messages,
    }


def test_graphmixer_scores_and_exposes_relation_contract() -> None:
    teacher = make_teacher().eval()
    first = make_batch()
    second = make_batch(
        event_indices=(1,),
        sources=(0,),
        positives=(2,),
        candidates=((2, 3),),
        targets=(0,),
    )

    empty = teacher.score_candidates(first, return_relations=True)
    assert empty["logits"].shape == (1, 2)
    assert not empty["relation_mask"].any()

    teacher.update_state(first)
    output = teacher.score_candidates(second, return_relations=True)
    assert output["logits"].shape == (1, 2)
    assert output["relation_tokens"].shape == (1, 3, 6)
    assert output["relation_ages"].shape == (1, 3)
    assert output["relation_mask"].shape == (1, 3)
    assert output["relation_mass"].shape == (1, 3)
    assert output["relation_mask"].sum() == 1
    assert torch.isfinite(output["relation_tokens"]).all()
    assert torch.allclose(output["relation_mass"].sum(dim=1), torch.ones(1))


def test_graphmixer_reset_replay_is_deterministic() -> None:
    teacher = make_teacher().eval()
    batch = make_batch(
        event_indices=(0, 1),
        sources=(0, 0),
        positives=(1, 2),
        candidates=((1, 3), (2, 3)),
        targets=(0, 0),
    )

    first = teacher(batch, update_memory=True)["logits"].detach().clone()
    teacher.reset_state()
    second = teacher(batch, update_memory=True)["logits"].detach().clone()

    assert torch.allclose(first, second)


def test_graphmixer_checkpoint_round_trip(tmp_path: Path) -> None:
    bundle = build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        num_candidates=3,
        seed=17,
    )
    teacher = GraphMixerTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=int(bundle.node_features.size(0)),
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    ).eval()
    path = tmp_path / "graphmixer.pt"
    save_model_checkpoint(
        path,
        teacher,
        model_name="GraphMixerTeacher",
        model_config={**teacher.model_config, "data_signature": "data"},
        args={},
    )

    loaded = load_teacher_from_checkpoint(
        path,
        bundle=bundle,
        data_signature="data",
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, GraphMixerTeacher)
    assert loaded.model_config["teacher_arch"] == "graphmixer"
    assert loaded.state_is_empty


def test_graphmixer_rejects_out_of_order_batches() -> None:
    teacher = make_teacher().eval()
    teacher.update_state(make_batch())

    with pytest.raises(ValueError, match="reset_state"):
        teacher.score_candidates(make_batch(), return_relations=False)
