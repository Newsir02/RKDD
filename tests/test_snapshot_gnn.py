import pytest
import torch
from model.snapshot_gnn import SnapshotGNN


def batch(index=2):
    return dict(src=torch.tensor([0]), positive_dst=torch.tensor([2]),
                candidates=torch.tensor([[2, 3]]), event_index=torch.tensor([index]))


def test_featureless_nodes_have_learnable_static_identity():
    model = SnapshotGNN(torch.zeros(5, 3), "sage", 8, 2, 2)
    output = model(batch())
    assert not torch.equal(output["candidate_embedding"][:, 0], output["candidate_embedding"][:, 1])


@pytest.mark.parametrize("architecture", ["gcn", "sage"])
def test_snapshot_prediction_is_causal_memoryless_and_batch_independent(architecture):
    torch.manual_seed(7)
    model = SnapshotGNN(torch.randn(5, 3), architecture, 4, 2, 2)
    query = batch()
    initial = model(query)["logits"].detach().clone()
    # Future timestamps/features are not part of the student's input.
    decorated = {**query, "timestamp": torch.tensor([999.]), "edge_features": torch.randn(1, 7)}
    assert torch.equal(initial, model(decorated)["logits"])
    model.update_state(dict(src=torch.tensor([0, 1]), positive_dst=torch.tensor([1, 2]),
                            event_index=torch.tensor([0, 1])))
    prediction = model(query)["logits"]
    duplicate = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in query.items()}
    assert torch.allclose(prediction, model(duplicate)["logits"][:1], atol=1e-6)
    prediction.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.update_state(query)
    with pytest.raises(ValueError, match="history"):
        model(query)
    model.reset_state()
    assert torch.equal(initial, model(query)["logits"])


@pytest.mark.parametrize("architecture", ["gcn", "sage"])
def test_unrelated_queries_do_not_change_snapshot_scores(architecture):
    torch.manual_seed(3)
    model = SnapshotGNN(torch.randn(10, 3), architecture, 7, 2, 3)
    model.update_state(dict(src=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            positive_dst=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                            event_index=torch.arange(8)))
    query = batch(index=8)
    other = dict(src=torch.tensor([7]), positive_dst=torch.tensor([8]),
                 candidates=torch.tensor([[8, 9]]), event_index=torch.tensor([9]))
    together = {key: torch.cat([query[key], other[key]], 0) for key in query}
    assert torch.allclose(model(query)["logits"], model(together)["logits"][:1], atol=1e-6)
