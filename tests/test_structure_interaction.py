import pytest
import torch

from loss.structure_interaction import (
    interaction, StructureInteractionObjective, StructureLogitObjective,
)
from model.snapshot_gnn import SnapshotGNN
from util.structure_interaction_training import pair_query, select_evidence, select_random_evidence
from model.temporal_neighbor_store import ExactTemporalNeighborStore


def historical_motif():
    # u=0, a=1, b=2, v=3, w=4. Repeated arms and an unshared teacher event.
    src = torch.tensor([2, 0, 2, 0, 2, 0])
    dst = torch.tensor([1, 1, 3, 1, 3, 1])
    batch = dict(src=src, positive_dst=dst, event_index=torch.arange(src.numel()))
    student = SnapshotGNN(torch.randn(5, 4), "gcn", 8, 2, 6)
    student.update_state(batch)
    teacher = ExactTemporalNeighborStore(5, 6)
    teacher.insert(src[:-1], dst[:-1], batch["event_index"][:-1])
    query = dict(src=torch.tensor([0]), candidates=torch.tensor([[4, 3]]),
                 target=torch.tensor([1]), event_index=torch.tensor([6]))
    return student, teacher, query


def test_signed_candidate_interaction_cancels_additive_effects_and_view_shifts():
    additive = torch.tensor([[[3., 1.]], [[4., 1.]], [[3., -1.]], [[4., -1.]]])
    # View order is 00, 10, 01, 11. Each row contains v and w.
    assert interaction(additive).item() == 0.
    additive[-1, 0, 0] += 2
    shifted = additive + torch.tensor([1., 3., -4., 0.]).view(4, 1, 1)
    assert interaction(shifted).item() == 2.


def test_structure_logit_control_uses_each_view_and_stops_teacher_gradient():
    student = torch.randn(4, 3, 2, requires_grad=True)
    teacher = torch.randn(4, 3, 2, requires_grad=True)
    objective = StructureLogitObjective(.5, temperature=2.)
    assert objective.kd_weight == 0.
    loss, student_delta, teacher_delta = objective(student, teacher)
    assert torch.isfinite(loss) and loss.item() > 0
    assert student_delta.shape == teacher_delta.shape == (3,)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_structure_logit_control_is_routed_by_make_objective():
    from loss.paper_baselines import make_objective

    objective = make_objective("structure_logits", 6, 8, relation_weight=.25, temperature=1.5)
    assert isinstance(objective, StructureLogitObjective)
    assert objective.weight == .25


def test_pair_query_supports_next_random_and_hard_references():
    import random

    batch = {
        "src": torch.tensor([0]),
        "candidates": torch.tensor([[1, 2, 3, 4]]),
        "target": torch.tensor([0]),
    }
    next_query = pair_query(batch, 0)
    assert next_query["candidates"].tolist() == [[1, 2]]
    random_query = pair_query(batch, 0, "random", random.Random(1))
    assert random_query["candidates"][0, 0].item() == 1
    assert random_query["candidates"][0, 1].item() in (2, 3, 4)
    hard_query = pair_query(batch, 0, "hard", teacher_scores=torch.tensor([[0., 1., 5., 2.]]))
    assert hard_query["candidates"].tolist() == [[1, 3]]


def test_random_evidence_keeps_group_budget_and_breaks_real_groups():
    import random

    student, teacher, query = historical_motif()
    real = list(select_evidence(student.neighbor_loader, teacher, query, max_events=2))
    random_rows = list(select_random_evidence(
        student.neighbor_loader, teacher, query, max_events=2, rng=random.Random(0)))
    assert real and random_rows
    for row, a, b in random_rows:
        assert a.numel() <= 2 and b.numel() <= 2
        assert set(a.tolist()).isdisjoint(b.tolist())
        assert set(a.tolist()) != set(real[0][1].tolist()) or set(b.tolist()) != set(real[0][2].tolist())


def test_evidence_groups_respect_motif_shared_teacher_support_and_max_size():
    student, teacher, query = historical_motif()
    (row, a, b), = select_evidence(student.neighbor_loader, teacher, query, max_events=2)
    assert row == 0 and a.tolist() == [3, 1] and b.tolist() == [4, 2]
    assert 0 not in a and 0 not in b  # common historical bridge
    (row, a, b), = select_evidence(student.neighbor_loader, teacher, query, max_events=1)
    assert a.tolist() == [3] and b.tolist() == [4]


def test_student_readout_mask_never_refills_or_mutates_the_live_cache():
    student, _, query = historical_motif()
    cache = student.neighbor_loader.e_id.clone()
    query = {**query, "history_mask_event_ids": torch.tensor([1, 3, 5])}
    actual = student(query)["logits"]
    assert torch.equal(cache, student.neighbor_loader.e_id)
    # On this fixture, direct removal from the stored slots is the reference.
    student.neighbor_loader.e_id.masked_fill_(torch.isin(cache, query["history_mask_event_ids"]), -1)
    expected = student({k: v for k, v in query.items() if k != "history_mask_event_ids"})["logits"]
    assert torch.allclose(actual, expected)
    with pytest.raises(ValueError, match="history"):
        student({**query, "history_mask_event_ids": torch.tensor([6])})


def test_teacher_gradient_is_stopped_and_all_four_student_views_are_used():
    student = torch.zeros(4, 1, 2, requires_grad=True)
    teacher = torch.tensor([[[0., 0.]], [[0., 0.]], [[0., 0.]], [[2., 0.]]], requires_grad=True)
    loss, _, _ = StructureInteractionObjective(1., beta=.5)(student, teacher)
    assert loss.item() == pytest.approx(.875)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, 0, 0].tolist() == [-.5, .5, .5, -.5]


@pytest.mark.parametrize("kd_weight", [0., .6])
def test_structure_epoch_distills_intact_full_candidate_batches(kd_weight):
    from types import SimpleNamespace
    from torch.nn import functional as F
    from loss.paper_baselines import make_objective
    from util.structure_interaction_training import train_structure_epoch

    torch.manual_seed(17)
    student = SnapshotGNN(torch.randn(6, 4), "gcn", 6, 2, 8)
    teacher = SnapshotGNN(torch.randn(6, 4), "gcn", 8, 2, 8)
    first = {
        "src": torch.tensor([2, 0, 2]),
        "positive_dst": torch.tensor([1, 1, 3]),
        "event_index": torch.arange(3),
        "candidates": torch.tensor([[1, 4, 5], [1, 4, 5], [3, 4, 5]]),
        "target": torch.zeros(3, dtype=torch.long),
    }
    second = {
        "src": torch.tensor([0]),
        "positive_dst": torch.tensor([3]),
        "event_index": torch.tensor([3]),
        "candidates": torch.tensor([[4, 3, 5]]),
        "target": torch.tensor([1]),
    }

    intact_calls = []

    def score(batch, return_relations=False):
        assert not torch.is_grad_enabled()
        assert teacher.last_event < int(batch["event_index"].min())
        logits = teacher(batch, return_relations=return_relations)["logits"]
        logits = logits + .3 * batch["candidates"]
        if batch["candidates"].size(1) == 3:
            assert "history_mask_event_ids" not in batch
            intact_calls.append((teacher.last_event, batch["candidates"], logits))
        return {"logits": logits}

    teacher.score_candidates = score
    args = SimpleNamespace(seed=42, structure_group_events=2, structure_queries_per_batch=4, grad_clip=5.)
    objective = make_objective("structure_interaction", 6, 8, temperature=1.6,
                               kd_weight=kd_weight, relation_weight=.25)
    optimizer = torch.optim.SGD(student.parameters(), lr=0)
    parts = train_structure_epoch(student, teacher, objective, [first, second], optimizer,
                                  torch.device("cpu"), args, epoch=1)

    assert parts["eligible_fraction"] == pytest.approx(.25)
    if kd_weight:
        # The first batch has no prior history and cannot supply any four-view pair.
        assert [call[0] for call in intact_calls] == [-1, 2]
        student.reset_state()
        expected_kd = 0.
        for batch, (_, candidates, t) in zip((first, second), intact_calls):
            assert torch.equal(candidates, batch["candidates"])
            with torch.no_grad():
                s = student(batch)["logits"]
            expected_kd += F.kl_div(F.log_softmax(s / 1.6, -1), F.softmax(t / 1.6, -1),
                                    reduction="sum").item() * 1.6**2 / 4
            student.update_state(batch)
        assert parts["kd"] == pytest.approx(expected_kd)
    else:
        assert not intact_calls
        assert parts["kd"] == 0.
    assert parts["total"] == pytest.approx(parts["task"] + kd_weight * parts["kd"] + .25 * parts["rel"])
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize("arch", ["tgn", "tgat", "graphmixer"])
def test_teacher_readout_masks_only_already_observed_events_without_writing_state(arch):
    from dataset import build_synthetic_temporal_graph, CandidateBatchLoader
    from train import parse_args, build_teacher, freeze_teacher
    from util.structure_interaction_training import score_views

    torch.manual_seed(42)
    bundle = build_synthetic_temporal_graph(num_nodes=12, num_events=30, feature_dim=4,
                                            edge_feature_dim=2, num_candidates=3, seed=17)
    args = parse_args(["--method", "trd", "--teacher-arch", arch, "--teacher-hidden", "8",
                       "--relation-dim", "8", "--time-dim", "4"])
    teacher = build_teacher(bundle, args)
    freeze_teacher(teacher)
    loader = iter(CandidateBatchLoader(bundle, "train", 4, 3, 17))
    first, batch = next(loader), next(loader)
    teacher.update_state(first)
    query = {k: v[:1] for k, v in batch.items()}
    query["candidates"] = query["candidates"][:, :2]
    before_cache = teacher.neighbor_loader.e_id.clone()
    before_score = teacher.score_candidates(query, return_relations=False)["logits"]
    if arch == "tgn":
        before_memory = teacher.memory.memory.clone()
    result = score_views(teacher.score_candidates, query, torch.tensor([0]), torch.tensor([1]))
    assert torch.isfinite(result).all()
    assert torch.equal(before_score, result[-1])
    assert torch.equal(before_cache, teacher.neighbor_loader.e_id)
    assert int(teacher.stream_event_index) == 3
    if arch == "tgn":
        assert torch.equal(before_memory, teacher.memory.memory)
    with pytest.raises(ValueError, match="history"):
        teacher.score_candidates({**query, "history_mask_event_ids": torch.tensor([4])})
