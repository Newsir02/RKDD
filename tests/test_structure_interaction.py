import pytest
import torch

from loss.structure_interaction import interaction, StructureInteractionObjective
from model.snapshot_gnn import SnapshotGNN
from util.structure_interaction_training import select_evidence
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
