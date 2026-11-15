import pytest
import torch
from torch.nn import functional as F

from loss.temporal_delta import TemporalDeltaDistillation, query_relations
from model.plain_gnn import PlainGNNStudent


def output(source, candidates):
    return {"source_embedding": source, "candidate_embedding": candidates,
            "logits": torch.zeros(source.size(0), candidates.size(1), requires_grad=True)}


def test_delta_is_coordinate_free_but_sensitive_to_time_direction():
    torch.manual_seed(3)
    before = output(torch.randn(3, 4), torch.randn(3, 5, 4))
    after = output(torch.randn(3, 4), torch.randn(3, 5, 4))
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    def transformed(x):
        return output(F.pad(3 * x["source_embedding"] @ rotation, (0, 2)),
                      F.pad(2 * x["candidate_embedding"] @ rotation, (0, 2)))
    criterion = TemporalDeltaDistillation(logit_weight=0, rkd_weight=1)
    targets = torch.zeros(3, dtype=torch.long)
    _, matched = criterion(transformed(after), after, targets, transformed(before), before)
    _, reversed_time = criterion(transformed(after), before, targets, transformed(before), after)
    assert matched["rkd"] < 1e-12
    assert reversed_time["rkd"] > 0.01


def test_only_three_terms_and_teacher_is_detached():
    torch.manual_seed(8)
    s0 = output(torch.randn(2, 3, requires_grad=True), torch.randn(2, 4, 3, requires_grad=True))
    s1 = output(torch.randn(2, 3, requires_grad=True), torch.randn(2, 4, 3, requires_grad=True))
    t0 = output(torch.randn(2, 7, requires_grad=True), torch.randn(2, 4, 7, requires_grad=True))
    t1 = output(torch.randn(2, 7, requires_grad=True), torch.randn(2, 4, 7, requires_grad=True))
    loss, parts = TemporalDeltaDistillation(0.7, 0.2)(s1, t1, torch.tensor([0, 2]), s0, t0)
    assert torch.allclose(loss.detach(), parts["task"] + .7 * parts["logit"] + .2 * parts["rkd"])
    loss.backward()
    assert s0["source_embedding"].grad.abs().sum() > 0
    assert s1["source_embedding"].grad.abs().sum() > 0
    assert all(v.grad is None for o in (t0, t1) for v in o.values())


def test_no_history_delta_and_zero_vectors_are_finite():
    x = output(torch.zeros(2, 3), torch.zeros(2, 4, 3))
    _, parts = TemporalDeltaDistillation()(x, x, torch.tensor([0, 1]))
    assert parts["rkd"] == 0
    assert torch.isfinite(query_relations(x)).all()


def test_plain_student_uses_graph_but_not_time_or_auxiliary_heads():
    torch.manual_seed(2)
    student = PlainGNNStudent(torch.randn(6, 4), hidden_dim=8, neighbors=2, num_blocks=1)
    batch = {"src": torch.tensor([0]), "candidates": torch.tensor([[1, 2]])}
    before = student(batch)["logits"]
    with_time = student({**batch, "timestamp": torch.tensor([999.]), "event_time": torch.tensor([999])})
    assert torch.equal(before, with_time["logits"])
    student.update_state({"src": torch.tensor([0]), "positive_dst": torch.tensor([3]), "event_index": torch.tensor([0])})
    assert not torch.allclose(before, student(batch)["logits"])
    names = " ".join(dict(student.named_parameters()))
    assert all(name not in names for name in ("time", "slot", "projector", "attention", "memory"))


@pytest.mark.parametrize("mode", ["temporal", "static", "rkd_distance", "rkd_angle", "shuffled", "reverse", "zero"])
def test_controls_are_finite(mode):
    x = output(torch.randn(2, 3), torch.randn(2, 4, 3))
    loss, parts = TemporalDeltaDistillation(mode=mode)(x, x, torch.tensor([0, 1]), x, x)
    assert torch.isfinite(loss)
    assert set(parts) == {"total", "task", "logit", "rkd"}


@pytest.mark.parametrize("arch", ["tgn", "tgat", "graphmixer"])
def test_paired_training_uses_only_previous_history_and_fixed_queries(arch):
    from dataset import build_synthetic_temporal_graph, CandidateBatchLoader
    from train import parse_args, apply_method_configuration, build_teacher, build_student, freeze_teacher
    from util.temporal_delta_training import train_temporal_delta_epoch
    bundle = build_synthetic_temporal_graph(num_nodes=12, num_events=20, feature_dim=4,
                                            edge_feature_dim=2, num_candidates=3, history_size=2, seed=17)
    args = parse_args(["--method", "trd", "--teacher-arch", arch, "--teacher-hidden", "8",
                       "--time-dim", "4", "--relation-dim", "8", "--student-hidden", "6"])
    apply_method_configuration(args)
    teacher = build_teacher(bundle, args).eval()
    if arch in {"tgat", "graphmixer"}:
        assert teacher.model_config["time_span"] == int(bundle.train_data.t[-1] - bundle.train_data.t[0])
    freeze_teacher(teacher)
    student = build_student(bundle.node_features, args)
    loader = list(CandidateBatchLoader(bundle, "train", 4, 3, 17))
    observations = []
    real_score = teacher.score_candidates
    def record_score(batch, **kwargs):
        observations.append((int(teacher.stream_event_index), batch["candidates"].clone()))
        return real_score(batch, **kwargs)
    teacher.score_candidates = record_score
    # Zero learning rate: this is a state/gradient contract test, not an experiment.
    result = train_temporal_delta_epoch(student, teacher, TemporalDeltaDistillation(), loader,
        torch.optim.SGD(student.parameters(), lr=0), torch.device("cpu"), 5.0)
    assert len(observations) == 2 * len(loader) - 1
    assert observations[0][0] == -1
    for k in range(1, len(loader)):
        old, new = observations[2*k-1], observations[2*k]
        assert old[0] == (-1 if k == 1 else int(loader[k-2]["event_index"][-1]))
        assert new[0] == int(loader[k-1]["event_index"][-1])
        assert torch.equal(old[1], new[1])
        assert new[0] < int(loader[k]["event_index"][0])
    assert int(teacher.stream_event_index) == int(loader[-1]["event_index"][-1])
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert all(p.grad is None for p in teacher.parameters())


def test_configuration_excludes_legacy_losses_and_checkpoint_roundtrip(tmp_path):
    from dataset import build_synthetic_temporal_graph
    from train import parse_args, apply_method_configuration, build_student, load_student_from_checkpoint
    from util import save_model_checkpoint
    args = parse_args(["--method", "trd", "--rank-weight", "9"])
    apply_method_configuration(args)
    assert args.rank_weight == args.diversity_weight == args.anchor_weight == 0
    assert args.task_weight == 1
    assert args.dropout == args.weight_decay == 0
    bundle = build_synthetic_temporal_graph(num_nodes=12, num_events=20, feature_dim=4,
                                            edge_feature_dim=2, num_candidates=3, history_size=2, seed=17)
    student = build_student(bundle.node_features, args)
    path = tmp_path / "plain.pt"
    save_model_checkpoint(path, student, model_name=type(student).__name__,
                          model_config=student.model_config, args=vars(args))
    restored = load_student_from_checkpoint(path, bundle, torch.device("cpu"))
    batch = {"src": torch.tensor([0]), "candidates": torch.tensor([[1, 2]])}
    assert torch.equal(student(batch)["logits"], restored(batch)["logits"])


def test_trd_training_entry_and_saved_student(tmp_path, capsys):
    from train import main
    main(["--method", "trd", "--quick", "--dataset", "synthetic", "--device", "cpu",
          "--num-nodes", "12", "--num-events", "20", "--num-candidates", "3",
          "--feature-dim", "4", "--edge-feature-dim", "2", "--batch-size", "4",
          "--teacher-hidden", "8", "--student-hidden", "6", "--time-dim", "4",
          "--output-dir", str(tmp_path)])
    payload = torch.load(tmp_path / "student.pt", weights_only=False)
    assert payload["model_name"] == "PlainGNNStudent"
    assert payload["model_config"]["num_nodes"] == 12
    assert payload["args"]["rank_weight"] == 0
    capsys.readouterr()
