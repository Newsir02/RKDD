import pytest
import torch
from torch.nn import functional as F

from loss.paper_baselines import StandardizedLogitKD, NOSMOGRelationKD, make_objective


def test_standardized_kd_matches_official_sample_std_formula():
    torch.manual_seed(13)
    s = torch.randn(3, 5, requires_grad=True)
    t = torch.randn(3, 5, requires_grad=True)
    normalize = lambda x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)
    expected = 4 * F.kl_div(F.log_softmax(normalize(s) / 2, -1),
                            F.softmax(normalize(t.detach()) / 2, -1), reduction="batchmean")
    actual = StandardizedLogitKD(2)(s, t)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_standardized_constant_logits_are_finite():
    s = torch.ones(2, 3, requires_grad=True)
    value = StandardizedLogitKD(2)(s, torch.ones_like(s))
    value.backward()
    assert value == 0 and torch.isfinite(s.grad).all()


def test_relation_kd_matches_projected_unnormalized_gram():
    s = torch.randn(4, 3, requires_grad=True)
    t = torch.randn(4, 5, requires_grad=True)
    objective = NOSMOGRelationKD(3, 5)
    projected = objective.projector(s)
    expected = ((projected @ projected.T - t.detach() @ t.detach().T) ** 2).mean()
    value = objective(s, t)
    assert torch.allclose(value, expected)
    value.backward()
    assert torch.isfinite(s.grad).all() and t.grad is None
    assert objective.projector[0].weight.grad is not None


@pytest.mark.parametrize("method", ["logits", "dkd", "rkd_distance", "rkd_angle", "logit_standard", "nosmog_rsd", "trd"])
def test_objectives_backward_with_unequal_widths(method):
    def output(width):
        return {"logits": torch.randn(2, 4, requires_grad=True),
                "source_embedding": torch.randn(2, width, requires_grad=True),
                "candidate_embedding": torch.randn(2, 4, width, requires_grad=True)}
    s, t, sb, tb = output(3), output(5), output(3), output(5)
    objective = make_objective(method, 3, 5)
    loss, _ = objective(s, t, torch.tensor([0, 1]), sb, tb)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(v.grad is None for v in t.values())
    assert all(v.grad is None for v in tb.values())


def test_invalid_method_is_rejected():
    with pytest.raises(ValueError):
        make_objective("unknown", 3, 5)


@pytest.mark.parametrize("candidates", [2, 4])
def test_paper_dkd_renormalizes_non_target_distribution(candidates):
    s = torch.tensor([[1., 3., -2., 5.]], requires_grad=True)[:, :candidates]
    t = torch.tensor([[8., 2., -1., 4.]])[:, :candidates]
    target = torch.tensor([0])
    ps, pt = F.softmax(s / 2, -1), F.softmax(t / 2, -1)
    bs = torch.stack([ps[:, 0], ps[:, 1:].sum(-1)], -1)
    bt = torch.stack([pt[:, 0], pt[:, 1:].sum(-1)], -1)
    binary = F.kl_div(bs.log(), bt, reduction='batchmean')
    non_target = F.kl_div(F.log_softmax(s[:, 1:] / 2, -1),
                          F.softmax(t[:, 1:] / 2, -1), reduction='batchmean')
    expected = 4 * (binary + 8 * non_target)
    objective = make_objective('dkd', 3, 5)
    objective.dkd_warmup = 0
    total, parts = objective({'logits': s}, {'logits': t}, target)
    assert torch.allclose(parts['kd'], expected, atol=1e-6)
    assert torch.allclose(total, F.cross_entropy(s, target) + expected)


def test_paper_rkd_distance_uses_huber_not_squared_error():
    from loss.paper_baselines import paper_rkd
    s = torch.tensor([[[0., 0.], [1., 0.], [9., 1.]]], requires_grad=True)
    t = torch.tensor([[[0., 0.], [2., 0.], [2., 1.]]])
    ds, dt = torch.cdist(s[0], s[0]), torch.cdist(t[0], t[0])
    expected = F.smooth_l1_loss(ds / ds[ds > 0].mean(), dt / dt[dt > 0].mean())
    actual = paper_rkd(s, t, "distance")
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(s.grad).all()
