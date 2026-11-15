import pytest
import torch
import torch.nn.functional as F

from loss.temporal_paper_adaptations import TEMPORAL_PAPER_METHODS, TemporalPaperObjective


def embeddings(width):
    return {"logits": torch.randn(2, 4, requires_grad=True),
            "source_embedding": torch.randn(2, width, requires_grad=True),
            "candidate_embedding": torch.randn(2, 4, width, requires_grad=True),
            "source_structure": torch.randn(2, width, requires_grad=True),
            "candidate_structure": torch.randn(2, 4, width, requires_grad=True)}


@pytest.mark.parametrize("name", TEMPORAL_PAPER_METHODS)
def test_objective_updates_student_and_never_teacher(name):
    torch.manual_seed(21)
    student, teacher = embeddings(3), embeddings(5)
    objective = TemporalPaperObjective(name, 3, 5)
    loss, metrics = objective(student, teacher, torch.tensor([0, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["relation"])
    assert student["logits"].grad is not None
    assert objective.projector.weight.grad is not None
    assert all(value.grad is None for value in teacher.values())
    feature = "source_structure" if name == "dyngkd_structure" else "source_embedding"
    assert student[feature].grad.abs().sum() > 0


def test_lightst_keeps_official_weighted_cosine_and_denominator_at_one_time():
    from loss.temporal_paper_adaptations import lightst_spatial_loss

    s = F.normalize(torch.tensor([[[1., 0.], [0., 1.]]]), dim=-1)
    t = F.normalize(torch.tensor([[[1., 1.], [1., -1.]]]), dim=-1)
    sc, tc = s[0], t[0]
    pos = (((sc * tc).sum(-1).square()) / .05).exp()
    neg_s, neg_t = ((sc @ sc.T) / .05).exp().sum(-1), ((tc @ tc.T) / .05).exp().sum(-1)
    expected = (torch.log1p(neg_s / pos).mean() + torch.log1p(neg_t / pos).mean()) / 2
    assert torch.allclose(lightst_spatial_loss(s, t), expected)


def test_lkd4dytag_uses_exponential_cosine_of_edge_embeddings():
    from loss.temporal_paper_adaptations import dy_tag_edge_loss

    student_edge = torch.tensor([[[1., 0.], [0., 1.]]])
    teacher_edge = torch.tensor([[[1., 0.], [0., -1.]]])
    expected = torch.tensor([-1., 1.]).exp().mean()
    assert torch.allclose(dy_tag_edge_loss(student_edge, teacher_edge), expected)


def test_distill2vec_reproduces_student_true_target_kl():
    from loss.temporal_paper_adaptations import latent_kl

    s, t = torch.tensor([[[1., 3.]]]), torch.tensor([[[4., 1.]]])
    ps, pt = s.softmax(-1), t.softmax(-1)
    expected = (ps * (ps.log() - pt.log())).sum(-1).mean()
    assert torch.allclose(latent_kl(s, t, 1.), expected)
