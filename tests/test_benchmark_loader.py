import torch

from dataset.benchmark_loader import sample_candidate_batch


def test_uniform_sampler_excludes_positive_source_and_duplicates():
    import random
    from torch_geometric.data import TemporalData
    raw = TemporalData(src=torch.tensor([0, 9]), dst=torch.tensor([1, 3]),
                       t=torch.tensor([10, 11]), msg=torch.zeros(2, 1))
    raw.event_index = torch.arange(2)
    raw.timestamp = torch.tensor([0., 1.])
    domain = list(range(6))
    output = sample_candidate_batch(raw, domain, set(domain), 5, random.Random(7))
    repeated = sample_candidate_batch(raw, domain, set(domain), 5, random.Random(7))
    assert torch.equal(output['candidates'], repeated['candidates'])
    for i, candidates in enumerate(output['candidates'].tolist()):
        assert len(set(candidates)) == 5
        assert candidates[output['target'][i]] == raw.dst[i]
        assert raw.src[i] not in candidates
        assert set(candidates) <= set(domain)


def test_large_domain_does_not_require_full_random_permutation(monkeypatch):
    import random
    from torch_geometric.data import TemporalData
    def forbidden(*args, **kwargs):
        raise AssertionError('Full permutation is not scalable')
    monkeypatch.setattr(torch, 'randperm', forbidden)
    raw = TemporalData(src=torch.tensor([1]), dst=torch.tensor([2]),
                       t=torch.tensor([1]), msg=torch.zeros(1, 1))
    raw.event_index, raw.timestamp = torch.tensor([0]), torch.tensor([0.])
    domain = range(1_000_000)
    result = sample_candidate_batch(raw, domain, domain, 16, random.Random(7))
    assert result['candidates'].shape == (1, 16)

