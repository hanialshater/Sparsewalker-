import torch

from sparsewalker.models.sparsewalker import SparseWalker, _coalesced_topk


def test_coalesced_topk_sums_duplicate_mass():
    ids = torch.tensor([[5, 5, 9, 11]])
    mass = torch.tensor([[0.2, 0.3, 0.4, 0.1]])
    out_ids, out_mass = _coalesced_topk(ids, mass, 3)
    got = {int(i): float(m) for i, m in zip(out_ids[0], out_mass[0])}
    assert len(set(out_ids[0].tolist())) == 3
    assert abs(got[5] - 0.5) < 1e-6
    assert abs(got[9] - 0.4) < 1e-6
    assert abs(got[11] - 0.1) < 1e-6


def test_fresh_merge_once_per_timestep_even_with_two_graph_hops():
    class CountWalker(SparseWalker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.merge_calls = 0

        def _merge(self, *args, **kwargs):
            self.merge_calls += 1
            return super()._merge(*args, **kwargs)

    model = CountWalker(
        50, 8, d=8, layers=2, side=8, h=4,
        active=4, top_side=2, degree=2,
    )
    x = torch.randint(1, 51, (2, 7))
    model.encode(x)
    assert model.merge_calls == 7
