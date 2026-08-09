import torch
from sparsewalker.models import SASRec, ESASRec
from sparsewalker.models.core import sampled_softmax_from_hidden


def test_ligr_has_biased_residual_gates_and_reference_swiglu_shape():
    m = ESASRec(100, 20, d=64, layers=1, heads=1, inner=256, dropout=.2)
    b = m.blocks[0]
    assert b.g1.bias is not None and b.g2.bias is not None
    assert b.ffn.up.bias is None and b.ffn.gate.bias is None and b.ffn.down.bias is None


def test_sampled_softmax_is_finite_and_differentiable():
    torch.manual_seed(3)
    h = torch.randn(2, 3, 8, requires_grad=True)
    y = torch.tensor([[2, 3, 4], [5, 6, 0]])
    w = torch.randn(11, 8, requires_grad=True)
    g = torch.Generator(); g.manual_seed(99)
    loss = sampled_softmax_from_hidden(
        h, y, w, 10, n_negs=4, chunk_size=2, generator=g
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None and w.grad is not None


def test_sasrec_and_ligr_forward_match_output_shape():
    x = torch.tensor([[1,2,3,0],[4,5,6,7]])
    a = SASRec(100, 4, d=64, layers=1, heads=1, inner=256, dropout=0.0)
    b = ESASRec(100, 4, d=64, layers=1, heads=1, inner=256, dropout=0.0)
    assert a.encode(x).shape == b.encode(x).shape == (2,4,64)
