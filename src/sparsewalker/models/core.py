import torch
import torch.nn as nn
import torch.nn.functional as F


def init_embedding(emb: nn.Embedding, std: float = 0.02):
    nn.init.normal_(emb.weight, mean=0.0, std=std)
    if emb.padding_idx is not None:
        with torch.no_grad():
            emb.weight[emb.padding_idx].zero_()


class ARRecommender(nn.Module):
    model_name = "AR"
    def __init__(self, n_items: int, max_len: int, d_model: int):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.d_model = d_model
    @property
    def item_weight(self):
        raise NotImplementedError
    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def score_hidden(self, h: torch.Tensor) -> torch.Tensor:
        z = h @ self.item_weight[: self.n_items + 1].T
        z[..., 0] = -1e9
        return z
    def last_hidden(self, seq: torch.Tensor, lengths: torch.Tensor):
        H = self.encode(seq)
        rows = torch.arange(seq.size(0), device=seq.device)
        return H[rows, lengths - 1]
    def full_scores(self, seq: torch.Tensor, lengths: torch.Tensor):
        return self.score_hidden(self.last_hidden(seq, lengths))


def autoregressive_inputs(tokens, lengths=None):
    return tokens[:, :-1], tokens[:, 1:]


def sampled_softmax_from_hidden(
    hidden,
    target,
    item_weight,
    n_items,
    n_negs=256,
    temperature=1.0,
    chunk_size=4096,
    generator=None,
):
    """Catalog-uniform sampled softmax over every valid next-item position.

    Negatives are sampled independently for each valid training position. We keep
    accidental target hits, matching the straightforward catalog-uniform sampler
    used by the eSASRec/RecTools implementation family.

    Candidate scoring is chunked to bound temporary [position, negative, dim]
    tensors while retaining a single backward pass for the batch.
    """
    valid = target != 0
    h = hidden[valid]
    y = target[valid]
    if h.numel() == 0:
        return hidden.sum() * 0

    total = h.new_zeros(())
    count = int(h.size(0))
    chunk_size = max(1, int(chunk_size))

    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        hc = h[start:end]
        yc = y[start:end]
        m = end - start

        neg = torch.randint(
            1,
            n_items + 1,
            (m, n_negs),
            device=h.device,
            generator=generator,
        )
        pos = (hc * item_weight[yc]).sum(-1, keepdim=True)
        neg_emb = item_weight[neg]
        neg_logits = torch.bmm(neg_emb, hc.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([pos, neg_logits], dim=1) / temperature
        labels = torch.zeros(m, dtype=torch.long, device=h.device)
        total = total + F.cross_entropy(logits, labels, reduction="sum")

    return total / count


def ar_training_loss(
    model,
    tokens,
    lengths=None,
    loss_mode="full",
    n_negs=256,
    temperature=1.0,
    ss_chunk_size=4096,
    generator=None,
):
    x, y = autoregressive_inputs(tokens, lengths)
    H = model.encode(x)
    valid = y != 0
    if loss_mode == "ss":
        return sampled_softmax_from_hidden(
            H,
            y,
            model.item_weight,
            model.n_items,
            n_negs=n_negs,
            temperature=temperature,
            chunk_size=ss_chunk_size,
            generator=generator,
        )
    return F.cross_entropy(model.score_hidden(H[valid]), y[valid])
