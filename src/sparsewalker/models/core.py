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


def sampled_softmax_from_hidden(hidden, target, item_weight, n_items, n_negs=256, temperature=1.0):
    valid = target != 0
    h = hidden[valid]
    y = target[valid]
    if h.numel() == 0:
        return hidden.sum() * 0
    neg = torch.randint(1, n_items + 1, (n_negs,), device=h.device)
    pos = (h * item_weight[y]).sum(-1, keepdim=True) / temperature
    neg_logits = (h @ item_weight[neg].T) / temperature
    neg_logits = neg_logits.masked_fill(neg[None, :] == y[:, None], -1e9)
    logits = torch.cat([pos, neg_logits], 1)
    labels = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
    return F.cross_entropy(logits, labels)


def ar_training_loss(model, tokens, lengths=None, loss_mode="full", n_negs=256, temperature=1.0):
    x, y = autoregressive_inputs(tokens, lengths)
    H = model.encode(x)
    valid = y != 0
    if loss_mode == "ss":
        return sampled_softmax_from_hidden(H, y, model.item_weight, model.n_items, n_negs, temperature)
    return F.cross_entropy(model.score_hidden(H[valid]), y[valid])
