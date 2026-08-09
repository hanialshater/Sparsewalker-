from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
from sparsewalker.data import collate_windows
from sparsewalker.models.core import ar_training_loss


def _length_bucket_batches(dataset, batch_size, generator):
    """Shuffle once, then group similar maximum window lengths into batches.

    WindowDataset still samples exactly the same per-user window for an epoch;
    this only changes batch composition so short users do not pay for a random
    200-token neighbor's padding/recurrent steps.
    """
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    indices.sort(key=lambda i: min(len(dataset.seqs[i]), dataset.max_len + 1))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if len(batches) > 1:
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]
    return batches


def train_epoch(
    name,
    model,
    dataset,
    optimizer,
    device,
    batch_size=512,
    epoch=1,
    loss_mode="full",
    n_negs=256,
    mask_prob=.2,
    grad_clip=5.0,
    temperature=1.0,
    ss_chunk_size=4096,
    negative_seed=None,
    bucket_by_length=False,
    use_bf16=False,
    return_stats=False,
):
    dataset.set_epoch(epoch)
    g = torch.Generator()
    g.manual_seed(dataset.seed + epoch)
    loader_kwargs = dict(
        dataset=dataset,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    if bucket_by_length:
        loader = DataLoader(
            batch_sampler=_length_bucket_batches(dataset, batch_size, g),
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            **loader_kwargs,
        )

    neg_generator = None
    if loss_mode == "ss" and negative_seed is not None:
        neg_generator = torch.Generator(device=device.type)
        neg_generator.manual_seed(int(negative_seed) + int(epoch))

    bf16_enabled = bool(
        use_bf16
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    model.train()
    total = 0.0
    n = 0
    examples = 0
    positions = 0
    padded_positions = 0
    if name == "SparseWalker-E2E":
        model.begin_terminal_epoch()

    for tokens, lengths in loader:
        examples += int(tokens.size(0))
        positions += int((lengths - 1).clamp_min(0).sum().item())
        padded_positions += int(tokens.size(0) * max(0, tokens.size(1) - 1))
        non_blocking = device.type == "cuda"
        tokens = tokens.to(device, non_blocking=non_blocking)
        lengths = lengths.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16_enabled else nullcontext()
        with amp_context:
            if name == "BERT4Rec":
                loss = model.masked_loss(tokens, lengths, mask_prob, loss_mode, n_negs)
            elif name == "SparseWalker-E2E":
                loss = model.sparse_training_loss(tokens)
            else:
                loss = ar_training_loss(
                    model,
                    tokens,
                    lengths,
                    loss_mode,
                    n_negs,
                    temperature=temperature,
                    ss_chunk_size=ss_chunk_size,
                    generator=neg_generator,
                )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach())
        n += 1

    avg_loss = total / max(1, n)
    if return_stats:
        return {
            "loss": avg_loss,
            "batches": n,
            "examples": examples,
            "positions": positions,
            "padded_positions": padded_positions,
            "padding_efficiency": positions / max(1, padded_positions),
            "bf16": bf16_enabled,
            "bucket_by_length": bool(bucket_by_length),
        }
    return avg_loss
