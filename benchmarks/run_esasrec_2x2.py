#!/usr/bin/env python
"""Controlled eSASRec 2x2 experiment.

Architecture x loss:
  SASRec vs LiGR  x  FullCE vs catalog-uniform SS256

The experiment uses the released eSASRec geometry for each dataset while the
primary evaluation remains Sparse Walker's full-catalog temporal protocol.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SASRec, ESASRec
from sparsewalker.training import train_epoch


CELLS = {
    "SASRec+FullCE": (SASRec, "full"),
    "SASRec+SS": (SASRec, "ss"),
    "LiGR+FullCE": (ESASRec, "full"),
    "eSASRec": (ESASRec, "ss"),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def build(cell, n_items, max_len, args):
    cls, loss = CELLS[cell]
    model = cls(
        n_items,
        max_len,
        d=args.d_model,
        layers=args.layers,
        heads=args.heads,
        inner=args.d_model * args.ff_mult,
        dropout=args.dropout,
    )
    return model, loss


def run_cell(cell, data, split, args, device, out_root):
    cell_dir = out_root / cell.replace("+", "_")
    cell_dir.mkdir(parents=True, exist_ok=True)
    done_path = cell_dir / "done.json"
    if done_path.exists() and not args.force:
        print("SKIP completed", cell)
        return json.loads(done_path.read_text())

    # Reset before every cell: paired loss comparisons start from identical
    # parameters for a fixed architecture and use identical training windows.
    seed_all(args.seed)
    max_len = args.max_len
    model, loss_mode = build(cell, data["n_items"], max_len, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ds = WindowDataset(split["train"], max_len, args.seed)

    best_ndcg = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    last_path = cell_dir / "last.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_ndcg = float(ckpt["best_ndcg"])
        best_epoch = int(ckpt["best_epoch"])
        bad_epochs = int(ckpt["bad_epochs"])
        history = ckpt.get("history", [])
        best_state = ckpt.get("best_state")
        print("RESUME", cell, "from epoch", start_epoch)
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.max_epochs + 1):
        loss = train_epoch(
            cell,
            model,
            ds,
            optimizer,
            device,
            batch_size=args.batch_size,
            epoch=epoch,
            loss_mode=loss_mode,
            n_negs=args.n_negs,
            temperature=args.temperature,
            ss_chunk_size=args.ss_chunk_size,
            negative_seed=args.seed * 1_000_003,
        )

        should_eval = epoch == 1 or epoch % args.eval_every == 0
        if not should_eval:
            continue

        val = evaluate_full(
            model,
            split["val_prefix"],
            split["val_target"],
            data["n_items"],
            max_len,
            device,
            topks=(10,),
            batch_size=args.eval_batch_size,
        )
        row = {"cell": cell, "epoch": epoch, "loss": loss, **val}
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(cell_dir / "history.csv", index=False)

        ndcg = float(val["NDCG@10"])
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch
            bad_epochs = 0
            best_state = cpu_state_dict(model)
            torch.save(
                {"model": best_state, "epoch": best_epoch, "val": val},
                cell_dir / "best.pt",
            )
        else:
            bad_epochs += args.eval_every

        # Durable restart point. Including best_state costs disk, but prevents
        # an interrupted runtime from losing the selected checkpoint.
        torch.save(
            {
                "model": cpu_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_ndcg": best_ndcg,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "best_state": best_state,
                "history": history,
            },
            last_path,
        )

        if bad_epochs >= args.patience:
            print("EARLY STOP", cell, "best epoch", best_epoch)
            break

    if best_state is None:
        raise RuntimeError(f"No validation checkpoint produced for {cell}")
    model.load_state_dict(best_state)

    test = evaluate_full(
        model,
        split["test_prefix"],
        split["test_target"],
        data["n_items"],
        max_len,
        device,
        topks=(10, 20, 50),
        batch_size=args.eval_batch_size,
    )
    result = {
        "cell": cell,
        "architecture": "LiGR" if cell.startswith("LiGR") or cell == "eSASRec" else "SASRec",
        "loss": "SS256" if loss_mode == "ss" else "FullCE",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_NDCG@10": best_ndcg,
        "params": sum(p.numel() for p in model.parameters()),
        **test,
    }
    save_json(done_path, result)
    print("TEST", result)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def add_decomposition(df):
    idx = df.set_index("cell")
    metric = "NDCG@10"
    if not all(x in idx.index for x in CELLS):
        return None
    a = float(idx.loc["SASRec+FullCE", metric])
    b = float(idx.loc["SASRec+SS", metric])
    c = float(idx.loc["LiGR+FullCE", metric])
    d = float(idx.loc["eSASRec", metric])
    return {
        "SASRec_SS_minus_FullCE": b - a,
        "LiGR_FullCE_minus_SASRec_FullCE": c - a,
        "eSASRec_minus_SASRec_SS": d - b,
        "eSASRec_minus_SASRec_FullCE": d - a,
        "interaction_LiGR_x_SS": (d - c) - (b - a),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=["beauty", "ml1m"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="*", default=list(CELLS))
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-negs", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ss-chunk-size", type=int, default=2048)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/sparsewalker_esasrec_2x2")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    bad = [m for m in args.models if m not in CELLS]
    if bad:
        raise ValueError(f"Unknown cells: {bad}; valid={list(CELLS)}")

    # Selected eSASRec geometry from the released benchmark.
    if args.dataset == "beauty":
        args.d_model = 64
        args.layers = 1
        args.heads = 1
        args.dropout = .2
        args.ff_mult = 4
        args.max_len = 50
        if args.max_epochs is None:
            args.max_epochs = 200
    else:
        args.d_model = 64
        args.layers = 2
        args.heads = 1
        args.dropout = .1
        args.ff_mult = 4
        args.max_len = 200
        if args.max_epochs is None:
            args.max_epochs = 100

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "config", vars(args))

    data = load_dataset(args.dataset, args.data_dir)
    split = split_data(data["sequences"])
    out_root = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "config.json", vars(args))

    results = []
    for cell in args.models:
        results.append(run_cell(cell, data, split, args, device, out_root))

    # Include already-completed cells in the final summary.
    for cell in CELLS:
        if cell in [x["cell"] for x in results]:
            continue
        done = out_root / cell.replace("+", "_") / "done.json"
        if done.exists():
            results.append(json.loads(done.read_text()))

    df = pd.DataFrame(results).sort_values("cell")
    df.to_csv(out_root / "summary.csv", index=False)
    print("\nSUMMARY")
    print(df.to_string(index=False))

    decomp = add_decomposition(df)
    if decomp is not None:
        save_json(out_root / "decomposition.json", decomp)
        print("\nNDCG@10 DECOMPOSITION")
        print(json.dumps(decomp, indent=2))


if __name__ == "__main__":
    main()
