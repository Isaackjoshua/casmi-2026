"""Train the peak transformer to predict the multi-fingerprint stack.

Same architecture that produced the best checkpoint to date (d_model 384,
6 layers), same BCE loss, same data. The one change is the target: 3239
bits across Morgan r2, Morgan r3 and MACCS instead of 2048 Morgan r2 bits
(see src/multi_fingerprint.py for why that is the change worth making
rather than a bigger Morgan predictor).

Two deliberate differences from scripts/train_peak_transformer.py:

  - every epoch is saved, not just the best by val loss and by
    Tanimoto@0.5. Those two selection criteria have each pointed the wrong
    way on this project, and the deployed checkpoint ended up being chosen
    by retrieval MRR after the fact. Saving all of them lets
    scripts/select_multi_checkpoint.py pick on the task itself.
  - per-block validation Tanimoto is reported, because the three blocks
    can be learned to very different standards and one average would hide
    it. If MACCS is predicted far better than Morgan, that is worth
    knowing before scoring weights it equally.

Run: PYTHONPATH=. python3 scripts/train_multi_fingerprint.py --epochs 8
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.multi_fingerprint import OFFSETS, TOTAL_BITS
from src.peak_transformer import PeakTransformer

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"


def targets_batch(packed, target_idx, idx):
    return np.unpackbits(packed[target_idx[idx]], axis=1)[:, :TOTAL_BITS].astype(np.float32)


def batch_tensors(d, idx, device):
    return (
        torch.from_numpy(d["mz"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["inten"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["mask"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["precursor"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["mode"][idx].astype(np.int64)).to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model, d, packed, target_idx, idx, device, batch_size=1024):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_n = 0.0, 0
    inter = {k: 0.0 for k in OFFSETS}
    union = {k: 0.0 for k in OFFSETS}
    for start in range(0, len(idx), batch_size):
        b = idx[start : start + batch_size]
        y = torch.from_numpy(targets_batch(packed, target_idx, b)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(*batch_tensors(d, b, device))
        logits = logits.float()
        total_loss += loss_fn(logits, y).item()
        total_n += y.numel()
        pred = (torch.sigmoid(logits) >= 0.5)
        true = y >= 0.5
        for k, (lo, hi) in OFFSETS.items():
            p, t = pred[:, lo:hi], true[:, lo:hi]
            inter[k] += (p & t).sum().item()
            union[k] += (p | t).sum().item()
    tans = {k: (inter[k] / union[k] if union[k] else 0.0) for k in OFFSETS}
    return total_loss / total_n, tans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--prefix", default=str(P / "multi_model"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    z = np.load(P / "fp_train_data.npz", allow_pickle=True)
    target_idx, is_val = z["target_idx"], z["is_val"]
    mf = np.load(P / "train_multi_fp.npz", allow_pickle=True)
    packed, ok = mf["packed"], mf["ok"]
    pk = np.load(P / "peak_train_data.npz")
    d = {k: pk[k] for k in ["mz", "inten", "mask", "precursor", "mode"]}
    assert len(d["mz"]) == len(target_idx)

    # drop spectra whose structure has no usable multi-fingerprint
    usable = ok[target_idx]
    train_idx = np.nonzero(~is_val & usable)[0]
    val_idx = np.nonzero(is_val & usable)[0]
    print(f"train spectra: {len(train_idx):,}  val spectra: {len(val_idx):,}  "
          f"(dropped {int((~usable).sum()):,} with no target)", flush=True)

    cfg = {"d_model": args.d_model, "n_layers": args.layers, "n_heads": args.heads,
           "dropout": args.dropout, "n_bits": TOTAL_BITS}
    model = PeakTransformer(**cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"config {cfg}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_idx) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)

    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        t0, running = time.time(), 0.0
        for step in range(steps_per_epoch):
            b = np.sort(perm[step * args.batch : (step + 1) * args.batch])
            y = torch.from_numpy(targets_batch(packed, target_idx, b)).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(*batch_tensors(d, b, device))
            loss = loss_fn(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if (step + 1) % 500 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{steps_per_epoch} "
                      f"loss {running/500:.4f} ({time.time()-t0:.0f}s)", flush=True)
                running = 0.0

        val_loss, tans = evaluate(model, d, packed, target_idx, val_idx, device)
        tan_str = "  ".join(f"{k} {v:.4f}" for k, v in tans.items())
        print(f"epoch {epoch+1}: val_loss={val_loss:.5f}  Tanimoto@0.5 by block: {tan_str}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        path = f"{args.prefix}_e{epoch+1}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg, "val_loss": val_loss,
                    "val_tanimoto_blocks": tans, "epoch": epoch + 1}, path)
        print(f"  saved {path}", flush=True)


if __name__ == "__main__":
    main()
