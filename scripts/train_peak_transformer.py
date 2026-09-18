"""Train the peak transformer (src/peak_transformer.py) on the padded peak
arrays (scripts/build_peak_training_data.py), reusing the targets and
structure split from fp_train_data.npz. Mixed precision on GPU.

Run: PYTHONPATH=. python3 scripts/train_peak_transformer.py --epochs 6
Output: data/processed/peak_model.pt (best val loss) and
        data/processed/peak_model_tan.pt (best val Tanimoto@0.5)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.candidates import FP_BITS
from src.peak_transformer import PeakTransformer

ROOT = Path(__file__).resolve().parents[1]


def targets_batch(fp_table, target_idx, idx):
    return np.unpackbits(fp_table[target_idx[idx]], axis=1)[:, :FP_BITS].astype(np.float32)


def batch_tensors(d, idx, device):
    return (
        torch.from_numpy(d["mz"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["inten"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["mask"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["precursor"][idx]).to(device, non_blocking=True),
        torch.from_numpy(d["mode"][idx].astype(np.int64)).to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model, d, fp_table, target_idx, idx, device, batch_size=2048):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_n, tan_sum = 0.0, 0, 0.0
    for start in range(0, len(idx), batch_size):
        b = idx[start : start + batch_size]
        y = torch.from_numpy(targets_batch(fp_table, target_idx, b)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(*batch_tensors(d, b, device))
        logits = logits.float()
        total_loss += loss_fn(logits, y).item()
        total_n += y.numel()
        pred = (logits > 0).float()
        inter = (pred * y).sum(1)
        union = pred.sum(1) + y.sum(1) - inter
        tan_sum += torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter)).sum().item()
    return total_loss / total_n, tan_sum / len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--finetune-from", default=None)
    ap.add_argument("--out", default=str(ROOT / "data/processed/peak_model.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    z = np.load(ROOT / "data/processed/fp_train_data.npz", allow_pickle=True)
    target_idx, fp_table, is_val, ingest_lib = z["target_idx"], z["fp_table"], z["is_val"], z["ingest_lib"]
    pk = np.load(ROOT / "data/processed/peak_train_data.npz")
    d = {k: pk[k] for k in ["mz", "inten", "mask", "precursor", "mode"]}
    assert len(d["mz"]) == len(target_idx)

    train_mask = ~is_val
    if args.sources:
        train_mask &= np.isin(ingest_lib, args.sources)
        print(f"restricting training to sources {args.sources}")
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(is_val)[0]
    print(f"train spectra: {len(train_idx)}, val spectra: {len(val_idx)}")

    if args.finetune_from:
        ckpt = torch.load(args.finetune_from, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = PeakTransformer(**cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"fine-tuning from {args.finetune_from}")
    else:
        cfg = {"d_model": args.d_model, "n_layers": args.layers, "n_heads": args.heads, "dropout": args.dropout}
        model = PeakTransformer(**cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, config {cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_idx) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    out_tan = args.out[:-3] + "_tan.pt"

    best_val, best_tan = float("inf"), -1.0
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        t0 = time.time()
        running = 0.0
        for step in range(steps_per_epoch):
            b = np.sort(perm[step * args.batch : (step + 1) * args.batch])
            y = torch.from_numpy(targets_batch(fp_table, target_idx, b)).to(device, non_blocking=True)
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
                print(f"  epoch {epoch+1} step {step+1}/{steps_per_epoch} loss {running/500:.4f} ({time.time()-t0:.0f}s)", flush=True)
                running = 0.0

        val_loss, val_tan = evaluate(model, d, fp_table, target_idx, val_idx, device)
        print(f"epoch {epoch+1}: val_loss={val_loss:.4f} val_tanimoto@0.5={val_tan:.4f} ({time.time()-t0:.0f}s)", flush=True)
        ckpt = {"state_dict": model.state_dict(), "config": cfg, "val_loss": val_loss, "val_tanimoto": val_tan, "epoch": epoch + 1}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.out)
            print(f"  saved {args.out} (best val loss)", flush=True)
        if val_tan > best_tan:
            best_tan = val_tan
            torch.save(ckpt, out_tan)
            print(f"  saved {out_tan} (best val Tanimoto)", flush=True)


if __name__ == "__main__":
    main()
