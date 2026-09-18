"""Train the spectrum -> fingerprint MLP (src/fingerprint_model.py) on the
cached featurized training set (scripts/build_fp_training_data.py).

Run: PYTHONPATH=. python3 scripts/train_fingerprint_model.py --epochs 5
Output: data/processed/fp_model.pt (best-val-loss checkpoint, with the
        config needed to rebuild the model at inference time)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.candidates import FP_BITS
from src.fingerprint_model import N_FEATURES, FingerprintMLP

ROOT = Path(__file__).resolve().parents[1]


def load_data(path: Path):
    z = np.load(path, allow_pickle=True)
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    return X, z["target_idx"], z["fp_table"], z["is_val"]


def dense_batch(X: sp.csr_matrix, idx: np.ndarray) -> np.ndarray:
    sub = X[idx]
    out = np.zeros((len(idx), X.shape[1]), dtype=np.float32)
    rows = np.repeat(np.arange(len(idx)), np.diff(sub.indptr))
    out[rows, sub.indices] = sub.data
    return out


def targets_batch(fp_table: np.ndarray, target_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
    packed = fp_table[target_idx[idx]]
    return np.unpackbits(packed, axis=1)[:, :FP_BITS].astype(np.float32)


@torch.no_grad()
def evaluate(model, X, target_idx, fp_table, idx, device, batch_size=4096):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_n = 0.0, 0
    tanimoto_sum = 0.0
    for start in range(0, len(idx), batch_size):
        b = idx[start : start + batch_size]
        x = torch.from_numpy(dense_batch(X, b)).to(device)
        y = torch.from_numpy(targets_batch(fp_table, target_idx, b)).to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        total_n += y.numel()
        pred = (logits > 0).float()
        inter = (pred * y).sum(1)
        union = pred.sum(1) + y.sum(1) - inter
        tanimoto_sum += torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter)).sum().item()
    return total_loss / total_n, tanimoto_sum / len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "data/processed/fp_model.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X, target_idx, fp_table, is_val = load_data(ROOT / "data/processed/fp_train_data.npz")
    train_idx = np.nonzero(~is_val)[0]
    val_idx = np.nonzero(is_val)[0]
    print(f"train spectra: {len(train_idx)}, val spectra: {len(val_idx)}, features: {X.shape[1]}")

    model = FingerprintMLP(n_features=N_FEATURES, hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = (len(train_idx) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        t0 = time.time()
        running = 0.0
        for step in range(steps_per_epoch):
            b = perm[step * args.batch : (step + 1) * args.batch]
            x = torch.from_numpy(dense_batch(X, b)).to(device, non_blocking=True)
            y = torch.from_numpy(targets_batch(fp_table, target_idx, b)).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if (step + 1) % 500 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{steps_per_epoch} loss {running/500:.4f} ({time.time()-t0:.0f}s)", flush=True)
                running = 0.0

        val_loss, val_tanimoto = evaluate(model, X, target_idx, fp_table, val_idx, device)
        print(f"epoch {epoch+1}: val_loss={val_loss:.4f} val_tanimoto@0.5={val_tanimoto:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"state_dict": model.state_dict(), "n_features": N_FEATURES, "hidden": args.hidden,
                 "dropout": args.dropout, "val_loss": val_loss, "val_tanimoto": val_tanimoto, "epoch": epoch + 1},
                args.out,
            )
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
