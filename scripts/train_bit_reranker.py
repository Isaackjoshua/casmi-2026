"""Learn to rank, not to predict bits.

Inference scores a candidate as `bits @ logit`, where logit = log p - log(1-p)
comes from the model (see fingerprint_model.fingerprint_loglik_scores). The
model, though, is trained with per-bit BCE: it optimizes how well each bit is
predicted, never whether the true structure outscores the few thousand
isobaric candidates it is actually compared against. The gating experiment
showed the cost of that mismatch directly.

Because the score is linear in the logits, the aligned objective is cheap to
test without retraining anything: reweight the logit vector,

    score(f) = f . (w * logit + b)

and fit the 4096 parameters (w, b) with a listwise softmax over the true
structure against hard decoys from its own mass window (scripts/build_decoys.py).
w = 1, b = 0 reproduces the current ranker exactly, so it is the initialization
and any improvement over it is unambiguous.

Run: PYTHONPATH=. python3 scripts/train_bit_reranker.py
Output: data/processed/bit_reranker.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.candidates import FP_BITS
from src.fingerprint_model import predict_probs
from src.pipeline_v3 import load_fingerprint_model

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4


def ensemble_logits(models, Xcsr, peaks, idx, device, batch=1024):
    """The exact logit vector inference uses: probabilities averaged across
    the ensemble, then turned into logits.
    """
    mlp, tf = models
    out = np.empty((len(idx), FP_BITS), dtype=np.float32)
    for s in range(0, len(idx), batch):
        b = idx[s : s + batch]
        probs = []
        if mlp is not None:
            probs.append(predict_probs(mlp, Xcsr[b], device))
        if tf is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                logits = tf(
                    torch.from_numpy(peaks["mz"][b]).to(device),
                    torch.from_numpy(peaks["inten"][b]).to(device),
                    torch.from_numpy(peaks["mask"][b]).to(device),
                    torch.from_numpy(peaks["precursor"][b]).to(device),
                    torch.from_numpy(peaks["mode"][b].astype(np.int64)).to(device),
                )
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        p = np.clip(np.mean(probs, axis=0).astype(np.float64), EPS, 1 - EPS)
        out[s : s + len(b)] = (np.log(p) - np.log1p(-p)).astype(np.float32)
    return out


def listwise_metrics(adj, true_bits, decoy_bits, valid):
    """top-1 rate and MRR of the true structure against its own decoys."""
    s_true = (true_bits * adj).sum(-1)
    s_dec = torch.einsum("bkd,bd->bk", decoy_bits, adj)
    s_dec = s_dec.masked_fill(~valid, float("-inf"))
    beaten = (s_dec > s_true[:, None]).sum(1)
    return (beaten == 0).float().mean().item(), (1.0 / (1 + beaten).float()).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra-per-structure", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--l2-to-init", type=float, default=1e-3,
                    help="pull (w,b) toward (1,0); the BCE-trained logits are a strong prior")
    ap.add_argument("--decoy-file", default=str(P / "decoys.npz"),
                    help="sampled (build_decoys.py) or mined (mine_decoys.py) negatives")
    ap.add_argument("--out", default=str(P / "bit_reranker.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = np.load(args.decoy_file, allow_pickle=True)
    dkeys, true_fp, decoy_fp, n_valid = dec["keys"], dec["true_fp"], dec["decoy_fp"], dec["n_valid"]
    K = decoy_fp.shape[1]
    print(f"{len(dkeys):,} structures x {K} decoys from {Path(args.decoy_file).name}", flush=True)

    d = np.load(P / "fp_train_data.npz", allow_pickle=True)
    Xcsr = sp.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]), shape=tuple(d["X_shape"]))
    peaks = {k: np.load(P / "peak_train_data.npz", allow_pickle=True)[k]
             for k in ("mz", "inten", "mask", "precursor", "mode")}
    spec_key = np.asarray(d["keys"])[d["target_idx"]]

    # one structure -> a couple of its spectra
    by_key = {}
    for i, k in enumerate(spec_key):
        lst = by_key.setdefault(k, [])
        if len(lst) < args.spectra_per_structure:
            lst.append(i)

    rows, struct_of = [], []
    for si, k in enumerate(dkeys):
        for i in by_key.get(k, ()):
            rows.append(i)
            struct_of.append(si)
    rows = np.asarray(rows)
    struct_of = np.asarray(struct_of)
    print(f"{len(rows):,} spectra covering {len(set(struct_of)):,} structures", flush=True)

    models = (load_fingerprint_model(str(P / "fp_model.pt"), device),
              load_fingerprint_model(str(P / "peak_model.pt"), device))
    t0 = time.time()
    Z = ensemble_logits(models, Xcsr, peaks, rows, device)
    print(f"logits computed in {time.time()-t0:.0f}s", flush=True)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # split by STRUCTURE, so a structure's other spectra cannot leak
    rng = np.random.default_rng(0)
    uniq = np.unique(struct_of)
    val_structs = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False).tolist())
    is_val = np.array([s in val_structs for s in struct_of])
    print(f"train {int((~is_val).sum()):,} spectra / val {int(is_val.sum()):,} spectra", flush=True)

    tb_all = torch.from_numpy(np.unpackbits(true_fp, axis=1)[:, :FP_BITS]).float()
    db_all = torch.from_numpy(np.unpackbits(decoy_fp.reshape(-1, decoy_fp.shape[2]), axis=1)[:, :FP_BITS]
                              .reshape(len(dkeys), K, FP_BITS)).float()
    valid_all = torch.from_numpy(np.arange(K)[None, :] < n_valid[:, None])

    w = torch.ones(FP_BITS, device=device, requires_grad=True)
    b = torch.zeros(FP_BITS, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)

    def batch_of(sel):
        z = torch.from_numpy(Z[sel]).to(device)
        s = struct_of[sel]
        return z, tb_all[s].to(device), db_all[s].to(device), valid_all[s].to(device)

    @torch.no_grad()
    def evaluate(sel, use_init=False):
        t1s, mrrs, n = 0.0, 0.0, 0
        for s in range(0, len(sel), 512):
            part = sel[s : s + 512]
            z, tb, db, vb = batch_of(part)
            adj = z if use_init else w * z + b
            a, m = listwise_metrics(adj, tb, db, vb)
            t1s += a * len(part); mrrs += m * len(part); n += len(part)
        return t1s / n, mrrs / n

    tr_idx = np.flatnonzero(~is_val)
    va_idx = np.flatnonzero(is_val)
    a0, m0 = evaluate(va_idx, use_init=True)
    print(f"\ncurrent ranker (w=1, b=0): val top-1 {a0:.4f}  MRR {m0:.4f}", flush=True)

    best = (m0, None)
    for epoch in range(args.epochs):
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            sel = perm[s : s + args.batch]
            z, tb, db, vb = batch_of(sel)
            adj = w * z + b
            s_true = (tb * adj).sum(-1)
            s_dec = torch.einsum("bkd,bd->bk", db, adj).masked_fill(~vb, float("-inf"))
            scores = torch.cat([s_true[:, None], s_dec], dim=1) / args.tau
            loss = nn.functional.cross_entropy(scores, torch.zeros(len(sel), dtype=torch.long, device=device))
            loss = loss + args.l2_to_init * ((w - 1) ** 2).mean() + args.l2_to_init * (b ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(sel)
        a, m = evaluate(va_idx)
        flag = ""
        if m > best[0]:
            best = (m, (w.detach().cpu().numpy().copy(), b.detach().cpu().numpy().copy()))
            flag = "  *"
        print(f"epoch {epoch+1:3d}: loss {tot/len(perm):.4f}  val top-1 {a:.4f}  MRR {m:.4f}{flag}", flush=True)

    print(f"\nbest val MRR {best[0]:.4f} vs {m0:.4f} for the current ranker "
          f"({100*(best[0]/m0-1):+.1f}%)")
    if best[1] is None:
        print("no improvement over w=1,b=0; nothing saved")
        return
    np.savez(args.out, w=best[1][0], b=best[1][1], tau=args.tau,
             val_mrr=best[0], baseline_mrr=m0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
