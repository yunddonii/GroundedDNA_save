"""Diagnose: measure cosine similarity matrix between text_part[m] embeddings
across slots m=0..5 (6 codebooks: cb0=global, cb1..cb5=local).

Stages:
  (a) RAW            — straight from CLIP text encoder, cached at text_part.f16.npy
  (b) WHITENED       — partial-whiten γ=0.25 (ZCA decorrelation)
  (c) ADAPTED        — RAW → partial_whiten → per-slot adapter (FINAL routing target)
                       Requires loading the trained model.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)


def cos_sim_matrix_np(X: np.ndarray) -> np.ndarray:
    """X: [N, M, D] → per-sample [N, M, M] cos sim → mean over N."""
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
    sim = np.einsum('nmd,nkd->nmk', Xn, Xn)
    return sim.mean(axis=0)  # [M, M]


def cos_sim_matrix_t(X: torch.Tensor) -> torch.Tensor:
    """X: [N, M, D] → per-sample [N, M, M] cos sim → mean over N."""
    Xn = torch.nn.functional.normalize(X, dim=-1)
    sim = torch.einsum('nmd,nkd->nmk', Xn, Xn)
    return sim.mean(dim=0)  # [M, M]


def report(name: str, sim: np.ndarray, local_only: bool = False):
    M = sim.shape[0]
    print(f"\n=== {name}  (M={M}) ===")
    header = "       " + "  ".join([f"m{i:>2d}" for i in range(M)])
    print(header)
    for i in range(M):
        row = "  ".join([f"{sim[i, j]:>5.3f}" for j in range(M)])
        print(f"  m{i:>2d}  {row}")
    start = 1 if local_only else 0
    off_idx = [(i, j) for i in range(start, M) for j in range(start, M) if i != j]
    off_vals = np.array([sim[i, j] for i, j in off_idx])
    print(f"  off-diag mean = {off_vals.mean():.4f}")
    print(f"  off-diag max  = {off_vals.max():.4f}")
    print(f"  off-diag min  = {off_vals.min():.4f}")
    print(f"  off-diag std  = {off_vals.std():.4f}")
    return {
        "mean": float(off_vals.mean()),
        "max": float(off_vals.max()),
        "min": float(off_vals.min()),
        "std": float(off_vals.std()),
        "matrix": sim.tolist(),
    }


def _parse_args_txt(path: str) -> SimpleNamespace:
    ns = SimpleNamespace()
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            for k_end in range(len(line)):
                if line[k_end] == "-": break
            else: continue
            v_start = k_end
            while v_start < len(line) and line[v_start] == "-": v_start += 1
            key = line[:k_end]
            val = line[v_start:].strip()
            if not key: continue
            if val == "None": val = None
            elif val == "True": val = True
            elif val == "False": val = False
            else:
                try: val = int(val)
                except ValueError:
                    try: val = float(val)
                    except ValueError:
                        if val.startswith("[") and val.endswith("]"):
                            inner = val[1:-1].strip()
                            if inner.startswith("'") and inner.endswith("'"):
                                val = [inner[1:-1]]
            setattr(ns, key, val)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache dir with text_part.f16.npy")
    ap.add_argument("--source_dir", default=None,
                    help="Trained run dir (for ADAPTED stage). If unset, skip stage (c).")
    ap.add_argument("--n_samples", type=int, default=2000,
                    help="Random subset for stage (c) ADAPTED if too large.")
    ap.add_argument("--device", default="cuda:0")
    args_cli = ap.parse_args()

    # ===== Stage (a) RAW =====
    text_path = os.path.join(args_cli.cache, "text_part.f16.npy")
    has_text_path = os.path.join(args_cli.cache, "has_text.bool.npy")
    text_raw = np.load(text_path).astype(np.float32)  # [N, M, 512]
    if os.path.exists(has_text_path):
        ht = np.load(has_text_path)
        text_raw = text_raw[ht]
        print(f"[diag] raw text loaded: {text_raw.shape} (filtered has_text)")
    else:
        print(f"[diag] raw text loaded: {text_raw.shape} (no has_text filter)")

    result = {}
    sim_raw = cos_sim_matrix_np(text_raw)
    result["raw"] = report("text_part_RAW (post-CLIP, pre-anything)", sim_raw, local_only=False)
    result["raw_local"] = report("text_part_RAW LOCAL m=1..M-1 (skip cb0=global)", sim_raw, local_only=True)

    # ===== Stage (b) WHITENED =====
    whiten_npz = os.path.join(args_cli.cache, "text_whiten.npz")
    if os.path.exists(whiten_npz):
        w = np.load(whiten_npz)
        mu = w["mu"]              # [D]
        U = w["U"]                # eigvecs [D, D]
        S = w["S"]                # eigvals (singular values) [D]
        # ZCA whitening matrix: U @ diag(1/sqrt(S + eps)) @ U.T
        eps = 1e-6
        W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
        gamma = 0.25
        # partial-whiten: x_w = mu + (1-γ)·(x − mu) + γ·(W·(x − mu))
        N, M, D = text_raw.shape
        flat = text_raw.reshape(-1, D) - mu
        whitened = mu + (1.0 - gamma) * flat + gamma * (flat @ W.T)
        text_whit = whitened.reshape(N, M, D)
        sim_whit = cos_sim_matrix_np(text_whit)
        result["whitened"] = report(f"text_part_WHITENED (partial γ={gamma}, ZCA via cached U/S)", sim_whit, local_only=False)
        result["whitened_local"] = report("text_part_WHITENED LOCAL", sim_whit, local_only=True)
    else:
        print(f"\n[diag] {whiten_npz} not found, skip stage (b)")
        W = None

    # ===== Stage (c) ADAPTED =====
    if args_cli.source_dir:
        args = _parse_args_txt(os.path.join(args_cli.source_dir, "args.txt"))
        args.siglip2_feature_cache_dir = args_cli.cache
        args.text_whiten_npz = whiten_npz
        args.device = args_cli.device

        from model_siglip2 import SigLIP2SemanticOTModel
        model = SigLIP2SemanticOTModel(args).to(args.device)
        ckpt = os.path.join(args_cli.source_dir, "model_state_dict.pth")
        sd = torch.load(ckpt, map_location=args.device, weights_only=False)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"\n[diag] ckpt loaded missing={len(miss)} unexpected={len(unexp)}")
        model.eval()

        # Sub-sample
        N = text_raw.shape[0]
        n = min(args_cli.n_samples, N)
        idx = np.random.choice(N, n, replace=False)
        text_subset = torch.from_numpy(text_raw[idx]).to(args.device)  # [n, M, 512]

        adapted_all = []
        bs = 64
        with torch.no_grad():
            for i in range(0, n, bs):
                tp_raw_batch = text_subset[i:i+bs]  # [b, M, 512]
                # Apply partial whitening manually (model does this internally too)
                W_t = torch.from_numpy(W).to(args.device).float() if W is not None else None
                mu_t = torch.from_numpy(mu).to(args.device).float()
                gamma = 0.25
                B_, M_, D_ = tp_raw_batch.shape
                flat = tp_raw_batch.reshape(-1, D_) - mu_t
                whitened = mu_t + (1.0 - gamma) * flat + gamma * (flat @ W_t.T)
                tp_whit = whitened.reshape(B_, M_, D_)
                # Apply per-slot adapter (text_adapter)
                # The adapter is a per-slot MLP — find it on model
                if isinstance(model.text_adapter, torch.nn.ModuleList):
                    per_slot = []
                    for m in range(M_):
                        adapter_m = model.text_adapter[m]
                        per_slot.append(adapter_m(tp_whit[:, m, :]))
                    tp_adapted = torch.stack(per_slot, dim=1)
                else:
                    tp_adapted = model.text_adapter(tp_whit)
                # text_token_ln (post-adapter LayerNorm applied in forward)
                if hasattr(model, "text_token_ln"):
                    tp_adapted = model.text_token_ln(tp_adapted)
                adapted_all.append(tp_adapted.cpu())
        adapted = torch.cat(adapted_all, dim=0)
        sim_adapted = cos_sim_matrix_t(adapted).numpy()
        result["adapted"] = report("text_part_ADAPTED (post per-slot text_adapter) — ROUTING TARGET", sim_adapted, local_only=False)
        result["adapted_local"] = report("text_part_ADAPTED LOCAL m=1..M-1", sim_adapted, local_only=True)

    # Save
    out_dir = args_cli.source_dir if args_cli.source_dir else args_cli.cache
    out_path = os.path.join(out_dir, "diagnose_text_part_cossim.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[diag] saved {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (off-diag mean cos sim across slots)")
    print("=" * 60)
    for key in ["raw_local", "whitened_local", "adapted_local"]:
        if key in result:
            print(f"  {key:20s}: {result[key]['mean']:.4f}  (max {result[key]['max']:.4f})")


if __name__ == "__main__":
    main()
