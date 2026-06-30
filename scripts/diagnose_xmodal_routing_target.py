"""Diagnose (a): cross-modal subspace disconnect.

For each (image i, codebook m=0..5, patch p=0..195):
  sim[i, m, p] = cos(text_part_adapted[i, m, :], visual_token[i, p, :])

This is the EXACT signal used to build target_routing in L_routing_text:
  target_routing[i, m, p] = softmax_p(sim[i, m, p] / tau)

Report:
  1. Per-(image, slot) sim distribution over P patches:
     - mean, std, max, min
     - top-5 / top-1 selectivity (sharpness)
  2. Cross-slot differential per image:
     - Per patch p, std(sim[:, :, p] over m): are sim values different across slots?
     - Or are they all similar? (the disconnect hypothesis)
  3. Correlation across m of sim values per patch:
     - mean over m-pair of corr(sim[m_i], sim[m_j]) over patches
     - high correlation -> slots produce same patch ranking -> disconnect
  4. Compare with cos(codebook[m, k_m*], visual_token[p]) — alternative routing target

Inference cache: whole-image (cub200_clip_v6bplus), 196 patches/img.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)


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
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--inference_cache", required=True)
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args_cli = ap.parse_args()

    args = _parse_args_txt(os.path.join(args_cli.source_dir, "args.txt"))
    args.siglip2_feature_cache_dir = args_cli.inference_cache
    args.text_whiten_npz = os.path.join(args_cli.inference_cache, "text_whiten.npz")
    args.device = args_cli.device

    from model_siglip2 import SigLIP2SemanticOTModel
    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = os.path.join(args_cli.source_dir, "model_state_dict.pth")
    sd = torch.load(ckpt, map_location=args.device, weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[diag] ckpt loaded missing={len(miss)} unexpected={len(unexp)}")
    # NOTE: use train() to force text path computation (text_part_tokens is None in eval).
    # All weights stay frozen via torch.no_grad() and BN/Dropout effects are minor for diagnostics.
    model.train()

    from dataloaders import ImgRtvCUB2011
    dataset_dir = getattr(args, "dataset_dir", "dataset")
    qwen_path = getattr(args, "qwen_text_cache_path", None)
    ds = ImgRtvCUB2011(root=os.path.join(dataset_dir, "CUB_200"),
                       mode="database", setting_name="setting1",
                       qwen_text_cache_path=qwen_path,
                       siglip2_feature_cache_dir=args_cli.inference_cache)
    loader = torch.utils.data.DataLoader(ds, batch_size=args_cli.batch_size,
                                          shuffle=True, num_workers=2)
    print(f"[diag] dataset N={len(ds)}, sampling ~{args_cli.n_images} images")

    sims_xt    = []   # [B, M, P] cos(text[m], visual[p])
    sims_xcb   = []   # [B, M, P] cos(codebook[m, k_m*], visual[p])
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args_cli.n_images: break
            batch_t = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Only the cached_ keys are forward args.
            allowed = {"cached_visual_tokens_raw", "cached_visual_global", "cached_text_part_raw",
                       "cached_has_text", "cached_text_tokens", "cached_text_token_mask"}
            in_kwargs = {}
            for k, v in batch_t.items():
                if k in allowed:
                    in_kwargs[k] = v
                elif k == "has_text":
                    in_kwargs["cached_has_text"] = v
            out = model(**in_kwargs)
            v_tok = out.get("visual_tokens")        # [B, P, D]
            t_tok = out.get("text_part_tokens")     # [B, M, D]
            if v_tok is None or t_tok is None:
                print("[diag] WARN: visual_tokens / text_part_tokens missing")
                continue
            # Normalize for cosine
            v_n = torch.nn.functional.normalize(v_tok, dim=-1)
            t_n = torch.nn.functional.normalize(t_tok, dim=-1)
            sim_xt = torch.einsum('bmd,bpd->bmp', t_n, v_n)        # [B, M, P]
            sims_xt.append(sim_xt.cpu())

            # codebook-vs-visual sim using nearest codeword per (b, m)
            codebooks = None
            for k in ("codebooks", "codebook"):
                if k in out and out[k] is not None:
                    codebooks = out[k]
                    break
            if codebooks is None and hasattr(model, "quantizer"):
                q = model.quantizer
                if hasattr(q, "codebooks"):
                    codebooks = q.codebooks
                elif hasattr(q, "codebook"):
                    codebooks = q.codebook
            if codebooks is not None:
                cb = torch.nn.functional.normalize(codebooks, dim=-1)  # [M, K, D]
                # Use semantic_visual_tokens to pick nearest codeword per (b, m)
                z = out.get("semantic_visual_tokens")  # [B, M, D]
                if z is not None:
                    z_n = torch.nn.functional.normalize(z, dim=-1)
                    sim_zc = torch.einsum('bmd,mkd->bmk', z_n, cb)   # [B, M, K]
                    kstar = sim_zc.argmax(dim=-1)                     # [B, M]
                    # Gather codeword used per (b, m)
                    cb_used = torch.stack([
                        torch.stack([cb[m, kstar[b, m]] for m in range(cb.size(0))], dim=0)
                        for b in range(kstar.size(0))
                    ], dim=0)                                          # [B, M, D]
                    sim_xcb = torch.einsum('bmd,bpd->bmp', cb_used, v_n)
                    sims_xcb.append(sim_xcb.cpu())

            seen += batch["label"].size(0) if "label" in batch else v_tok.size(0)
            print(f"  batch: total seen={seen}")

    sim_xt = torch.cat(sims_xt, dim=0).numpy() if sims_xt else None
    sim_xcb = torch.cat(sims_xcb, dim=0).numpy() if sims_xcb else None
    print(f"\n[diag] sim_xt (text-vs-visual): {None if sim_xt is None else sim_xt.shape}")
    print(f"[diag] sim_xcb (codebook-vs-visual): {None if sim_xcb is None else sim_xcb.shape}")

    def report(name, sim):
        if sim is None:
            print(f"\n=== {name} ===\n  (no data)")
            return None
        N, M, P = sim.shape
        print(f"\n=== {name}  (N={N} images, M={M} slots, P={P} patches) ===")

        # 1) Per-(image, slot) over patches: mean, std, top-1, top5 mass
        ps_mean = sim.mean(axis=2)      # [N, M]
        ps_std  = sim.std(axis=2)       # [N, M]
        # Top-5 mass concentration
        sim_sorted = -np.sort(-sim, axis=2)  # [N, M, P] desc
        top5_mass = sim_sorted[:, :, :5].sum(axis=2) / sim.sum(axis=2).clip(min=1e-6)  # fraction of TOTAL
        print(f"  per-(img, slot) sim mean: avg over (N, M) = {ps_mean.mean():.4f}")
        print(f"  per-(img, slot) sim std:  avg over (N, M) = {ps_std.mean():.4f}")
        print(f"  per-(img, slot) max sim:  avg over (N, M) = {sim.max(axis=2).mean():.4f}")
        print(f"  per-(img, slot) min sim:  avg over (N, M) = {sim.min(axis=2).mean():.4f}")
        print(f"  per-(img, slot) top-5/total mass: avg over (N, M) = {top5_mass.mean():.4f}")

        # 2) Cross-slot differential per (image, patch):
        # std over m of sim[n, :, p] -> how different are the M slots at the SAME patch
        cs_std = sim.std(axis=1)        # [N, P]
        print(f"  cross-slot std (per image-patch, across m): mean = {cs_std.mean():.4f}")
        print(f"      |  if << per-(img, slot) std -> slots indistinguishable at patch level")

        # 3) Pearson correlation across m: for each (image, m_pair), corr(sim[m_i], sim[m_j]) over patches
        # Average over images and pairs.
        Mp = M
        corrs = []
        for n in range(min(N, 100)):
            for i in range(Mp):
                for j in range(i+1, Mp):
                    a, b = sim[n, i], sim[n, j]
                    a = a - a.mean(); b = b - b.mean()
                    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + 1e-9
                    corrs.append((a*b).sum() / denom)
        print(f"  cross-slot Pearson corr (mean over pairs, ~100 imgs): {np.mean(corrs):.4f}  std {np.std(corrs):.4f}")
        print(f"      |  > 0.7 -> slots produce nearly-identical patch ranking (DISCONNECT)")
        print(f"      |  < 0.3 -> slots produce differential ranking (HEALTHY)")

        # 4) Top-1 patch selectivity differential
        argmax_per_slot = sim.argmax(axis=2)  # [N, M]
        # Number of distinct argmax-patches across slots per image
        dist_top1 = np.array([len(set(argmax_per_slot[n].tolist())) for n in range(N)])
        print(f"  per-image distinct top-1 patches across M={M} slots:")
        print(f"      mean = {dist_top1.mean():.3f}  (max possible = {M})")
        print(f"      |  ~M -> each slot prefers a different patch (HEALTHY)")
        print(f"      |  ~1 -> all slots prefer same patch (DISCONNECT)")

        return {
            "per_slot_mean": float(ps_mean.mean()),
            "per_slot_std": float(ps_std.mean()),
            "per_slot_max": float(sim.max(axis=2).mean()),
            "per_slot_min": float(sim.min(axis=2).mean()),
            "per_slot_top5_mass": float(top5_mass.mean()),
            "cross_slot_std": float(cs_std.mean()),
            "cross_slot_pearson": float(np.mean(corrs)),
            "distinct_top1_per_img": float(dist_top1.mean()),
            "max_possible_distinct": int(M),
        }

    out_dict = {}
    out_dict["text_vs_visual"] = report("cos(text_part_adapted[m], visual_token[p])  ★ ROUTING TARGET ANALYZED", sim_xt)
    out_dict["codebook_vs_visual"] = report("cos(codebook[m, k_m*], visual_token[p])  (alternative routing target)", sim_xcb)

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON: text-target vs codebook-target")
    print("=" * 70)
    if out_dict["text_vs_visual"] and out_dict["codebook_vs_visual"]:
        t = out_dict["text_vs_visual"]
        c = out_dict["codebook_vs_visual"]
        print(f"  cross_slot_pearson:       text={t['cross_slot_pearson']:.4f}  codebook={c['cross_slot_pearson']:.4f}")
        print(f"  distinct_top1_per_img:    text={t['distinct_top1_per_img']:.3f}  codebook={c['distinct_top1_per_img']:.3f}  (max=6)")
        print(f"  per_slot_max sim:         text={t['per_slot_max']:.4f}  codebook={c['per_slot_max']:.4f}")
        print(f"  per_slot_std:             text={t['per_slot_std']:.4f}  codebook={c['per_slot_std']:.4f}")

    out_path = os.path.join(args_cli.source_dir, "diagnose_xmodal_routing_target.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n[diag] saved {out_path}")


if __name__ == "__main__":
    main()
