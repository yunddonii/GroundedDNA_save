#!/usr/bin/env python
"""PER-IMAGE: does any slot receive zero visual tokens, making its codon content-free?

Every earlier dead-slot measurement averaged the transported mass over the
batch, which cannot separate two very different situations for a slot at 0.9 %
mean share:

    (a) it receives a thin slice of mass on EVERY image  -> pooled feature is a
        real, image-varying direction; the codon means something.
    (b) it receives EXACTLY zero on most images          -> `denom.clamp_min(1e-12)`
        (semantic_router.py:498) makes the pooled feature the ZERO VECTOR, so the
        codon is a constant filler carrying no information about that image.

(b) would be a design defect: the slot's three bases would be reserved capacity
spent on a symbol that says nothing. The adaptive top-p mask makes it possible,
because `keep_sorted[...,0] = True` only guarantees that each PATCH keeps its
own rank-1 slot — nothing guarantees a given SLOT is anyone's rank-1.

Reports per (image, slot):
  n_tokens  number of patches with strictly positive routing weight
  mass      column mass after masking
and aggregates the fraction of images where a slot is completely empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]

#: Everything whose bytes decide what this tool reports.  Recorded in the output
#: so a table built from it can be re-derived rather than believed -- including
#: this file, which is edited more often than the model it measures.
_PROVENANCE_SOURCES = (
    "scripts/diagnose_per_image_empty_slots.py",
    "scripts/regen_viz_routing.py",
    "model_siglip2.py",
    "models/semantic_router.py",
    "dna_utils/visualization.py",
)


def _sha256_file(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--record", required=True, metavar="PATH", help=(
        "the cell's sealed selection record. The epoch to measure at and the "
        "checkpoint to measure are BOTH taken from here; there is no flag to "
        "state them by hand."))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--disable_adaptive_topp", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # The epoch decides the Sinkhorn epsilon, and the epsilon decides the
    # routing distribution this tool measures.  `--epoch` used to be an
    # unchecked integer that was simply believed, and omitting it left the
    # model at epoch 0 -- which is how a whole diagnostic table ended up
    # measured at the wrong operating point.  Both now come from the cell's own
    # sealed record, and a cell without one is not measured.
    record_path = os.path.abspath(a.record)
    with open(record_path, "rb") as handle:
        record_bytes = handle.read()
    record = json.loads(record_bytes.decode("utf-8"))
    completion = record.get("completion") or {}
    epoch = completion.get("final_checkpoint_epoch_zero_based")
    expected_ckpt = completion.get("final_checkpoint_sha256")
    if not isinstance(epoch, int) or isinstance(epoch, bool) \
            or not isinstance(expected_ckpt, str) or len(expected_ckpt) != 64:
        raise SystemExit(
            f"{record_path}: record carries no terminal epoch and checkpoint "
            f"digest; refusing to guess an operating point")
    if os.path.abspath(str(record.get("run_dir", ""))) \
            != os.path.abspath(a.result_dir):
        raise SystemExit(
            f"record names run_dir {record.get('run_dir')!r}, which is not "
            f"{a.result_dir!r}")

    checkpoint = os.path.join(a.result_dir, "model_state_dict.pth")
    actual_ckpt = _sha256_file(checkpoint)
    if actual_ckpt != expected_ckpt:
        raise SystemExit(
            f"{checkpoint}: sha256 {actual_ckpt} is not the sealed terminal "
            f"checkpoint {expected_ckpt}")

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    if a.disable_adaptive_topp:
        args.routing_adaptive_topp = False
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # `strict=False` silently tolerated a partially loaded model, which would
    # be measured as if it were the trained one.
    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise SystemExit(
            f"{checkpoint}: state dict does not match the model: "
            f"{len(missing)} missing {missing[:5]}, "
            f"{len(unexpected)} unexpected {unexpected[:5]}")
    model.set_current_epoch(epoch)
    effective_epsilon = model._current_sinkhorn_epsilon()

    want_train = (a.split == "train")
    tr, te, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=want_train, load_database=False, load_test=not want_train,
        return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    ds = tr if want_train else te

    g = np.random.default_rng(a.sample_seed)
    order = g.permutation(len(ds))[:min(a.n, len(ds))].tolist()

    ntok, mass, codons = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            samples = [ds[i] for i in order[i0:i0 + 32]]
            batch = {}
            for k in samples[0]:
                vals = [s_[k] for s_ in samples]
                if torch.is_tensor(vals[0]):
                    batch[k] = torch.stack(vals)
                elif isinstance(vals[0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(vals))
                else:
                    batch[k] = vals
            o = _forward_text_routed(model, batch, a.device)
            P = o["routing_matrix"].float()
            ntok.append((P > 0).sum(dim=1).cpu().numpy())      # [B, M]
            mass.append(P.sum(dim=1).cpu().numpy())            # [B, M]
            # codon symbol per (image, slot), to test whether the images an
            # empty slot cannot see all collapse onto ONE constant filler codon
            for k in ("base_indices", "dna_base_indices", "bases", "codon_indices"):
                if k in o and o[k] is not None:
                    bi = o[k]
                    bi = bi.argmax(-1) if bi.dim() == 3 else bi
                    codons.append(bi.long().cpu().numpy()); break
    ntok = np.concatenate(ntok); mass = np.concatenate(mass)
    codons = np.concatenate(codons) if codons else None
    B, M = ntok.shape
    N_patch = None

    print(f"{args.dataset} [{a.split}] images={B} slots={M} "
          f"topp={'OFF' if a.disable_adaptive_topp else 'ON'} epoch={a.epoch}")
    print(f"  {'slot':20s}{'empty imgs %':>14s}{'n_tok med':>11s}"
          f"{'n_tok p10':>11s}{'mass med':>11s}{'mass<1e-6 %':>13s}")
    rows = {}
    for m in range(M):
        empty = float((ntok[:, m] == 0).mean() * 100)
        tiny = float((mass[:, m] < 1e-6).mean() * 100)
        name = SLOTS[m] if m < len(SLOTS) else str(m)
        print(f"  {name:20s}{empty:13.2f}%{np.median(ntok[:, m]):11.1f}"
              f"{np.percentile(ntok[:, m], 10):11.1f}"
              f"{np.median(mass[:, m]):11.5f}{tiny:12.2f}%")
        rows[name] = {"empty_image_pct": round(empty, 4),
                      "mass_below_1e-6_pct": round(tiny, 4),
                      "n_tokens_median": float(np.median(ntok[:, m])),
                      "n_tokens_p10": float(np.percentile(ntok[:, m], 10)),
                      "n_tokens_min": int(ntok[:, m].min()),
                      "mass_median": float(np.median(mass[:, m])),
                      "mass_min": float(mass[:, m].min())}
    print(f"  (patches per image = {int(ntok.sum(axis=1).max())} at most across slots)")

    if codons is not None and codons.shape[1] >= 3 * M:
        print(f"\n  constant-filler check: among images a slot CANNOT see, do the "
              f"codons collapse to one symbol?")
        print(f"  {'slot':20s}{'empty n':>9s}{'codons(empty)':>15s}{'top1%':>8s}"
              f"{'codons(seen)':>14s}{'top1%':>8s}")
        for m in range(M):
            e = ntok[:, m] == 0
            if e.sum() < 5:
                continue
            c = codons[:, 3*m]*16 + codons[:, 3*m+1]*4 + codons[:, 3*m+2]
            ue, ce = np.unique(c[e], return_counts=True)
            us, cs = np.unique(c[~e], return_counts=True)
            name = SLOTS[m] if m < len(SLOTS) else str(m)
            print(f"  {name:20s}{int(e.sum()):9d}{len(ue):15d}"
                  f"{ce.max()/ce.sum()*100:7.1f}%{len(us):14d}"
                  f"{cs.max()/cs.sum()*100:7.1f}%")
            rows[name]["codons_when_empty"] = int(len(ue))
            rows[name]["top1_codon_when_empty_pct"] = round(float(ce.max()/ce.sum()*100), 2)
            rows[name]["codons_when_seen"] = int(len(us))

    out = {"dataset": args.dataset, "dir": a.result_dir, "split": a.split,
           "epoch": epoch, "images": int(B),
           "adaptive_topp_disabled": bool(a.disable_adaptive_topp),
           # `adaptive_topp_disabled` above is this TOOL's flag, not the cell's
           # setting -- it was misread as the latter once.  The cell's own
           # effective value is stated separately, composed the way
           # model_siglip2.py:2300-2301 composes it.
           "cell_adaptive_topp_effective": bool(
               getattr(args, "routing_adaptive_topp", False)) and not bool(
               getattr(args, "no_routing_adaptive_topp", False)),
           "cell_adaptive_topp_min": getattr(
               args, "routing_adaptive_topp_min", None),
           "cell_adaptive_topp_max": getattr(
               args, "routing_adaptive_topp_max", None),
           # The epoch is only a label; epsilon is what the routing actually
           # ran at.  Recording it makes "measured at the terminal epoch" a
           # property of the model state rather than of a command line.
           "effective_sinkhorn_epsilon": (
               None if effective_epsilon is None else float(effective_epsilon)),
           "provenance": {
               "record_path": record_path,
               "record_sha256": __import__("hashlib").sha256(
                   record_bytes).hexdigest(),
               "checkpoint_path": checkpoint,
               "checkpoint_sha256": actual_ckpt,
               "source_sha256": {
                   rel: _sha256_file(os.path.join(_REPO, rel))
                   for rel in _PROVENANCE_SOURCES
                   if os.path.exists(os.path.join(_REPO, rel))},
               "sample_seed": int(a.sample_seed),
               "sample_indices": [int(i) for i in order],
           },
           "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
