"""The confidence-adaptive top-p retune, as one frozen plan (protocol stage 1).

WHY THIS EXISTS. The four trainers hardcode `--routing_adaptive_topp_min 0.3
--routing_adaptive_topp_max 0.7` in their command bodies, and Phase 3 passed no
override, so all sixteen selection cells chose N at those values. The draft
(§4.10b) says that pair is the M=6 tuning: at M=5 the cut is pinned to k=3/5 in
99.7% of patches, which starves CIFAR's `color_texture` slot in 23.44% of
images -- a slot whose pooled feature is then the zero vector, so three of the
fifteen bases stop depending on the image at all. CIFAR was retuned to 0.6/0.95
and the qualitative figures were regenerated there; the draft records the other
three datasets as "재튜닝 미완".

So N was chosen for a model the paper does not report. This plan re-tunes top-p
on the three outstanding datasets before N is chosen again.

SELECTION CRITERION, and why it differs from CIFAR's. On CIFAR the codon
decoding probe is INVALID (§4.7c): with exactly one positive per image, label
ranking AP collapses to reciprocal rank, every slot competes for the same
target, and the saturated global gate wins -- so decoding rises monotonically
with the starvation rate (.9033 < .9053 < .9158 < .9175 as empty images go
0% -> 43.36%). It rewards the defect, which is why the empty-image rate had to
decide there. Flickr25k, NUS-WIDE and MS-COCO are multi-label (3.74 / 3.33 /
2.92 positives per image) and show ~0% starvation already, so the defect
criterion has no discriminating power and decoding is not invalidated. Those
three are therefore chosen on mAP@R and held-out codon decoding.

WHAT IS HELD FIXED. Everything except top-p, including the incumbent per-dataset
`--lambda_codon_joint`: this is coordinate descent from the current operating
point, and lambda is re-swept afterwards at the winning top-p. That ordering is
not cosmetic -- adding `noGumbel` already moved MS-COCO's lambda optimum from
0.05 to 0.03, so the axes interact and sweeping lambda first would be
invalidated by this stage.

The plan is emitted in the same schema as the ablation campaign, so
scripts/_ablation_exec.py and scripts/run_ablation_campaign.sh run it unchanged.

Usage:
    python scripts/topp_retune_plan.py --print
    python scripts/topp_retune_plan.py --out artifacts/topp_retune_plan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.ablation_spec import preflight_ablation  # noqa: E402
from scripts.ablation_campaign_plan import (  # noqa: E402
    CACHE_ROOT, ENV_PASSTHROUGH, RUNNERS, SLOTS, BASES_PER_SLOT,
    WHITEN_VARIANT, _PINNED, PlanRefused, _load_selection, _sha, cache_state)

#: CIFAR is absent on purpose: it was retuned to 0.6/0.95 on the empty-image
#: criterion (draft §4.10b) and the §4.10c figures were regenerated there.
DATASETS = {
    "flickr_A_v4": dict(canon="Flickr25k", slug="flickr25k", k=128,
                        cache="flickr25k_clip_tokens", cibnt="1.0",
                        qwen="./cache/flickr25k_qwen3_v4_trainset.jsonl",
                        joint="0.02"),
    "nuswide_A_v4": dict(canon="NUSWIDE", slug="nuswide", k=128,
                         cache="nuswide_clip_tokens", cibnt="1.5",
                         qwen="./cache/nuswide_qwen3_v4_trainset.jsonl",
                         joint="0.05"),
    "mscoco_A_v5b": dict(canon="MSCOCO", slug="mscoco", k=128,
                         cache="mscoco_clip_tokens", cibnt="1.5",
                         qwen="./cache/mscoco_qwen3_v5b_trainset.jsonl",
                         joint="0.03"),
}

#: The grid. `.3-.7` is the incumbent and is re-run rather than reused, because
#: the existing runs are the Phase-3 SELECTION cells: different tag, different
#: `selection_mode`, and comparing a new cell against an old one across that
#: boundary is the kind of handoff this project keeps getting wrong.
TOPP_GRID = (("0.3", "0.7"), ("0.4", "0.8"), ("0.6", "0.95"))

#: Held fixed across the grid. Top-p is deliberately absent -- it is the axis.
BASE_FLAGS = [
    "--num_semantic_parts", str(SLOTS), "--num_codebooks", str(SLOTS),
    "--no_gumbel_softmax", "--lambda_codeword_codon_sinkhorn", "0.0",
]

STAGE = "topp_retune"


def _tag_fragment(lo: str, hi: str) -> str:
    return f"topp{lo.replace('.', '')}{hi.replace('.', '')}"


def build_plan(*, smoke: bool = False) -> dict:
    """`smoke=True` is one epoch on Flickr only, and says so in every artefact.

    The marker is not decoration. `run_identity._EXCLUDED_FRAGMENTS` refuses to
    resolve a directory whose name contains "smoke", the stage string carries
    it, and the reducer refuses a record that is not a candidate cell -- because
    the last time a smoke run was mistaken for the real thing it was an 18-base
    run committed as "Phase 3 smoke passes".
    """
    chosen, selection_sha = _load_selection()
    stage = f"{STAGE}_SMOKE" if smoke else STAGE
    cells = []
    datasets = ({"flickr_A_v4": DATASETS["flickr_A_v4"]} if smoke
                else DATASETS)
    for exp, spec in datasets.items():
        cache = cache_state(spec)
        n = 0 if smoke else chosen[exp]
        for lo, hi in TOPP_GRID:
            frag = _tag_fragment(lo, hi)
            if smoke:
                frag = f"smoke_{frag}"
            flags = list(BASE_FLAGS) + [
                "--routing_adaptive_topp",
                "--routing_adaptive_topp_min", lo,
                "--routing_adaptive_topp_max", hi,
                "--lambda_codon_joint", spec["joint"],
                # Recorded in the run identity, so a retune cell cannot resolve
                # as -- or share a directory with -- a Phase-3 selection cell.
                "--selection_mode", f"{stage}_{frag}",
                "-e", str(n + 1),
            ]
            preflight_ablation(flags, dataset=spec["canon"])
            env = {
                "GDNA_NUM_SEMANTIC_PARTS": str(SLOTS),
                "NUM_CODONS": str(BASES_PER_SLOT),
                "K": str(spec["k"]),
                "CIBNT": spec["cibnt"],
                "VIZ": "0",
                "CURVE": "",
                "WHITEN_VARIANT": WHITEN_VARIANT,
                "A_SKIPS": ("--xmodal_commit_skip_global "
                            "--cibhash_dynamic_tau_skip_global"),
                "LBU": "0.02",
                "FIXED_N": str(n),
                "EVERY": "1",
                "TAG_SUFFIX": f"_{frag}",
                "AUX_ARGS": " ".join(shlex.quote(f) for f in flags),
                "CACHE_OVERRIDE": cache["cache"],
                "WDIR_OVERRIDE": cache["foils"],
                "QWEN_OVERRIDE": cache["qwen"],
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            cells.append({
                "exp": exp, "dataset": spec["canon"], "cell": frag,
                "stage": stage, "is_candidate_cell": not smoke, "N": n,
                "topp_min": lo, "topp_max": hi,
                "lambda_codon_joint": spec["joint"],
                "tag": f"promptAblA_{exp}_{frag}",
                "flags": flags, "env": env, "cache": cache,
                "runner": "scripts/prompt_ablation_A_cell_fixedN.sh",
            })

    tags = [c["tag"] for c in cells]
    if len(set(tags)) != len(tags):
        raise PlanRefused(
            f"two cells share a tag {sorted({t for t in tags if tags.count(t) > 1})}; "
            f"they would share a result directory")

    plan = {
        "schema_version": 2,
        "what_this_is": (
            "The confidence-adaptive top-p retune on the three datasets the "
            "draft records as outstanding. CIFAR is absent: it was retuned to "
            "0.6/0.95 on the empty-image criterion. Chosen on mAP@R and "
            "held-out codon decoding, which are valid here and are not on "
            "single-label CIFAR (draft §4.7c)."),
        "stage": stage,
        "is_candidate_matrix": not smoke,
        "expected_cells": len(datasets) * len(TOPP_GRID),
        "topp_grid": [list(pair) for pair in TOPP_GRID],
        "selected_n": chosen,
        "selection_sha256": selection_sha,
        "selection_caveat": (
            "These N were chosen at top-p 0.3/0.7, the very setting under test. "
            "They are the tuning HORIZON here, not a result: N is re-selected "
            "after this stage and after the lambda sweep."),
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        "env_passthrough_values": {
            k: os.environ[k] for k in sorted(ENV_PASSTHROUGH) if k in os.environ},
        "env_pinned": list(_PINNED),
        "runners": sorted(RUNNERS),
        "cells": cells,
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--print", action="store_true", dest="show")
    parser.add_argument("--smoke", action="store_true",
                        help="one epoch on Flickr only; marked in every field")
    args = parser.parse_args()

    try:
        plan = build_plan(smoke=args.smoke)
    except PlanRefused as error:
        print(f"[topp-plan] REFUSED: {error}", file=sys.stderr)
        return 1

    if len(plan["cells"]) != plan["expected_cells"]:
        print(f"[topp-plan] REFUSED: {len(plan['cells'])} cells, expected "
              f"{plan['expected_cells']}", file=sys.stderr)
        return 1

    # Provenance is required, not optional. The campaign planner made this a
    # flag and the documented command omitted it, so a missing cache root wrote
    # 24 unprovenanced cells at rc 0.
    unaccounted = [c["tag"] for c in plan["cells"]
                   if not c["cache"]["provenanced"]]
    if unaccounted:
        print(f"[topp-plan] REFUSED: {len(unaccounted)} of "
              f"{len(plan['cells'])} cells use inputs that cannot be accounted "
              f"for: {plan['cells'][0]['cache']['refusal']}", file=sys.stderr)
        return 1

    if args.show:
        for cell in plan["cells"]:
            print(f"  {cell['dataset']:<10} topp {cell['topp_min']:>4}-"
                  f"{cell['topp_max']:<5} N={cell['N']:<3} "
                  f"jd={cell['lambda_codon_joint']:<5} tag={cell['tag']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out)
        print(f"wrote {out}: {len(plan['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
