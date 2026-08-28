"""The whole 24-cell A-series campaign, as one frozen plan (F09).

WHY A PLAN AND NOT SHELL. The chain used to compute each cell in bash: grep the
composed flags for `--codebook_size`, splice it back with `${CELL_K:+K=...}`,
read the cell list through a process substitution. Every one of those was
broken, and none of it was visible from the outside:

  * `CELL_K=$(printf ... | grep ...)` returns 1 when the cell has no
    `--codebook_size`, and under `set -Eeuo pipefail` the script dies at the
    assignment -- before the `|| CELL_K=""` fallback on the next line. A2 and
    A5 could never start.
  * `${CELL_K:+K="$CELL_K"} cmd` is not an assignment: bash expands it and then
    looks for a command called `K=320`. A4 died with rc 127. This is the same
    expansion mistake as `${FOIL:+FOIL_JSONL=...}` in the overlay launcher.
  * `mapfile -t CELLS < <(producer) || ...` checks mapfile's status, not the
    producer's, so a producer that printed six names and then failed was
    accepted.

So the plan is computed once, in Python, where a missing key is a KeyError
rather than a silent empty string; it is written out; and the shell iterates it
without deciding anything. That also makes the campaign inspectable before a
GPU is touched, and testable by reading the plan instead of the trainer's logs.

The plan pins, per cell: the exact child argv, the exact environment (an
allow-list -- an inherited `NUM_CODONS=4` turns the paper's 15 bases into 20,
and `CURVE=1` drops FINAL_EPOCH and switches to the test-monitored path), the
cache paths with their provenance state, and the tag.

Usage:
    python scripts/ablation_campaign_plan.py --out artifacts/ablation_plan.json
    python scripts/ablation_campaign_plan.py --print
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

from dna_utils.ablation_spec import _BUILDERS, preflight_ablation  # noqa: E402

CACHE_ROOT = os.environ.get(
    "GDNA_CACHE_ROOT", "/data/yschoi/groundeddna_cache_v6prov")

#: The datasets, their canonical names and the per-slot codebook size.
DATASETS = {
    "cifar_A_v4": dict(canon="CIFAR10", slug="cifar10", k=64,
                       cache="cifar10_clip_tokens", cibnt="1.0",
                       qwen="./cache/cifar10_qwen_v4.jsonl"),
    "flickr_A_v4": dict(canon="Flickr25k", slug="flickr25k", k=128,
                        cache="flickr25k_clip_tokens", cibnt="1.0",
                        qwen="./cache/flickr25k_qwen3_v4_trainset.jsonl"),
    "nuswide_A_v4": dict(canon="NUSWIDE", slug="nuswide", k=128,
                         cache="nuswide_clip_tokens", cibnt="1.5",
                         qwen="./cache/nuswide_qwen3_v4_trainset.jsonl"),
    "mscoco_A_v5b": dict(canon="MSCOCO", slug="mscoco", k=128,
                         cache="mscoco_clip_tokens", cibnt="1.5",
                         qwen="./cache/mscoco_qwen3_v5b_trainset.jsonl"),
}

ORDER = ("A2", "A4", "A5")
SLOTS, BASES_PER_SLOT = 5, 3

#: The champion recipe every cell starts from.
BASE_FLAGS = [
    "--num_semantic_parts", str(SLOTS), "--num_codebooks", str(SLOTS),
    "--no_gumbel_softmax", "--lambda_codeword_codon_sinkhorn", "0.0",
    "--routing_adaptive_topp_min", "0.6", "--routing_adaptive_topp_max", "0.95",
]

#: Axes a cell owns whether or not it names them. A5_none and A5_joint mean
#: "no noGumbel", which they express by leaving the flag out -- so the base
#: must not put it back, or the factorial collapses to two configurations.
_A5_AXES = ("--no_gumbel_softmax", "--lambda_codon_joint")
_TAKES_VALUE = {
    "--lambda_codon_joint", "--codebook_size", "--selection_mode",
    "--lambda_text_code_kl", "--lambda_text_hash_ntxent",
    "--lambda_xmodal_commit", "--num_codebooks", "--num_semantic_parts",
    "--routing_adaptive_topp_min", "--routing_adaptive_topp_max",
    "--lambda_codeword_codon_sinkhorn",
}

#: What the child may inherit. The fixedN wrapper reads a dozen variables from
#: the environment, and several change the experiment rather than the logging:
#: NUM_CODONS changes the code length, CURVE removes FINAL_EPOCH, K changes the
#: capacity, WHITEN_VARIANT changes the transform.
ENV_PASSTHROUGH = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "PYTHONPATH", "PYTHONUNBUFFERED", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "LD_LIBRARY_PATH", "HF_HOME", "GDNA_CACHE_ROOT", "PY",
})

#: Set explicitly on every child, so an inherited value cannot decide it.
_PINNED = ("GDNA_NUM_SEMANTIC_PARTS", "NUM_CODONS", "K", "CIBNT", "VIZ",
           "CURVE", "WHITEN_VARIANT", "A_SKIPS", "LBU", "FIXED_N", "EVERY",
           "TAG_SUFFIX", "AUX_ARGS", "CACHE_OVERRIDE", "HF_HUB_OFFLINE",
           "TRANSFORMERS_OFFLINE")


class PlanRefused(RuntimeError):
    """The campaign cannot be described, so it must not be started."""


def _owned_by(cell) -> set:
    owned = {f for f in cell.flags if f.startswith("--")}
    if cell.name.startswith("A5_"):
        owned.update(_A5_AXES)
    return owned


def _strip_owned(base: list, owned: set) -> list:
    out, i = [], 0
    while i < len(base):
        token = base[i]
        if token in owned:
            i += 2 if token in _TAKES_VALUE else 1
            continue
        out.append(token)
        i += 1
    return out


def _value(flags: list, option: str, default=None):
    return flags[flags.index(option) + 1] if option in flags else default


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_n() -> dict:
    """The N Phase 3 chose, per dataset. Not the pre-audit table."""
    path = REPO / "artifacts" / "phase3_selection" / "selected_n.json"
    if not path.is_file():
        raise PlanRefused(
            f"{path} does not exist; the ablations run at the N the train-only "
            f"selection chose, so that has to exist first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chosen = payload.get("selected") or {}
    out = {}
    for exp, spec in DATASETS.items():
        entry = chosen.get(spec["slug"])
        if not entry or "selected_N" not in entry:
            raise PlanRefused(f"{path} has no chosen N for {spec['slug']}")
        out[exp] = int(entry["selected_N"])
    return out, _sha(path)


def cache_state(spec: dict) -> dict:
    """Where the features come from, and whether they can be accounted for."""
    from dna_utils.cache_provenance import (
        CacheProvenanceMissing, require_cache_provenance)

    cache = Path(CACHE_ROOT) / spec["cache"]
    meta = cache / "meta.json"
    record = {"cache": str(cache), "foils": f"{cache}_foils",
              "qwen": spec["qwen"], "provenanced": False, "refusal": None}
    if not meta.is_file():
        record["refusal"] = f"{meta} does not exist"
        return record
    try:
        require_cache_provenance(str(cache),
                                 json.loads(meta.read_text(encoding="utf-8")))
        record["provenanced"] = True
    except CacheProvenanceMissing as error:
        record["refusal"] = str(error)
    return record


def build_plan() -> dict:
    chosen, selection_sha = selected_n()
    cells = []
    for exp, spec in DATASETS.items():
        cache = cache_state(spec)
        n = chosen[exp]
        for name in ORDER:
            for cell in _BUILDERS[name](spec["canon"]):
                flags = _strip_owned(BASE_FLAGS, _owned_by(cell)) \
                    + list(cell.flags)
                if "--lambda_codon_joint" not in flags:
                    flags += ["--lambda_codon_joint", _joint_for(exp)]
                flags += ["-e", str(n + 1)]
                preflight_ablation(flags, dataset=spec["canon"])
                # The codebook size is read from the composed flags HERE, where
                # a missing key is a default and not an error that kills the
                # run, and it is written into the child's environment as data.
                k = _value(flags, "--codebook_size", str(spec["k"]))
                env = {
                    "GDNA_NUM_SEMANTIC_PARTS": str(SLOTS),
                    "NUM_CODONS": str(BASES_PER_SLOT),
                    "K": str(k),
                    "CIBNT": spec["cibnt"],
                    "VIZ": "0",
                    "CURVE": "",
                    "WHITEN_VARIANT": "_localOnly",
                    "A_SKIPS": ("--xmodal_commit_skip_global "
                                "--cibhash_dynamic_tau_skip_global"),
                    "LBU": "0.02",
                    "FIXED_N": str(n),
                    "EVERY": "1",
                    "TAG_SUFFIX": f"_{cell.name}",
                    "AUX_ARGS": " ".join(shlex.quote(f) for f in flags),
                    "CACHE_OVERRIDE": cache["cache"],
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
                cells.append({
                    "exp": exp, "dataset": spec["canon"], "cell": cell.name,
                    "ablation": name, "N": n,
                    "tag": f"promptAblA_{exp}_{cell.name}",
                    "flags": flags,
                    "env": env,
                    "cache": cache,
                    "runner": "scripts/prompt_ablation_A_cell_fixedN.sh",
                })
    return {
        "schema_version": 1,
        "what_this_is": (
            "Every A-series cell, frozen: the exact child argv, the exact "
            "environment, and the caches. The shell iterates this and decides "
            "nothing."),
        "expected_cells": len(DATASETS) * 6,
        "selected_n": chosen,
        "selection_sha256": selection_sha,
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        "env_pinned": list(_PINNED),
        "cells": cells,
    }


def _joint_for(exp: str) -> str:
    table = json.loads(
        (REPO / "docs" / "newmodel_analysis" / "fixed_N.json").read_text())
    return str(table[exp]["lambda_codon_joint"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--print", action="store_true", dest="show")
    parser.add_argument("--require-provenance", action="store_true",
                        help=("refuse to write a plan whose caches cannot be "
                              "accounted for"))
    args = parser.parse_args()

    try:
        plan = build_plan()
    except PlanRefused as error:
        print(f"[ablation-plan] REFUSED: {error}", file=sys.stderr)
        return 1

    if len(plan["cells"]) != plan["expected_cells"]:
        print(f"[ablation-plan] REFUSED: {len(plan['cells'])} cells, expected "
              f"{plan['expected_cells']}", file=sys.stderr)
        return 1

    unprovenanced = sorted({c["dataset"] for c in plan["cells"]
                            if not c["cache"]["provenanced"]})
    if unprovenanced:
        blocked = sum(1 for c in plan["cells"]
                      if not c["cache"]["provenanced"])
        print(f"[ablation-plan] {blocked} of {len(plan['cells'])} cells use a "
              f"cache that cannot be accounted for: {unprovenanced}",
              file=sys.stderr)
        if args.require_provenance:
            return 1

    if args.show:
        for cell in plan["cells"]:
            print(f"  {cell['cell']:<20} {cell['dataset']:<10} N={cell['N']:<3} "
                  f"K={cell['env']['K']:<4} "
                  f"prov={'ok' if cell['cache']['provenanced'] else 'REFUSED'}")
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
