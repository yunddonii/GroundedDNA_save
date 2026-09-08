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

#: The A recipe fits the text transform on the local slots only. It is a
#: constant here rather than an environment read, because it names the files the
#: plan has to check for leakage-free provenance.
WHITEN_VARIANT = "_localOnly"

#: The only program a cell may start. A forged plan naming a different runner
#: executed it, with the plan's environment, at rc 0.
RUNNERS = frozenset({"scripts/prompt_ablation_A_cell_fixedN.sh"})

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
    "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "PYTHONUNBUFFERED", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "HF_HOME", "GDNA_CACHE_ROOT", "PY",
})

#: Set explicitly on every child, so an inherited value cannot decide it.
_PINNED = ("GDNA_NUM_SEMANTIC_PARTS", "NUM_CODONS", "K", "CIBNT", "VIZ",
           "CURVE", "WHITEN_VARIANT", "A_SKIPS", "LBU", "FIXED_N", "EVERY",
           "TAG_SUFFIX", "AUX_ARGS", "CACHE_OVERRIDE", "WDIR_OVERRIDE",
           "QWEN_OVERRIDE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


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


#: What a selection artefact has to prove before 24 GPU cells are pinned to it.
#: The first version read four `selected_N` fields and recorded the file's SHA,
#: which is a receipt for whatever it was handed: a forged file carrying
#: `schema_version: 999`, an empty record map and the values -7, 999, true and
#: 123 was accepted, and the campaign would have run at N=-7.
_SELECTION_SCHEMA = 1
_SELECTION_METRIC = "eval_mAP_at_R"
_SELECTION_DISTANCE = "base_hamming"
_CANDIDATE_N = (4, 9, 19, 39)


def _load_selection() -> tuple:
    """The N Phase 3 chose, per dataset -- authenticated, not merely present."""
    path = REPO / "artifacts" / "phase3_selection" / "selected_n.json"
    if not path.is_file():
        raise PlanRefused(
            f"{path} does not exist; the ablations run at the N the train-only "
            f"selection chose, so that has to exist first")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise PlanRefused(f"{path}: {error}") from None
    if not isinstance(payload, dict):
        raise PlanRefused(f"{path} is not an object")

    if payload.get("schema_version") != _SELECTION_SCHEMA:
        raise PlanRefused(
            f"{path}: schema_version {payload.get('schema_version')!r}, not "
            f"{_SELECTION_SCHEMA}")
    for key, want in (("selection_metric", _SELECTION_METRIC),
                      ("selection_distance", _SELECTION_DISTANCE)):
        if payload.get(key) != want:
            raise PlanRefused(
                f"{path}: {key} is {payload.get(key)!r}, not {want!r}; this N "
                f"was not chosen by the protocol the ablations assume")
    if tuple(payload.get("candidate_n") or ()) != _CANDIDATE_N:
        raise PlanRefused(
            f"{path}: candidate_n is {payload.get('candidate_n')!r}, not "
            f"{list(_CANDIDATE_N)}")

    # The aggregator that wrote it has to be the one in this tree, and it has
    # to name the sixteen records it reduced. Sixteen is not a style choice:
    # a partial matrix picks N from whichever cells happened to finish.
    aggregator = REPO / "scripts" / "phase3_select_n.py"
    if payload.get("aggregator_sha256") != _sha(aggregator):
        raise PlanRefused(
            f"{path} was written by a different {aggregator.name} than the one "
            f"in this tree; re-run the aggregator or check out the commit that "
            f"produced the file")
    records = payload.get("record_sha256")
    if not isinstance(records, dict) or len(records) != 16:
        raise PlanRefused(
            f"{path}: record_sha256 names "
            f"{len(records) if isinstance(records, dict) else 'no'} records, "
            f"not the sixteen cells of the matrix")
    bad = sorted(k for k, v in records.items()
                 if not isinstance(v, str) or len(v) != 64)
    if bad:
        raise PlanRefused(f"{path}: record_sha256 entries are not digests: {bad}")

    # The protocol the records were produced under must still be the protocol
    # in this tree, and the file must say so about itself.
    from scripts.phase3_selection_matrix import protocol_digests
    current = protocol_digests()
    for key in ("protocol_sources", "current_protocol_sources"):
        if payload.get(key) != current:
            differing = sorted(
                k for k in set(current) | set(payload.get(key) or {})
                if (payload.get(key) or {}).get(k) != current.get(k))
            raise PlanRefused(
                f"{path}: {key} differs from this tree: {differing}. The N was "
                f"chosen under a different protocol than the one the ablations "
                f"would run.")

    chosen = payload.get("selected")
    if not isinstance(chosen, dict):
        raise PlanRefused(f"{path}: no `selected` map")
    wanted = {spec["slug"] for spec in DATASETS.values()}
    if set(chosen) != wanted:
        raise PlanRefused(
            f"{path}: `selected` covers {sorted(chosen)}, not exactly "
            f"{sorted(wanted)}")

    out = {}
    for exp, spec in DATASETS.items():
        entry = chosen[spec["slug"]]
        if not isinstance(entry, dict):
            raise PlanRefused(f"{path}: {spec['slug']} is not an object")
        n = entry.get("selected_N")
        # `isinstance(True, int)` is True, and `int("999")` is happy to make a
        # number out of anything -- so the type is checked before the value.
        if isinstance(n, bool) or not isinstance(n, int) or n not in _CANDIDATE_N:
            raise PlanRefused(
                f"{path}: {spec['slug']} selected_N is {n!r}, not one of "
                f"{list(_CANDIDATE_N)}")
        value = entry.get("selection_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not 0.0 <= float(value) <= 1.0:
            raise PlanRefused(
                f"{path}: {spec['slug']} selection_value is {value!r}, not a "
                f"proportion")
        out[exp] = n
    return out, _sha(path)


def cache_state(spec: dict) -> dict:
    """Every input a cell consumes, and whether it can be accounted for.

    The first version gated the feature cache's `meta.json` and then recorded a
    GUESSED `${cache}_foils` and a relative Qwen path that nothing checked. That
    left the transform out of the plan entirely: the wrapper's whitening
    directory was hardcoded to `./cache/...`, and the Flickr and MS-COCO
    trainOnly matrices differ in BYTES between the legacy root and the rebuilt
    one, so twelve of the twenty-four cells would have trained on a transform
    the plan never named.

    It also recorded `provenanced: true` for a cache the gate had only warned
    about: with `GDNA_ALLOW_UNPROVENANCED_CACHE` set, `require_cache_provenance`
    prints and returns instead of raising, and the returned record's own
    `unprovenanced_opt_out` field was thrown away. A plan is a paper artefact;
    the opt-out is refused here rather than merely announced.
    """
    from dna_utils.cache_provenance import (
        OPT_OUT_ENV, CacheProvenanceMissing, opted_out,
        require_cache_provenance, require_leakage_free_whitening)

    cache = Path(CACHE_ROOT) / spec["cache"]
    foils = Path(f"{cache}_foils")
    qwen = REPO / spec["qwen"]
    variant = WHITEN_VARIANT
    whiten = {
        "trainOnly": foils / f"text_whiten_trainOnly{variant}.npz",
        "optTrain": foils / f"text_whiten_optTrain{variant}.npz",
    }
    record = {
        "cache": str(cache), "foils": str(foils), "qwen": str(qwen),
        "whiten_trainOnly": str(whiten["trainOnly"]),
        "whiten_optTrain": str(whiten["optTrain"]),
        "provenanced": False, "refusal": None, "digests": {},
    }

    if opted_out():
        # Not a warning. A plan written under the opt-out would carry
        # `provenanced: true` for caches the gate refused.
        record["refusal"] = (
            f"{OPT_OUT_ENV} is set in the planning process; a campaign plan is "
            f"a paper artefact and cannot be written under the opt-out")
        return record

    problems = []
    meta = cache / "meta.json"
    if not meta.is_file():
        problems.append(f"{meta} does not exist")
    else:
        try:
            record["provenance"] = require_cache_provenance(
                str(cache), json.loads(meta.read_text(encoding="utf-8")))
            record["digests"]["cache_meta"] = _sha(meta)
        except (CacheProvenanceMissing, ValueError) as error:
            problems.append(str(error))

    # The foils directory is a real input -- the text-foil arrays are what the
    # cross-modal terms contrast against -- so it is checked, not guessed.
    foil_meta = foils / "meta.json"
    if not foils.is_dir():
        problems.append(f"{foils} does not exist")
    elif not foil_meta.is_file():
        problems.append(f"{foil_meta} does not exist")
    else:
        try:
            require_cache_provenance(
                str(foils), json.loads(foil_meta.read_text(encoding="utf-8")))
            record["digests"]["foils_meta"] = _sha(foil_meta)
        except (CacheProvenanceMissing, ValueError) as error:
            problems.append(str(error))

    for which, path in whiten.items():
        if not path.is_file():
            problems.append(f"{path} does not exist")
            continue
        try:
            require_leakage_free_whitening(str(path))
        except CacheProvenanceMissing as error:
            problems.append(str(error))
            continue
        record["digests"][f"whiten_{which}"] = _sha(path)
        record["digests"][f"whiten_{which}_meta"] = _sha(
            Path(str(path) + ".meta.json"))

    if not qwen.is_file():
        problems.append(f"{qwen} does not exist")
    else:
        record["digests"]["qwen"] = _sha(qwen)

    if problems:
        record["refusal"] = "; ".join(problems)
    else:
        record["provenanced"] = True
    return record


def build_plan() -> dict:
    chosen, selection_sha = _load_selection()
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
                    "WHITEN_VARIANT": WHITEN_VARIANT,
                    "A_SKIPS": ("--xmodal_commit_skip_global "
                                "--cibhash_dynamic_tau_skip_global"),
                    "LBU": "0.02",
                    "FIXED_N": str(n),
                    "EVERY": "1",
                    "TAG_SUFFIX": f"_{cell.name}",
                    "AUX_ARGS": " ".join(shlex.quote(f) for f in flags),
                    "CACHE_OVERRIDE": cache["cache"],
                    # Pinned for the same reason CACHE is: the wrapper's
                    # defaults are the legacy root, whose Flickr and MS-COCO
                    # whitening matrices differ in bytes from the rebuilt ones.
                    "WDIR_OVERRIDE": cache["foils"],
                    "QWEN_OVERRIDE": cache["qwen"],
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
    tags = [c["tag"] for c in cells]
    if len(set(tags)) != len(tags):
        duplicated = sorted({t for t in tags if tags.count(t) > 1})
        raise PlanRefused(
            f"two cells share a tag {duplicated}; they would share a result "
            f"directory and the second would be read as the first")

    plan = {
        "schema_version": 2,
        "what_this_is": (
            "Every A-series cell, frozen: the exact child argv, the exact "
            "environment, and the caches. The shell iterates this and decides "
            "nothing."),
        "expected_cells": len(DATASETS) * 6,
        "selected_n": chosen,
        "selection_sha256": selection_sha,
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        # The VALUES, not just the names. The executor used to fill these from
        # its own live environment, so a plan reviewed on one PATH could run on
        # another -- including a PATH whose `python` is a different interpreter.
        "env_passthrough_values": {
            k: os.environ[k] for k in sorted(ENV_PASSTHROUGH)
            if k in os.environ},
        "env_pinned": list(_PINNED),
        "runners": sorted(RUNNERS),
        "cells": cells,
    }
    # ``PY`` is commonly a non-exported shell variable (including in the
    # repository's chain), so reading only os.environ omitted it from otherwise
    # valid plans.  Production refuses to guess an interpreter: record the
    # exact interpreter that built the plan every time.
    plan["env_passthrough_values"]["PY"] = str(
        Path(sys.executable).resolve())
    # A digest OF the plan, IN the plan: the executor is handed the digest the
    # chain computed at build time on its command line and re-derives this one,
    # so a plan edited between `--out` and the twenty-fourth cell is refused
    # rather than executed. Self-reference is why it is added last.
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


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
