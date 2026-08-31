"""Phase 3 stage 1: the exact 16-cell train-only (N) selection matrix (D1/D2).

WHY THIS EXISTS, and why the existing wrappers are not it:

* `prompt_ablation_A_cell_fixedN.sh` picks N by watching the OFFICIAL TEST split
  every epoch. Its own header says so. That is defect F02.
* `prompt_ablation_A_cell.sh` does select on held-out train, but it hands the
  SAME `AUX_ARGS` to both stages, and D2 needs different ones: the search cells
  run `-e 60 --stop_after_epoch N --lr_schedule_horizon 60
  --sinkhorn_schedule_horizon N+1`, while the later refit runs `-e N+1
  --stop_after_epoch N` so all three horizons collapse together. It also starts
  a refit after every call, so driving it sixteen times would add sixteen
  unwanted refits, and it reads its answer by grepping a log line.
* Neither passes the geometry. `--num_semantic_parts`, `--num_codebooks` and
  `GDNA_NUM_SEMANTIC_PARTS` all default to 6, so the 2026-08-27 smoke ran at
  18 bases and was reported as a 15-base pass. Nothing asserted otherwise.

So this launcher owns the protocol, and the per-dataset trainers keep owning the
recipe -- it drives them through env and `EXTRA_ARGS` without editing them.

Every cell is checked BEFORE it runs (no artefact may already exist under its
tag) and AFTER (the effective `args.txt` and the run manifest must both say
M=5, L=3, 15 bases, 30 bits). The selection value is read from the best
checkpoint's runtime sidecar, which records `selection_metric` and
`selection_value` next to the checkpoint's own SHA -- not from a log line.

Usage:
    python scripts/phase3_selection_matrix.py --plan
    python scripts/phase3_selection_matrix.py --smoke --gpu 0
    python scripts/phase3_selection_matrix.py --run --gpu 0 --only cifar10:4
    python scripts/phase3_selection_matrix.py --run --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.run_identity import (  # noqa: E402
    MANIFEST_NAME, RunIdentity, load_run_manifest)

PY = os.environ.get("PY", "/home/yschoi/.conda/envs/dna_hashing/bin/python")
CACHE_ROOT = os.environ.get(
    "GDNA_CACHE_ROOT", "/data/yschoi/groundeddna_cache_v6prov")

#: D1: one grid for every dataset.
CANDIDATE_N = (4, 9, 19, 39)
SEED = 42
VAL_RATIO = 0.1
VAL_SEED = 42
#: D2 search stage: the LR schedule is held at 60 for every candidate, so
#: comparing N compares training length rather than three coupled knobs.
LR_HORIZON = 60
EPOCH_BUDGET = 60
#: `--eval_every`'s default. The retrieval score exists only on this cadence
#: (or at the nominal final epoch), so it decides whether a candidate's own
#: terminal epoch is scored at all.
EVAL_EVERY = 5

#: The RECIPE axes, and what the original sixteen cells effectively ran at.
#:
#: Neither was ever passed: the four trainers hardcode
#: `--routing_adaptive_topp_min 0.3 --max 0.7` in their command bodies, and
#: `--lambda_codon_joint` defaults to 0.0 in config.py, so those are what all
#: sixteen selection cells used -- confirmed in each winner's args.txt. The
#: draft calls that top-p window the M=6 tuning, so the incumbent is the anchor
#: of the sweep rather than the answer.
#: `(0.5, 0.9)` is the ORIGINAL adopted window -- v81a, chosen on Flickr25k
#: final test mAP, and the one the v82a/b/c interval sweep failed to beat. The
#: first grid took the three windows from the CIFAR correction cells and left it
#: out, which would have meant sweeping without the incumbent that every earlier
#: decision rested on. Flickr and NUS then moved 0.3/0.7 -> 0.4/0.8, i.e. WIDER,
#: so the omission was in the direction the data was already pointing.
TOPP_GRID = (("0.3", "0.7"), ("0.4", "0.8"), ("0.5", "0.9"), ("0.6", "0.95"))
TOPP_INCUMBENT = TOPP_GRID[0]
#: The lambda grid, swept at the winning top-p. 0.0 is not in it: the top-p
#: stage runs at 0.0, so its winning cell IS the off-control, at the same N,
#: seed, split and protocol as every lambda cell.
JOINT_GRID = ("0.02", "0.03", "0.05", "0.07", "0.10")
JOINT_INCUMBENT = "0.0"

#: CIFAR is not swept on this metric, and pinning it is not a shortcut.
#: The draft settled CIFAR at 0.6/0.95 on the per-image empty-slot rate, because
#: on a single-label dataset the codon decoding probe REWARDS slot starvation
#: (decoding .9033 < .9053 < .9158 < .9175 as empty images go 0% -> 43%) and
#: mAP@R prefers the starved cell too (.9019 at 0.3/0.7 vs .8875 at 0.6/0.95).
#: Selecting CIFAR by mAP@R here would therefore overturn a structural decision
#: with the very metric that decision was made to overrule. The other three are
#: multi-label, so that inversion does not apply and mAP@R is valid.
TOPP_SWEEP_DATASETS = ("flickr25k", "nuswide", "mscoco")
TOPP_PINNED = {"cifar10": ("0.6", "0.95")}

#: The N the sweeps hold fixed. These came out of the first sixteen cells, i.e.
#: at the incumbent recipe, so they are a HORIZON here and not a result -- N is
#: chosen again, by D1's own rule, once the recipe is settled.
INCUMBENT_N = {"cifar10": 39, "flickr25k": 4, "nuswide": 4, "mscoco": 39}

#: The paper's geometry, asserted before and after every cell.
SLOTS, BASES_PER_SLOT = 5, 3
TOTAL_BASES = SLOTS * BASES_PER_SLOT
TOTAL_BITS = 2 * TOTAL_BASES

#: The established S5 recipe (`auto_chain_after_3seed.sh`): five slots, no
#: gumbel, codon-Sinkhorn off. The 18-base smoke used none of it.
S5_FLAGS = [
    "--num_semantic_parts", str(SLOTS),
    "--num_codebooks", str(SLOTS),
    "--no_gumbel_softmax",
    "--lambda_codeword_codon_sinkhorn", "0.0",
]

#: The A recipe's global-slot skips, as the prompt-A wrapper passes them.
A_FLAGS = ["--xmodal_commit_skip_global", "--cibhash_dynamic_tau_skip_global"]

#: Nothing here should draw pictures or run the compositional post-eval.
QUIET_FLAGS = ["--no-post_eval_compositional", "--no_visualize"]

DATASETS = {
    "cifar10": dict(
        canon="CIFAR10", exp="cifar_A_v4",
        trainer="scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh",
        cache=f"{CACHE_ROOT}/cifar10_clip_tokens",
        foils=f"{CACHE_ROOT}/cifar10_clip_tokens_foils",
        qwen="./cache/cifar10_qwen_v4.jsonl",
        K=64, env=dict(CIBNT="1.0", CCS="0.1")),
    "flickr25k": dict(
        canon="Flickr25k", exp="flickr_A_v4",
        trainer="scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
        cache=f"{CACHE_ROOT}/flickr25k_clip_tokens",
        foils=f"{CACHE_ROOT}/flickr25k_clip_tokens_foils",
        qwen="./cache/flickr25k_qwen3_v4_trainset.jsonl",
        K=128, env=dict(CIBNT="1.0", BIDIR_MODE="legacy")),
    "nuswide": dict(
        canon="NUSWIDE", exp="nuswide_A_v4",
        trainer="scripts/train_nuswide_v185_sweep_clip.sh",
        cache=f"{CACHE_ROOT}/nuswide_clip_tokens",
        foils=f"{CACHE_ROOT}/nuswide_clip_tokens_foils",
        qwen="./cache/nuswide_qwen3_v4_trainset.jsonl",
        K=128, env=dict(CIBNT="1.5", CELL="Aprompt")),
    "mscoco": dict(
        canon="MSCOCO", exp="mscoco_A_v5b",
        trainer="scripts/train_mscoco_F2_sweep_clip.sh",
        cache=f"{CACHE_ROOT}/mscoco_clip_tokens",
        foils=f"{CACHE_ROOT}/mscoco_clip_tokens_foils",
        qwen="./cache/mscoco_qwen3_v5b_trainset.jsonl",
        K=128, env=dict(CIBNT="1.5", CELL="Aprompt")),
}

#: Kept from the caller: the things a shell needs to run at all, plus the
#: cache root override the tests use. Everything else is set by this launcher.
_ENV_PASSTHROUGH = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "PYTHONPATH", "PYTHONUNBUFFERED", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "GDNA_CACHE_ROOT", "PHASE3_NAMESPACE", "PY",
})

#: Recipe variables the trainers read. Cleared explicitly so a leftover export
#: cannot survive even if it slipped through the allow-list.
_RECIPE_ENV = (
    "WASS", "XMODAL", "THASH", "TCKL", "CIBNT", "CCS", "SLA", "GATE", "LBU",
    "BI_V", "BI_T", "BIDIR_MODE", "WHITEN_GAMMA", "DISABLE_GATE",
    "DISABLE_TEXT", "SHARE_CB", "SEQRES", "SEQRES_G", "CELL", "K",
    "NUM_CODONS", "TAG", "TAG_SUFFIX", "EXTRA_ARGS", "AUX_ARGS", "A_SKIPS",
    "GLOBAL_SKIPS", "FINAL_EPOCH", "STOP_EP", "VAL_RATIO", "VAL_SEED",
    "CACHE", "EVAL_CACHE", "QWEN", "WHITEN_NPZ", "VIZ", "FIXED_N", "EVERY",
)

NAMESPACE = os.environ.get("PHASE3_NAMESPACE", "phase3sel")
RECORD_DIR = REPO / "artifacts" / "phase3_selection"
RECORD_SCHEMA = 1

#: Files whose bytes decide what a cell is. Recorded per cell so the aggregator
#: can refuse a matrix assembled from more than one protocol.
PROTOCOL_SOURCES = (
    "scripts/phase3_selection_matrix.py",
    "train_siglip2.py",
    "model_siglip2.py",
    "config.py",
    "dna_utils/run_identity.py",
)


def _sha(path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: The shell that actually launches the cell. It was not in PROTOCOL_SOURCES,
#: and it is the file that hardcodes the top-p window this stage is sweeping --
#: so the one source whose edit would silently change what a cell ran was the
#: one the record did not name. Bash reads a script's top level incrementally,
#: so editing a trainer WHILE cells run gives a hybrid execution; the digest is
#: taken at launch and again after the child exits, and a change is a refusal.
def protocol_digests(dataset: str | None = None) -> dict:
    out = {rel: _sha(REPO / rel) for rel in PROTOCOL_SOURCES}
    if dataset is not None:
        trainer = DATASETS[dataset]["trainer"]
        out[trainer] = _sha(REPO / trainer)
    return out


class CellRefused(RuntimeError):
    """A cell cannot be run, or its output cannot be admitted."""


def cell_keys() -> list:
    """The exact 16 keys. No more, no fewer, in a fixed order."""
    return [(ds, n) for ds in ("cifar10", "flickr25k", "nuswide", "mscoco")
            for n in CANDIDATE_N]


def recipe_fragment(topp=None, joint=None) -> str:
    """How a non-incumbent recipe shows up in the tag, or "" for the incumbent.

    The empty string matters: it keeps the existing sixteen cells' tags exactly
    as they were, so this file still describes the matrix it already ran.
    """
    parts = []
    if topp is not None and tuple(topp) != tuple(TOPP_INCUMBENT):
        parts.append("P" + "".join(str(v).replace(".", "") for v in topp))
    if joint is not None and str(joint) != JOINT_INCUMBENT:
        parts.append("JD" + str(joint).replace(".", ""))
    return ("_" + "_".join(parts)) if parts else ""


def tag_for(dataset: str, n: int, *, namespace: str = NAMESPACE,
            topp=None, joint=None) -> str:
    return (f"{namespace}_{DATASETS[dataset]['exp']}_N{n}_s{SEED}"
            f"{recipe_fragment(topp, joint)}")


def _existing_artifacts(tag: str) -> list:
    hits = [str(p) for p in (REPO / "result").glob(f"*{tag}*")]
    log = REPO / "logs" / f"{tag}.log"
    if log.exists():
        hits.append(str(log))
    return sorted(hits)


def _stage1_flags(n: int) -> list:
    """D2's search cell. `-e 60` and the LR horizon are shared by every N."""
    return [
        "-e", str(EPOCH_BUDGET),
        "--lr_schedule_horizon", str(LR_HORIZON),
        "--sinkhorn_schedule_horizon", str(n + 1),
        "--random_seed", str(SEED),
        "--dna_distance_mode", "base",
        # Stage 1 IS the search. Left at its default the run identity records
        # `refit`, so a selection cell and the later refit at the same N
        # describe themselves the same way.
        "--selection_mode", "select",
        # The recorded score comes from epoch N; without this the trainer
        # replaces the final checkpoint with an earlier best, so the cell's
        # weights and its number describe different epochs. CIFAR N9 did
        # exactly that: terminal 0.8494 at epoch 9, weights from epoch 4.
        "--keep_final_checkpoint",
    ]


#: D2's final stage. `-e N+1 --stop_after_epoch N` and nothing else: with both
#: horizon flags unset they fall back to `--epoch`, which is what collapses the
#: three schedules onto N+1 together. Passing them explicitly would be the same
#: numbers by a longer route, but leaving them out is the decision as written.
REFIT_SEEDS = (42, 43, 44)


def _refit_flags(n: int, seed: int) -> list:
    return [
        "-e", str(n + 1),
        "--random_seed", str(seed),
        "--dna_distance_mode", "base",
        "--selection_mode", "refit",
    ]


def _override_flags(flags: list, replacements: dict) -> list:
    """Set each named flag to a new value, adding it if it is absent.

    Positional rewriting of an argument list only works for the one layout it
    was written against.
    """
    out = list(flags)
    for name, value in replacements.items():
        if name in out:
            out[out.index(name) + 1] = value
        else:
            out += [name, value]
    return out


def _whitening(spec: dict, *, stage: str = "select") -> str:
    """Stage 1 fits on the optimization-train rows only; the refit uses the
    full designated train, because it no longer holds any of it out."""
    variant = "optTrain" if stage == "select" else "trainOnly"
    return f"{spec['foils']}/text_whiten_{variant}_localOnly.npz"


def refit_tag_for(dataset: str, n: int, seed: int, *,
                  namespace: str = NAMESPACE, topp=None, joint=None) -> str:
    return (f"{namespace}_{DATASETS[dataset]['exp']}_refit_N{n}_s{seed}"
            f"{recipe_fragment(topp, joint)}")


def build_command(dataset: str, n: int, gpu: int, *,
                  epochs: int | None = None,
                  namespace: str = NAMESPACE,
                  stage: str = "select", seed: int = SEED,
                  topp=None, joint=None) -> tuple:
    spec = DATASETS[dataset]
    if stage == "refit":
        tag = refit_tag_for(dataset, n, seed, namespace=namespace,
                            topp=topp, joint=joint)
        flags = _refit_flags(n, seed) + S5_FLAGS + A_FLAGS + QUIET_FLAGS
    else:
        tag = tag_for(dataset, n, namespace=namespace, topp=topp, joint=joint)
        flags = _stage1_flags(n) + S5_FLAGS + A_FLAGS + QUIET_FLAGS
    # EXTRA_ARGS lands at the END of every trainer's command, after the
    # hardcoded window, and argparse keeps the last occurrence -- verified
    # against config.py's own parser rather than assumed from the layout.
    if topp is not None:
        flags = _override_flags(flags, {
            "--routing_adaptive_topp_min": str(topp[0]),
            "--routing_adaptive_topp_max": str(topp[1]),
        })
        if "--routing_adaptive_topp" not in flags:
            flags = flags + ["--routing_adaptive_topp"]
    if joint is not None:
        flags = _override_flags(flags, {"--lambda_codon_joint": str(joint)})
    stop = n
    if epochs is not None:
        # Smoke: shorten everything CONSISTENTLY, by NAME. Slicing off the
        # first six tokens was tuned to the selection layout, so a refit smoke
        # -- whose first six are `-e`, `--random_seed <seed>` and
        # `--dna_distance_mode base` -- silently lost its seed and its distance
        # mode and ran as the trainer's defaults.
        stop = max(epochs - 1, 0)
        flags = _override_flags(flags, {
            "-e": str(epochs),
            "--lr_schedule_horizon": str(epochs),
            "--sinkhorn_schedule_horizon": str(stop + 1),
        })
    # A clean environment plus exactly what this protocol sets. The trainers
    # read a dozen recipe variables -- WASS, XMODAL, DISABLE_TEXT, SHARE_CB,
    # CCS, GATE and more -- so inheriting the caller's shell would let a stale
    # export from an unrelated experiment redefine a matrix cell.
    env = {k: v for k, v in os.environ.items() if k in _ENV_PASSTHROUGH}
    for name in _RECIPE_ENV:
        env.pop(name, None)
    env.update(spec["env"])
    env.update(
        # The geometry is baked in at import time, so it has to be in the
        # environment before python starts -- passing the CLI flag alone aborts.
        GDNA_NUM_SEMANTIC_PARTS=str(SLOTS),
        CUDA_VISIBLE_DEVICES=str(gpu),
        CACHE=spec["cache"], EVAL_CACHE=spec["cache"], QWEN=spec["qwen"],
        WHITEN_NPZ=_whitening(spec, stage=stage), K=str(spec["K"]),
        NUM_CODONS=str(BASES_PER_SLOT),
        TAG=tag,
        EXTRA_ARGS=" ".join(shlex.quote(f) for f in flags),
    )
    env["STOP_EP"] = str(stop)
    if stage == "refit":
        # The refit trains on the FULL designated train -- there is nothing to
        # hold out once N is chosen -- and it is the run that finally evaluates
        # the official test split, exactly once.
        env["VAL_RATIO"] = "0.0"
        env["VAL_SEED"] = str(VAL_SEED)
        env["FINAL_EPOCH"] = "1"
    else:
        # Stage 1 never evaluates the official test split: no FINAL_EPOCH, and
        # `--stop_after_epoch N` stops the run at the candidate epoch.
        env["VAL_RATIO"] = str(VAL_RATIO)
        env["VAL_SEED"] = str(VAL_SEED)
        env.pop("FINAL_EPOCH", None)
    return ["bash", spec["trainer"], str(gpu)], env, tag


#: Every file whose bytes decide what a sweep cell computes, beyond the four
#: PROTOCOL_SOURCES and the per-dataset trainer. A sweep spans hours; these are
#: hashed ONCE when the plan is made and re-checked after every cell, so a
#: campaign cannot be a mixture of two trees.
_SNAPSHOT_SOURCES = (
    "config.py", "train_siglip2.py", "model_siglip2.py", "loss_siglip2.py",
    "dataloaders.py", "extraction_siglip2.py", "evaluation_siglip2.py",
    "dna_utils/run_identity.py", "dna_utils/extraction_validation.py",
    "dna_utils/cache_provenance.py",
    "scripts/phase3_selection_matrix.py",
)


def canonical_plan(axis: str, *, at_topp=None) -> list:
    """The plan the SOURCE declares for an axis.

    The selector must compare a snapshot against this, not against the set the
    snapshot names for itself: a Flickr-only snapshot claiming to be a complete
    sweep was accepted as one.
    """
    return sweep_cells(axis, at_topp=at_topp)


def plan_snapshot(datasets=None, *, plan=None, executed=None, axis=None,
                  namespace=None) -> dict:
    """The sources AND the inputs, frozen at plan time.

    The launcher-crash lesson generalises: bash reads a script incrementally,
    Python loads a module once, and neither notices a cache being rebuilt
    underneath it. Recording the digests is not the point -- COMPARING them
    after each cell is, which is what `verify_snapshot` does.
    """
    datasets = sorted(DATASETS) if datasets is None else sorted(set(datasets))
    # `#absent` rather than an error, the same marker RunIdentity._artifact
    # uses: a file that is not there is RECORDED as not there, so it still
    # differs from the same file appearing later. Refusing outright would make
    # the snapshot unusable in any tree that does not carry the whole
    # repository, and would hide the drift rather than name it.
    def _digest(rel: str) -> str:
        path = REPO / rel
        return _sha(path) if path.is_file() else "#absent"

    sources = {rel: _digest(rel) for rel in _SNAPSHOT_SOURCES}
    for ds in datasets:
        sources[DATASETS[ds]["trainer"]] = _digest(DATASETS[ds]["trainer"])
    inputs = {}
    for ds in datasets:
        spec = DATASETS[ds]
        # A feature cache is tens of gigabytes; `meta.json` is the file the
        # provenance gate already checks and it carries the backbone revision,
        # the transform and the row count. The whitening matrix and the caption
        # file are small enough to hash whole, and both change the numbers.
        for label, path in (
                ("feature_cache_meta", Path(spec["cache"]) / "meta.json"),
                ("foils_meta", Path(spec["foils"]) / "meta.json"),
                ("whiten_select", Path(_whitening(spec, stage="select"))),
                ("whiten_refit", Path(_whitening(spec, stage="refit"))),
                ("qwen", REPO / spec["qwen"])):
            inputs[f"{ds}/{label}"] = (_sha(path) if Path(path).is_file()
                                       else "#absent")
    payload = {"schema_version": 2, "sources": sources, "inputs": inputs}
    # WHAT is being run, not only WHAT IT IS RUN WITH. Without this the
    # one-cell smoke and the nine-cell production sweep produced a
    # byte-identical snapshot, so the digest in a receipt could not say which
    # plan it sealed.
    def _cells(rows):
        return [{"dataset": ds, "N": n,
                 "topp": list(topp) if topp is not None else None,
                 "joint": jd}
                for ds, n, topp, jd in rows]

    if plan is not None:
        executed = plan if executed is None else executed
        payload["plan"] = {
            "axis": axis,
            "namespace": namespace,
            # What the axis declares, and what THIS invocation actually runs.
            # One field could not express a `--only` smoke: the snapshot said
            # twelve while its receipt said one, and nothing compared them.
            "declared_cells": _cells(plan),
            "declared_count": len(plan),
            "executed_cells": _cells(executed),
            "executed_count": len(executed),
        }
    return payload


def verify_snapshot(snapshot: dict, *, datasets=None) -> None:
    """Refuse when anything the PLAN pinned has moved.

    The comparison is over the snapshot's OWN keys, re-hashed now. Comparing
    against `plan_snapshot(one_dataset)` compared a nine-cell plan's key set
    against a one-cell key set, so every multi-dataset plan refused its own
    first cell -- with nothing changed -- because the other datasets' trainers
    were "missing" from the narrower recomputation. The snapshot is the
    authority; the subset is not a second snapshot to diff against it.
    """
    del datasets  # the snapshot's keys decide what is re-checked
    now_sources = {rel: (_sha(REPO / rel) if (REPO / rel).is_file()
                         else "#absent")
                   for rel in (snapshot.get("sources") or {})}
    now_inputs = {}
    for key in (snapshot.get("inputs") or {}):
        ds, label = key.split("/", 1)
        spec = DATASETS[ds]
        path = {
            "feature_cache_meta": Path(spec["cache"]) / "meta.json",
            "foils_meta": Path(spec["foils"]) / "meta.json",
            "whiten_select": Path(_whitening(spec, stage="select")),
            "whiten_refit": Path(_whitening(spec, stage="refit")),
            "qwen": REPO / spec["qwen"],
        }[label]
        now_inputs[key] = _sha(path) if Path(path).is_file() else "#absent"

    for kind, after in (("sources", now_sources), ("inputs", now_inputs)):
        before = snapshot.get(kind) or {}
        changed = sorted(k for k in before if before[k] != after.get(k))
        if changed:
            raise CellRefused(
                f"{kind} changed while the sweep was running: {changed}. What "
                f"ran is a mixture of two trees, so no cell in this sweep is "
                f"admissible against one protocol. Re-run against one tree.")


# --------------------------------------------------------------- checking

def _flag_value(flags: list, option: str, default: str) -> str:
    return flags[flags.index(option) + 1] if option in flags else default


def _arg_value(args_txt: Path, field: str) -> str:
    prefix = field + "-"
    for line in args_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].lstrip("-").strip()
    raise CellRefused(f"{args_txt} has no field {field!r}")


def assert_geometry(run_dir: Path, *, dataset: str, n: int,
                    stop: int | None = None, seed: int = SEED,
                    mode: str = "select", val_ratio: float = VAL_RATIO,
                    budget: int | None = None,
                    lr_horizon: int | None = None,
                    sinkhorn_horizon: int | None = None,
                    topp=None, joint=None) -> dict:
    """The check the 18-base smoke did not have.

    Both the effective arguments and the sealed run manifest must say the
    paper's geometry. Either alone can be right while the other is not: the CLI
    flag without the environment variable aborts, and the environment variable
    without the flag would leave `args.txt` describing something else.
    """
    stop = n if stop is None else stop
    budget = EPOCH_BUDGET if budget is None else budget
    lr_horizon = LR_HORIZON if lr_horizon is None else lr_horizon
    sinkhorn_horizon = (stop + 1) if sinkhorn_horizon is None \
        else sinkhorn_horizon
    args_txt = run_dir / "args.txt"
    if not args_txt.is_file():
        raise CellRefused(f"{run_dir}: no args.txt")
    # The swept axes are read from `args.txt` -- what the trainer actually
    # parsed -- rather than from the command this process composed. The three
    # top-p cells differ ONLY here, and until the identity carried these axes
    # they all produced one digest, so a cell that silently kept the trainer's
    # hardcoded 0.3/0.7 would have been indistinguishable from one that did not.
    topp = TOPP_INCUMBENT if topp is None else topp
    joint = JOINT_INCUMBENT if joint is None else joint
    effective = {
        "num_semantic_parts": int(_arg_value(args_txt, "num_semantic_parts")),
        "num_codebooks": int(_arg_value(args_txt, "num_codebooks")),
        "num_codons_per_codebook": int(
            _arg_value(args_txt, "num_codons_per_codebook")),
        "routing_adaptive_topp_min": float(
            _arg_value(args_txt, "routing_adaptive_topp_min")),
        "routing_adaptive_topp_max": float(
            _arg_value(args_txt, "routing_adaptive_topp_max")),
        "lambda_codon_joint": float(_arg_value(args_txt, "lambda_codon_joint")),
    }
    want = {"num_semantic_parts": SLOTS, "num_codebooks": SLOTS,
            "num_codons_per_codebook": BASES_PER_SLOT,
            "routing_adaptive_topp_min": float(topp[0]),
            "routing_adaptive_topp_max": float(topp[1]),
            "lambda_codon_joint": float(joint)}
    bad = {k: (v, want[k]) for k, v in effective.items() if v != want[k]}
    if bad:
        raise CellRefused(
            f"{run_dir}: effective recipe is wrong {bad} -- geometry like this "
            f"is the 18-base failure, and a wrong top-p or lambda is a cell "
            f"reported under a recipe it did not run")

    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise CellRefused(f"{run_dir}: no readable {MANIFEST_NAME}")
    spec = DATASETS[dataset]
    # EVERY axis the protocol fixes, not the handful that used to be checked.
    # A `/tmp` probe with mode=refit, budget 7, horizons 3/999, val split
    # .9/999, fake caches, batch 1 and projection LR 99 passed the old subset.
    expected = {
        "num_slots": SLOTS, "bases_per_slot": BASES_PER_SLOT,
        "total_bases": TOTAL_BASES, "total_bits": TOTAL_BITS,
        "dataset": spec["canon"], "seed": seed,
        "stop_after_epoch": stop, "codebook_size": spec["K"],
        "selection_mode": mode,
        "val_split_ratio": val_ratio, "val_split_seed": VAL_SEED,
        "epoch_budget": budget,
        "lr_schedule_horizon": lr_horizon,
        "sinkhorn_schedule_horizon": sinkhorn_horizon,
        # Through the SAME normaliser the identity used, or the comparison is
        # between a raw path and a content-bound one and never matches.
        "feature_cache": RunIdentity._artifact(spec["cache"]),
        "eval_cache": RunIdentity._artifact(spec["cache"]),
        "routing_adaptive_topp_min": float(topp[0]),
        "routing_adaptive_topp_max": float(topp[1]),
        "lambda_codon_joint": float(joint),
    }
    manifest_bad = {k: (getattr(identity, k), v) for k, v in expected.items()
                    if getattr(identity, k) != v}
    if manifest_bad:
        raise CellRefused(f"{run_dir}: manifest disagrees {manifest_bad}")
    return {"identity_digest": identity.digest, **effective}


def assert_refit_outputs(run_dir: Path, *, dataset: str) -> dict:
    """A refit is complete only when its official evaluation exists and binds.

    The selection stage produces no extraction, so this applies to the refit
    alone -- and the refit IS the paper number. Checking the checkpoint and the
    CSV said nothing about whether the official test split was ever extracted
    or scored: a run whose evaluation raised used to print the failure and exit
    0, leaving a cell that looked finished and had no metrics.

    The shared strict validator is the same one Phase 2 uses: it opens every
    file the manifests name, recomputes every digest and checks the code arrays
    themselves, rather than trusting that a file with the right name is the
    right file.
    """
    from dna_utils.extraction_validation import (
        ExpectedIdentity, ExtractionInvalid, validate_extraction_run)

    spec = DATASETS[dataset]
    try:
        run = validate_extraction_run(
            str(run_dir), required_splits=("db", "query"),
            allow_backfilled=False,
            expected=ExpectedIdentity(
                dataset=spec["canon"], num_slots=SLOTS,
                bases_per_slot=BASES_PER_SLOT,
                codebook_size=spec["K"]))
    except ExtractionInvalid as error:
        raise CellRefused(f"{run_dir}: extraction does not validate: {error}")

    metrics = run_dir / "evaluation_siglip2_base.json"
    if not metrics.is_file():
        raise CellRefused(
            f"{run_dir}: no {metrics.name}; the official evaluation did not "
            f"produce a metric, so this cell has no number to report")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if "mAP" not in payload:
        raise CellRefused(f"{metrics}: records no mAP")
    return {
        "extraction_validated": True,
        "npz_sha256": {split: m["npz_sha256"]
                       for split, m in sorted(run.splits.items())},
        "evaluation_sha256": _sha(metrics),
        "map": payload.get("mAP"),
        "map_at_R": payload.get("mAP_at_R"),
    }


def _manifest_facts(run_dir: Path) -> dict:
    """What the run itself recorded, verbatim."""
    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise CellRefused(f"{run_dir}: no readable {MANIFEST_NAME}")
    return {
        "seed": identity.seed,
        "val_split_ratio": identity.val_split_ratio,
        "val_split_seed": identity.val_split_seed,
        "lr_schedule_horizon": identity.lr_schedule_horizon,
        "sinkhorn_schedule_horizon": identity.sinkhorn_schedule_horizon,
        "epoch_budget": identity.epoch_budget,
        "stop_after_epoch": identity.stop_after_epoch,
        "selection_mode": identity.selection_mode,
        "codebook_size": identity.codebook_size,
        "identity_digest": identity.digest,
    }


def assert_completed(run_dir: Path, *, terminal_epoch: int) -> dict:
    """The cell finished and nobody still owns it.

    A partial run leaves args, a manifest and a checkpoint behind, which is
    enough to pass a files-exist check. The checkpoint is re-hashed against
    what its own sidecar claims, and an active claim means the writer is still
    going -- or died holding it.
    """
    from dna_utils.run_identity import ACTIVE_CLAIM_NAME

    if (run_dir / ACTIVE_CLAIM_NAME).exists():
        raise CellRefused(
            f"{run_dir}: still carries an active claim, so the run either has "
            f"not finished or died holding it")
    final = run_dir / "model_state_dict.pth"
    if not final.is_file():
        raise CellRefused(f"{run_dir}: no final checkpoint")
    sidecar = Path(str(final) + ".runtime.json")
    if not sidecar.is_file():
        raise CellRefused(f"{run_dir}: final checkpoint has no runtime sidecar")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    claimed = payload.get("checkpoint_sha256")
    actual = _sha(final)
    if claimed != actual:
        raise CellRefused(
            f"{run_dir}: final checkpoint hashes to {actual[:12]} but its "
            f"sidecar says {str(claimed)[:12]}")

    # Self-consistency is not identity. A sidecar whose own SHA matched but
    # which named a different budget, stop epoch or schedule was admitted, so
    # it is bound to the run manifest it sits beside.
    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise CellRefused(f"{run_dir}: no readable {MANIFEST_NAME}")
    disagree = {
        name: (payload.get(name), expected)
        for name, expected in (
            ("training_epoch_budget", identity.epoch_budget),
            ("stop_after_epoch", identity.stop_after_epoch),
            ("lr_schedule_horizon", identity.lr_schedule_horizon),
            ("sinkhorn_schedule_horizon", identity.sinkhorn_schedule_horizon),
        )
        if name in payload and payload.get(name) != expected}
    if disagree:
        raise CellRefused(
            f"{run_dir}: the checkpoint sidecar describes a different run than "
            f"the manifest beside it: {disagree}")
    # Whether the weights on disk are the ones the terminal metric describes.
    # Without `--final_epoch_eval` -- which the selection stage may not use,
    # because it evaluates the official test split -- the trainer overwrites the
    # final checkpoint with an earlier best. The metric still comes from epoch
    # N's row in log.csv, and these weights are discarded anyway (the refit is
    # from scratch), but the record must not imply otherwise.
    recorded_epoch = json.loads(sidecar.read_text(encoding="utf-8")).get(
        "checkpoint_epoch_zero_based")
    preserved = recorded_epoch == terminal_epoch
    if not preserved:
        raise CellRefused(
            f"{run_dir}: the metric is read from epoch {terminal_epoch} but "
            f"the surviving checkpoint is from epoch {recorded_epoch}. The "
            f"score and the weights describe different epochs, so the cell is "
            f"not self-consistent evidence -- pass --keep_final_checkpoint.")
    return {"final_checkpoint_sha256": actual,
            "final_checkpoint_epoch_zero_based": recorded_epoch,
            "terminal_weights_preserved": preserved,
            "log_csv_sha256": _sha(run_dir / "log.csv")}


def read_selection(run_dir: Path, *, n: int) -> dict:
    """The candidate's score at ITS OWN terminal epoch N.

    D1 compares candidates by what training to N produces, so the number has to
    come from epoch N. The first version read
    `model_state_dict_best.pth.runtime.json`, which holds the best value over
    the whole prefix -- so an N=39 cell whose best epoch was 4 would report the
    N=4 answer, and every candidate could collapse onto the same early epoch.
    The live N=4 smoke reported its epoch-0 score, which is that bug in
    miniature.

    `log.csv` carries `eval_mAP_at_R` per epoch, so the terminal row is read
    directly. The best-checkpoint sidecar is still recorded, for comparison, but
    it is not the answer.
    """
    csv_path = run_dir / "log.csv"
    if not csv_path.is_file():
        raise CellRefused(f"{run_dir}: no log.csv, so no per-epoch metric")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CellRefused(f"{csv_path} has no rows")
    if "eval_mAP_at_R" not in rows[0]:
        raise CellRefused(
            f"{csv_path} has no eval_mAP_at_R column; this run predates the "
            f"per-epoch selection metric and its terminal value is not "
            f"recoverable")

    # The epochs must be exactly 0..N, once each, in order. A single matching
    # row used to be enough, so a file with gaps, repeats or rows beyond the
    # stop point -- an append-only log two runs had written to -- still yielded
    # an answer.
    epochs = [str(r.get("epoch", "")).strip() for r in rows]
    if epochs != [str(e) for e in range(n + 1)]:
        raise CellRefused(
            f"{csv_path}: epochs are {epochs[:8]}{'...' if len(epochs) > 8 else ''}, "
            f"expected exactly 0..{n} once each. The cell did not train to its "
            f"candidate epoch, or more than one run wrote this file.")
    terminal = [rows[n]]
    raw = (terminal[0].get("eval_mAP_at_R") or "").strip()
    if not raw:
        raise CellRefused(
            f"{csv_path}: epoch {n} has no eval_mAP_at_R, so the candidate was "
            f"never scored at its own terminal epoch")
    try:
        value = float(raw)
    except ValueError:
        raise CellRefused(f"{csv_path}: eval_mAP_at_R={raw!r} is not a number")
    if not 0.0 <= value <= 1.0:
        raise CellRefused(f"{csv_path}: eval_mAP_at_R={value} is not a "
                          f"proportion")

    cutoff = (terminal[0].get("eval_mAP_R_cutoff") or "").strip()
    if not cutoff:
        raise CellRefused(
            f"{csv_path}: epoch {n} records no mAP@R cutoff, so the metric "
            f"cannot be compared across datasets")
    record = {
        "selection_metric": "eval_mAP_at_R",
        "selection_value": value,
        "selection_epoch_zero_based": n,
        "map_r_cutoff": cutoff,
        "epochs_logged": len(rows),
        # The bytes the number came from. `log.csv` is append-only and nothing
        # else binds it to the run.
        "log_csv_sha256": _sha(csv_path),
    }
    # Recorded, never used to select: it is the best over the prefix, and
    # seeing it differ from the terminal value is exactly what a reader wants.
    sidecar = run_dir / "model_state_dict_best.pth.runtime.json"
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        extra = payload.get("extra") or {}
        record["best_over_prefix"] = {
            "value": extra.get("selection_value"),
            "epoch_zero_based": payload.get("checkpoint_epoch_zero_based"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
        }
    return record


def _resolve(tag: str, *, expected_digest: str | None = None) -> Path:
    """The directory this cell produced, identified rather than searched for.

    A tag substring re-search races a concurrent writer with a similar tag and
    compares nothing about what it finds. When the expected identity is known,
    the candidates are filtered by their manifest digest, so the answer is the
    directory that carries THIS run's identity -- not merely one whose name
    happens to contain the tag.
    """
    hits = sorted(p for p in (REPO / "result").glob(f"*{tag}*") if p.is_dir())
    if expected_digest is not None:
        matched = [p for p in hits
                   if (load_run_manifest(str(p)) or None) is not None
                   and load_run_manifest(str(p)).digest == expected_digest]
        if len(matched) == 1:
            return matched[0]
        if hits and not matched:
            raise CellRefused(
                f"{len(hits)} directories match {tag} and none carries this "
                f"run's identity {expected_digest[:12]}; the cell did not "
                f"write where it was expected to")
        if len(matched) > 1:
            raise CellRefused(
                f"{len(matched)} directories carry identity "
                f"{expected_digest[:12]}: {[p.name for p in matched]}")
    if len(hits) != 1:
        raise CellRefused(
            f"expected exactly one result for {tag}, found {len(hits)}: "
            f"{[p.name for p in hits]}")
    return hits[0]


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def run_cell(dataset: str, n: int, gpu: int, *, epochs=None,
             namespace: str = NAMESPACE, stage: str = "select",
             seed: int = SEED, topp=None, joint=None,
             snapshot: dict | None = None) -> dict:
    tag = (refit_tag_for(dataset, n, seed, namespace=namespace,
                         topp=topp, joint=joint)
           if stage == "refit"
           else tag_for(dataset, n, namespace=namespace,
                        topp=topp, joint=joint))
    existing = _existing_artifacts(tag)
    if existing:
        raise CellRefused(
            f"{tag} already has artefacts: {existing[:3]}. Give the run its own "
            f"namespace with PHASE3_NAMESPACE.")

    cmd, env, _ = build_command(dataset, n, gpu, epochs=epochs,
                                namespace=namespace, stage=stage, seed=seed,
                                topp=topp, joint=joint)
    # What THIS cell is supposed to end up being, derived once so the
    # post-checks and the record cannot drift from the command.
    _stop = int(env["STOP_EP"])
    _flags = env["EXTRA_ARGS"].split()
    _budget = int(_flags[_flags.index("-e") + 1])
    _lr_h = (int(_flags[_flags.index("--lr_schedule_horizon") + 1])
             if "--lr_schedule_horizon" in _flags else _budget)
    _sk_h = (int(_flags[_flags.index("--sinkhorn_schedule_horizon") + 1])
             if "--sinkhorn_schedule_horizon" in _flags else _budget)

    # The retrieval eval runs on the `--eval_every` cadence or at the nominal
    # final epoch, and the search stage stops early -- so a terminal row exists
    # only when (stop + 1) is a multiple of the cadence. It happens to hold for
    # every candidate in the current grid, which is exactly the kind of
    # coincidence that stops holding when someone edits the grid.
    # The trainer evaluates when `(e+1) % eval_every == 0` OR when `e+1` is the
    # nominal final epoch. The guard used to require only the first, so a smoke
    # that stops at `budget - 1` -- which IS the nominal final -- was refused
    # before it ran even though the trainer would have scored it.
    _scored = ((_stop + 1) % EVAL_EVERY == 0) or (_stop + 1 == _budget)
    if not _scored:
        raise CellRefused(
            f"{tag}: terminal epoch {_stop} would not be evaluated -- "
            f"(epoch+1)={_stop + 1} is neither a multiple of "
            f"--eval_every={EVAL_EVERY} nor the nominal final epoch "
            f"({_budget}), so log.csv would carry no score for it")
    # Taken BEFORE the child starts, compared after it exits. The plan-wide
    # snapshot is the stronger check -- it spans the whole sweep -- and this
    # narrower one still applies when a cell is run on its own.
    if snapshot is not None:
        verify_snapshot(snapshot, datasets=[dataset])
    sources_before = protocol_digests(dataset)
    started = time.time()
    # The trainers end in `python ... | tee "$LOG"` under `set -eu` with no
    # `pipefail`, so the script's status is tee's, not python's: a crashed
    # trainer would look like a finished cell.
    #
    # `bash -o pipefail -c 'bash trainer.sh'` does NOT fix that -- the option
    # is set on the outer shell while the pipeline runs in the inner one, and
    # shell options are not inherited across an exec. The option has to be on
    # the shell that runs the trainer's own pipeline, so the script is invoked
    # directly with it.
    proc = subprocess.run(["bash", "-o", "pipefail", *cmd[1:]],
                          cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise CellRefused(f"{tag}: trainer exited {proc.returncode}")
    if snapshot is not None:
        verify_snapshot(snapshot, datasets=[dataset])
    sources_after = protocol_digests(dataset)
    if sources_after != sources_before:
        changed = sorted(k for k in sources_before
                         if sources_before[k] != sources_after.get(k))
        raise CellRefused(
            f"{tag}: {changed} changed while this cell was running, so what "
            f"executed is a mixture of two versions -- the same way editing "
            f"the campaign launcher mid-run produced a hybrid parse. The run "
            f"is not admissible; re-run it against one tree.")

    # §54.3 asks for a precomputed expected digest here. Reconstructing one
    # means restating the batch size, projection LR, Sinkhorn epsilon endpoints
    # and every lambda the trainer sets -- values this file does not own, and
    # guessing them wrong would refuse every honest cell. The hole it names --
    # "a sole directory with the wrong P/JD is accepted and its own manifest is
    # copied into the record" -- is closed instead by `assert_geometry` below,
    # which now compares the effective top-p window and lambda from args.txt
    # AND from the manifest against what this cell asked for. Same property,
    # from values that were verified rather than assumed.
    run_dir = _resolve(tag)
    record = {
        "schema_version": RECORD_SCHEMA,
        "namespace": namespace,
        "matrix": {"candidate_n": list(CANDIDATE_N),
                   "datasets": sorted(DATASETS),
                   "cells": len(cell_keys())},
        # The swept coordinate, read back from the composed command rather than
        # from the caller's arguments, so a record cannot claim a recipe the
        # trainer did not receive.
        "recipe": {
            "routing_adaptive_topp_min": _flag_value(_flags,
                                                     "--routing_adaptive_topp_min",
                                                     TOPP_INCUMBENT[0]),
            "routing_adaptive_topp_max": _flag_value(_flags,
                                                     "--routing_adaptive_topp_max",
                                                     TOPP_INCUMBENT[1]),
            "lambda_codon_joint": _flag_value(_flags, "--lambda_codon_joint",
                                              JOINT_INCUMBENT),
        },
        "protocol_sources": sources_after,
        "plan_snapshot_sha256": (hashlib.sha256(
            json.dumps(snapshot, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
            if snapshot is not None else None),
        "inputs": {
            "feature_cache": DATASETS[dataset]["cache"],
            "eval_cache": DATASETS[dataset]["cache"],
            "qwen": DATASETS[dataset]["qwen"],
            "whitening": _whitening(DATASETS[dataset], stage=stage),
            "codebook_size": DATASETS[dataset]["K"],
        },
        "stage": stage,
        "dataset": dataset, "N": n, "tag": tag,
        "run_dir": str(run_dir),
        # Copied from the run's OWN manifest, not from this launcher's
        # constants. Writing the intended values would describe a
        # wrong-protocol run as a right-protocol one, which is the opposite of
        # what a record is for.
        **_manifest_facts(run_dir),
        "completion": {
            **assert_completed(run_dir, terminal_epoch=_stop),
            **(assert_refit_outputs(run_dir, dataset=dataset)
               if stage == "refit" else {}),
        },
        "geometry": assert_geometry(
            run_dir, dataset=dataset, n=n, stop=_stop, seed=seed,
            mode="refit" if stage == "refit" else "select",
            val_ratio=0.0 if stage == "refit" else VAL_RATIO,
            budget=_budget, lr_horizon=_lr_h, sinkhorn_horizon=_sk_h,
            topp=topp, joint=joint),
        "selection": (
            read_selection(run_dir, n=_stop)
            if stage == "select" else
            {"selection_metric": None,
             "note": "a refit has no held-out validation to select on"}),
        "wall_seconds": round(time.time() - started, 1),
        # A smoke is plumbing evidence, never a candidate score.
        "smoke": epochs is not None,
        "is_candidate_cell": epochs is None and stage == "select",
    }
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORD_DIR / f"{tag}.json"
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)
    return record


def _load_recipe(path: Path) -> dict:
    """Per-dataset {topp, joint} chosen by a recipe reduction.

    Authenticated the same way the N artefact is: schema, declared reduction,
    the aggregator's own digest, the grid, and that every value is ON the grid.
    A file that merely has the right shape is a receipt for whatever wrote it.
    """
    if not path.is_file():
        raise CellRefused(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CellRefused(f"{path}: unknown schema_version "
                          f"{payload.get('schema_version')!r}")
    axis = payload.get("axis")
    if axis not in ("topp", "joint"):
        raise CellRefused(f"{path}: axis is {axis!r}")
    aggregator = REPO / "scripts" / "phase3_select_n.py"
    if payload.get("aggregator_sha256") != _sha(aggregator):
        raise CellRefused(
            f"{path} was written by a different {aggregator.name} than the one "
            f"in this tree")
    # A recipe that names no records is a receipt for whatever wrote it. A
    # forged file carrying the current aggregator's digest, an empty
    # `record_sha256` and `protocol_sources`, and `stability.confirmed: false`
    # was accepted for all four datasets.
    records = payload.get("record_sha256")
    if not isinstance(records, dict) or not records:
        raise CellRefused(
            f"{path}: record_sha256 names no cells, so this choice cannot be "
            f"traced to any run")
    bad = sorted(k for k, v in records.items()
                 if not isinstance(v, str) or len(v) != 64)
    if bad:
        raise CellRefused(f"{path}: record_sha256 entries are not digests {bad}")
    from scripts.phase3_select_n import RECIPE_REDUCTION
    if payload.get("reduction") != RECIPE_REDUCTION:
        raise CellRefused(
            f"{path}: the reduction it was produced under is not this tree's "
            f"declared rule; the winner was chosen by a different policy")
    if payload.get("protocol_sources") != protocol_digests():
        raise CellRefused(
            f"{path}: produced under a different protocol than this tree")
    grid = [tuple(c) if isinstance(c, list) else c
            for c in (payload.get("grid") or [])]
    want = ([tuple(g) for g in TOPP_GRID] if axis == "topp"
            else list(JOINT_GRID))
    if grid != want:
        raise CellRefused(f"{path}: grid {grid} is not this tree's {want}")
    selected = payload.get("selected") or {}
    out = {}
    for dataset, entry in selected.items():
        if dataset not in DATASETS:
            raise CellRefused(f"{path}: {dataset!r} is not a dataset")
        value = entry.get("selected")
        value = tuple(value) if isinstance(value, list) else value
        if value not in grid:
            raise CellRefused(
                f"{path}: {dataset} selects {value!r}, which is not on the "
                f"grid {grid}")
        out[dataset] = {axis: value}
    missing = sorted(set(DATASETS) - set(out))
    if missing:
        raise CellRefused(f"{path}: no choice for {missing}")
    return out


def _load_selection(path: Path) -> dict:
    """The chosen N per dataset, as the aggregator wrote it.

    D1 reuses one N per dataset across all three seeds, so the refit reads the
    decision rather than re-deriving it -- re-deriving per seed is what
    `queue_ours_multiseed.sh` did, and it is the opposite of the rule.
    """
    if not path.is_file():
        raise CellRefused(
            f"{path} does not exist; run scripts/phase3_select_n.py on the "
            f"completed matrix first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected") or {}
    missing = sorted(d for d in DATASETS if d not in selected)
    extra = sorted(d for d in selected if d not in DATASETS)
    if missing or extra:
        raise CellRefused(
            f"{path}: the selection must name exactly {sorted(DATASETS)}; "
            f"missing {missing}, unexpected {extra}")
    out = {}
    for dataset, entry in selected.items():
        n = entry.get("selected_N")
        if n not in CANDIDATE_N:
            raise CellRefused(
                f"{path}: {dataset} selects N={n}, not one of "
                f"{list(CANDIDATE_N)}")
        out[dataset] = n
    return out


def sweep_cells(axis: str, *, at_topp=None) -> list:
    """(dataset, N, topp, joint) for a recipe sweep. Exactly one axis moves.

    The N is INCUMBENT_N, i.e. what the first sixteen cells chose at the
    incumbent recipe. It is the horizon the candidates share, not the answer:
    N is chosen again by D1's rule once the recipe is settled.
    """
    if axis == "topp":
        # The lambda stays at the incumbent 0.0, so the only difference between
        # these cells and the original sixteen is the window under test.
        return [(ds, INCUMBENT_N[ds], topp, JOINT_INCUMBENT)
                for ds in TOPP_SWEEP_DATASETS for topp in TOPP_GRID]
    if at_topp is None:
        raise CellRefused(
            "--sweep joint needs --at-topp MIN,MAX: the two axes interact "
            "(adding noGumbel already moved MS-COCO's lambda optimum from 0.05 "
            "to 0.03), so a lambda swept at an unstated window measures "
            "nothing transferable")
    return [(ds, INCUMBENT_N[ds],
             TOPP_PINNED.get(ds, at_topp), jd)
            for ds in sorted(DATASETS) for jd in JOINT_GRID]


def _only_cell(spec: str, plan: list, *, axis: str) -> tuple:
    """Resolve `--only` to exactly one sweep cell, or refuse.

    A selector that can quietly mean "several" is how a three-trainer run got
    launched and reported as one cell.
    """
    if ":" not in spec:
        raise CellRefused(
            f"{spec!r} names no coordinate. For --sweep topp use "
            f"dataset:MIN,MAX (e.g. flickr25k:0.6,0.95); for --sweep joint use "
            f"dataset:LAMBDA (e.g. flickr25k:0.05)")
    dataset, coord = spec.split(":", 1)
    if dataset not in DATASETS:
        raise CellRefused(f"{dataset!r} is not one of {sorted(DATASETS)}")

    if axis == "topp":
        want = tuple(part.strip() for part in coord.split(","))
        if len(want) != 2:
            raise CellRefused(
                f"{coord!r} is not a top-p window; write MIN,MAX")
        matches = [c for c in plan
                   if c[0] == dataset and tuple(c[2]) == want]
        offered = sorted({tuple(c[2]) for c in plan if c[0] == dataset})
    else:
        matches = [c for c in plan
                   if c[0] == dataset and str(c[3]) == coord.strip()]
        offered = sorted({str(c[3]) for c in plan if c[0] == dataset})

    if len(matches) != 1:
        raise CellRefused(
            f"{spec!r} matches {len(matches)} cells; {dataset} offers "
            f"{offered}")
    return matches[0]


def _run_sweep(args, at_topp) -> int:
    try:
        full_plan = sweep_cells(args.sweep, at_topp=at_topp)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
    plan = full_plan
    if args.only:
        # ONE cell, named by its coordinate -- `dataset:MIN,MAX` for a top-p
        # sweep, `dataset:JD` for a lambda sweep. `--only flickr25k` used to
        # select the dataset, i.e. three cells, so "a one-cell smoke" ran three
        # trainers and the description and the act disagreed.
        try:
            plan = [_only_cell(args.only, plan, axis=args.sweep)]
        except CellRefused as error:
            print(f"[phase3] REFUSED --only: {error}", file=sys.stderr)
            return 2

    if args.plan or not (args.run or args.smoke):
        print(f"{len(plan)} cells, sweeping {args.sweep} at the incumbent N")
        for ds, n, topp, jd in plan:
            print(f"  {ds:<10} N={n:<3} topp={topp[0]}/{topp[1]:<5} "
                  f"jd={jd:<5} -> "
                  f"{tag_for(ds, n, namespace=args.namespace, topp=topp, joint=jd)}")
        return 0

    # A namespace is claimed by ONE sweep. Two sweeps sharing it overwrite each
    # other's records and receipt, and the survivor looks complete: the reducer
    # would then be reading a mixture whose parts nothing distinguishes. This is
    # checked before the snapshot, so a collision costs no GPU time.
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    occupied = sorted(
        p.name for p in RECORD_DIR.glob(f"{args.namespace}_*.json"))
    if occupied:
        print(f"[phase3] REFUSED: namespace {args.namespace!r} already holds "
              f"{len(occupied)} artefacts {occupied[:3]}. A namespace belongs "
              f"to one sweep; give this one its own, or move the old ones "
              f"aside.", file=sys.stderr)
        return 2

    epochs = args.epochs if args.smoke else None
    # One snapshot for the whole sweep, taken before the first trainer starts.
    # Every cell is checked against THIS, not against the tree as it was when
    # that cell began -- otherwise a drift introduced between cell 1 and cell 2
    # becomes the new baseline and the sweep silently spans two trees.
    # The snapshot spans the DECLARED sweep, not the subset `--only` narrowed
    # it to: a one-cell run that pins only its own dataset is a new baseline
    # each time, which is what nine `--only` processes produced.
    snapshot = plan_snapshot(sorted({c[0] for c in full_plan}),
                             plan=full_plan, executed=plan, axis=args.sweep,
                             namespace=args.namespace)
    # The snapshot file is named by its OWN digest. Three streams each running
    # `--only` wrote one shared `<namespace>_snapshot.json` and atomically
    # replaced each other's, so the file ended up holding one dataset's inputs
    # and the other two plans were unrecoverable. A content-addressed name
    # cannot be overwritten by a different plan.
    snap_digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    snap_path = RECORD_DIR / f"{args.namespace}_snapshot_{snap_digest[:16]}.json"
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(snap_path, snapshot)
    print(f"[phase3] snapshot {len(snapshot['sources'])} sources, "
          f"{len(snapshot['inputs'])} inputs -> {snap_path}")

    # Datasets run concurrently, one GPU each; the coordinates within a dataset
    # run in sequence. All of it is ONE process against ONE snapshot, because
    # nine separate `--only` invocations each took a fresh snapshot and so could
    # not fail on drift between cells -- the drift simply became the next
    # process's baseline.
    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus
                             else [str(args.gpu)])]
    by_dataset = {}
    for cell in plan:
        by_dataset.setdefault(cell[0], []).append(cell)
    if len(gpus) < len(by_dataset):
        print(f"[phase3] {len(by_dataset)} datasets need {len(by_dataset)} "
              f"GPUs, --gpus gave {len(gpus)}", file=sys.stderr)
        return 2

    import threading
    results, lock = {}, threading.Lock()

    def _stream(dataset, cells, gpu):
        for ds, n, topp, jd in cells:
            key = f"{ds}/{topp}/{jd}"
            try:
                record = run_cell(ds, n, gpu, epochs=epochs,
                                  namespace=args.namespace, topp=topp,
                                  joint=jd, snapshot=snapshot)
            except Exception as error:                 # noqa: BLE001
                with lock:
                    results[key] = ("failed", str(error))
                # The rest of this dataset is abandoned: a stream that keeps
                # going after a refusal is the `|| true` that turned a failed
                # cell into a successful sweep.
                return
            with lock:
                results[key] = ("ok", record)

    threads = [threading.Thread(target=_stream, args=(ds, cells, gpus[i]),
                                daemon=False)
               for i, (ds, cells) in enumerate(sorted(by_dataset.items()))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for key in sorted(results):
        status, payload = results[key]
        if status == "ok":
            sel = payload["selection"]
            print(f"[phase3] {key} ok  mAP@R={sel['selection_value']:.4f} "
                  f"@epoch {sel['selection_epoch_zero_based']}")
        else:
            print(f"[phase3] REFUSED {key}: {payload}", file=sys.stderr)

    done = [k for k, (st, _) in results.items() if st == "ok"]
    if len(done) != len(plan):
        missing = sorted({f"{c[0]}/{c[2]}/{c[3]}" for c in plan} - set(done))
        print(f"[phase3] {len(done)} of {len(plan)} sweep cells complete; "
              f"missing {missing}", file=sys.stderr)
        return 1
    # The receipt seals the EVIDENCE, not just a count. It used to record
    # `key -> tag` and a number, which a reducer cannot use to check that the
    # cells it is reading are the cells that ran.
    _atomic_json(RECORD_DIR / f"{args.namespace}_sweep_complete.json", {
        "schema_version": 2, "axis": args.sweep,
        "namespace": args.namespace,
        "plan_snapshot_sha256": snap_digest,
        "plan_snapshot_file": snap_path.name,
        "expected_cells": len(plan),
        "declared_cells": len(full_plan),
        "cell_count": len(done),
        "cells": {
            k: {
                "tag": results[k][1]["tag"],
                "run_dir": results[k][1]["run_dir"],
                "identity_digest": results[k][1]["geometry"]["identity_digest"],
                "record": f"{results[k][1]['tag']}.json",
                "record_sha256": _sha(RECORD_DIR / f"{results[k][1]['tag']}.json"),
                "recipe": results[k][1]["recipe"],
            }
            for k in sorted(results)},
    })
    print(f"[phase3] {len(plan)} of {len(plan)} sweep cells complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true",
                        help="print the exact cell set and one command")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help=("a short run of the SELECTED cells (use --only "
                              "for one), to prove the plumbing and the "
                              "assertions. Never a candidate score."))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default=None, metavar="A,B,C",
                        help=("one GPU per dataset for a sweep; the sweep runs "
                              "them concurrently in ONE process against ONE "
                              "snapshot"))
    parser.add_argument("--epochs", type=int, default=1,
                        help="--smoke only")
    parser.add_argument("--only", default=None,
                        help="dataset:N, e.g. cifar10:4")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument(
        "--refit", action="store_true",
        help=("run the 12 scratch refits at the N chosen by "
              "scripts/phase3_select_n.py, for seeds 42/43/44"))
    parser.add_argument("--selection", default=str(
        REPO / "artifacts" / "phase3_selection" / "selected_n.json"))
    parser.add_argument(
        "--sweep", choices=("topp", "joint"), default=None,
        help=("sweep a RECIPE axis instead of N, at the incumbent N for each "
              "dataset. The cells are ordinary stage-1 selection cells: same "
              "90/10 train split, same optTrain whitening, no official test."))
    parser.add_argument(
        "--recipe", default=None, metavar="PATH", action="append",
        help=("a selected_topp.json / selected_joint.json from "
              "scripts/phase3_select_n.py --axis. The N matrix and the refit "
              "read the chosen coordinates from it instead of being told them "
              "again by hand, which is how the ablation chain came to run at "
              "an N nobody reported. Repeatable: pass the top-p artefact AND "
              "the lambda artefact, because a single file carries one axis and "
              "handing over only one silently returns the other to its "
              "default."))
    parser.add_argument(
        "--at-topp", default=None, metavar="MIN,MAX",
        help="hold top-p here (required by --sweep joint)")
    args = parser.parse_args()

    recipe = {}
    if args.recipe:
        try:
            for one in args.recipe:
                for dataset, axes in _load_recipe(Path(one)).items():
                    overlap = set(axes) & set(recipe.get(dataset, {}))
                    if overlap:
                        raise CellRefused(
                            f"two --recipe files both choose {sorted(overlap)} "
                            f"for {dataset}; one axis, one artefact")
                    recipe.setdefault(dataset, {}).update(axes)
        except CellRefused as error:
            print(f"[phase3] REFUSED --recipe: {error}", file=sys.stderr)
            return 2

    sweep_topp = None
    if args.at_topp:
        try:
            lo, hi = args.at_topp.split(",")
            sweep_topp = (lo.strip(), hi.strip())
        except ValueError:
            print(f"[phase3] --at-topp must be MIN,MAX, got {args.at_topp!r}",
                  file=sys.stderr)
            return 2
        if sweep_topp not in TOPP_GRID:
            print(f"[phase3] --at-topp {sweep_topp} is not one of the swept "
                  f"windows {list(TOPP_GRID)}", file=sys.stderr)
            return 2

    if args.sweep:
        return _run_sweep(args, sweep_topp)

    keys = cell_keys()
    if args.only:
        try:
            ds, raw = args.only.split(":")
            key = (ds, int(raw))
        except ValueError:
            print(f"[phase3] --only must be dataset:N, got {args.only!r}",
                  file=sys.stderr)
            return 2
        # "Exactly sixteen cells" has to bind what actually runs, not only what
        # `--plan` prints: `--only cifar10:5` used to build an N=5 command.
        if key not in cell_keys():
            print(f"[phase3] {args.only} is not one of the {len(keys)} cells; "
                  f"N must be one of {list(CANDIDATE_N)} and the dataset one "
                  f"of {sorted(DATASETS)}", file=sys.stderr)
            return 2
        keys = [key]

    if args.refit and not (args.run or args.smoke):
        print("[phase3] --refit runs cells, so pass --run (or --smoke) with "
              "it; on its own it used to print the selection plan and exit 0",
              file=sys.stderr)
        return 2

    if args.plan or not (args.run or args.smoke):
        print(f"{len(cell_keys())} cells: "
              f"{sorted({d for d, _ in cell_keys()})} x {list(CANDIDATE_N)}")
        print(f"geometry M={SLOTS} L={BASES_PER_SLOT} "
              f"bases={TOTAL_BASES} bits={TOTAL_BITS}")
        print(f"stage 1: -e {EPOCH_BUDGET} --stop_after_epoch N "
              f"--lr_schedule_horizon {LR_HORIZON} "
              f"--sinkhorn_schedule_horizon N+1")
        ds, n = keys[0]
        cmd, env, tag = build_command(ds, n, args.gpu, namespace=args.namespace)
        print(f"\nexample cell {ds}/N{n} -> {tag}")
        print("  " + " ".join(cmd))
        for key in ("GDNA_NUM_SEMANTIC_PARTS", "CACHE", "EVAL_CACHE",
                    "WHITEN_NPZ", "STOP_EP", "VAL_RATIO", "EXTRA_ARGS"):
            print(f"  {key}={env[key]}")
        return 0

    epochs = args.epochs if args.smoke else None
    snapshot = plan_snapshot(sorted({d for d, _ in keys}))

    if args.refit:
        # Narrowing is for smoking the plumbing, never for producing the
        # twelve: a partial run that reported success would be a matrix nobody
        # completed. Checked before anything is read, so an argument error is
        # reported as one.
        if args.only and not args.smoke:
            print("[phase3] --only narrows the refit to one dataset, which "
                  "cannot produce the twelve cells; use it with --smoke",
                  file=sys.stderr)
            return 2
        chosen = _load_selection(Path(args.selection))
        plan = [(ds, chosen[ds], seed)
                for ds in sorted(DATASETS) for seed in REFIT_SEEDS]
        if args.only:
            plan = [p for p in plan if p[0] == keys[0][0]]
        failures = []
        for ds, n, seed in plan:
            try:
                record = run_cell(ds, n, args.gpu, epochs=epochs,
                                  namespace=args.namespace, stage="refit",
                                  seed=seed,
                                  topp=recipe.get(ds, {}).get("topp"),
                                  joint=recipe.get(ds, {}).get("joint"),
                                  snapshot=snapshot)
            except CellRefused as error:
                print(f"[phase3] REFUSED refit {ds}/N{n}/s{seed}: {error}",
                      file=sys.stderr)
                failures.append(f"{ds}/N{n}/s{seed}")
                continue
            print(f"[phase3] refit {ds}/N{n}/s{seed} ok -> {record['run_dir']}")
        if failures:
            print(f"[phase3] {len(failures)} of {len(plan)} refits refused",
                  file=sys.stderr)
            return 1
        print(f"[phase3] {len(plan)} of {len(plan)} refits complete")
        return 0

    failures = []
    for ds, n in keys:
        try:
            record = run_cell(ds, n, args.gpu, epochs=epochs,
                              namespace=args.namespace,
                              topp=recipe.get(ds, {}).get("topp"),
                              joint=recipe.get(ds, {}).get("joint"),
                              snapshot=snapshot)
        except CellRefused as error:
            print(f"[phase3] REFUSED {ds}/N{n}: {error}", file=sys.stderr)
            failures.append(f"{ds}/N{n}")
            continue
        print(f"[phase3] {ds}/N{n} ok  bases={record['geometry']}  "
              f"mAP@R={record['selection']['selection_value']:.4f} "
              f"@epoch {record['selection']['selection_epoch_zero_based']}")

    if failures:
        print(f"[phase3] {len(failures)} of {len(keys)} cells refused: "
              f"{failures}", file=sys.stderr)
        return 1
    print(f"[phase3] {len(keys)} of {len(keys)} cells complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
