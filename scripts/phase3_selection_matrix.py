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
    "dna_utils/run_identity.py",
)


def _sha(path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_digests() -> dict:
    return {rel: _sha(REPO / rel) for rel in PROTOCOL_SOURCES}


class CellRefused(RuntimeError):
    """A cell cannot be run, or its output cannot be admitted."""


def cell_keys() -> list:
    """The exact 16 keys. No more, no fewer, in a fixed order."""
    return [(ds, n) for ds in ("cifar10", "flickr25k", "nuswide", "mscoco")
            for n in CANDIDATE_N]


def tag_for(dataset: str, n: int, *, namespace: str = NAMESPACE) -> str:
    return f"{namespace}_{DATASETS[dataset]['exp']}_N{n}_s{SEED}"


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
                  namespace: str = NAMESPACE) -> str:
    return f"{namespace}_{DATASETS[dataset]['exp']}_refit_N{n}_s{seed}"


def build_command(dataset: str, n: int, gpu: int, *,
                  epochs: int | None = None,
                  namespace: str = NAMESPACE,
                  stage: str = "select", seed: int = SEED) -> tuple:
    spec = DATASETS[dataset]
    if stage == "refit":
        tag = refit_tag_for(dataset, n, seed, namespace=namespace)
        flags = _refit_flags(n, seed) + S5_FLAGS + A_FLAGS + QUIET_FLAGS
    else:
        tag = tag_for(dataset, n, namespace=namespace)
        flags = _stage1_flags(n) + S5_FLAGS + A_FLAGS + QUIET_FLAGS
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


# --------------------------------------------------------------- checking

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
                    sinkhorn_horizon: int | None = None) -> dict:
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
    effective = {
        "num_semantic_parts": int(_arg_value(args_txt, "num_semantic_parts")),
        "num_codebooks": int(_arg_value(args_txt, "num_codebooks")),
        "num_codons_per_codebook": int(
            _arg_value(args_txt, "num_codons_per_codebook")),
    }
    want = {"num_semantic_parts": SLOTS, "num_codebooks": SLOTS,
            "num_codons_per_codebook": BASES_PER_SLOT}
    bad = {k: (v, want[k]) for k, v in effective.items() if v != want[k]}
    if bad:
        raise CellRefused(
            f"{run_dir}: effective geometry is wrong {bad} -- this is the "
            f"18-base failure, not the paper's {TOTAL_BASES}-base cell")

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
    return {"final_checkpoint_sha256": actual,
            "final_checkpoint_epoch_zero_based": recorded_epoch,
            "terminal_weights_preserved": recorded_epoch == terminal_epoch,
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


def run_cell(dataset: str, n: int, gpu: int, *, epochs=None,
             namespace: str = NAMESPACE, stage: str = "select",
             seed: int = SEED) -> dict:
    tag = (refit_tag_for(dataset, n, seed, namespace=namespace)
           if stage == "refit" else tag_for(dataset, n, namespace=namespace))
    existing = _existing_artifacts(tag)
    if existing:
        raise CellRefused(
            f"{tag} already has artefacts: {existing[:3]}. Give the run its own "
            f"namespace with PHASE3_NAMESPACE.")

    cmd, env, _ = build_command(dataset, n, gpu, epochs=epochs,
                                namespace=namespace, stage=stage, seed=seed)
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

    run_dir = _resolve(tag)
    record = {
        "schema_version": RECORD_SCHEMA,
        "namespace": namespace,
        "matrix": {"candidate_n": list(CANDIDATE_N),
                   "datasets": sorted(DATASETS),
                   "cells": len(cell_keys())},
        "protocol_sources": protocol_digests(),
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
            budget=_budget, lr_horizon=_lr_h, sinkhorn_horizon=_sk_h),
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
    args = parser.parse_args()

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
                                  seed=seed)
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
                              namespace=args.namespace)
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
