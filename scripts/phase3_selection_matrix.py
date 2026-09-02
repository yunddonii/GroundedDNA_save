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
M=5, L=3, 15 bases, 30 bits). The selection value is read from the terminal
epoch row in ``log.csv``; best-checkpoint runtime metadata is diagnostic
only. Its sidecar records `selection_metric` and `selection_value` next to the
checkpoint's own SHA, but it is not the selection authority.

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
import math
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# All local Python/shell bytes reachable from the Phase-3 entrypoints.  Keep
# this tuple above the first local import: the production entrypoint hashes it,
# self-execs, and verifies the same bundle again *before* Python can import one
# set of bytes and later snapshot another.  ``dna_utils.__init__`` imports its
# helpers eagerly, hence the apparently broad dna_utils closure.
_BOOTSTRAP_SOURCE_PATHS = (
    "config.py", "train_siglip2.py", "model_siglip2.py", "loss_siglip2.py",
    "dataloaders.py", "extraction_siglip2.py", "evaluation_siglip2.py",
    "val_split.py", "p0_protocol.py",
    "dna_utils/__init__.py", "dna_utils/run_identity.py",
    "dna_utils/extraction_validation.py", "dna_utils/cache_provenance.py",
    "dna_utils/bio_constraints.py", "dna_utils/csv_logger.py",
    "dna_utils/dna_code_utils.py",
    "dna_utils/text_description_processor.py",
    "dna_utils/training_utils.py", "dna_utils/visualization.py",
    "dna_utils/vlm_qwen25_descriptions.py", "dna_utils/runtime_state.py",
    "dna_utils/runtime_environment.py", "dna_utils/gpu_lease.py",
    "dna_utils/gc_policy.py",
    "models/adapters.py", "models/cluster_attention_router.py",
    "models/__init__.py",
    "models/pretrained_backbone.py", "models/pretrained_backbone_clip.py",
    "models/semantic_router.py", "models/text_cross_attention_router.py",
    "models/text_encoder.py", "models/visual_encoder.py",
    "scripts/__init__.py", "scripts/phase3_selection_matrix.py",
    "scripts/phase3_select_n.py",
    "scripts/seal_phase3_inputs.py",
    "scripts/build_text_whiten_matrix.py",
    "scripts/extract_train_split.py", "scripts/eval_cell_bioproj.py",
    "scripts/pairwise_nmi.py", "scripts/seal_cell_analysis.py",
    "scripts/phase3_launch_matrix.sh",
    "scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
    "scripts/train_nuswide_v185_sweep_clip.sh",
    "scripts/train_mscoco_F2_sweep_clip.sh",
)
_BOOTSTRAP_ENV = "GDNA_PHASE3_PREIMPORT_SOURCE_BUNDLE"


def _stdlib_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except FileNotFoundError:
        return "#absent"
    return digest.hexdigest()


def _source_authority_bundle(paths=_BOOTSTRAP_SOURCE_PATHS) -> dict:
    """HEAD blob and worktree hashes for the exact executable source closure."""
    head_proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=str(REPO),
        text=True, capture_output=True, check=False)
    head = head_proc.stdout.strip() if head_proc.returncode == 0 else "#no-head"
    entries = {}
    for rel in sorted(set(paths)):
        oid_proc = subprocess.run(
            ["git", "rev-parse", f"HEAD:{rel}"], cwd=str(REPO),
            text=True, capture_output=True, check=False)
        tracked = oid_proc.returncode == 0
        head_oid = oid_proc.stdout.strip() if tracked else "#untracked"
        if tracked:
            blob_proc = subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=str(REPO),
                capture_output=True, check=False)
            head_sha = (hashlib.sha256(blob_proc.stdout).hexdigest()
                        if blob_proc.returncode == 0 else "#unreadable")
        else:
            head_sha = "#untracked"
        worktree_sha = _stdlib_file_sha256(REPO / rel)
        entries[rel] = {
            "tracked": tracked,
            "head_blob_oid": head_oid,
            "head_sha256": head_sha,
            "worktree_sha256": worktree_sha,
            "clean": bool(tracked and head_sha == worktree_sha),
        }
    return {"schema_version": 1, "head_commit": head, "entries": entries}


def _semantic_digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# Only the executable entrypoint self-execs. Imports used by tests/tools still
# capture a bundle, but production admission below refuses them because no
# pre-import handshake was proven.
_BOOTSTRAP_PREIMPORT_BUNDLE = _source_authority_bundle()
_BOOTSTRAP_PREIMPORT_VERIFIED = False
if __name__ == "__main__":
    _sealed = os.environ.get(_BOOTSTRAP_ENV)
    if _sealed is None:
        _env = dict(os.environ)
        # A caller-controlled PYTHONPATH can shadow both local and third-party
        # imports before they enter any source/package digest.  The verified
        # child starts with Python's ordinary script-directory/sys.path rules.
        _env.pop("PYTHONPATH", None)
        _env[_BOOTSTRAP_ENV] = json.dumps(
            _BOOTSTRAP_PREIMPORT_BUNDLE, sort_keys=True, separators=(",", ":"))
        os.execve(sys.executable, [sys.executable, *sys.argv], _env)
    if os.environ.get("PYTHONPATH"):
        print("[phase3] REFUSED: PYTHONPATH must be unset at the verified "
              "pre-import boundary", file=sys.stderr)
        raise SystemExit(2)
    try:
        _expected_preimport = json.loads(_sealed)
    except (TypeError, ValueError):
        print("[phase3] REFUSED: malformed pre-import source bundle",
              file=sys.stderr)
        raise SystemExit(2)
    if _expected_preimport != _BOOTSTRAP_PREIMPORT_BUNDLE:
        print("[phase3] REFUSED: protocol sources changed across the "
              "pre-import self-exec boundary", file=sys.stderr)
        raise SystemExit(2)
    _BOOTSTRAP_PREIMPORT_VERIFIED = True

from dna_utils.run_identity import (  # noqa: E402
    MANIFEST_NAME, RunIdentity, load_run_manifest)
from dna_utils.runtime_environment import (  # noqa: E402
    EnvironmentAttestationError, capture_parent_environment,
    expected_child_environment)

PY = os.environ.get("PY", "/home/yschoi/.conda/envs/dna_hashing/bin/python")
CACHE_ROOT = os.environ.get(
    "GDNA_CACHE_ROOT", "/data/yschoi/groundeddna_cache_v6prov")
_QWEN_ROOT_RAW = os.environ.get(
    "GDNA_QWEN_ROOT", "/data/yschoi/dataset/deephashing/cache")
if not Path(_QWEN_ROOT_RAW).expanduser().is_absolute():
    raise RuntimeError(
        f"GDNA_QWEN_ROOT must be absolute, got {_QWEN_ROOT_RAW!r}")
QWEN_ROOT = Path(_QWEN_ROOT_RAW).expanduser().resolve()

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

PHASE3_CLIP_CHECKPOINT = "openai/clip-vit-base-patch16"
PHASE3_CLIP_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
PHASE3_CLIP_WEIGHT_SHA256 = (
    "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
)

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
    "--text_hash_counterfactual_weight", "0.0",
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
        qwen=str(QWEN_ROOT / "cifar10_qwen_v4.jsonl"),
        K=64, env=dict(CIBNT="1.0", CCS="0.1")),
    "flickr25k": dict(
        canon="Flickr25k", exp="flickr_A_v4",
        trainer="scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
        cache=f"{CACHE_ROOT}/flickr25k_clip_tokens",
        foils=f"{CACHE_ROOT}/flickr25k_clip_tokens_foils",
        qwen=str(QWEN_ROOT / "flickr25k_qwen3_v4_trainset.jsonl"),
        K=128, env=dict(CIBNT="1.0", BIDIR_MODE="legacy")),
    "nuswide": dict(
        canon="NUSWIDE", exp="nuswide_A_v4",
        trainer="scripts/train_nuswide_v185_sweep_clip.sh",
        cache=f"{CACHE_ROOT}/nuswide_clip_tokens",
        foils=f"{CACHE_ROOT}/nuswide_clip_tokens_foils",
        qwen=str(QWEN_ROOT / "nuswide_qwen3_v4_trainset.jsonl"),
        K=128, env=dict(CIBNT="1.5", CELL="Aprompt")),
    "mscoco": dict(
        canon="MSCOCO", exp="mscoco_A_v5b",
        trainer="scripts/train_mscoco_F2_sweep_clip.sh",
        cache=f"{CACHE_ROOT}/mscoco_clip_tokens",
        foils=f"{CACHE_ROOT}/mscoco_clip_tokens_foils",
        qwen=str(QWEN_ROOT / "mscoco_qwen3_v5b_trainset.jsonl"),
        K=128, env=dict(CIBNT="1.5", CELL="Aprompt")),
}

# Exact setting-1 cardinalities.  MAIN numbers are meaningful only for these
# published query/database populations and the designated training split used
# by train-only analyses.
PAPER_SPLIT_ROWS = {
    "cifar10": {"db": 59000, "query": 1000, "train": 5000},
    "flickr25k": {"db": 23000, "query": 2000, "train": 5000},
    "nuswide": {"db": 193734, "query": 2100, "train": 10500},
    "mscoco": {"db": 107218, "query": 5000, "train": 10000},
}
MAP_R_CUTOFF = {
    "cifar10": 1000, "flickr25k": 5000,
    "nuswide": 5000, "mscoco": 5000,
}

#: Kept from the caller: the things a shell needs to run at all, plus the
#: cache root override the tests use. Everything else is set by this launcher.
_ENV_PASSTHROUGH = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "PYTHONUNBUFFERED", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "GDNA_CACHE_ROOT", "GDNA_QWEN_ROOT", "PHASE3_NAMESPACE", "PY",
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
SWEEP_SNAPSHOT_SCHEMA = 4
SWEEP_RECEIPT_SCHEMA = 4
CAMPAIGN_RESERVATION_SCHEMA = 1
CAMPAIGN_RESERVATION_SUFFIX = "_campaign_reservation.json"
N_SELECTION_RECEIPT_SUFFIX = "_n_selection_complete.json"
REFIT_RECEIPT_SUFFIX = "_refit_complete.json"
RECIPE_RECEIPT_SUFFIX = "_sweep_complete.json"
RECIPE_AUTHORITY_SCHEMA = 2

#: Whether the campaign REHASHES every sealed input byte one last time before
#: it writes its receipt.  It does not, and the admission pass is still
#: `full=True`.
#:
#: The seals bind 452 GB for the three swept datasets and 526 GB for all four.
#: At the 115 MB/s this host actually sustains on `/data`, one pass is 65-76
#: minutes, so a second one costs about as much as every trainer in the
#: campaign put together.  It buys nothing: `verify_snapshot` already runs the
#: stats-only recheck around EVERY cell (see its `full=False` call), so by the
#: time control reaches the receipt the last cell's own recheck has already
#: covered the same inputs.  A second full rehash can only catch a mutation
#: that (a) landed after that recheck and (b) preserved every file's lstat,
#: link text, resolved target stat and directory inventory exactly -- which is
#: deliberate forgery, not the drift this bookend exists to catch.
#:
#: `verify_seal_stats` is written for exactly this: "A campaign should call
#: verify_seal once at admission, retain its aggregate digest, and pass that
#: value here before and after cells."  The admission digest is what the
#: stats path is checked against, so a swapped or truncated seal is still
#: refused here.
FINAL_BOOKEND_FULL = False
SELECTED_N_AUTHORITY_SCHEMA = 2

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


def _json_digest(payload: dict) -> str:
    """Digest the semantic JSON value, independent of pretty-printing."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _seal_stage(stage: str) -> str:
    return "refit" if stage == "refit" else "stage1"


def parse_input_seal_specs(values) -> dict:
    """Parse repeatable DATASET:STAGE=/absolute/seal.json declarations."""
    result = {}
    for raw in values or ():
        try:
            key, path = raw.split("=", 1)
            dataset, stage = key.split(":", 1)
        except ValueError:
            raise CellRefused(
                f"--input-seal must be DATASET:STAGE=ABS_PATH, got {raw!r}"
            ) from None
        if dataset not in DATASETS or stage not in ("stage1", "refit"):
            raise CellRefused(f"unsupported input-seal coordinate {key!r}")
        seal_path = Path(path).expanduser()
        if not seal_path.is_absolute():
            raise CellRefused(f"input seal path must be absolute: {path!r}")
        coord = (dataset, stage)
        if coord in result:
            raise CellRefused(f"duplicate input seal declaration for {key}")
        result[coord] = str(seal_path.resolve())
    return result


def verify_campaign_input_seals(specs: dict, plan: list, *, full: bool,
                                expected: dict | None = None) -> dict:
    """Verify exactly the dataset/stage seals consumed by one plan."""
    from scripts.seal_phase3_inputs import SealError, verify_seal_authority

    required = {(_campaign_cell_parts(cell)[0],
                 _seal_stage(_campaign_cell_parts(cell)[4])) for cell in plan}
    if set(specs) != required:
        raise CellRefused(
            "input seal declarations are not the exact campaign coordinate set: "
            f"missing={sorted(required-set(specs))}, extra={sorted(set(specs)-required)}"
        )
    authorities = {}
    for coord in sorted(required):
        key = f"{coord[0]}:{coord[1]}"
        try:
            authority = verify_seal_authority(
                specs[coord], expected=(expected or {}).get(key), full=full
            )
        except SealError as error:
            raise CellRefused(f"input seal {key} refused: {error}") from None
        if (authority.get("dataset"), authority.get("stage")) != coord:
            raise CellRefused(
                f"input seal {key} contains {authority.get('dataset')}:"
                f"{authority.get('stage')}"
            )
        hf = authority.get("hf_runtime") or {}
        if hf.get("checkpoint") != PHASE3_CLIP_CHECKPOINT \
                or hf.get("revision") != PHASE3_CLIP_REVISION \
                or hf.get("weight_sha256") != PHASE3_CLIP_WEIGHT_SHA256 \
                or hf.get("local_files_only") is not True:
            raise CellRefused(
                f"input seal {key} is not the canonical immutable Phase-3 CLIP snapshot"
            )
        authorities[key] = authority
    return authorities


def verify_snapshot_input_seals(snapshot: dict, *, full: bool) -> None:
    sealed = snapshot.get("input_seals")
    if not isinstance(sealed, dict) or not sealed:
        raise CellRefused("snapshot has no verified dataset/stage input seals")
    specs = {}
    for key, authority in sealed.items():
        try:
            dataset, stage = key.split(":", 1)
        except ValueError:
            raise CellRefused(f"malformed snapshot input-seal key {key!r}") from None
        specs[(dataset, stage)] = authority.get("seal_path")
    plan_rows = []
    for cell in ((snapshot.get("plan") or {}).get("declared_cells") or []):
        plan_rows.append((cell["dataset"], cell["N"], cell["topp"],
                          cell["joint"], cell["stage"], cell["seed"]))
    verified = verify_campaign_input_seals(
        specs, plan_rows, full=full, expected=sealed
    )
    if verified != sealed:
        raise CellRefused("snapshot input-seal authority changed")


def environment_fingerprint(gpu_ids=()) -> dict:
    """Read-only runtime provenance used by plan, child binding and reducer."""
    return capture_parent_environment(gpu_ids)


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8").strip()
    except OSError:
        return ""


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


def _canonical_result_root(value=None) -> Path:
    root = Path(value if value is not None else REPO / "result").expanduser()
    if not root.is_absolute():
        raise CellRefused(
            f"--result-root must be an absolute path, got {str(root)!r}")
    return root.resolve()


def _existing_artifacts(tag: str, *, result_root=None) -> list:
    root = _canonical_result_root(result_root)
    hits = [str(p) for p in root.glob(f"*{tag}*")]
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
    env.pop("PYTHONPATH", None)
    env["PY"] = os.path.realpath(sys.executable)
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


def _bind_campaign_gpu_selector(
        cmd: list[str], env: dict, *, gpu: int,
        campaign_binding: dict) -> list[str]:
    """Make the trainer shell and its Python child use one sealed GPU UUID."""
    physical_uuid = str(
        campaign_binding["expected_child_environment"]["physical_gpu"]["uuid"])
    env["CUDA_VISIBLE_DEVICES"] = physical_uuid
    if len(cmd) < 3 or cmd[-1] != str(gpu):
        raise CellRefused(
            "trainer command does not expose the planned GPU selector as its "
            "final positional argument")
    return [*cmd[:-1], physical_uuid]


_CHILDREN_LOCK = threading.RLock()
_ACTIVE_CHILDREN: set[subprocess.Popen] = set()
_CHILD_LAUNCH_BLOCKED = False
_CAMPAIGN_LEASE_FDS: tuple[int, ...] = ()


def _run_managed_process(command, *, cwd, env) -> subprocess.CompletedProcess:
    """Run a campaign child in a registered, independently killable group."""
    global _ACTIVE_CHILDREN
    # Python handlers always execute in the main thread.  Popen from that same
    # thread would leave a re-entrant signal window between kernel spawn and
    # registry insertion. Production streams already run every trainer and
    # postprocess in worker threads; make that invariant executable so the
    # handler must wait on this lock until the new PID/PGID is registered.
    if threading.current_thread() is threading.main_thread():
        raise CellRefused(
            "managed campaign children must launch from a worker thread")
    with _CHILDREN_LOCK:
        if _CHILD_LAUNCH_BLOCKED:
            raise CellRefused(
                "campaign shutdown has begun; refusing a new child process")
        proc = subprocess.Popen(
            command, cwd=cwd, env=env, start_new_session=True,
            # The inherited flock descriptions keep every campaign GPU leased
            # even if the launcher itself is SIGKILLed while a trainer lives.
            pass_fds=_CAMPAIGN_LEASE_FDS)
        _ACTIVE_CHILDREN.add(proc)
    try:
        return subprocess.CompletedProcess(command, proc.wait())
    finally:
        with _CHILDREN_LOCK:
            _ACTIVE_CHILDREN.discard(proc)


def _terminate_active_children(*, grace_seconds: float = 5.0) -> None:
    """Stop every live campaign process group, TERM then bounded KILL."""
    global _CHILD_LAUNCH_BLOCKED
    with _CHILDREN_LOCK:
        _CHILD_LAUNCH_BLOCKED = True
        children = tuple(_ACTIVE_CHILDREN)
    for proc in children:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    pending = [proc for proc in children if proc.poll() is None]
    while pending and time.monotonic() < deadline:
        # A short bounded poll keeps signal cleanup responsive without a
        # blocking multi-second sleep.
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        pending = [proc for proc in pending if proc.poll() is None]
    for proc in pending:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in children:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"campaign child process group {proc.pid} survived SIGKILL")
    with _CHILDREN_LOCK:
        for proc in children:
            if proc.poll() is not None:
                _ACTIVE_CHILDREN.discard(proc)


def _campaign_signal_handler(signum, _frame) -> None:
    _terminate_active_children()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + int(signum))


def _with_campaign_gpu_leases(args, callback):
    """Hold the shared host-global physical-UUID leases for one campaign."""
    global _CAMPAIGN_LEASE_FDS, _CHILD_LAUNCH_BLOCKED
    from dna_utils.gpu_lease import acquire_gpu_leases

    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus
                             else [str(args.gpu)])]
    environment = environment_fingerprint(gpus)
    if environment.get("errors"):
        print("[phase3] REFUSED GPU lease environment: "
              f"{environment['errors']}", file=sys.stderr)
        return 2
    assignments = environment.get("selected_gpus") or []
    if len(assignments) != len(gpus):
        print("[phase3] REFUSED GPU lease: parent inventory does not contain "
              "every requested physical GPU exactly once", file=sys.stderr)
        return 2
    try:
        leases = acquire_gpu_leases(
            assignments,
            owner_metadata={
                "campaign": "phase3_selection",
                "namespace": str(args.namespace),
                "repo": str(REPO),
            })
    except RuntimeError as error:
        print(f"[phase3] REFUSED GPU lease: {error}", file=sys.stderr)
        return 2
    if threading.current_thread() is not threading.main_thread():
        leases.release()
        print("[phase3] REFUSED GPU lease: campaign signal supervision must "
              "start in the main thread", file=sys.stderr)
        return 2
    evidence = tuple(sorted({str(row["uuid"]) for row in assignments}))
    args._phase3_gpu_lease_uuids = evidence
    with _CHILDREN_LOCK:
        if _ACTIVE_CHILDREN or _CAMPAIGN_LEASE_FDS:
            leases.release()
            del args._phase3_gpu_lease_uuids
            raise RuntimeError("another managed campaign is active in this process")
        _CHILD_LAUNCH_BLOCKED = False
        _CAMPAIGN_LEASE_FDS = tuple(
            int(handle.fileno()) for handle in leases.handles)
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous_handlers:
        signal.signal(signum, _campaign_signal_handler)
    try:
        return callback()
    finally:
        # This ordering is the contract: no live/orphan child may outlast the
        # host-global lease held on its physical GPU.
        _terminate_active_children()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        with _CHILDREN_LOCK:
            _CAMPAIGN_LEASE_FDS = ()
            _CHILD_LAUNCH_BLOCKED = False
        leases.release()
        del args._phase3_gpu_lease_uuids


def _assert_snapshot_gpu_leases(snapshot: dict, args) -> None:
    """The lock keys must be the same physical UUIDs the plan attests."""
    held = getattr(args, "_phase3_gpu_lease_uuids", None)
    # Internal smoke/unit callers cannot claim production admission because
    # they lack the pre-import handshake.  The executable entrypoint and every
    # wrapper-mediated call must carry live lease evidence.
    if held is None and not _BOOTSTRAP_PREIMPORT_VERIFIED:
        return
    planned = tuple(sorted(
        str(row.get("uuid"))
        for row in (snapshot.get("environment") or {}).get(
            "selected_gpus", ())))
    if held is None or tuple(held) != planned:
        raise CellRefused(
            "campaign does not hold the shared physical-GPU UUID leases "
            f"sealed by its snapshot: held={held}, planned={planned}")


# Values in RunIdentity that are fixed by each canonical trainer shell rather
# than by the generic sweep flags.  Keeping this small is deliberate: these are
# exactly the RunIdentity fields whose dataset-specific values differ.  The
# trainer shell itself is in the plan's source digest set, and the child refuses
# before training if the identity it actually parses differs from this expected
# one.
_IDENTITY_PROFILE = {
    "cifar10": dict(eps_init=1.0, text_code_kl=0.05,
                    text_hash_ntxent=0.05, xmodal_commit=0.05,
                    # S5_FLAGS is appended after the trainer's historical
                    # CCS=0.1 default, so argparse's last value is 0.0.
                    codeword_codon_sinkhorn=0.0),
    "flickr25k": dict(eps_init=1.0, text_code_kl=0.05,
                      text_hash_ntxent=0.05, xmodal_commit=0.05,
                      codeword_codon_sinkhorn=0.0),
    "nuswide": dict(eps_init=0.5, text_code_kl=0.05,
                    text_hash_ntxent=0.05, xmodal_commit=0.05,
                    codeword_codon_sinkhorn=0.0),
    "mscoco": dict(eps_init=1.0, text_code_kl=0.10,
                   text_hash_ntxent=0.10, xmodal_commit=0.10,
                   codeword_codon_sinkhorn=0.0),
}


def expected_run_identity(dataset: str, n: int, *, stage: str = "select",
                          seed: int = SEED, topp=None, joint=None,
                          epochs: int | None = None,
                          input_authority: dict | None = None) -> RunIdentity:
    """Compute the identity a canonical child must parse before it is started.

    This is an admission value, not a reconstruction after the fact.  The
    launcher seals it in the plan, passes its digest to the trainer, and the
    trainer compares it with ``RunIdentity.from_args(actual_args)`` before
    claiming a result directory or touching a GPU.
    """
    from types import SimpleNamespace

    spec = DATASETS[dataset]
    profile = _IDENTITY_PROFILE[dataset]
    topp = TOPP_INCUMBENT if topp is None else tuple(topp)
    joint = JOINT_INCUMBENT if joint is None else str(joint)
    if stage == "refit":
        budget = n + 1
        stop = n
        lr_horizon = budget
        sinkhorn_horizon = budget
        val_ratio = 0.0
        whitening = _whitening(spec, stage="refit")
    else:
        budget = EPOCH_BUDGET
        stop = n
        lr_horizon = LR_HORIZON
        sinkhorn_horizon = n + 1
        val_ratio = VAL_RATIO
        whitening = _whitening(spec, stage="select")
    if epochs is not None:
        budget = int(epochs)
        stop = max(budget - 1, 0)
        lr_horizon = budget
        sinkhorn_horizon = stop + 1

    authority = input_authority or {}
    hf = authority.get("hf_runtime") or {}
    tokenizer_json = (json.dumps(
        hf.get("tokenizer_files_sha256") or {},
        sort_keys=True, separators=(",", ":")) if authority else "")
    return RunIdentity.from_args(SimpleNamespace(
        dataset=spec["canon"], setting="setting1", random_seed=seed,
        epoch=budget, stop_after_epoch=stop,
        num_semantic_parts=SLOTS,
        num_codons_per_codebook=BASES_PER_SLOT,
        codebook_size=spec["K"], selection_mode=stage,
        val_split_ratio=val_ratio, val_split_seed=VAL_SEED,
        lr_schedule_horizon=lr_horizon,
        sinkhorn_schedule_horizon=sinkhorn_horizon,
        siglip2_feature_cache_dir=spec["cache"],
        eval_cache_dir=spec["cache"],
        qwen_text_cache_path=str(Path(spec["qwen"]).resolve()),
        text_whiten_npz=whitening,
        sinkhorn_epsilon_init=profile["eps_init"],
        sinkhorn_epsilon_final=0.1,
        batch_size=64, proj_lr=1e-3,
        routing_adaptive_topp=True,
        no_routing_adaptive_topp=False,
        routing_adaptive_topp_min=float(topp[0]),
        routing_adaptive_topp_max=float(topp[1]),
        routing_adaptive_topp_entropy=False,
        routing_perplexity_topk=False,
        codon_joint_slots="", codon_joint_floor=1e-6,
        share_codebook=False, disable_text_supervision=False,
        use_gumbel_softmax=False,
        lambda_codon_joint=float(joint),
        lambda_text_code_kl=profile["text_code_kl"],
        lambda_text_hash_ntxent=profile["text_hash_ntxent"],
        lambda_xmodal_commit=profile["xmodal_commit"],
        lambda_codeword_codon_sinkhorn=profile["codeword_codon_sinkhorn"],
        phase3_input_seal=authority.get("seal_path"),
        phase3_input_aggregate_sha256=authority.get("aggregate_sha256", ""),
        phase3_split_identity_sha256=authority.get(
            "split_identity_sha256", ""),
        phase3_hf_identity_sha256=hf.get("identity_sha256", ""),
        clip_snapshot_dir=hf.get("snapshot_dir", ""),
        clip_snapshot_revision=hf.get("revision", ""),
        clip_snapshot_weight_file=hf.get("weight_file", ""),
        clip_snapshot_weight_sha256=hf.get("weight_sha256", ""),
        clip_snapshot_config_sha256=hf.get("config_sha256", ""),
        clip_snapshot_tokenizers_sha256_json=tokenizer_json,
    ))


def campaign_cell_id(dataset: str, n: int, *, topp, joint,
                     stage: str = "select", seed: int = SEED) -> str:
    """Stable, human-readable primary key for one planned recipe cell."""
    return (f"{dataset}|N={int(n)}|P={str(topp[0])},{str(topp[1])}|"
            f"JD={str(joint)}|stage={stage}|seed={int(seed)}")


def expected_cell_binding(dataset: str, n: int, *, namespace: str,
                          campaign_nonce: str, topp, joint,
                          epochs: int | None = None,
                          stage: str = "select", seed: int = SEED,
                          result_root=None,
                          environment_sha256: str | None = None,
                          child_environment: dict | None = None,
                          input_authority: dict | None = None) -> dict:
    identity = expected_run_identity(
        dataset, n, stage=stage, seed=seed, topp=topp, joint=joint,
        epochs=epochs, input_authority=input_authority)
    tag = (refit_tag_for(dataset, n, seed, namespace=namespace,
                         topp=topp, joint=joint)
           if stage == "refit" else
           tag_for(dataset, n, namespace=namespace, topp=topp, joint=joint))
    authority = input_authority or {}
    hf = authority.get("hf_runtime") or {}
    child_environment_sha256 = (
        _semantic_digest(child_environment) if child_environment is not None
        else None)
    return {
        "cell_id": campaign_cell_id(
            dataset, n, topp=topp, joint=joint, stage=stage, seed=seed),
        "campaign_nonce": campaign_nonce,
        "dataset": dataset,
        "N": int(n),
        "stage": stage,
        "seed": int(seed),
        "recipe": {
            "routing_adaptive_topp_min": str(topp[0]),
            "routing_adaptive_topp_max": str(topp[1]),
            "lambda_codon_joint": str(joint),
        },
        "expected_tag": tag,
        "result_root": str(_canonical_result_root(result_root)),
        "environment_sha256": environment_sha256,
        "physical_gpu_index": (
            (child_environment or {}).get("physical_gpu", {}).get("index")),
        "expected_child_environment": child_environment,
        "child_environment_sha256": child_environment_sha256,
        "input_authority_sha256": authority.get("authority_sha256"),
        "input_seal_sha256": authority.get("seal_file_sha256"),
        "input_aggregate_sha256": authority.get("aggregate_sha256"),
        "split_identity_sha256": authority.get("split_identity_sha256"),
        "hf_identity_sha256": hf.get("identity_sha256"),
        "expected_identity_digest": identity.digest,
        "expected_run_identity": identity.as_record(),
    }


CAMPAIGN_LAUNCH_FIELDS = (
    "campaign_nonce", "cell_id", "expected_identity_digest", "expected_tag",
    "result_root", "environment_sha256", "input_authority_sha256",
    "input_seal_sha256", "input_aggregate_sha256",
    "split_identity_sha256", "hf_identity_sha256",
    "physical_gpu_index", "expected_child_environment",
    "child_environment_sha256",
)


def launch_binding_from_expected(binding: dict, plan_snapshot_sha256: str) -> dict:
    """Single canonical projection used by producer and every reducer."""
    missing = [field for field in CAMPAIGN_LAUNCH_FIELDS
               if field not in binding or binding[field] is None
               or binding[field] == ""]
    if missing:
        raise CellRefused(
            f"expected cell binding lacks launch authority fields {missing}"
        )
    return {
        "campaign_nonce": binding["campaign_nonce"],
        "plan_snapshot_sha256": plan_snapshot_sha256,
        **{field: binding[field] for field in CAMPAIGN_LAUNCH_FIELDS
           if field != "campaign_nonce"},
    }


def _campaign_cell_parts(cell) -> tuple:
    """Normalise legacy four-tuples and stage-aware six-tuples."""
    if len(cell) == 4:
        dataset, n, topp, joint = cell
        return dataset, n, topp, joint, "select", SEED
    if len(cell) == 6:
        dataset, n, topp, joint, stage, seed = cell
        return dataset, n, topp, joint, stage, int(seed)
    raise CellRefused(
        f"campaign cell {cell!r} has {len(cell)} fields, expected 4 or 6")


#: Every file whose bytes decide what a sweep cell computes, beyond the four
#: PROTOCOL_SOURCES and the per-dataset trainer. A sweep spans hours; these are
#: hashed ONCE when the plan is made and re-checked after every cell, so a
#: campaign cannot be a mixture of two trees.
_SNAPSHOT_SOURCES = (
    *_BOOTSTRAP_SOURCE_PATHS,
)


def canonical_plan(axis: str, *, at_topp=None) -> list:
    """The plan the SOURCE declares for an axis.

    The selector must compare a snapshot against this, not against the set the
    snapshot names for itself: a Flickr-only snapshot claiming to be a complete
    sweep was accepted as one.
    """
    return sweep_cells(axis, at_topp=at_topp)


def plan_snapshot(datasets=None, *, plan=None, executed=None, axis=None,
                  namespace=None, campaign_nonce=None,
                  epochs: int | None = None, campaign_kind=None,
                  authorities: dict | None = None,
                  input_seals: dict | None = None,
                  result_root: str | Path | None = None,
                  gpu_ids=(), environment_provenance: dict | None = None) -> dict:
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
                ("qwen", Path(spec["qwen"]))):
            inputs[f"{ds}/{label}"] = (_sha(path) if Path(path).is_file()
                                       else "#absent")
    source_authority = _source_authority_bundle(sources)
    effective_gpu_ids = tuple(int(gpu) for gpu in gpu_ids)
    if plan is not None and not effective_gpu_ids:
        # The CLI's canonical default is physical GPU 0.  Keeping that default
        # here also makes programmatic plan construction bind a real child
        # assignment instead of producing an unattested plan.
        effective_gpu_ids = (0,)
    environment = (environment_fingerprint(effective_gpu_ids)
                   if environment_provenance is None else
                   json.loads(json.dumps(environment_provenance)))
    environment_sha = _semantic_digest(environment)
    payload = {
        "schema_version": SWEEP_SNAPSHOT_SCHEMA,
        "sources": sources,
        "inputs": inputs,
        "qwen_root": str(QWEN_ROOT),
        "source_authority": source_authority,
        "source_authority_sha256": _semantic_digest(source_authority),
        "preimport_source_authority_sha256": (
            _semantic_digest(_BOOTSTRAP_PREIMPORT_BUNDLE)
            if _BOOTSTRAP_PREIMPORT_VERIFIED else None),
        "environment": environment,
        "environment_sha256": environment_sha,
        "input_seals": json.loads(json.dumps(input_seals or {}, sort_keys=True)),
    }
    # WHAT is being run, not only WHAT IT IS RUN WITH. Without this the
    # one-cell smoke and the nine-cell production sweep produced a
    # byte-identical snapshot, so the digest in a receipt could not say which
    # plan it sealed.
    def _cells(rows):
        return [{"dataset": ds, "N": n,
                 "topp": list(topp) if topp is not None else None,
                 "joint": jd, "stage": stage, "seed": seed}
                for ds, n, topp, jd, stage, seed in
                (_campaign_cell_parts(row) for row in rows)]

    if plan is not None:
        if not isinstance(campaign_nonce, str) or len(campaign_nonce) < 32:
            raise CellRefused(
                "a sweep snapshot needs a prelaunch campaign nonce of at least "
                "128 bits; a plan without one cannot bind trainer evidence")
        executed = plan if executed is None else executed
        canonical_result_root = _canonical_result_root(result_root)
        requested_gpus = [int(gpu) for gpu in
                          environment.get("requested_gpu_indices", ())]
        planned_datasets = sorted({_campaign_cell_parts(row)[0]
                                   for row in plan})
        if not requested_gpus:
            raise CellRefused("a campaign plan has no physical GPU assignment")
        if len(requested_gpus) == 1:
            dataset_gpu = {dataset: requested_gpus[0]
                           for dataset in planned_datasets}
        elif len(requested_gpus) >= len(planned_datasets):
            dataset_gpu = {dataset: requested_gpus[index]
                           for index, dataset in enumerate(planned_datasets)}
        else:
            raise CellRefused(
                f"{len(planned_datasets)} planned datasets cannot be bound to "
                f"{len(requested_gpus)} physical GPUs")
        bindings = {}
        for row in plan:
            ds, n, topp, jd, stage, seed = _campaign_cell_parts(row)
            try:
                child_environment = expected_child_environment(
                    environment, dataset_gpu[ds])
            except EnvironmentAttestationError as error:
                raise CellRefused(str(error)) from None
            binding = expected_cell_binding(
                ds, n, namespace=namespace, campaign_nonce=campaign_nonce,
                topp=topp, joint=jd, epochs=epochs, stage=stage, seed=seed,
                result_root=canonical_result_root,
                environment_sha256=environment_sha,
                child_environment=child_environment,
                input_authority=(input_seals or {}).get(
                    f"{ds}:{_seal_stage(stage)}"))
            if binding["cell_id"] in bindings:
                raise CellRefused(
                    f"duplicate campaign cell id {binding['cell_id']!r}")
            bindings[binding["cell_id"]] = binding
        payload["plan"] = {
            "axis": axis,
            "campaign_kind": campaign_kind or f"recipe_{axis}",
            "namespace": namespace,
            "result_root": str(canonical_result_root),
            "execution_kind": "smoke" if epochs is not None else "production",
            "campaign_nonce": campaign_nonce,
            # What the axis declares, and what THIS invocation actually runs.
            # One field could not express a `--only` smoke: the snapshot said
            # twelve while its receipt said one, and nothing compared them.
            "declared_cells": _cells(plan),
            "declared_count": len(plan),
            "executed_cells": _cells(executed),
            "executed_count": len(executed),
            # This mapping is fixed BEFORE a child starts.  A coordinate set is
            # permutation-invariant; this mapping is not.  It names the tag and
            # full RunIdentity that every canonical cell must parse, while the
            # nonce later appears in the trainer-owned checkpoint sidecar.
            "cell_bindings": bindings,
            "dataset_gpu_assignments": dataset_gpu,
            "authorities": authorities or {},
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
    if snapshot.get("qwen_root") != str(QWEN_ROOT):
        raise CellRefused(
            "the canonical absolute Qwen root changed after planning")
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
            "qwen": Path(spec["qwen"]),
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

    before_authority = snapshot.get("source_authority")
    if not isinstance(before_authority, dict) \
            or snapshot.get("source_authority_sha256") != \
            _semantic_digest(before_authority):
        raise CellRefused("snapshot has no intact source-authority bundle")
    after_authority = _source_authority_bundle(now_sources)
    if after_authority != before_authority:
        raise CellRefused(
            "HEAD/blob/worktree source authority changed while the campaign "
            "was running; start/end hashes are not identical")
    before_environment = snapshot.get("environment")
    if not isinstance(before_environment, dict) \
            or snapshot.get("environment_sha256") != \
            _semantic_digest(before_environment):
        raise CellRefused("snapshot has no intact execution environment fingerprint")
    after_environment = environment_fingerprint(
        before_environment.get("requested_gpu_indices") or ())
    if after_environment != before_environment:
        raise CellRefused(
            "Python/package/GPU/library execution environment changed while "
            "the campaign was running")
    if snapshot.get("input_seals"):
        verify_snapshot_input_seals(snapshot, full=False)


def assert_production_source_authority(snapshot: dict) -> None:
    """Admission gate: executed imports, HEAD blobs and worktree must agree."""
    verify_snapshot(snapshot)
    authority = snapshot.get("source_authority") or {}
    entries = authority.get("entries") or {}
    dirty = sorted(rel for rel, state in entries.items()
                   if not state.get("tracked") or not state.get("clean"))
    if dirty:
        raise CellRefused(
            "production protocol sources are untracked or differ from HEAD: "
            f"{dirty}. Commit the exact executable source closure first.")
    if not _BOOTSTRAP_PREIMPORT_VERIFIED:
        raise CellRefused(
            "production admission requires the stdlib-only pre-import "
            "self-reexec handshake; invoke this file as its entrypoint")
    digest = _semantic_digest(_BOOTSTRAP_PREIMPORT_BUNDLE)
    if snapshot.get("preimport_source_authority_sha256") != digest \
            or snapshot.get("source_authority_sha256") != digest:
        raise CellRefused(
            "the pre-import, snapshot and launch source bundles are not one "
            "identical hash")
    environment = snapshot.get("environment") or {}
    if environment.get("errors"):
        raise CellRefused(
            f"execution environment provenance is incomplete: "
            f"{environment['errors']}")
    if environment.get("pythonpath") not in (None, ""):
        raise CellRefused("production child PYTHONPATH must be unset")
    if environment.get("python_executable") != os.path.realpath(sys.executable):
        raise CellRefused("snapshot names a different canonical Python executable")
    verify_snapshot_input_seals(snapshot, full=False)


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
        "text_hash_counterfactual_weight": float(
            _arg_value(args_txt, "text_hash_counterfactual_weight")),
    }
    want = {"num_semantic_parts": SLOTS, "num_codebooks": SLOTS,
            "num_codons_per_codebook": BASES_PER_SLOT,
            "routing_adaptive_topp_min": float(topp[0]),
            "routing_adaptive_topp_max": float(topp[1]),
            "lambda_codon_joint": float(joint),
            "text_hash_counterfactual_weight": 0.0}
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


def _read_json_bound(path: Path) -> tuple[dict, str]:
    """Parse finite JSON from the same descriptor whose bytes are hashed."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise CellRefused(f"{path}: cannot open authoritative JSON: {error}") \
            from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CellRefused(f"{path}: authoritative JSON is not regular")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    current = os.stat(path, follow_symlinks=True)
    if any(getattr(before, field) != getattr(after, field)
           or getattr(after, field) != getattr(current, field)
           for field in stat_fields):
        raise CellRefused(f"{path}: file changed or pathname was replaced while read")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise CellRefused(f"{path}: invalid finite JSON: {error}") from None
    if not isinstance(payload, dict):
        raise CellRefused(f"{path}: JSON root is not an object")
    return payload, digest


def _read_npz_bound(path: Path) -> tuple[dict, dict]:
    """Load arrays and attest the exact canonical pathname/inode consumed."""
    import numpy as np

    absolute = path.resolve(strict=True)
    if absolute != path:
        raise CellRefused(f"{path}: extraction path is not canonical")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise CellRefused(f"{path}: cannot open authoritative NPZ: {error}") \
            from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CellRefused(f"{path}: authoritative NPZ is not regular")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw_sha = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                raw_sha.update(block)
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as stored:
                arrays = {key: np.asarray(stored[key]) for key in stored.files}
            handle.seek(0)
            after_sha = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                after_sha.update(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = os.stat(path, follow_symlinks=True)
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if raw_sha.hexdigest() != after_sha.hexdigest() or any(
           getattr(before, field) != getattr(after, field)
           or getattr(after, field) != getattr(current, field)
           for field in stat_fields):
        raise CellRefused(f"{path}: extraction changed or was replaced while read")
    binding = {
        "path": str(path), "resolved_path": str(absolute),
        "sha256": raw_sha.hexdigest(), "size_bytes": int(after.st_size),
        "st_dev": int(after.st_dev), "st_ino": int(after.st_ino),
        "st_mtime_ns": int(after.st_mtime_ns),
        "st_ctime_ns": int(after.st_ctime_ns),
    }
    return arrays, binding


def _finite_proportion(payload: dict, key: str, *, what: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise CellRefused(f"{what}.{key}={value!r} is not a finite proportion")
    return float(value)


def _assert_evaluation_payload(payload: dict, *, path: Path, dataset: str,
                               npz_binding: dict, bio_project: bool) -> None:
    from evaluation_siglip2 import (
        AP_NORMALIZATION_POLICY, EVALUATION_SCHEMA_VERSION,
        RANKING_POLICY, RELEVANCE_POLICY)

    spec = DATASETS[dataset]
    rows = PAPER_SPLIT_ROWS[dataset]
    cutoff = MAP_R_CUTOFF[dataset]
    exact = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": spec["canon"], "distance_mode": "base",
        "codebook_size": spec["K"], "n_query": rows["query"],
        "n_db": rows["db"], "query_chunk_size": 64,
        "bio_project": bio_project, "mAP_R_cutoff": cutoff,
    }
    wrong = {key: (payload.get(key), value) for key, value in exact.items()
             if payload.get(key) != value}
    policy = payload.get("metric_policy")
    expected_policy = {
        "ranking": RANKING_POLICY,
        "ap_normalization": AP_NORMALIZATION_POLICY,
        "relevance": RELEVANCE_POLICY,
        "multi_label_relevance_threshold": 0.0,
        "remove_self_match": False,
        "map_at_r": cutoff,
        "query_chunk_size": 64,
    }
    if policy != expected_policy:
        wrong["metric_policy"] = (policy, expected_policy)
    expected_inputs = {split: npz_binding[split]
                       for split in ("db", "query")}
    if payload.get("input_artifacts") != expected_inputs:
        wrong["input_artifacts"] = (
            payload.get("input_artifacts"), expected_inputs)
    if wrong:
        raise CellRefused(f"{path}: evaluation protocol/identity differs: {wrong}")
    _finite_proportion(payload, "mAP", what=path.name)
    _finite_proportion(payload, "mAP_at_R", what=path.name)


def assert_refit_outputs(run_dir: Path, *, dataset: str) -> dict:
    """Reopen exact db/query/train, raw metric and post-BIO/NMI evidence."""
    import importlib.metadata
    import numpy as np
    from sklearn.metrics import normalized_mutual_info_score
    from dna_utils.extraction_validation import (
        ExpectedIdentity, ExtractionInvalid, read_analysis_marker,
        validate_extraction_run)
    from dna_utils.gc_policy import resolve_gc_policy

    run_dir = run_dir.resolve()
    spec = DATASETS[dataset]
    rows = PAPER_SPLIT_ROWS[dataset]
    checkpoint = run_dir / "model_state_dict.pth"
    if not checkpoint.is_file():
        raise CellRefused(
            f"{run_dir}: extraction does not validate: final checkpoint absent")
    try:
        run = validate_extraction_run(
            str(run_dir), required_splits=("db", "query", "train"),
            allow_backfilled=False,
            expected=ExpectedIdentity(
                dataset=spec["canon"], num_slots=SLOTS,
                bases_per_slot=BASES_PER_SLOT, codebook_size=spec["K"],
                checkpoint_sha256=_sha(checkpoint), n_rows=rows))
    except ExtractionInvalid as error:
        raise CellRefused(f"{run_dir}: extraction does not validate: {error}") \
            from None

    arrays, npz_binding, manifest_sha = {}, {}, {}
    for split in ("db", "query", "train"):
        canonical_npz = run_dir / f"extract_{split}.npz"
        manifest = run.splits[split]
        manifest_path = run_dir / f"extraction_manifest_{split}.json"
        bound_manifest, manifest_sha[split] = _read_json_bound(manifest_path)
        if bound_manifest != manifest \
                or manifest_sha[split] != run.manifest_sha256.get(split):
            raise CellRefused(
                f"{manifest_path}: manifest bytes changed after strict validation")
        if manifest.get("npz_path") != str(canonical_npz) \
                or manifest.get("checkpoint_path") != str(checkpoint) \
                or manifest.get("config_path") != str(run_dir / "config.pt"):
            raise CellRefused(
                f"{run_dir}: {split} manifest does not name canonical run files")
        arrays[split], npz_binding[split] = _read_npz_bound(canonical_npz)
        bases = np.asarray(arrays[split].get("base_indices"))
        if bases.shape != (rows[split], TOTAL_BASES):
            raise CellRefused(
                f"{canonical_npz}: base_indices shape {bases.shape}, expected "
                f"{(rows[split], TOTAL_BASES)}")
        if manifest.get("npz_sha256") != npz_binding[split]["sha256"]:
            raise CellRefused(f"{canonical_npz}: manifest binds different bytes")
    completion_marker, extraction_complete_sha = _read_json_bound(
        run_dir / "extraction_complete.json")
    if extraction_complete_sha != run.completion_marker_sha256:
        raise CellRefused(
            f"{run_dir}/extraction_complete.json: marker bytes changed after "
            "strict validation")
    if completion_marker != {
            "schema_version": 1,
            "splits": ["db", "query", "train"],
            "manifest_sha256": manifest_sha}:
        raise CellRefused(
            f"{run_dir}/extraction_complete.json: not the exact bound "
            "three-split transaction")

    raw_path = run_dir / "evaluation_siglip2_base.json"
    post_path = run_dir / "evaluation_siglip2_base_bioproj.json"
    raw, raw_sha = _read_json_bound(raw_path)
    post, post_sha = _read_json_bound(post_path)
    _assert_evaluation_payload(
        raw, path=raw_path, dataset=dataset, npz_binding=npz_binding,
        bio_project=False)
    _assert_evaluation_payload(
        post, path=post_path, dataset=dataset, npz_binding=npz_binding,
        bio_project=True)

    # Recompute the paper metrics from the same descriptor-bound arrays.  File
    # hashes and mutually consistent JSON are provenance, not evidence that the
    # numbers were actually computed; a hand-written raw/post/result/marker
    # bundle otherwise passed every binding check.
    from evaluation_siglip2 import (
        _apply_bio_projection, evaluate_code_collapse, evaluate_retrieval)
    cutoff = MAP_R_CUTOFF[dataset]
    raw_retrieval = evaluate_retrieval(
        arrays["query"], arrays["db"], distance_mode="base",
        multi_label_relevance_threshold=0.0, remove_self_match=False,
        map_at_r=cutoff, query_chunk_size=64)
    raw_collapse = evaluate_code_collapse(
        arrays["db"], codebook_size=spec["K"])
    for key, expected_value in {
            "mAP": raw_retrieval["mAP"],
            "mAP_at_R": raw_retrieval["mAP_at_R"],
            "mAP_R_cutoff": raw_retrieval["mAP_R_cutoff"],
            "unique_code_ratio": raw_collapse["unique_code_ratio"],
            "duplicate_rate": raw_collapse["duplicate_rate"],
            }.items():
        if raw.get(key) != expected_value:
            raise CellRefused(
                f"{raw_path}: {key} does not equal deterministic recomputation")

    projected_db = {key: np.array(value, copy=True)
                    for key, value in arrays["db"].items()}
    projected_query = {key: np.array(value, copy=True)
                       for key, value in arrays["query"].items()}
    recomputed_bio_stats = {
        "mAP_pre_projection": raw_retrieval["mAP"],
        "mAP_at_R_pre_projection": raw_retrieval["mAP_at_R"],
        "mAP_R_cutoff_pre_projection": raw_retrieval["mAP_R_cutoff"],
    }
    recomputed_bio_stats.update(_apply_bio_projection(
        projected_db, 0.4, 0.6, 3, tag="db"))
    recomputed_bio_stats.update(_apply_bio_projection(
        projected_query, 0.4, 0.6, 3, tag="qy"))
    post_retrieval = evaluate_retrieval(
        projected_query, projected_db, distance_mode="base",
        multi_label_relevance_threshold=0.0, remove_self_match=False,
        map_at_r=cutoff, query_chunk_size=64)
    post_collapse = evaluate_code_collapse(
        projected_db, codebook_size=spec["K"])
    for key, expected_value in {
            "mAP": post_retrieval["mAP"],
            "mAP_at_R": post_retrieval["mAP_at_R"],
            "mAP_R_cutoff": post_retrieval["mAP_R_cutoff"],
            "unique_code_ratio": post_collapse["unique_code_ratio"],
            "duplicate_rate": post_collapse["duplicate_rate"],
            "bio_stats": recomputed_bio_stats,
            }.items():
        if post.get(key) != expected_value:
            raise CellRefused(
                f"{post_path}: {key} does not equal deterministic recomputation")

    policy = resolve_gc_policy(TOTAL_BASES)
    for key, value in {
            "bio_gc_min_frac": policy.gc_min_frac,
            "bio_gc_max_frac": policy.gc_max_frac,
            "bio_max_homopolymer_run": policy.max_run}.items():
        if post.get(key) != value:
            raise CellRefused(f"{post_path}: {key}={post.get(key)!r}, expected {value}")
    stats = post.get("bio_stats")
    if not isinstance(stats, dict):
        raise CellRefused(f"{post_path}: no bio_stats")
    expected_stats = {
        "mAP_pre_projection", "mAP_at_R_pre_projection",
        "mAP_R_cutoff_pre_projection",
        *(f"{prefix}_{suffix}" for prefix in ("db", "qy") for suffix in (
            "pre_compliance", "post_compliance", "mean_edit_distance",
            "max_edit_distance", "num_total", "num_pre_invalid",
            "num_proj_failed")),
    }
    if set(stats) != expected_stats:
        raise CellRefused(
            f"{post_path}: bio_stats keys differ: "
            f"missing={sorted(expected_stats-set(stats))}, "
            f"extra={sorted(set(stats)-expected_stats)}")
    if stats["mAP_pre_projection"] != raw["mAP"] \
            or stats["mAP_at_R_pre_projection"] != raw["mAP_at_R"] \
            or stats["mAP_R_cutoff_pre_projection"] != MAP_R_CUTOFF[dataset]:
        raise CellRefused(f"{post_path}: pre-projection metric is not the raw result")
    for prefix, split in (("db", "db"), ("qy", "query")):
        total = rows[split]
        num_total = stats[f"{prefix}_num_total"]
        num_failed = stats[f"{prefix}_num_proj_failed"]
        post_compliance = stats[f"{prefix}_post_compliance"]
        if isinstance(num_total, bool) or not isinstance(num_total, int) \
                or num_total != total \
                or isinstance(num_failed, bool) \
                or not isinstance(num_failed, int) or num_failed != 0 \
                or isinstance(post_compliance, bool) \
                or not isinstance(post_compliance, (int, float)) \
                or post_compliance != 1.0:
            raise CellRefused(f"{post_path}: {prefix} BIO projection did not fully succeed")
        invalid = stats[f"{prefix}_num_pre_invalid"]
        maximum = stats[f"{prefix}_max_edit_distance"]
        mean = stats[f"{prefix}_mean_edit_distance"]
        if isinstance(invalid, bool) or not isinstance(invalid, int) \
                or not 0 <= invalid <= total \
                or isinstance(maximum, bool) or not isinstance(maximum, int) \
                or not 0 <= maximum <= TOTAL_BASES \
                or isinstance(mean, bool) or not isinstance(mean, (int, float)) \
                or not math.isfinite(float(mean)) or not 0 <= mean <= TOTAL_BASES:
            raise CellRefused(f"{post_path}: invalid {prefix} BIO statistics")
        _finite_proportion(stats, f"{prefix}_pre_compliance", what=post_path.name)

    expected_protocol = {
        "dataset": spec["canon"], "codebook_size": spec["K"],
        "total_bases": TOTAL_BASES, "map_r_cutoff": MAP_R_CUTOFF[dataset],
        "gc_policy_version": policy.policy_version,
        "gc_count_min_inclusive": policy.gc_min_count,
        "gc_count_max_inclusive": policy.gc_max_count,
        "gc_min_frac": policy.gc_min_frac, "gc_max_frac": policy.gc_max_frac,
        "bio_max_homopolymer_run": policy.max_run,
        "nmi_average_method": "arithmetic",
        "sklearn_version": importlib.metadata.version("scikit-learn"),
    }
    marker_path = run_dir / "analysis_complete.json"
    bound_marker, marker_sha = _read_json_bound(marker_path)
    try:
        marker = read_analysis_marker(
            str(run_dir), allow_backfilled=False,
            expected=ExpectedIdentity(
                dataset=spec["canon"], num_slots=SLOTS,
                bases_per_slot=BASES_PER_SLOT, codebook_size=spec["K"],
                checkpoint_sha256=_sha(checkpoint), n_rows=rows),
            expected_protocol=expected_protocol,
            required_splits=("db", "query", "train"),
            expected_sha256=marker_sha)
    except ExtractionInvalid as error:
        raise CellRefused(f"{run_dir}: analysis marker refusal: {error}") from None
    if marker != bound_marker:
        raise CellRefused(
            f"{marker_path}: marker changed while strict verification ran")

    cell_result, cell_result_sha = _read_json_bound(run_dir / "cell_result.json")
    nmi_result, nmi_sha = _read_json_bound(run_dir / "pairwise_nmi.json")
    ci = np.asarray(arrays["db"].get("codebook_indices"))
    if ci.shape != (rows["db"], SLOTS) or not np.issubdtype(ci.dtype, np.integer):
        raise CellRefused(f"{run_dir}/extract_db.npz: invalid codebook_indices")
    nmi_matrix = np.empty((SLOTS, SLOTS), dtype=np.float64)
    for left in range(SLOTS):
        for right in range(SLOTS):
            nmi_matrix[left, right] = normalized_mutual_info_score(
                ci[:, left], ci[:, right], average_method="arithmetic")
    off_diag = nmi_matrix[~np.eye(SLOTS, dtype=bool)]
    recomputed_nmi = float(off_diag.mean())
    marker_metrics = marker["metrics"]
    cross = {
        "map_at_R_bioproj": post["mAP_at_R"],
        "full_map_bioproj": post["mAP"],
        "full_map_pre_projection": raw["mAP"],
        "dna_unique_db": post["unique_code_ratio"],
        "mean_off_diag_nmi": recomputed_nmi,
    }
    if marker_metrics != cross \
            or cell_result.get("mAP_at_R_bioproj") != post["mAP_at_R"] \
            or cell_result.get("full_mAP_bioproj") != post["mAP"] \
            or cell_result.get("full_mAP_pre_projection") != raw["mAP"] \
            or cell_result.get("DNA_unique_DB") != post["unique_code_ratio"] \
            or nmi_result.get("mean_off_diag_nmi") != recomputed_nmi:
        raise CellRefused(
            f"{run_dir}: analysis marker/producer results do not equal strict "
            "evaluation and recomputed NMI")

    return {
        "extraction_validated": True,
        "required_splits": ["db", "query", "train"],
        "npz_sha256": {split: npz_binding[split]["sha256"]
                       for split in sorted(npz_binding)},
        "extraction_manifest_sha256": manifest_sha,
        "extraction_complete_sha256": extraction_complete_sha,
        "evaluation_sha256": raw_sha,
        "bio_evaluation_sha256": post_sha,
        "cell_result_sha256": cell_result_sha,
        "pairwise_nmi_sha256": nmi_sha,
        "analysis_complete_sha256": marker_sha,
        "map": raw["mAP"], "map_at_R": raw["mAP_at_R"],
        "map_R_cutoff": MAP_R_CUTOFF[dataset],
        "bio_map": post["mAP"], "bio_map_at_R": post["mAP_at_R"],
        "mean_off_diag_nmi": recomputed_nmi,
        "analysis_metrics": marker_metrics,
        "analysis_protocol": marker["protocol"],
        "label_zero_positive_rows": {
            split: (int(np.sum(np.asarray(arrays[split][
                "multi_hot_labels"]).sum(axis=1) == 0))
                    if "multi_hot_labels" in arrays[split] else 0)
            for split in ("db", "query", "train")},
        "checkpoint_exact_load": True,
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


def assert_completed(run_dir: Path, *, terminal_epoch: int,
                     campaign_binding: dict | None = None) -> dict:
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
    try:
        payload = json.loads(
            sidecar.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (OSError, ValueError, TypeError) as error:
        raise CellRefused(f"{sidecar}: invalid runtime metadata: {error}") \
            from None
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
    if campaign_binding is not None:
        from dna_utils.runtime_state import annealed_epsilon

        required_runtime_fields = {
            "schema_version", "checkpoint_path", "checkpoint_sha256",
            "checkpoint_epoch_zero_based", "training_epoch_budget",
            "stop_after_epoch", "lr_schedule_horizon",
            "sinkhorn_schedule_horizon", "sinkhorn_epsilon_init",
            "sinkhorn_epsilon_final", "effective_sinkhorn_epsilon",
            "lr_scheduler", "extra",
        }
        if set(payload) != required_runtime_fields:
            raise CellRefused(
                f"{sidecar}: Phase-3 runtime schema fields differ: "
                f"missing={sorted(required_runtime_fields-set(payload))}, "
                f"extra={sorted(set(payload)-required_runtime_fields)}")
        expected_scalars = {
            "schema_version": 1,
            "checkpoint_path": str(final.resolve()),
            "checkpoint_sha256": actual,
            "checkpoint_epoch_zero_based": int(terminal_epoch),
            "training_epoch_budget": identity.epoch_budget,
            "stop_after_epoch": identity.stop_after_epoch,
            "lr_schedule_horizon": identity.lr_schedule_horizon,
            "sinkhorn_schedule_horizon": identity.sinkhorn_schedule_horizon,
            "sinkhorn_epsilon_init": identity.sinkhorn_epsilon_init,
            "sinkhorn_epsilon_final": identity.sinkhorn_epsilon_final,
            "lr_scheduler": "cosine",
        }
        runtime_wrong = {key: (payload.get(key), value)
                         for key, value in expected_scalars.items()
                         if payload.get(key) != value}
        effective = annealed_epsilon(
            terminal_epoch, identity.sinkhorn_schedule_horizon,
            identity.sinkhorn_epsilon_init, identity.sinkhorn_epsilon_final)
        if not isinstance(payload.get("effective_sinkhorn_epsilon"),
                          (int, float)) \
                or isinstance(payload.get("effective_sinkhorn_epsilon"), bool) \
                or not math.isfinite(float(payload["effective_sinkhorn_epsilon"])) \
                or payload.get("effective_sinkhorn_epsilon") != effective:
            runtime_wrong["effective_sinkhorn_epsilon"] = (
                payload.get("effective_sinkhorn_epsilon"), effective)
        expected_extra = {
            "tag": campaign_binding["expected_tag"],
            "dataset": identity.dataset,
            "random_seed": identity.seed,
            "num_semantic_parts": identity.num_slots,
            "num_codons_per_codebook": identity.bases_per_slot,
        }
        extra = payload.get("extra")
        if not isinstance(extra, dict) \
                or set(extra) != set(expected_extra) | {"phase3_campaign"}:
            runtime_wrong["extra.keys"] = (
                sorted(extra) if isinstance(extra, dict) else type(extra).__name__,
                sorted(set(expected_extra) | {"phase3_campaign"}))
        elif any(extra.get(key) != value
                 for key, value in expected_extra.items()):
            runtime_wrong["extra.identity"] = (extra, expected_extra)
        if runtime_wrong:
            raise CellRefused(
                f"{run_dir}: Phase-3 checkpoint runtime metadata is not the "
                f"exact terminal run schema: {runtime_wrong}")
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
    campaign_evidence = None
    campaign_evidence_sha = None
    if campaign_binding is not None:
        from dna_utils.run_identity import (
            PHASE3_CAMPAIGN_BINDING_NAME, RunCollision,
            phase3_campaign_from_checkpoint)

        evidence_path = run_dir / PHASE3_CAMPAIGN_BINDING_NAME
        if not evidence_path.is_file():
            raise CellRefused(
                f"{run_dir}: checkpoint belongs to a planned campaign cell but "
                f"has no trainer-owned {PHASE3_CAMPAIGN_BINDING_NAME}")
        try:
            campaign_evidence = json.loads(
                evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CellRefused(
                f"{evidence_path}: unreadable campaign evidence: {error}")
        core_fields = tuple(dict.fromkeys(
            ("plan_snapshot_sha256", *CAMPAIGN_LAUNCH_FIELDS)))
        wrong = {
            field: (campaign_evidence.get(field), campaign_binding.get(field))
            for field in core_fields
            if campaign_evidence.get(field) != campaign_binding.get(field)}
        if wrong:
            raise CellRefused(
                f"{run_dir}: trainer-owned campaign evidence belongs to a "
                f"different planned cell: {wrong}")
        if campaign_evidence.get("actual_identity_digest") != identity.digest:
            raise CellRefused(
                f"{run_dir}: campaign evidence names actual identity "
                f"{campaign_evidence.get('actual_identity_digest')}, manifest "
                f"is {identity.digest}")
        from dna_utils.runtime_environment import (
            EnvironmentAttestationError, semantic_digest,
            verify_child_environment)
        child_expected = campaign_evidence.get("expected_child_environment")
        child_actual = campaign_evidence.get("actual_child_environment")
        try:
            if not isinstance(child_expected, dict) \
                    or semantic_digest(child_expected) != \
                    campaign_binding.get("child_environment_sha256"):
                raise EnvironmentAttestationError(
                    "child environment digest mismatch")
            verify_child_environment(child_expected, actual=child_actual)
        except EnvironmentAttestationError as error:
            raise CellRefused(
                f"{run_dir}: invalid trainer child environment attestation: "
                f"{error}") from None
        runtime_campaign = (payload.get("extra") or {}).get("phase3_campaign")
        if runtime_campaign != campaign_evidence:
            raise CellRefused(
                f"{run_dir}: checkpoint runtime metadata does not contain the "
                "exact trainer-owned campaign evidence")
        try:
            checkpoint_campaign = phase3_campaign_from_checkpoint(str(final))
        except RunCollision as error:
            raise CellRefused(str(error)) from None
        if checkpoint_campaign != campaign_evidence:
            raise CellRefused(
                f"{run_dir}: checkpoint serialization does not commit the "
                "exact trainer-owned campaign evidence")
        campaign_evidence_sha = _sha(evidence_path)

    result = {"final_checkpoint": final.name,
              "final_checkpoint_sha256": actual,
              "final_checkpoint_epoch_zero_based": recorded_epoch,
              "terminal_weights_preserved": preserved,
              "run_identity_sha256": _sha(run_dir / MANIFEST_NAME),
              "log_csv_sha256": _sha(run_dir / "log.csv"),
              "checkpoint_runtime_sha256": _sha(sidecar)}
    args_txt = run_dir / "args.txt"
    if args_txt.is_file():
        result["args_txt_sha256"] = _sha(args_txt)
    if campaign_evidence is not None:
        result.update(
            phase3_campaign=campaign_evidence,
            phase3_campaign_evidence_sha256=campaign_evidence_sha)
    return result


def read_selection(run_dir: Path, *, dataset: str, n: int) -> dict:
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
    if dataset not in MAP_R_CUTOFF:
        raise CellRefused(
            f"{csv_path}: dataset {dataset!r} has no canonical mAP@R contract")
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
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CellRefused(f"{csv_path}: eval_mAP_at_R={raw!r} is not a "
                          "finite proportion")

    cutoff = (terminal[0].get("eval_mAP_R_cutoff") or "").strip()
    if not cutoff:
        raise CellRefused(
            f"{csv_path}: epoch {n} records no mAP@R cutoff, so the metric "
            f"cannot be compared across datasets")
    try:
        cutoff_value = int(cutoff)
    except ValueError:
        raise CellRefused(
            f"{csv_path}: epoch {n} mAP@R cutoff {cutoff!r} is not an integer") \
            from None
    expected_cutoff = MAP_R_CUTOFF[dataset]
    if cutoff != str(expected_cutoff) or cutoff_value != expected_cutoff:
        raise CellRefused(
            f"{csv_path}: epoch {n} mAP@R cutoff is {cutoff!r}, expected "
            f"canonical {expected_cutoff} for {dataset}")
    metric_distance = (terminal[0].get("eval_distance_mode") or "").strip()
    if metric_distance != "base":
        raise CellRefused(
            f"{csv_path}: epoch {n} distance mode is {metric_distance!r}, "
            "expected canonical raw base Hamming mode 'base'")
    record = {
        "selection_metric": "eval_mAP_at_R",
        "selection_value": value,
        "selection_epoch_zero_based": n,
        "map_r_cutoff": cutoff_value,
        "distance_mode": metric_distance,
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


def _resolve(tag: str, *, expected_digest: str | None = None,
             result_root=None) -> Path:
    """The directory this cell produced, identified rather than searched for.

    A tag substring re-search races a concurrent writer with a similar tag and
    compares nothing about what it finds. When the expected identity is known,
    the candidates are filtered by their manifest digest, so the answer is the
    directory that carries THIS run's identity -- not merely one whose name
    happens to contain the tag.
    """
    root = _canonical_result_root(result_root)
    hits = sorted(p for p in root.glob(f"*{tag}*") if p.is_dir())
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
    """Atomic replace with a temp name unique across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _publish_json_exclusive(path: Path, payload: dict) -> None:
    """Create a JSON artefact once; never replace bytes another owner wrote."""
    blob = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as error:
        raise CellRefused(f"{path} already exists and is immutable") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())


def campaign_reservation_path(namespace: str) -> Path:
    return RECORD_DIR / f"{namespace}{CAMPAIGN_RESERVATION_SUFFIX}"


def reserve_sweep_namespace(namespace: str, *, snapshot: dict,
                            plan_digest: str, campaign_nonce: str,
                            snapshot_file: str) -> dict:
    """Atomically give one process ownership of a fresh sweep namespace.

    The old ``glob(); if empty`` gate was check-then-act: two fresh callers
    both passed it.  The reservation itself is the linearisation point.  It is
    retained after success (and after a crash) because downstream verification
    uses it as the prelaunch anchor; recovery therefore requires an explicit
    archival decision, never automatic PID-based stealing.
    """
    if not namespace or any(c not in
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                            for c in namespace):
        raise CellRefused(
            f"namespace {namespace!r} contains characters unsafe for a "
            "reservation filename")
    if _json_digest(snapshot) != plan_digest:
        raise CellRefused("the reservation was given a digest for other plan bytes")
    plan = snapshot.get("plan") or {}
    if plan.get("namespace") != namespace \
            or plan.get("campaign_nonce") != campaign_nonce:
        raise CellRefused(
            "the namespace/campaign nonce in the plan does not match the "
            "reservation request")

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = campaign_reservation_path(namespace)
    payload = {
        "schema_version": CAMPAIGN_RESERVATION_SCHEMA,
        "namespace": namespace,
        "owner_pid": os.getpid(),
        "owner_boot_id": _boot_id(),
        "owner_host": socket.gethostname(),
        "campaign_nonce": campaign_nonce,
        "plan_digest": plan_digest,
        "plan_snapshot_file": snapshot_file,
        "campaign_kind": plan.get("campaign_kind"),
        "result_root": plan.get("result_root"),
        "qwen_root": snapshot.get("qwen_root"),
        "source_authority_sha256": snapshot.get(
            "source_authority_sha256"),
        "environment_sha256": snapshot.get("environment_sha256"),
        "input_seals_sha256": _semantic_digest(snapshot.get("input_seals") or {}),
        "head_commit": (snapshot.get("source_authority") or {}).get(
            "head_commit"),
        "declared_cells": (snapshot.get("plan") or {}).get("declared_count"),
        "executed_cells": (snapshot.get("plan") or {}).get("executed_count"),
    }
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            held = {}
        raise CellRefused(
            f"namespace {namespace!r} is already reserved by pid "
            f"{held.get('owner_pid')} on boot {held.get('owner_boot_id')} for "
            f"campaign {str(held.get('campaign_nonce'))[:12]}... and plan "
            f"{str(held.get('plan_digest'))[:12]}...") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())

    # Claim first, then inspect legacy artefacts.  Only the winning caller can
    # reach this point, so a file appearing after the inspection must belong to
    # this owner under the new protocol.  The reservation is intentionally kept
    # when legacy residue is found: silently freeing it would invite another
    # caller to repeat the same unsafe admission.
    occupied = sorted(
        p.name for p in RECORD_DIR.glob(f"{namespace}_*.json") if p != path)
    if occupied:
        raise CellRefused(
            f"namespace {namespace!r} already holds {len(occupied)} artefacts "
            f"{occupied[:3]}. A namespace belongs to one sweep; archive the "
            "old evidence and choose a fresh namespace.")
    return payload


def assert_reservation_owner(namespace: str, expected: dict) -> None:
    """Reopen the immutable reservation before producing further evidence."""
    path = campaign_reservation_path(namespace)
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CellRefused(f"cannot reopen namespace reservation {path}: {error}")
    if actual != expected:
        raise CellRefused(
            f"namespace reservation {path.name} changed after admission")


def _run_refit_postprocess(run_dir: Path, *, dataset: str, env: dict,
                           snapshot: dict | None = None) -> None:
    """Produce the exact three-split, BIO and NMI evidence for a MAIN refit."""
    spec = DATASETS[dataset]
    commands = (
        [sys.executable, str(REPO / "scripts/extract_train_split.py"),
         "--config_path", str(run_dir)],
        [sys.executable, str(REPO / "scripts/eval_cell_bioproj.py"),
         "--dir", str(run_dir), "--dataset", spec["canon"],
         "--K", str(spec["K"]), "--require-train"],
        [sys.executable, str(REPO / "scripts/pairwise_nmi.py"),
         "--results", str(run_dir), "--require-train"],
        [sys.executable, str(REPO / "scripts/seal_cell_analysis.py"),
         "--dir", str(run_dir), "--require-train"],
    )
    for command in commands:
        proc = _run_managed_process(command, cwd=str(REPO), env=env)
        if proc.returncode != 0:
            raise CellRefused(
                f"{run_dir}: refit post-process {Path(command[1]).name} "
                f"exited {proc.returncode}")
        if snapshot is not None:
            # Stats-level input seal, source, package and physical-GPU
            # authority are re-opened between every producer. The final full
            # seal verification is campaign-wide after all cells finish.
            verify_snapshot(snapshot)


def run_cell(dataset: str, n: int, gpu: int, *, epochs=None,
             namespace: str = NAMESPACE, stage: str = "select",
             seed: int = SEED, topp=None, joint=None,
             snapshot: dict | None = None,
             campaign_binding: dict | None = None,
             result_root=None) -> dict:
    result_root = _canonical_result_root(result_root)
    tag = (refit_tag_for(dataset, n, seed, namespace=namespace,
                         topp=topp, joint=joint)
           if stage == "refit"
           else tag_for(dataset, n, namespace=namespace,
                        topp=topp, joint=joint))
    existing = _existing_artifacts(tag, result_root=result_root)
    if existing:
        raise CellRefused(
            f"{tag} already has artefacts: {existing[:3]}. Give the run its own "
            f"namespace with PHASE3_NAMESPACE.")

    cmd, env, _ = build_command(dataset, n, gpu, epochs=epochs,
                                namespace=namespace, stage=stage, seed=seed,
                                topp=topp, joint=joint)
    input_authority = None
    if snapshot is not None:
        input_authority = (snapshot.get("input_seals") or {}).get(
            f"{dataset}:{_seal_stage(stage)}")
    if input_authority is not None:
        hf = input_authority["hf_runtime"]
        input_flags = [
            "--phase3_input_seal", input_authority["seal_path"],
            "--phase3_input_seal_sha256", input_authority["seal_file_sha256"],
            "--phase3_input_aggregate_sha256", input_authority["aggregate_sha256"],
            "--phase3_split_identity_sha256",
            input_authority["split_identity_sha256"],
            "--phase3_hf_identity_sha256", hf["identity_sha256"],
            "--clip_snapshot_dir", hf["snapshot_dir"],
            "--clip_snapshot_revision", hf["revision"],
            "--clip_snapshot_weight_file", hf["weight_file"],
            "--clip_snapshot_weight_sha256", hf["weight_sha256"],
            "--clip_snapshot_config_sha256", hf["config_sha256"],
            "--clip_snapshot_tokenizers_sha256_json", json.dumps(
                hf["tokenizer_files_sha256"], sort_keys=True,
                separators=(",", ":")),
        ]
        env["EXTRA_ARGS"] += " " + " ".join(
            shlex.quote(value) for value in input_flags)
    if campaign_binding is not None:
        if snapshot is None:
            raise CellRefused(
                f"{tag}: campaign binding supplied without its plan snapshot")
        snap_digest = _json_digest(snapshot)
        planned_binding = ((snapshot.get("plan") or {}).get(
            "cell_bindings") or {}).get(campaign_binding.get("cell_id"))
        if not isinstance(planned_binding, dict) or campaign_binding != \
                launch_binding_from_expected(planned_binding, snap_digest):
            raise CellRefused(
                f"{tag}: launch binding is not the exact prelaunch cell binding")
        expected = {
            "expected_tag": tag,
            "plan_snapshot_sha256": snap_digest,
            "campaign_nonce": (snapshot.get("plan") or {}).get(
                "campaign_nonce"),
            "result_root": str(result_root),
            "environment_sha256": snapshot.get("environment_sha256"),
            "physical_gpu_index": int(gpu),
            "child_environment_sha256": campaign_binding.get(
                "child_environment_sha256"),
            "input_authority_sha256": input_authority.get(
                "authority_sha256") if input_authority else None,
            "input_seal_sha256": input_authority.get(
                "seal_file_sha256") if input_authority else None,
            "input_aggregate_sha256": input_authority.get(
                "aggregate_sha256") if input_authority else None,
            "split_identity_sha256": input_authority.get(
                "split_identity_sha256") if input_authority else None,
            "hf_identity_sha256": (input_authority.get("hf_runtime") or {}).get(
                "identity_sha256") if input_authority else None,
        }
        wrong = {field: (campaign_binding.get(field), value)
                 for field, value in expected.items()
                 if campaign_binding.get(field) != value}
        if wrong:
            raise CellRefused(
                f"{tag}: launch binding disagrees with this cell/plan: {wrong}")
        env.update({
            "GDNA_PHASE3_CAMPAIGN_NONCE": campaign_binding["campaign_nonce"],
            "GDNA_PHASE3_PLAN_DIGEST":
                campaign_binding["plan_snapshot_sha256"],
            "GDNA_PHASE3_CELL_ID": campaign_binding["cell_id"],
            "GDNA_PHASE3_EXPECTED_IDENTITY_DIGEST":
                campaign_binding["expected_identity_digest"],
            "GDNA_PHASE3_EXPECTED_TAG": campaign_binding["expected_tag"],
            "GDNA_PHASE3_RESULT_ROOT": campaign_binding["result_root"],
            "GDNA_PHASE3_ENVIRONMENT_DIGEST":
                campaign_binding["environment_sha256"],
            "GDNA_PHASE3_CHILD_ENVIRONMENT_DIGEST":
                campaign_binding["child_environment_sha256"],
            "GDNA_PHASE3_PHYSICAL_GPU_INDEX": str(
                campaign_binding["physical_gpu_index"]),
            "GDNA_PHASE3_EXPECTED_CHILD_ENVIRONMENT_JSON": json.dumps(
                campaign_binding["expected_child_environment"],
                sort_keys=True, separators=(",", ":"), allow_nan=False),
            "GDNA_PHASE3_INPUT_AUTHORITY_DIGEST":
                campaign_binding["input_authority_sha256"],
            "GDNA_PHASE3_INPUT_SEAL_DIGEST":
                campaign_binding["input_seal_sha256"],
            "GDNA_PHASE3_INPUT_AGGREGATE_DIGEST":
                campaign_binding["input_aggregate_sha256"],
            "GDNA_PHASE3_SPLIT_IDENTITY_DIGEST":
                campaign_binding["split_identity_sha256"],
            "GDNA_PHASE3_HF_IDENTITY_DIGEST":
                campaign_binding["hf_identity_sha256"],
        })
        # Every canonical trainer shell assigns CUDA_VISIBLE_DEVICES from its
        # positional GPU argument.  Leaving that argument numeric silently
        # overwrites the UUID-bound environment above and reintroduces CUDA
        # enumeration ambiguity.  Production campaign children receive the
        # exact sealed UUID through both channels; legacy direct launches keep
        # their historical numeric argument.
        try:
            cmd = _bind_campaign_gpu_selector(
                cmd, env, gpu=gpu, campaign_binding=campaign_binding)
        except CellRefused as error:
            raise CellRefused(f"{tag}: {error}") from None
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
    proc = _run_managed_process(
        ["bash", "-o", "pipefail", *cmd[1:]], cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise CellRefused(f"{tag}: trainer exited {proc.returncode}")
    if snapshot is not None:
        verify_snapshot(snapshot, datasets=[dataset])
    sources_after_trainer = protocol_digests(dataset)
    if sources_after_trainer != sources_before:
        changed = sorted(k for k in sources_before
                         if sources_before[k] != sources_after_trainer.get(k))
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
    run_dir = _resolve(
        tag, expected_digest=(campaign_binding or {}).get(
            "expected_identity_digest"), result_root=result_root)
    completion = assert_completed(
        run_dir, terminal_epoch=_stop, campaign_binding=campaign_binding)
    geometry = assert_geometry(
        run_dir, dataset=dataset, n=n, stop=_stop, seed=seed,
        mode="refit" if stage == "refit" else "select",
        val_ratio=0.0 if stage == "refit" else VAL_RATIO,
        budget=_budget, lr_horizon=_lr_h, sinkhorn_horizon=_sk_h,
        topp=topp, joint=joint)
    if campaign_binding is not None and geometry["identity_digest"] != \
            campaign_binding["expected_identity_digest"]:
        raise CellRefused(
            f"{tag}: completed identity {geometry['identity_digest'][:12]}... "
            f"differs from prelaunch identity "
            f"{campaign_binding['expected_identity_digest'][:12]}...")
    if stage == "refit":
        _run_refit_postprocess(
            run_dir, dataset=dataset, env=env, snapshot=snapshot)
    if snapshot is not None:
        verify_snapshot(snapshot, datasets=[dataset])
    sources_after = protocol_digests(dataset)
    if sources_after != sources_before:
        changed = sorted(k for k in sources_before
                         if sources_before[k] != sources_after.get(k))
        raise CellRefused(
            f"{tag}: {changed} changed while the trainer/refit post-chain was "
            "running; the evidence spans multiple source versions")
    record = {
        "schema_version": RECORD_SCHEMA,
        "namespace": namespace,
        "result_root": str(result_root),
        "environment_sha256": (
            snapshot.get("environment_sha256") if snapshot is not None else None),
        "input_authority": input_authority,
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
        "plan_snapshot_sha256": (_json_digest(snapshot)
            if snapshot is not None else None),
        **({"campaign": dict(campaign_binding)}
           if campaign_binding is not None else {}),
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
            **completion,
            **(assert_refit_outputs(run_dir, dataset=dataset)
               if stage == "refit" else {}),
        },
        "geometry": geometry,
        "selection": (
            read_selection(run_dir, dataset=dataset, n=_stop)
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
    _publish_json_exclusive(out, record)
    return record


def _load_recipe(path: Path) -> dict:
    """Return choices only after replaying the complete fixed-point evidence."""
    if not path.is_file():
        raise CellRefused(f"{path} does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CellRefused(f"{path}: unreadable recipe: {error}") from None
    # Keep the old/provisional error explicit: changing its boolean to true is
    # not an upgrade path and must never make old mAP-only files authoritative.
    if payload.get("artifact_kind") != "phase3_recipe_stability":
        stability = payload.get("stability") or {}
        if stability.get("confirmed") is not True:
            raise CellRefused(
                f"{path}: recipe stability is not confirmed; a provisional "
                "sweep cannot define the production N/refit recipe")
        raise CellRefused(
            f"{path}: a boolean stability claim is not a replayable "
            "state-machine authority")
    try:
        from scripts.phase3_select_n import verify_recipe_stability_artifact
        choices = verify_recipe_stability_artifact(path)
    except Exception as error:                         # noqa: BLE001
        raise CellRefused(f"{path}: {error}") from None
    expected = set(DATASETS)
    if set(choices) != expected:
        raise CellRefused(
            f"{path}: stable recipe covers {sorted(choices)}, not "
            f"{sorted(expected)}")
    return choices


def load_recipe_authority(paths) -> dict:
    """Authenticate the one final fixed-point artefact (never axis fragments)."""
    if not paths:
        raise CellRefused(
            "production N/refit needs one confirmed recipe-stability artefact")
    if len(paths) != 1:
        raise CellRefused(
            f"production recipe is one fixed-point authority, got {len(paths)} files")
    path = Path(paths[0]).resolve()
    choices = _load_recipe(path)
    return {"schema_version": RECIPE_AUTHORITY_SCHEMA,
            "artifact": {"path": str(path), "sha256": _sha(path)},
            "choices": choices}


def verify_recipe_authority(authority: dict) -> dict:
    """Reopen recipe bytes instead of trusting a snapshot's copied choices."""
    if not isinstance(authority, dict) \
            or authority.get("schema_version") != RECIPE_AUTHORITY_SCHEMA:
        raise CellRefused("recipe authority has an unknown schema")
    entry = authority.get("artifact") or {}
    path = Path(str(entry.get("path") or ""))
    if not path.is_file() or _sha(path) != entry.get("sha256"):
        raise CellRefused(
            f"recipe stability artefact is absent or changed: {path}")
    rebuilt = load_recipe_authority([path])
    if rebuilt != authority:
        raise CellRefused("recipe authority changed when its bytes were reopened")
    return rebuilt["choices"]


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
    from scripts.phase3_select_n import (
        SelectionRefused, verify_selection_artifact)
    try:
        selection = verify_selection_artifact(path)
    except SelectionRefused as error:
        raise CellRefused(f"{path}: selected-N authority refused: {error}") \
            from None
    selected = selection.get("selected_n") or {}
    missing = sorted(set(DATASETS) - set(selected))
    extra = sorted(set(selected) - set(DATASETS))
    bad = {dataset: n for dataset, n in selected.items()
           if n not in CANDIDATE_N}
    if missing or extra or bad:
        raise CellRefused(
            f"{path}: selected-N verifier returned invalid coverage; missing "
            f"{missing}, unexpected {extra}, out-of-grid {bad}")
    recipe_authority = selection.get("recipe_authority")
    recipe_choices = selection.get("recipe_choices")
    if not isinstance(recipe_authority, dict) or not isinstance(recipe_choices, dict):
        raise CellRefused(
            f"{path}: selected-N verifier returned no recipe lineage")
    return {
        "selected_n": selected,
        "recipe_authority": recipe_authority,
        "recipe_choices": recipe_choices,
    }


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


def _run_sweep(args, at_topp, *, full_plan=None,
               authorities: dict | None = None) -> int:
    if full_plan is None:
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

    epochs = args.epochs if args.smoke else None
    # Validate resource cardinality before consuming a namespace reservation.
    # Once reserved, a namespace is deliberately never auto-released: the
    # reservation is the reducer's prelaunch authority and a crash requires an
    # explicit archive/recovery decision.
    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus
                             else [str(args.gpu)])]
    by_dataset = {}
    for cell in plan:
        by_dataset.setdefault(cell[0], []).append(cell)
    if len(gpus) < len(by_dataset):
        print(f"[phase3] {len(by_dataset)} datasets need {len(by_dataset)} "
              f"GPUs, --gpus gave {len(gpus)}", file=sys.stderr)
        return 2

    # One snapshot for the whole sweep, taken before the first trainer starts.
    # Every cell is checked against THIS, not against the tree as it was when
    # that cell began -- otherwise a drift introduced between cell 1 and cell 2
    # becomes the new baseline and the sweep silently spans two trees.
    # The snapshot spans the DECLARED sweep, not the subset `--only` narrowed
    # it to: a one-cell run that pins only its own dataset is a new baseline
    # each time, which is what nine `--only` processes produced.
    campaign_nonce = secrets.token_hex(32)
    try:
        input_seals = verify_campaign_input_seals(
            args.input_seal_specs, full_plan, full=True)
        snapshot = plan_snapshot(
            sorted({c[0] for c in full_plan}), plan=full_plan, executed=plan,
            axis=args.sweep, namespace=args.namespace,
            campaign_nonce=campaign_nonce, epochs=epochs,
            campaign_kind=f"recipe_{args.sweep}",
            authorities=authorities,
            input_seals=input_seals,
            result_root=getattr(args, "result_root", REPO / "result"),
            gpu_ids=gpus)
        _assert_snapshot_gpu_leases(snapshot, args)
        assert_production_source_authority(snapshot)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
    # The snapshot file is named by its OWN digest. Three streams each running
    # `--only` wrote one shared `<namespace>_snapshot.json` and atomically
    # replaced each other's, so the file ended up holding one dataset's inputs
    # and the other two plans were unrecoverable. A content-addressed name
    # cannot be overwritten by a different plan.
    snap_digest = _json_digest(snapshot)
    snap_path = RECORD_DIR / f"{args.namespace}_snapshot_{snap_digest[:16]}.json"
    try:
        reservation = reserve_sweep_namespace(
            args.namespace, snapshot=snapshot, plan_digest=snap_digest,
            campaign_nonce=campaign_nonce, snapshot_file=snap_path.name)
        # Published only by the process that won O_EXCL.  A second caller never
        # gets a chance to replace or even create a competing snapshot.
        _publish_json_exclusive(snap_path, snapshot)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"[phase3] snapshot {len(snapshot['sources'])} sources, "
          f"{len(snapshot['inputs'])} inputs, campaign "
          f"{campaign_nonce[:12]}... -> {snap_path}")

    # Datasets run concurrently, one GPU each; the coordinates within a dataset
    # run in sequence. All of it is ONE process against ONE snapshot, because
    # nine separate `--only` invocations each took a fresh snapshot and so could
    # not fail on drift between cells -- the drift simply became the next
    # process's baseline.
    import threading
    results, lock = {}, threading.Lock()

    def _stream(dataset, cells, gpu):
        for ds, n, topp, jd in cells:
            key = campaign_cell_id(ds, n, topp=topp, joint=jd)
            try:
                assert_reservation_owner(args.namespace, reservation)
                cell_id = campaign_cell_id(ds, n, topp=topp, joint=jd)
                planned = snapshot["plan"]["cell_bindings"][cell_id]
                launch_binding = launch_binding_from_expected(
                    planned, snap_digest)
                record = run_cell(ds, n, gpu, epochs=epochs,
                                  namespace=args.namespace, topp=topp,
                                  joint=jd, snapshot=snapshot,
                                  campaign_binding=launch_binding,
                                  result_root=snapshot["plan"]["result_root"])
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
        missing = sorted({campaign_cell_id(
            c[0], c[1], topp=c[2], joint=c[3]) for c in plan} - set(done))
        print(f"[phase3] {len(done)} of {len(plan)} sweep cells complete; "
              f"missing {missing}", file=sys.stderr)
        return 1
    try:
        verify_snapshot_input_seals(snapshot, full=FINAL_BOOKEND_FULL)
    except CellRefused as error:
        print(f"[phase3] REFUSED final input-seal verification: {error}",
              file=sys.stderr)
        return 1
    # The receipt seals the EVIDENCE, not just a count. It used to record
    # `key -> tag` and a number, which a reducer cannot use to check that the
    # cells it is reading are the cells that ran.
    assert_reservation_owner(args.namespace, reservation)
    receipt = {
        "schema_version": SWEEP_RECEIPT_SCHEMA, "axis": args.sweep,
        "campaign_kind": f"recipe_{args.sweep}",
        "namespace": args.namespace,
        "campaign_nonce": campaign_nonce,
        "result_root": snapshot["plan"]["result_root"],
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals": snapshot["input_seals"],
        "input_seals_sha256": _semantic_digest(snapshot["input_seals"]),
        "campaign_reservation_file":
            campaign_reservation_path(args.namespace).name,
        "campaign_reservation_sha256": _sha(
            campaign_reservation_path(args.namespace)),
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
                "campaign": results[k][1]["campaign"],
                "completion": {
                    field: results[k][1]["completion"].get(field)
                    for field in (
                        "final_checkpoint_sha256", "log_csv_sha256",
                        "run_identity_sha256", "args_txt_sha256",
                        "checkpoint_runtime_sha256",
                        "phase3_campaign_evidence_sha256")
                },
            }
            for k in sorted(results)},
    }
    _publish_json_exclusive(
        RECORD_DIR / f"{args.namespace}{RECIPE_RECEIPT_SUFFIX}", receipt)
    print(f"[phase3] {len(plan)} of {len(plan)} sweep cells complete")
    return 0


def n_selection_cells(recipe_choices: dict) -> list:
    """The production D1 plan: chosen recipe, four N values, seed 42."""
    return [
        (dataset, n, tuple(recipe_choices[dataset]["topp"]),
         str(recipe_choices[dataset]["joint"]), "select", SEED)
        for dataset in sorted(DATASETS) for n in CANDIDATE_N]


def refit_cells(selected_n: dict, recipe_choices: dict) -> list:
    """The production D2 plan: one selected N and three fixed seeds."""
    return [
        (dataset, int(selected_n[dataset]),
         tuple(recipe_choices[dataset]["topp"]),
         str(recipe_choices[dataset]["joint"]), "refit", seed)
        for dataset in sorted(DATASETS) for seed in REFIT_SEEDS]


def _run_exact_campaign(args, *, full_plan: list, executed_plan: list,
                        campaign_kind: str, receipt_suffix: str,
                        authorities: dict, epochs: int | None) -> int:
    """Run one exact N/refit campaign under one reservation and snapshot."""
    expected_count = {"n_selection": 16, "refit": 12}[campaign_kind]
    if len(full_plan) != expected_count:
        print(f"[phase3] REFUSED: {campaign_kind} declares {len(full_plan)} "
              f"cells, expected exactly {expected_count}", file=sys.stderr)
        return 2
    if epochs is None and executed_plan != full_plan:
        print(f"[phase3] REFUSED: a production {campaign_kind} must execute all "
              f"{expected_count} cells in one process", file=sys.stderr)
        return 2

    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus
                             else [str(args.gpu)])]
    by_dataset = {}
    for cell in executed_plan:
        by_dataset.setdefault(_campaign_cell_parts(cell)[0], []).append(cell)
    if len(gpus) != 1 and len(gpus) < len(by_dataset):
        print(f"[phase3] {len(by_dataset)} datasets need {len(by_dataset)} GPUs, "
              f"or one GPU for a sequential campaign; --gpus gave {len(gpus)}",
              file=sys.stderr)
        return 2

    campaign_nonce = secrets.token_hex(32)
    try:
        input_seals = verify_campaign_input_seals(
            args.input_seal_specs, full_plan, full=True)
        snapshot = plan_snapshot(
            sorted({_campaign_cell_parts(c)[0] for c in full_plan}),
            plan=full_plan, executed=executed_plan,
            axis="n" if campaign_kind == "n_selection" else "refit",
            namespace=args.namespace, campaign_nonce=campaign_nonce,
            epochs=epochs, campaign_kind=campaign_kind,
            authorities=authorities,
            input_seals=input_seals,
            result_root=getattr(args, "result_root", REPO / "result"),
            gpu_ids=gpus)
        _assert_snapshot_gpu_leases(snapshot, args)
        assert_production_source_authority(snapshot)
        snap_digest = _json_digest(snapshot)
        snap_path = RECORD_DIR / (
            f"{args.namespace}_snapshot_{snap_digest[:16]}.json")
        reservation = reserve_sweep_namespace(
            args.namespace, snapshot=snapshot, plan_digest=snap_digest,
            campaign_nonce=campaign_nonce, snapshot_file=snap_path.name)
        _publish_json_exclusive(snap_path, snapshot)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"[phase3] {campaign_kind} campaign {campaign_nonce[:12]}... "
          f"reserved {len(executed_plan)}/{len(full_plan)} cells -> {snap_path}")
    import threading
    results, lock = {}, threading.Lock()

    def _stream(cells, gpu):
        for cell in cells:
            ds, n, topp, jd, stage, seed = _campaign_cell_parts(cell)
            key = campaign_cell_id(
                ds, n, topp=topp, joint=jd, stage=stage, seed=seed)
            try:
                assert_reservation_owner(args.namespace, reservation)
                planned = snapshot["plan"]["cell_bindings"][key]
                launch_binding = launch_binding_from_expected(
                    planned, snap_digest)
                record = run_cell(
                    ds, n, gpu, epochs=epochs, namespace=args.namespace,
                    stage=stage, seed=seed, topp=topp, joint=jd,
                    snapshot=snapshot, campaign_binding=launch_binding,
                    result_root=snapshot["plan"]["result_root"])
            except Exception as error:                 # noqa: BLE001
                with lock:
                    results[key] = ("failed", str(error))
                return
            with lock:
                results[key] = ("ok", record)

    streams = ([([cell for _, cells in sorted(by_dataset.items())
                 for cell in cells], gpus[0])]
               if len(gpus) == 1 else
               [(cells, gpus[i])
                for i, (_, cells) in enumerate(sorted(by_dataset.items()))])
    threads = [threading.Thread(target=_stream, args=stream, daemon=False)
               for stream in streams]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    planned_keys = {campaign_cell_id(
        ds, n, topp=topp, joint=jd, stage=stage, seed=seed)
        for ds, n, topp, jd, stage, seed in
        (_campaign_cell_parts(c) for c in executed_plan)}
    done = {key for key, (status, _) in results.items() if status == "ok"}
    if done != planned_keys:
        missing = sorted(planned_keys - done)
        print(f"[phase3] {len(done)} of {len(planned_keys)} {campaign_kind} "
              f"cells complete; missing {missing}", file=sys.stderr)
        return 1

    try:
        verify_snapshot_input_seals(snapshot, full=FINAL_BOOKEND_FULL)
    except CellRefused as error:
        print(f"[phase3] REFUSED final input-seal verification: {error}",
              file=sys.stderr)
        return 1

    assert_reservation_owner(args.namespace, reservation)
    receipt = {
        "schema_version": SWEEP_RECEIPT_SCHEMA,
        "axis": "n" if campaign_kind == "n_selection" else "refit",
        "campaign_kind": campaign_kind,
        "namespace": args.namespace,
        "campaign_nonce": campaign_nonce,
        "result_root": snapshot["plan"]["result_root"],
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals": snapshot["input_seals"],
        "input_seals_sha256": _semantic_digest(snapshot["input_seals"]),
        "campaign_reservation_file":
            campaign_reservation_path(args.namespace).name,
        "campaign_reservation_sha256": _sha(
            campaign_reservation_path(args.namespace)),
        "plan_snapshot_sha256": snap_digest,
        "plan_snapshot_file": snap_path.name,
        "expected_cells": len(executed_plan),
        "declared_cells": len(full_plan),
        "cell_count": len(done),
        "cells": {},
    }
    for key in sorted(done):
        record = results[key][1]
        receipt["cells"][key] = {
            "dataset": record["dataset"], "N": record["N"],
            "seed": record["seed"], "stage": record["stage"],
            "tag": record["tag"], "run_dir": record["run_dir"],
            "identity_digest": record["geometry"]["identity_digest"],
            "record": f"{record['tag']}.json",
            "record_sha256": _sha(RECORD_DIR / f"{record['tag']}.json"),
            "recipe": record["recipe"], "campaign": record["campaign"],
            # Exact copy, rather than a permissive subset.  For a refit this
            # includes extraction manifests/NPZs/evaluation identity as well as
            # checkpoint/runtime/manifest/args hashes.
            "completion": record["completion"],
        }
    try:
        _publish_json_exclusive(
            RECORD_DIR / f"{args.namespace}{receipt_suffix}", receipt)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
    if epochs is None:
        try:
            if campaign_kind == "n_selection":
                stability_authority = (snapshot["plan"].get("authorities")
                                       or {}).get("stability_stage")
                if stability_authority is not None:
                    from scripts.phase3_select_n import (
                        _stability_n_matrix, load_stability_stage_authority)
                    stage_path = Path(stability_authority["path"])
                    reopened = load_stability_stage_authority(stage_path)
                    if reopened != stability_authority:
                        raise CellRefused(
                            "stability N stage authority changed after launch")
                    stage_payload = json.loads(
                        stage_path.read_text(encoding="utf-8"))
                    _stability_n_matrix(
                        RECORD_DIR, namespace=args.namespace,
                        state=stage_payload["state"],
                        stage_authority=stability_authority)
                else:
                    from scripts.phase3_select_n import load_matrix
                    load_matrix(RECORD_DIR, namespace=args.namespace)
            else:
                from scripts.phase3_select_n import (
                    REFIT_AGGREGATE_SUFFIX, verify_refit_aggregate,
                    verify_refit_campaign)
                aggregate = verify_refit_campaign(
                    RECORD_DIR, namespace=args.namespace)
                aggregate_path = RECORD_DIR / (
                    f"{args.namespace}{REFIT_AGGREGATE_SUFFIX}")
                _publish_json_exclusive(aggregate_path, aggregate)
                verify_refit_aggregate(
                    aggregate_path, records_dir=RECORD_DIR,
                    namespace=args.namespace)
        except Exception as error:                     # noqa: BLE001
            print(f"[phase3] REFUSED completed {campaign_kind} receipt: {error}",
                  file=sys.stderr)
            return 1
    print(f"[phase3] {len(done)} of {len(full_plan)} {campaign_kind} cells complete")
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
        "--result-root", default=str(REPO / "result"), metavar="ABS_PATH",
        help=("exact root for newly produced Phase-3 run directories; the "
              "default remains repository/result, while production may use "
              "a fresh absolute /data root"))
    parser.add_argument(
        "--input-seal", action="append", default=[],
        metavar="DATASET:STAGE=ABS_PATH",
        help=("exact immutable input seal for every dataset/stage in the "
              "declared campaign; repeat once per required coordinate"))
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
        help=("the one final phase3_recipe_stability artefact. Production N "
              "and refit reopen every P/J/N campaign and replay the fixed "
              "point; provisional per-axis files are refused."))
    parser.add_argument(
        "--stability-plan", default=None, metavar="PATH",
        help=("prelaunch stage plan emitted for one bootstrap/confirmation "
              "P, JD, or N campaign. It carries per-dataset current N/P/J, "
              "so no global --at-topp/manual multi-namespace composition is "
              "used by the stability protocol."))
    parser.add_argument(
        "--at-topp", default=None, metavar="MIN,MAX",
        help="hold top-p here (required by --sweep joint)")
    args = parser.parse_args()

    try:
        args.input_seal_specs = parse_input_seal_specs(args.input_seal)
    except CellRefused as error:
        print(f"[phase3] REFUSED --input-seal: {error}", file=sys.stderr)
        return 2

    try:
        args.result_root = str(_canonical_result_root(args.result_root))
    except CellRefused as error:
        print(f"[phase3] REFUSED --result-root: {error}", file=sys.stderr)
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

    if args.stability_plan:
        try:
            from scripts.phase3_select_n import (
                load_stability_stage_authority, stability_stage_cells)
            stage_path = Path(args.stability_plan).resolve()
            stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
            stage_authority = load_stability_stage_authority(stage_path)
            stage_plan = stability_stage_cells(stage_payload)
        except Exception as error:                     # noqa: BLE001
            print(f"[phase3] REFUSED --stability-plan: {error}",
                  file=sys.stderr)
            return 2
        stage_axis = stage_authority["axis"]
        if stage_axis in ("topp", "joint"):
            if args.sweep != stage_axis or args.refit or args.recipe:
                print("[phase3] REFUSED: a P/J stability plan requires the "
                      f"matching --sweep {stage_axis} and no recipe/refit",
                      file=sys.stderr)
                return 2
            if args.at_topp:
                print("[phase3] REFUSED: a stability plan carries exact "
                      "per-dataset top-p; --at-topp is forbidden",
                      file=sys.stderr)
                return 2
            if args.plan or not (args.run or args.smoke):
                return _run_sweep(
                    args, None, full_plan=stage_plan,
                    authorities={"stability_stage": stage_authority})
            if args.only and not args.smoke:
                print("[phase3] REFUSED: --only cannot narrow a production "
                      "P/J stability campaign; use it only with --smoke",
                      file=sys.stderr)
                return 2
            return _with_campaign_gpu_leases(
                args, lambda: _run_sweep(
                    args, None, full_plan=stage_plan,
                    authorities={"stability_stage": stage_authority}))
        if args.sweep or args.refit or args.recipe or args.at_topp:
            print("[phase3] REFUSED: an N stability plan cannot be combined "
                  "with sweep/refit/recipe/at-topp", file=sys.stderr)
            return 2
        if args.plan or not (args.run or args.smoke):
            print(f"{len(stage_plan)} cells for stability "
                  f"{stage_authority['phase']} round "
                  f"{stage_authority['round_index']}")
            for dataset, n, topp, joint, _, _ in stage_plan:
                print(f"  {dataset:<10} N={n:<3} "
                      f"topp={topp[0]}/{topp[1]} jd={joint}")
            return 0
        executed = stage_plan
        if args.only:
            try:
                ds, raw_n = args.only.split(":")
                key = (ds, int(raw_n))
            except ValueError:
                print(f"[phase3] --only must be dataset:N, got "
                      f"{args.only!r}", file=sys.stderr)
                return 2
            executed = [row for row in stage_plan
                        if (row[0], row[1]) == key]
            if len(executed) != 1 or not args.smoke:
                print("[phase3] REFUSED: --only may narrow a stability N plan "
                      "only for a one-cell smoke", file=sys.stderr)
                return 2
        return _with_campaign_gpu_leases(
            args, lambda: _run_exact_campaign(
                args, full_plan=stage_plan, executed_plan=executed,
                campaign_kind="n_selection",
                receipt_suffix=N_SELECTION_RECEIPT_SUFFIX,
                authorities={"stability_stage": stage_authority},
                epochs=(args.epochs if args.smoke else None)))

    if args.sweep:
        if args.plan or not (args.run or args.smoke):
            return _run_sweep(args, sweep_topp)
        if args.smoke:
            if not args.only:
                print("[phase3] REFUSED: a direct P/J smoke must name exactly "
                      "one coordinate with --only", file=sys.stderr)
                return 2
            return _with_campaign_gpu_leases(
                args, lambda: _run_sweep(args, sweep_topp))
        print("[phase3] REFUSED: an executable P/J sweep requires an exact "
              "--stability-plan; direct candidate-shaped campaigns cannot "
              "participate in the fixed-point authority", file=sys.stderr)
        return 2

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
    if args.only and not args.smoke:
        if args.refit:
            print("[phase3] --only narrows the refit to one dataset, which "
                  "cannot produce the twelve cells; use it with --smoke",
                  file=sys.stderr)
        else:
            print("[phase3] --only cannot narrow the exact-16 production N "
                  "selection; use it only with --smoke", file=sys.stderr)
        return 2
    try:
        recipe_authority = load_recipe_authority(args.recipe)
        recipe = verify_recipe_authority(recipe_authority)
    except CellRefused as error:
        print(f"[phase3] REFUSED --recipe: {error}", file=sys.stderr)
        return 2

    if args.refit:
        # Production is exactly twelve. A narrowed invocation is only a smoke,
        # and its snapshot records that it executed a strict subset.
        try:
            selection_path = Path(args.selection).resolve()
            selection = _load_selection(selection_path)
        except CellRefused as error:
            print(f"[phase3] REFUSED --selection: {error}", file=sys.stderr)
            return 2
        chosen = selection["selected_n"]
        if selection["recipe_authority"] != recipe_authority \
                or selection["recipe_choices"] != recipe:
            print("[phase3] REFUSED --selection: selected N was derived under "
                  "a different stable recipe authority", file=sys.stderr)
            return 2
        selection_authority = {
            "schema_version": SELECTED_N_AUTHORITY_SCHEMA,
            "path": str(selection_path),
            "sha256": _sha(selection_path), "selected_n": chosen,
            "recipe_authority": selection["recipe_authority"],
            "recipe_choices": selection["recipe_choices"],
        }
        full_plan = refit_cells(chosen, recipe)
        executed = full_plan
        if args.only:
            executed = [cell for cell in full_plan
                        if _campaign_cell_parts(cell)[0] == keys[0][0]]
        return _with_campaign_gpu_leases(
            args, lambda: _run_exact_campaign(
                args, full_plan=full_plan, executed_plan=executed,
                campaign_kind="refit", receipt_suffix=REFIT_RECEIPT_SUFFIX,
                authorities={"recipe": recipe_authority,
                             "selected_n": selection_authority},
                epochs=epochs))

    full_plan = n_selection_cells(recipe)
    executed = full_plan
    if args.only:
        executed = [cell for cell in full_plan
                    if (_campaign_cell_parts(cell)[0],
                        _campaign_cell_parts(cell)[1]) == keys[0]]
    return _with_campaign_gpu_leases(
        args, lambda: _run_exact_campaign(
            args, full_plan=full_plan, executed_plan=executed,
            campaign_kind="n_selection",
            receipt_suffix=N_SELECTION_RECEIPT_SUFFIX,
            authorities={"recipe": recipe_authority}, epochs=epochs))


if __name__ == "__main__":
    raise SystemExit(main())
