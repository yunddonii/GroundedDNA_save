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
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.run_identity import MANIFEST_NAME, load_run_manifest  # noqa: E402

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

NAMESPACE = os.environ.get("PHASE3_NAMESPACE", "phase3sel")
RECORD_DIR = REPO / "artifacts" / "phase3_selection"


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


def _whitening(spec: dict) -> str:
    """Stage 1 fits on the optimization-train rows only."""
    return f"{spec['foils']}/text_whiten_optTrain_localOnly.npz"


def build_command(dataset: str, n: int, gpu: int, *,
                  epochs: int | None = None,
                  namespace: str = NAMESPACE) -> tuple:
    spec = DATASETS[dataset]
    tag = tag_for(dataset, n, namespace=namespace)
    flags = _stage1_flags(n) + S5_FLAGS + A_FLAGS + QUIET_FLAGS
    if epochs is not None:                      # smoke only
        flags = ["-e", str(epochs), "--sinkhorn_schedule_horizon",
                 str(epochs), "--lr_schedule_horizon", str(epochs)] + \
            flags[6:]
    env = dict(os.environ)
    env.update(spec["env"])
    env.update(
        # The geometry is baked in at import time, so it has to be in the
        # environment before python starts -- passing the CLI flag alone aborts.
        GDNA_NUM_SEMANTIC_PARTS=str(SLOTS),
        CUDA_VISIBLE_DEVICES=str(gpu),
        CACHE=spec["cache"], EVAL_CACHE=spec["cache"], QWEN=spec["qwen"],
        WHITEN_NPZ=_whitening(spec), K=str(spec["K"]),
        NUM_CODONS=str(BASES_PER_SLOT),
        VAL_RATIO=str(VAL_RATIO), VAL_SEED=str(VAL_SEED),
        TAG=tag,
        EXTRA_ARGS=" ".join(shlex.quote(f) for f in flags),
    )
    # Stage 1 never evaluates the official test split: no FINAL_EPOCH, and
    # `--stop_after_epoch N` stops the run at the candidate epoch.
    env["STOP_EP"] = str(n)
    env.pop("FINAL_EPOCH", None)
    return ["bash", spec["trainer"], str(gpu)], env, tag


# --------------------------------------------------------------- checking

def _arg_value(args_txt: Path, field: str) -> str:
    prefix = field + "-"
    for line in args_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].lstrip("-").strip()
    raise CellRefused(f"{args_txt} has no field {field!r}")


def assert_geometry(run_dir: Path, *, dataset: str, n: int) -> dict:
    """The check the 18-base smoke did not have.

    Both the effective arguments and the sealed run manifest must say the
    paper's geometry. Either alone can be right while the other is not: the CLI
    flag without the environment variable aborts, and the environment variable
    without the flag would leave `args.txt` describing something else.
    """
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
    manifest_bad = {
        k: (got, exp) for k, got, exp in (
            ("num_slots", identity.num_slots, SLOTS),
            ("bases_per_slot", identity.bases_per_slot, BASES_PER_SLOT),
            ("total_bases", identity.total_bases, TOTAL_BASES),
            ("total_bits", identity.total_bits, TOTAL_BITS),
            ("dataset", identity.dataset, DATASETS[dataset]["canon"]),
            ("seed", identity.seed, SEED),
            ("stop_after_epoch", identity.stop_after_epoch, n),
            ("codebook_size", identity.codebook_size, DATASETS[dataset]["K"]),
        ) if got != exp}
    if manifest_bad:
        raise CellRefused(f"{run_dir}: manifest disagrees {manifest_bad}")
    return {"identity_digest": identity.digest, **effective}


def read_selection(run_dir: Path) -> dict:
    """The candidate's score, from the checkpoint that scored it.

    `model_state_dict_best.pth.runtime.json` records `selection_metric` and
    `selection_value` beside the checkpoint's own SHA, so the number is bound to
    the weights it describes. The previous runner grepped a log line instead.
    """
    sidecar = run_dir / "model_state_dict_best.pth.runtime.json"
    if not sidecar.is_file():
        raise CellRefused(
            f"{run_dir}: no best-checkpoint sidecar, so the cell selected "
            f"nothing")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    extra = payload.get("extra") or {}
    metric = extra.get("selection_metric")
    value = extra.get("selection_value")
    if metric != "eval_mAP_at_R":
        raise CellRefused(
            f"{run_dir}: selection metric is {metric!r}; D1 requires raw "
            f"base-Hamming mAP@R")
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not 0.0 <= float(value) <= 1.0:
        raise CellRefused(f"{run_dir}: selection value {value!r} is not a "
                          f"proportion")
    return {
        "selection_metric": metric,
        "selection_value": float(value),
        "best_epoch_zero_based": payload.get("checkpoint_epoch_zero_based"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "effective_sinkhorn_epsilon": payload.get("effective_sinkhorn_epsilon"),
        "lr_schedule_horizon": payload.get("lr_schedule_horizon"),
        "sinkhorn_schedule_horizon": payload.get("sinkhorn_schedule_horizon"),
    }


def _resolve(tag: str) -> Path:
    hits = sorted(p for p in (REPO / "result").glob(f"*{tag}*") if p.is_dir())
    if len(hits) != 1:
        raise CellRefused(
            f"expected exactly one result for {tag}, found {len(hits)}: "
            f"{[p.name for p in hits]}")
    return hits[0]


def run_cell(dataset: str, n: int, gpu: int, *, epochs=None,
             namespace: str = NAMESPACE) -> dict:
    tag = tag_for(dataset, n, namespace=namespace)
    existing = _existing_artifacts(tag)
    if existing:
        raise CellRefused(
            f"{tag} already has artefacts: {existing[:3]}. Give the run its own "
            f"namespace with PHASE3_NAMESPACE.")

    cmd, env, _ = build_command(dataset, n, gpu, epochs=epochs,
                                namespace=namespace)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise CellRefused(f"{tag}: trainer exited {proc.returncode}")

    run_dir = _resolve(tag)
    record = {
        "dataset": dataset, "N": n, "tag": tag,
        "run_dir": str(run_dir),
        "seed": SEED, "val_split_ratio": VAL_RATIO, "val_split_seed": VAL_SEED,
        "lr_schedule_horizon": LR_HORIZON if epochs is None else epochs,
        "sinkhorn_schedule_horizon": (n + 1) if epochs is None else epochs,
        "epoch_budget": EPOCH_BUDGET if epochs is None else epochs,
        "geometry": assert_geometry(run_dir, dataset=dataset, n=n),
        "selection": read_selection(run_dir),
        "wall_seconds": round(time.time() - started, 1),
        "smoke": epochs is not None,
    }
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORD_DIR / f"{tag}.json"
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true",
                        help="print the exact cell set and one command")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short cell, to prove the geometry asserts")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1,
                        help="--smoke only")
    parser.add_argument("--only", default=None,
                        help="dataset:N, e.g. cifar10:4")
    parser.add_argument("--namespace", default=NAMESPACE)
    args = parser.parse_args()

    keys = cell_keys()
    if args.only:
        ds, n = args.only.split(":")
        keys = [(ds, int(n))]

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
              f"@epoch {record['selection']['best_epoch_zero_based']}")

    if failures:
        print(f"[phase3] {len(failures)} of {len(keys)} cells refused: "
              f"{failures}", file=sys.stderr)
        return 1
    print(f"[phase3] {len(keys)} of {len(keys)} cells complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
