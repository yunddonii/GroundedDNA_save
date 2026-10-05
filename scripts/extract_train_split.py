#!/usr/bin/env python
"""Additively emit `extract_train.npz` for a finished run (REQUIRED_EXPERIMENTS §2.4).

`extraction_siglip2.py` only encodes db + query. Held-out codon decoding also
needs the TRAIN rows, because the `(slot, code) -> concept` dictionary is built
on train and evaluated on test.

For Flickr25k and NUS-WIDE `train` is a subset of the DB extraction, so the
train codes can simply be sliced out by image basename and this script is not
needed. MSCOCO's train split is disjoint from its DB, so its train codes have
to be encoded explicitly -- that is what this is for.

The db/query schema is untouched; this only adds a file.

    python scripts/extract_train_split.py --config_path <result_dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, set_random_seed
from dataloaders import load_dataset
from dna_utils import get_transform
from extraction_siglip2 import (
    encode_split, _find_model_checkpoint, _resume_args_flat_or_legacy,
)
from model_siglip2 import SigLIP2SemanticOTModel

#: Stage T (generation v9; audits 756, 759-760): the stage-T launcher names here the pinned identities
#: of the stage-R cell this run consumes and its admitted terminal epoch. Outside stage T none is set
#: and nothing below changes.
T_INPUT_PINS = {"config": "GDNA_T_EXPECT_CONFIG_SHA256",
                "checkpoint": "GDNA_T_EXPECT_CHECKPOINT_SHA256",
                "runtime": "GDNA_T_EXPECT_RUNTIME_SHA256",
                "epoch": "GDNA_T_EXPECT_TERMINAL_EPOCH"}


def stage_t_pins():
    """The three stage-T input pins, or None outside stage T; a partial set refuses."""
    values = {key: os.environ.get(name) for key, name in T_INPUT_PINS.items()}
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise SystemExit("[extract-train] REFUSED: incomplete stage-T input pins")
    return values


def _pinned_bytes(path, digest, what):
    import hashlib
    with open(path, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"[extract-train] REFUSED: {what} {path} is not the stage-R cell's pinned bytes")
    return raw


def stage_t_inputs(args, pins):
    """Stage T: config.pt, the checkpoint and its runtime witness, each read once and verified against
    the pins BEFORE any deserialization or model construction. The arguments are built from the
    verified config object (the shared resume helper's flat step, never a second read; audit 754.2);
    the checkpoint and witness are bound at the admitted terminal epoch (audits 759-760), so the
    weights load from the verified bytes and the epoch resolves from the verified witness.
    Returns (checkpoint path, dna_utils.runtime_state.VerifiedRuntime)."""
    import io
    from dna_utils.runtime_state import CheckpointMetadata, RuntimeBindingRefused, verified_runtime
    from extraction_siglip2 import _apply_saved_config, _reapply_explicit_cli
    cli = Config.get_config()
    run = cli.config_path
    raw_config = _pinned_bytes(os.path.join(run, "config.pt"), pins["config"], "config.pt")
    ckpt = _find_model_checkpoint(os.path.join(run, ""))
    raw_ckpt = _pinned_bytes(ckpt, pins["checkpoint"], "checkpoint")
    raw_witness = _pinned_bytes(CheckpointMetadata.sidecar_path(ckpt), pins["runtime"], "runtime witness")
    try:
        runtime = verified_runtime(ckpt, raw_ckpt, raw_witness, checkpoint_sha256=pins["checkpoint"],
                                   witness_sha256=pins["runtime"], terminal_epoch=int(pins["epoch"]))
    except (RuntimeBindingRefused, ValueError) as error:
        raise SystemExit(f"[extract-train] REFUSED: {error}") from None
    _apply_saved_config(args, torch.load(io.BytesIO(raw_config), map_location="cpu"), run)
    _reapply_explicit_cli(args, cli)
    return ckpt, runtime


def recheck_stage_t_inputs(ckpt, pins):
    """Before anything is written: the three consumed files are still their pinned bytes."""
    from dna_utils.runtime_state import CheckpointMetadata
    run = os.path.dirname(ckpt)
    _pinned_bytes(os.path.join(run, "config.pt"), pins["config"], "config.pt")
    _pinned_bytes(ckpt, pins["checkpoint"], "checkpoint")
    _pinned_bytes(CheckpointMetadata.sidecar_path(ckpt), pins["runtime"], "runtime witness")


def main() -> None:
    set_random_seed(42)
    args = Config()
    pins = stage_t_pins()
    if pins is None:
        _resume_args_flat_or_legacy(args)
    else:
        pinned_ckpt, runtime = stage_t_inputs(args, pins)

    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = _find_model_checkpoint(args.save_model_state_path)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    if pins is not None and os.path.realpath(ckpt) != os.path.realpath(pinned_ckpt):
        raise SystemExit(f"[extract-train] REFUSED: {ckpt} is not the verified checkpoint {pinned_ckpt}")
    from dna_utils.run_identity import load_model_state_dict_for_extraction
    import io
    missing, unexpected, phase3_binding = load_model_state_dict_for_extraction(
        model, ckpt if pins is None else io.BytesIO(runtime.checkpoint_bytes), map_location=args.device)
    print(f"[extract-train] loaded {ckpt} "
          f"(missing={len(missing)} unexpected={len(unexpected)} "
          f"phase3_exact={phase3_binding is not None})")
    # F01: same fail-closed epoch restore as extraction_siglip2, via the SAME
    # helper. Duplicating the resolution logic is how the two paths drifted into
    # producing train and query/DB codes at different operating points.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dna_utils.runtime_state import apply_inference_epoch
    if pins is None:
        _resolved = apply_inference_epoch(model, ckpt, args)
    else:   # the verified witness, at the admitted terminal epoch, before any dataset access
        _resolved = apply_inference_epoch(model, ckpt, args, verified=runtime)
    print(f"[extract-train] inference epoch={_resolved.epoch} "
          f"(source={_resolved.source}) "
          f"effective_sinkhorn_epsilon={_resolved.effective_sinkhorn_epsilon}")

    transform = get_transform("test")           # no augmentation for extraction
    trainset, _, _ = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=transform, test_transform=transform,
        load_train=True, load_database=False, load_test=False,
        return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None),
    )

    loader = torch.utils.data.DataLoader(
        trainset, batch_size=int(getattr(args, "extract_batch_size", 256)),
        shuffle=False, num_workers=args.num_workers, drop_last=False,
    )
    out = encode_split(model, loader, args.device, split_name="train")

    if pins is not None:
        recheck_stage_t_inputs(ckpt, pins)
    dest = os.path.join(args.save_result_path, "extract_train.npz")
    np.savez(dest, **out)
    # F01: the train split shares the resolver, so it must share the manifest --
    # otherwise the train NPZ has no recorded operating point and the three
    # splits cannot be shown to have run under the same runtime state.
    from extraction_siglip2 import (
        _write_completion_marker, _write_split_manifest)
    from dna_utils.runtime_state import sha256_file

    # §17.7: the train split used to be written outside the run marker, so a
    # DB/query marker could be valid while train was partial, stale, or from a
    # different runtime. Adding train re-commits the marker as a new
    # transaction over all three splits.
    train_manifest = _write_split_manifest(
        args.save_result_path, "train", "extract_train.npz", out,
        args=args, checkpoint_path=ckpt, resolved=_resolved)
    manifests = {"train": train_manifest}
    for split in ("db", "query"):
        path = os.path.join(args.save_result_path,
                            f"extraction_manifest_{split}.json")
        if os.path.isfile(path):
            manifests[split] = path
    _write_completion_marker(args.save_result_path, manifests,
                             required_splits=tuple(sorted(manifests)))
    print(f"[extract-train] {out['base_indices'].shape[0]} rows -> {dest}")


if __name__ == "__main__":
    main()
