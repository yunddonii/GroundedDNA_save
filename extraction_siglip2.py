"""Extraction pipeline for ``SigLIP2SemanticOTModel``.

Replaces the legacy `extraction.py` (which uses the old `model.Model`).
Loads a checkpoint, runs the model in eval mode (codebook_mean routing,
no text input), and saves both the per-base index code and the 2-bit
DB-compatible code for db / query splits.

Output (per split, saved as .npz):
    base_indices       [N, 18]
    hash_2bit          [N, 36]   (uint8, DB-compatible)
    codebook_indices   [N, 6]
    labels             [N]                     (if single-label)
    multi_hot_labels   [N, num_classes]        (if multi-hot)
    image_paths        [N]                     (if available)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from config import Config, set_random_seed
from dataloaders import load_dataset
from model_siglip2 import SigLIP2SemanticOTModel
# `get_transform` is now mirrored in dna_utils so we don't have to import the
# legacy `utils.py` (which depends on DWKM / sinkhornbarycenters).
from dna_utils import base_indices_to_2bit_flat, get_transform


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _get_pixel_values(batch: Dict[str, Any]) -> torch.Tensor:
    pv = batch.get("pixel_values", batch.get("img"))
    if pv is None:
        raise RuntimeError(
            "[extraction_siglip2] batch must contain 'pixel_values' or 'img' key."
        )
    return pv


def _get_labels(batch: Dict[str, Any]):
    """Resolve labels with auto-detection of multi-hot vs single-class.

    The legacy `ImgRtvDataset` (Flickr25k / MSCOCO / NUSWIDE / ImageNet100)
    puts a multi-hot vector ``[num_classes]`` in ``batch['label']`` --
    the same key that single-class datasets use for the int class id. This
    helper routes the tensor to the correct slot so the evaluation Jaccard
    relevance matrix uses multi-hot semantics for multi-label datasets.
    """
    labels = batch.get("labels", batch.get("label", batch.get("target", None)))
    multi_hot_labels = batch.get(
        "multi_hot_labels", batch.get("multi_label", None)
    )
    # Auto-promote: if `labels` is [B, C] with C > 1, treat as multi-hot.
    if (labels is not None and hasattr(labels, "ndim")
            and labels.ndim == 2 and labels.shape[1] > 1):
        if multi_hot_labels is None:
            multi_hot_labels = labels
        labels = None
    return labels, multi_hot_labels


def encode_split(
    model: SigLIP2SemanticOTModel,
    loader: torch.utils.data.DataLoader,
    device,
    split_name: str = "",
) -> Dict[str, np.ndarray]:
    """Run the model on a split and accumulate extraction tensors.

    v122: shape M*L (=18 at L=3, =24 at L=4) and M*L*2 (=36, =48) are
    inferred from the model's `num_codons_per_codebook` instead of being
    hardcoded so this function transparently supports the v122a L=4 path.
    """
    n_samples = len(loader.dataset)
    M = int(getattr(model, "num_codebooks", 6))
    L = int(getattr(model, "num_codons_per_codebook", 3))
    base_idx_all = np.zeros((n_samples, M * L),     dtype=np.int64)
    hash2bit_all = np.zeros((n_samples, M * L * 2), dtype=np.uint8)
    cb_idx_all   = np.zeros((n_samples, M),         dtype=np.int64)
    labels_list:    List[np.ndarray] = []
    mh_list:        List[np.ndarray] = []
    paths_list:     List[str]        = []

    write_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract[{split_name}]"):
            cached_vt = batch.get("cached_visual_tokens_raw", None)
            if cached_vt is not None:
                cached_vt = cached_vt.to(device)
                cached_vg = batch.get("cached_visual_global", None)
                cached_vg = cached_vg.to(device) if cached_vg is not None else None
                B = cached_vt.shape[0]
                outputs = model(
                    pixel_values=None,
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                    cached_visual_tokens_raw=cached_vt,
                    cached_visual_global=cached_vg,
                    cached_text_part_raw=None,    # not needed at eval
                    cached_has_text=None,
                )
            else:
                pixel_values = _get_pixel_values(batch).to(device)
                B = pixel_values.shape[0]
                outputs = model(
                    pixel_values=pixel_values,
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                )
            assert outputs["routing_mode"] == "codebook_mean", (
                f"[extraction] expected codebook_mean routing at eval, got "
                f"{outputs['routing_mode']!r}"
            )
            base_indices = outputs["base_indices"]               # [B, 18]
            cb_indices   = outputs["codebook_indices"]           # [B, 6]
            hash_2bit    = base_indices_to_2bit_flat(base_indices)  # [B, 36]

            base_idx_all[write_idx:write_idx + B] = _to_numpy(base_indices)
            cb_idx_all  [write_idx:write_idx + B] = _to_numpy(cb_indices)
            hash2bit_all[write_idx:write_idx + B] = _to_numpy(hash_2bit).astype(np.uint8)

            labels, multi_hot_labels = _get_labels(batch)
            if labels is not None:
                labels_list.append(_to_numpy(labels))
            if multi_hot_labels is not None:
                mh_list.append(_to_numpy(multi_hot_labels))
            if "image_path" in batch:
                # default collate makes this a list of strings already
                paths_list.extend(list(batch["image_path"]))

            write_idx += B

    out: Dict[str, np.ndarray] = {
        "base_indices":     base_idx_all[:write_idx],
        "hash_2bit":        hash2bit_all[:write_idx],
        "codebook_indices": cb_idx_all  [:write_idx],
    }
    if labels_list:
        labels_arr = np.concatenate(labels_list, axis=0)
        # Existing dataloaders return labels as [N, C] one-hot/multi-hot.
        # Detect single-label by max-1-on-row and emit both forms.
        if labels_arr.ndim == 2:
            out["multi_hot_labels"] = labels_arr.astype(np.int64)
            row_sums = labels_arr.sum(axis=1)
            if (row_sums == 1).all():
                out["labels"] = labels_arr.argmax(axis=1).astype(np.int64)
        else:
            out["labels"] = labels_arr.astype(np.int64)
    if mh_list:
        out["multi_hot_labels"] = np.concatenate(mh_list, axis=0).astype(np.int64)
    if paths_list:
        out["image_paths"] = np.asarray(paths_list, dtype=np.str_)
    return out


def _find_model_checkpoint(save_model_state_path: str) -> str:
    """Locate the trained model checkpoint, preferring the new flat name.

    Preference order:
        1. ``<dir>/model_state_dict.pth``     (current SigLIP2 layout)
        2. ``<dir>/model_siglip2.pt``         (legacy SigLIP2 name)
    """
    candidates = [
        os.path.join(save_model_state_path, "model_state_dict.pth"),
        os.path.join(save_model_state_path, "model_siglip2.pt"),  # legacy
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # return preferred path even if missing (for warn msg)


class MissingCheckpoint(RuntimeError):
    """Extraction was asked to run without weights."""


def _require_checkpoint(path: str) -> None:
    """Abort rather than extract from a freshly initialised model.

    The previous behaviour printed a warning and continued, so a run with no
    checkpoint produced complete-looking NPZs from random weights -- and nothing
    downstream could tell them apart from a real extraction.
    """
    if not os.path.isfile(path):
        raise MissingCheckpoint(
            f"no checkpoint at {path}. Extraction from an untrained model "
            f"would emit artefacts indistinguishable from a real run; fix the "
            f"path or the run rather than proceeding.")


class ManifestBindingError(RuntimeError):
    """A manifest disagreed with the artefacts it claims to describe."""


def _write_split_manifest(out_dir: str, split: str, npz_name: str,
                          payload: dict, *, args: Config,
                          checkpoint_path: str, resolved) -> str:
    """Record what this split was extracted from, next to the NPZ itself.

    Every field is checked against the files before anything is written. The
    first version recorded whatever it was handed, so a manifest could declare a
    checkpoint SHA matching no file, 7 rows against a 2-row NPZ and 15 bases
    against a 12-wide code -- and all of it was accepted. A manifest that can
    disagree with its own artefacts is not provenance.
    """
    from dna_utils.runtime_state import sha256_file, write_extraction_manifest

    npz_path = os.path.join(out_dir, npz_name)
    if not os.path.isfile(npz_path):
        raise ManifestBindingError(
            f"{npz_path} does not exist; the manifest is written after the NPZ "
            f"so that its digest describes the file on disk")
    if not os.path.isfile(checkpoint_path):
        raise ManifestBindingError(
            f"checkpoint {checkpoint_path} does not exist")

    declared_sha = getattr(resolved, "checkpoint_sha256", None)
    actual_sha = sha256_file(checkpoint_path)
    if not declared_sha:
        raise ManifestBindingError(
            f"the resolver returned no checkpoint SHA for {checkpoint_path}. A "
            f"manifest that cannot name the weights it used records nothing.")
    if declared_sha != actual_sha:
        raise ManifestBindingError(
            f"declared checkpoint sha256 {declared_sha[:12]}... does not match "
            f"{checkpoint_path} ({actual_sha[:12]}...)")

    # Row count and geometry come from the NPZ, not from the in-memory payload:
    # the file is what downstream reads.
    with np.load(npz_path, allow_pickle=False) as stored:
        if "base_indices" not in stored:
            raise ManifestBindingError(f"{npz_path} has no base_indices")
        stored_codes = np.asarray(stored["base_indices"])
        stored_hash = (np.asarray(stored["hash_2bit"])
                       if "hash_2bit" in stored else None)
        stored_cb = (np.asarray(stored["codebook_indices"])
                     if "codebook_indices" in stored else None)
    # A scalar or 1-D array used to raise IndexError on `.shape[1]`; that is a
    # domain error about the artefact, not a crash.
    if stored_codes.ndim != 2:
        raise ManifestBindingError(
            f"{npz_path} base_indices has rank {stored_codes.ndim}, expected 2")
    n_rows = int(stored_codes.shape[0])
    payload_codes = payload.get("base_indices")
    if payload_codes is not None:
        declared_rows = int(np.asarray(payload_codes).shape[0])
        if declared_rows != n_rows:
            raise ManifestBindingError(
                f"{split}: payload declares {declared_rows} rows but "
                f"{npz_name} holds {n_rows}")

    _flat = os.path.join(out_dir, "config.pt")
    _nested = os.path.join(out_dir, "model_state", "config.pt")
    _config_path = _flat if os.path.isfile(_flat) else _nested
    if not os.path.isfile(_config_path):
        raise ManifestBindingError(
            f"no config.pt at {_flat} or {_nested}; the manifest cannot record "
            f"which configuration produced this extraction")

    slots = int(getattr(args, "num_semantic_parts", 0) or 0)
    per_slot = int(getattr(args, "num_codons_per_codebook", 0) or 0)
    expected_bases = slots * per_slot
    actual_bases = int(stored_codes.shape[1]) if stored_codes.ndim == 2 else -1
    if expected_bases != actual_bases:
        raise ManifestBindingError(
            f"{split}: config declares {slots}x{per_slot} = {expected_bases} "
            f"bases but {npz_name} holds base_indices of width {actual_bases}")

    if stored_codes.size and (stored_codes.min() < 0 or stored_codes.max() > 3):
        raise ManifestBindingError(
            f"{split}: base_indices holds values outside 0..3 "
            f"[{stored_codes.min()}, {stored_codes.max()}]")

    # The companion arrays are what downstream retrieval and codebook analyses
    # read, so their geometry is part of the same contract.
    if stored_hash is None:
        raise ManifestBindingError(f"{npz_path} has no hash_2bit")
    if stored_hash.shape != (n_rows, 2 * expected_bases):
        raise ManifestBindingError(
            f"{split}: hash_2bit is {stored_hash.shape}, expected "
            f"{(n_rows, 2 * expected_bases)}")
    if stored_cb is None:
        raise ManifestBindingError(f"{npz_path} has no codebook_indices")
    if stored_cb.shape != (n_rows, slots):
        raise ManifestBindingError(
            f"{split}: codebook_indices is {stored_cb.shape}, expected "
            f"{(n_rows, slots)}")

    codes = stored_codes
    return write_extraction_manifest(
        os.path.join(out_dir, f"extraction_manifest_{split}.json"),
        checkpoint_path=checkpoint_path,
        resolved=resolved,
        num_slots=int(getattr(args, "num_semantic_parts", 0) or 0),
        bases_per_slot=int(getattr(args, "num_codons_per_codebook", 0) or 0),
        split=split,
        n_rows=n_rows,
        lr_schedule_horizon=getattr(args, "lr_schedule_horizon", None)
        or getattr(args, "epoch", None),
        training_epoch_budget=getattr(args, "epoch", None),
        # `stop_after_epoch` is unset on a normal full run, where the effective
        # stop is `epoch - 1`; recording the raw flag wrote null into the
        # manifest for exactly the runs that matter most.
        training_stop_epoch=_effective_stop_epoch(args),
        extra={
            "npz_path": os.path.abspath(npz_path),
            "npz_sha256": sha256_file(npz_path),
            # Resume supports a flat `<out>/config.pt` and a legacy nested
            # `<out>/model_state/config.pt`; recording only the flat one left a
            # nested run with a wrong path and a null SHA.
            "config_path": os.path.abspath(_config_path),
            "config_sha256": sha256_file(_config_path),
            # Stated, not omitted: a missing field used to default to false,
            # so deleting it promoted a backfilled cell to a native one.
            "backfilled": False,
            "dataset": getattr(args, "dataset", None),
            "random_seed": getattr(args, "random_seed", None),
            "codebook_size": getattr(args, "codebook_size", None),
        },
    )


def _effective_stop_epoch(args: Config):
    """The last epoch that actually ran, however it was specified."""
    explicit = getattr(args, "stop_after_epoch", None)
    if explicit is not None:
        return int(explicit)
    budget = getattr(args, "epoch", None)
    return None if budget is None else int(budget) - 1


def _atomic_savez(path: str, payload: dict) -> str:
    """Write to a temp file on the same filesystem, then rename.

    `np.savez` straight onto the final path leaves a truncated NPZ if the
    process dies mid-write, and the marker cannot help because the file it
    names already looks present.
    """
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **payload)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _write_completion_marker(out_dir: str, manifests: dict, *,
                             required_splits=("db", "query")) -> str:
    """Mark the run complete only after every required split has validated.

    The first version wrote whatever split set it was handed, so a marker naming
    only `db` was produced and looked authoritative to every consumer.
    """
    import json as _json
    from dna_utils.runtime_state import sha256_file

    missing = sorted(set(required_splits) - set(manifests))
    extra = sorted(set(manifests) - set(required_splits))
    if missing or extra:
        raise ManifestBindingError(
            f"completion marker needs exactly {sorted(required_splits)}; "
            f"missing={missing} unexpected={extra}")

    path = os.path.join(out_dir, "extraction_complete.json")
    payload = {
        "schema_version": 1,
        "splits": sorted(manifests),
        "manifest_sha256": {
            split: sha256_file(p) for split, p in sorted(manifests.items())
        },
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        _json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def extract_code(args: Config) -> None:
    model = SigLIP2SemanticOTModel(args).to(args.device)

    ckpt_path = _find_model_checkpoint(args.save_model_state_path)
    if os.path.exists(ckpt_path):
        from dna_utils.run_identity import load_model_state_dict_for_extraction
        missing, unexpected, phase3_binding = \
            load_model_state_dict_for_extraction(
                model, ckpt_path, map_location=args.device)
        # Only legacy checkpoints may predate newer keys. A checkpoint carrying
        # sealed Phase-3 metadata was produced by this committed architecture
        # and the shared loader makes any mismatch fatal.
        if missing or unexpected:
            print(f"[extraction] load_state_dict: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[extraction] loaded checkpoint from {ckpt_path} "
              f"(phase3_exact={phase3_binding is not None})")
        # F01: restore the epoch BEFORE the first forward. `_current_epoch` is
        # a plain int and is absent from the state dict, so a fresh model sits
        # at 0 and the router runs at the INITIAL Sinkhorn epsilon instead of
        # the annealed value these weights were trained with. Fail-closed: if
        # annealing is on and the epoch cannot be established, abort rather
        # than silently produce codes from an operating point never trained.
        from dna_utils.runtime_state import apply_inference_epoch
        _resolved = apply_inference_epoch(model, ckpt_path, args)
        print(f"[extraction] inference epoch={_resolved.epoch} "
              f"(source={_resolved.source}) "
              f"effective_sinkhorn_epsilon={_resolved.effective_sinkhorn_epsilon}")
    else:
        _require_checkpoint(ckpt_path)

    transform = get_transform("test")
    qwen_text_cache_path = getattr(args, "qwen_text_cache_path", None)
    feature_cache_dir    = getattr(args, "siglip2_feature_cache_dir", None)

    _, queryset, dbset = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=transform, test_transform=transform,
        load_train=False, load_database=True, load_test=True,
        return_index=True,
        qwen_text_cache_path=qwen_text_cache_path,
        siglip2_feature_cache_dir=feature_cache_dir,
    )
    if feature_cache_dir is not None:
        print(f"[extraction] using cached SigLIP2 features from {feature_cache_dir}; "
              f"encoder pass will be skipped.")

    bs = int(getattr(args, "extract_batch_size", 256))
    db_loader = torch.utils.data.DataLoader(
        dbset, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )
    query_loader = torch.utils.data.DataLoader(
        queryset, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    print("[extraction] encoding db ...")
    db_out = encode_split(model, db_loader, args.device, split_name="db")
    print("[extraction] encoding query ...")
    qy_out = encode_split(model, query_loader, args.device, split_name="query")

    print(f"[extraction] db    -> base_indices {db_out['base_indices'].shape}")
    print(f"[extraction] query -> base_indices {qy_out['base_indices'].shape}")

    out_dir = args.save_result_path
    os.makedirs(out_dir, exist_ok=True)
    # Invalidate FIRST: the previous order overwrote the NPZs and only then
    # removed the old marker, so a crash in between left an old marker and old
    # manifests sitting beside new or partial data, and every consumer read that
    # as a complete run.
    # The train manifest goes too: a db/query rerun leaves the old train NPZ
    # describing a different forward pass, and the held-out decoder would
    # happily mix the two.
    for _stale in ("extraction_complete.json",
                   "extraction_manifest_db.json",
                   "extraction_manifest_query.json",
                   "extraction_manifest_train.json"):
        _path = os.path.join(out_dir, _stale)
        if os.path.exists(_path):
            os.remove(_path)

    _atomic_savez(os.path.join(out_dir, "extract_db.npz"), db_out)
    _atomic_savez(os.path.join(out_dir, "extract_query.npz"), qy_out)
    # F01: without these a table cannot be traced back to the operating point
    # that produced it. Written after the NPZs so the recorded digest is the
    # digest of the file on disk.
    manifests = {}
    for split, name, payload in (("db", "extract_db.npz", db_out),
                                 ("query", "extract_query.npz", qy_out)):
        manifests[split] = _write_split_manifest(
            out_dir, split, name, payload, args=args,
            checkpoint_path=ckpt_path, resolved=_resolved)

    # Written last and atomically: a consumer that sees this knows both splits
    # validated. Without it, an interrupted run leaves one manifest and one
    # NPZ and looks partially complete to everything downstream.
    _write_completion_marker(out_dir, manifests)
    print(f"[extraction] saved to {out_dir}")


#: Flags that describe THIS inference run, not the training run the config.pt
#: came from. Restoring the saved config must not silently overwrite them.
_INFERENCE_TIME_OVERRIDES = ("inference_epoch", "selection_mode", "device")


def _reapply_explicit_cli(args: Config, cli: Config) -> None:
    """Re-apply inference-time flags after a saved config is restored.

    `main` builds `args = Config()`, which is the raw object with NO argv
    parsing -- the parsed values live on `Config.get_config()`. Every CLI flag
    except `--config_path` was therefore dropped on this path, including
    `--inference_epoch`, which F01 introduced as the PRIMARY way to resolve the
    extraction epoch. The failure was not silent (the resolver fails closed),
    but the documented escape hatch could not be used at all.
    """
    for name in _INFERENCE_TIME_OVERRIDES:
        value = getattr(cli, name, None)
        if value is None:
            continue
        if name == "device":
            continue          # set from num_devices just above; do not clobber
        setattr(args, name, value)
        print(f"[extraction] CLI override kept after config restore: "
              f"{name}={value}")


def _resume_args_flat_or_legacy(args: Config) -> None:
    """Populate ``args`` from a saved ``config.pt``, supporting both layouts.

    Accepts:
      - Flat (new SigLIP2 layout): ``<config_path>/config.pt``
      - Nested (legacy):           ``<config_path>/model_state/config.pt``

    Mirrors what ``Config.load_args`` does, but additionally checks the flat
    location used by ``train_siglip2._resolve_save_path``.
    """
    cli = Config.get_config()
    config_path = cli.config_path
    if not config_path:
        raise SystemExit("[extraction] --config_path is required for standalone run.")

    flat_pt   = os.path.join(config_path, "config.pt")
    nested_pt = os.path.join(config_path, "model_state", "config.pt")

    if os.path.exists(flat_pt):
        sd = torch.load(flat_pt, map_location="cpu")
        for k, v in sd.items():
            setattr(args, k, v)
        args.save_result_path      = config_path
        args.save_log_path         = os.path.join(config_path, "")
        args.save_model_state_path = os.path.join(config_path, "")
        nd = int(getattr(args, "num_devices", 0) or 0)
        args.device = torch.device(
            f"cuda:{nd}" if torch.cuda.is_available() else "cpu"
        )
        print(f"[extraction] loaded flat config.pt from {flat_pt}")
        _reapply_explicit_cli(args, cli)
    elif os.path.exists(nested_pt):
        # legacy nested layout
        args.load_args()
        print(f"[extraction] loaded legacy nested config.pt from {nested_pt}")
        _reapply_explicit_cli(args, cli)
    else:
        raise FileNotFoundError(
            f"[extraction] no config.pt found at\n  {flat_pt}\nor\n  {nested_pt}"
        )


if __name__ == "__main__":
    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)
    args.print_info()
    extract_code(args)
