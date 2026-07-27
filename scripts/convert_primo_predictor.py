#!/usr/bin/env python3
"""Convert PRIMO's official Keras yield predictor to a small PyTorch ``npz``.

The source artifact is available through the authors' Git-LFS/Zenodo release:

    https://github.com/uwmisl/primo-similarity-search
    https://doi.org/10.5281/zenodo.5090717

This script copies only learned numeric tensors supplied by the user.  It does
not contain, install, invoke, or redistribute NUPACK/CuPyCK.  The output is
consumed by ``baseline.native_dna.PRIMOYieldPredictor``.

Usage::

    python3 scripts/convert_primo_predictor.py \
      --keras_h5 /path/to/primo/data/models/yield-model.h5 \
      --out artifacts/primo_yield_predictor.npz

``h5py`` is an optional conversion-time dependency.  A 133-byte Git-LFS pointer
is not a model; run ``git lfs pull`` or obtain the Zenodo artifact first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_datasets(path: str):
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on optional tool
        raise RuntimeError("install optional dependency `h5py` for conversion") from error
    arrays = {}
    with h5py.File(path, "r") as handle:
        def visitor(name, value):
            if isinstance(value, h5py.Dataset):
                arrays[name] = np.asarray(value)
        handle.visititems(visitor)
    return arrays


def _pick(arrays, *, shape, name_fragments=()):
    candidates = []
    for name, value in arrays.items():
        if tuple(value.shape) != tuple(shape):
            continue
        score = sum(fragment.lower() in name.lower() for fragment in name_fragments)
        if "model_weights" in name.lower():
            score += 2
        if "optimizer" in name.lower():
            score -= 100
        candidates.append((score, name, value))
    if not candidates:
        raise KeyError(f"no HDF5 tensor with shape {shape}")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score = candidates[0][0]
    tied = [item for item in candidates if item[0] == best_score]
    if len(tied) > 1:
        names = [item[1] for item in tied]
        raise RuntimeError(f"ambiguous shape {shape}: {names}")
    return candidates[0][1], candidates[0][2]


def _to_pytorch_arrays(arrays):
    """Select official Keras tensors and convert their storage layouts."""
    conv_name, conv_kernel = _pick(
        arrays, shape=(3, 36, 36), name_fragments=("conv", "kernel")
    )
    conv_bias_name, conv_bias = _pick(
        arrays, shape=(36,), name_fragments=("conv", "bias")
    )
    dense_name, dense_kernel = _pick(
        arrays, shape=(36, 1), name_fragments=("logit", "kernel")
    )
    dense_bias_name, dense_bias = _pick(
        arrays, shape=(1,), name_fragments=("logit", "bias")
    )

    # Keras Conv1D is [kernel,in,out]; PyTorch is [out,in,kernel].
    converted = {
        "conv_weight": np.transpose(conv_kernel, (2, 1, 0)).astype(np.float32),
        "conv_bias": np.asarray(conv_bias, dtype=np.float32),
        # Keras Dense is [in,out]; PyTorch Linear is [out,in].
        "dense_weight": dense_kernel.T.astype(np.float32),
        "dense_bias": np.asarray(dense_bias, dtype=np.float32),
    }
    tensor_names = {
        "conv_kernel": conv_name,
        "conv_bias": conv_bias_name,
        "dense_kernel": dense_name,
        "dense_bias": dense_bias_name,
    }
    return converted, tensor_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_h5", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.keras_h5)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size < 1024:
        head = source.read_text(errors="ignore")[:200]
        if "git-lfs" in head:
            raise RuntimeError(f"{source} is a Git-LFS pointer; fetch the actual model first")

    arrays = _all_datasets(str(source))
    converted, tensor_names = _to_pytorch_arrays(arrays)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **converted)
    metadata = {
        "source": str(source.resolve()),
        "source_size": source.stat().st_size,
        "source_sha256": _sha256(source),
        "keras_tensors": tensor_names,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "note": "PRIMO official predictor; calibrated for 80-nt NUPACK setup",
    }
    with output.with_suffix(output.suffix + ".json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
