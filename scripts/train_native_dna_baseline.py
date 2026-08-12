#!/usr/bin/env python3
"""Train/evaluate the direct native-DNA predecessor baselines.

Examples
--------
Stage 1 (select E* without touching test)::

    python3 scripts/train_native_dna_baseline.py \
      --method koike2024 --dataset Flickr25k \
      --cache_dir cache/flickr25k_clip_v4plus \
      --out params_native_dna/koike2024_flickr_s1 \
      --val_split_ratio .1 --epochs 60 --eval_period 5

Stage 2 (refit all designated train rows to the selected zero-based epoch)::

    python3 scripts/train_native_dna_baseline.py \
      --method koike2024 --dataset Flickr25k \
      --cache_dir cache/flickr25k_clip_v4plus \
      --out result_native_dna/koike2024_flickr_e14 \
      --epochs 15 --eval_period 5 --save_train_extract

PRIMO requires a frozen NUPACK-trained official yield predictor converted with
``scripts/convert_primo_predictor.py``.  We deliberately do not substitute an
untrained/random predictor or vendor NUPACK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.base_model import (  # noqa: E402
    MAP_AT_R_BY_DATASET,
    NUM_CLASS,
    CachedFeatureDataset,
)
from baseline.native_dna import (  # noqa: E402
    METHODS,
    Bee2018Encoder,
    DenseDNAEncoder,
    PRIMOYieldPredictor,
    bee2018_approximate_yield,
    bee2018_cosine_distance,
    calibrate_feature_threshold,
    entropy_regularizer,
    evaluate_base_retrieval,
    extract_native_codes,
    extraction_payload,
    fit_pca,
    keras_activity_entropy_loss,
    koike_objective,
    project_codes_memoized,
    sample_balanced_feature_pairs,
    sample_random_feature_pairs,
)
from val_split import carve_val_indices  # noqa: E402
from dna_utils.bio_constraints import is_valid_batch  # noqa: E402

_MATCHED_LENGTH = int(os.environ.get('GDNA_NATIVE_DNA_BASES', '15'))



DEFAULTS = {
    "bee2018": {"batch_size": 500, "epochs": 65, "steps_per_epoch": 1000,
                "lr": 1e-3, "length": 30, "pair_sampling": "random",
                "optimizer": "adam"},
    "bee2021": {"batch_size": 100, "epochs": 100, "steps_per_epoch": 1000,
                "lr": 1e-3, "length": 80, "pair_sampling": "balanced",
                "optimizer": "adagrad"},
    "koike2024": {"batch_size": 128, "epochs": 150, "steps_per_epoch": None,
                  "lr": 1e-2, "length": 80, "pair_sampling": None,
                  "optimizer": "adagrad"},
    "koike2026": {"batch_size": 128, "epochs": 1000, "steps_per_epoch": None,
                  "lr": 1e-2, "length": 80, "pair_sampling": None,
                  "optimizer": "adagrad"},
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def _paths(dataset, dataset_root: str) -> np.ndarray:
    if dataset.dataset_name == "CIFAR10":
        # ``dataset.paths`` is split-local (``cifar10:0`` repeats between query
        # and database).  Cache-row ids remain globally unique for safe joins.
        return np.asarray(
            [f"cifar10-cache-row:{int(row)}" for row in dataset.rows], dtype=object
        )
    root = Path(dataset_root) / dataset.dataset_name
    return np.asarray([str(root / path) for path in dataset.paths], dtype=object)


def _make_dataset(args, mode: str) -> CachedFeatureDataset:
    return CachedFeatureDataset(
        args.dataset,
        args.setting,
        mode,
        args.dataset_root,
        args.cache_dir,
        paired_aug=False,
        return_index=False,
    )


def _feature_matrix(dataset: CachedFeatureDataset) -> np.ndarray:
    return np.asarray(dataset.visual_global[dataset.rows], dtype=np.float32)


def _pair_metric_features(model, features: np.ndarray, method: str) -> np.ndarray:
    """Feature space in which each predecessor defines pair similarity."""
    if method != "bee2018":
        return features
    mean = model.pca_mean.detach().cpu().numpy()
    components = model.pca_components.detach().cpu().numpy()
    return (features - mean).dot(components).astype(np.float32)


def _build_model(
    args,
    input_dim: int,
    train_features: Optional[np.ndarray] = None,
    loading: bool = False,
):
    if args.method == "bee2018":
        if loading:
            mean = torch.zeros(input_dim)
            components = torch.zeros(input_dim, args.pca_dim)
        else:
            if train_features is None:
                raise ValueError("Bee 2018 needs train features to fit PCA")
            mean, components = fit_pca(
                train_features, n_components=args.pca_dim, seed=args.seed
            )
        return Bee2018Encoder(
            input_dim=input_dim,
            length=args.length,
            pca_dim=args.pca_dim,
            channels=args.bee_channels,
            pca_mean=mean,
            pca_components=components,
        )
    return DenseDNAEncoder(input_dim, args.length, hidden_dim=args.hidden_dim)


def _load_model_from_checkpoint(args, checkpoint: dict, input_dim: int):
    saved = checkpoint["config"]
    if saved.get("method") != args.method:
        raise ValueError(
            f"checkpoint method={saved.get('method')} but --method={args.method}"
        )
    if saved.get("dataset") not in (None, args.dataset):
        raise ValueError(
            f"checkpoint dataset={saved.get('dataset')} but --dataset={args.dataset}"
        )
    context_mismatches = []
    for key in ("setting", "cache_dir", "dataset_root"):
        if saved.get(key) is not None and saved.get(key) != getattr(args, key):
            context_mismatches.append(
                f"{key}: checkpoint={saved.get(key)!r}, current={getattr(args, key)!r}"
            )
    if context_mismatches and not args.allow_checkpoint_context_mismatch:
        raise ValueError(
            "checkpoint data context mismatch (pass "
            "--allow_checkpoint_context_mismatch only if intentional): "
            + "; ".join(context_mismatches)
        )
    for key in ("length", "pca_dim", "bee_channels", "hidden_dim"):
        if key in saved:
            setattr(args, key, saved[key])
    model = _build_model(args, input_dim, loading=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _save_checkpoint(model, optimizer, args, epoch: int, threshold: Optional[float]) -> Path:
    path = Path(args.out) / f"epoch_{epoch:03d}.pth"
    config = dict(vars(args))
    config["feature_distance_threshold_resolved"] = threshold
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "config": config,
        },
        path,
    )
    return path


def _append_log(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _extract_dataset(model, dataset, args):
    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    hp = args.paper_hp_max_run if args.method == "koike2026" and args.paper_hp_postprocess else None
    return extract_native_codes(model, loader, args.device, koike_hp_postprocess=hp)


def _evaluate_pair(model, query_set, database_set, args, save_prefix: Optional[str] = None):
    query_neural_raw, query_codes, query_labels, query_probs = _extract_dataset(
        model, query_set, args
    )
    db_neural_raw, db_codes, db_labels, db_probs = _extract_dataset(
        model, database_set, args
    )
    neural_raw = evaluate_base_retrieval(
        query_neural_raw,
        db_neural_raw,
        query_labels,
        db_labels,
        map_at_r=args.map_at_r,
        device=args.device,
        query_chunk=args.query_chunk,
    )
    query_projection = project_codes_memoized(
        query_codes, args.gc_min, args.gc_max, args.max_run
    )
    db_projection = project_codes_memoized(
        db_codes, args.gc_min, args.gc_max, args.max_run
    )
    projected = evaluate_base_retrieval(
        query_projection["projected_codes"],
        db_projection["projected_codes"],
        query_labels,
        db_labels,
        map_at_r=args.map_at_r,
        device=args.device,
        query_chunk=args.query_chunk,
    )
    metrics = {
        "neural_raw": neural_raw,
        "projected": projected,
        "query": {
            "n": int(len(query_codes)),
            "neural_raw_compliance": float(
                is_valid_batch(
                    query_neural_raw, args.gc_min, args.gc_max, args.max_run
                ).mean()
            ),
            "deployment_pre_dp_compliance": query_projection["pre_compliance"],
            "post_compliance": query_projection["post_compliance"],
            "mean_dp_edit_distance": query_projection["mean_edit_distance"],
            "unique_ratio_neural_raw": float(
                len(np.unique(query_neural_raw, axis=0)) / max(len(query_codes), 1)
            ),
            "unique_ratio_deployment_pre_dp": float(
                len(np.unique(query_codes, axis=0)) / max(len(query_codes), 1)
            ),
        },
        "database": {
            "n": int(len(db_codes)),
            "neural_raw_compliance": float(
                is_valid_batch(
                    db_neural_raw, args.gc_min, args.gc_max, args.max_run
                ).mean()
            ),
            "deployment_pre_dp_compliance": db_projection["pre_compliance"],
            "post_compliance": db_projection["post_compliance"],
            "mean_dp_edit_distance": db_projection["mean_edit_distance"],
            "unique_ratio_neural_raw": float(
                len(np.unique(db_neural_raw, axis=0)) / max(len(db_codes), 1)
            ),
            "unique_ratio_deployment_pre_dp": float(
                len(np.unique(db_codes, axis=0)) / max(len(db_codes), 1)
            ),
            "unique_ratio_projected": float(
                len(np.unique(db_projection["projected_codes"], axis=0)) / max(len(db_codes), 1)
            ),
        },
    }
    if args.method == "koike2026" and args.paper_hp_postprocess:
        metrics["paper_hp_postprocessed"] = evaluate_base_retrieval(
            query_codes,
            db_codes,
            query_labels,
            db_labels,
            map_at_r=args.map_at_r,
            device=args.device,
            query_chunk=args.query_chunk,
        )
    if save_prefix is not None:
        output = Path(args.out)
        np.savez(
            output / f"extract_{save_prefix}_query_neural_raw.npz",
            **extraction_payload(
                query_neural_raw, query_labels, _paths(query_set, args.dataset_root),
                soft_probs=query_probs if args.save_soft_probs else None,
            ),
        )
        np.savez(
            output / f"extract_{save_prefix}_db_neural_raw.npz",
            **extraction_payload(
                db_neural_raw, db_labels, _paths(database_set, args.dataset_root),
                soft_probs=db_probs if args.save_soft_probs else None,
            ),
        )
        np.savez(
            output / f"extract_{save_prefix}_query.npz",
            **extraction_payload(
                query_codes, query_labels, _paths(query_set, args.dataset_root),
                soft_probs=query_probs if args.save_soft_probs else None,
                neural_raw_base_indices=query_neural_raw,
            ),
        )
        np.savez(
            output / f"extract_{save_prefix}_db.npz",
            **extraction_payload(
                db_codes, db_labels, _paths(database_set, args.dataset_root),
                soft_probs=db_probs if args.save_soft_probs else None,
                neural_raw_base_indices=db_neural_raw,
            ),
        )
        np.savez(
            output / f"extract_{save_prefix}_query_bioproj.npz",
            **extraction_payload(
                query_projection["projected_codes"], query_labels,
                _paths(query_set, args.dataset_root),
            ),
        )
        np.savez(
            output / f"extract_{save_prefix}_db_bioproj.npz",
            **extraction_payload(
                db_projection["projected_codes"], db_labels,
                _paths(database_set, args.dataset_root),
            ),
        )
    return metrics


def _train_bee(model, train_set, optimizer, predictor, threshold: float, args, epoch: int):
    model.train()
    if predictor is not None:
        predictor.eval()
    features = _feature_matrix(train_set)
    metric_features = _pair_metric_features(model, features, args.method)
    rng = np.random.RandomState(args.seed + epoch * 100_003)
    sums: Dict[str, float] = {}
    steps = args.steps_per_epoch or int(np.ceil(len(features) / args.batch_size))
    for _ in range(steps):
        if args.bee_pair_sampling == "random":
            _, targets, pair_features = sample_random_feature_pairs(
                features, threshold, args.batch_size, rng,
                distance_features=metric_features,
            )
        else:
            _, targets, pair_features = sample_balanced_feature_pairs(
                features, threshold, args.batch_size, rng,
                distance_features=metric_features,
            )
        first = torch.from_numpy(pair_features[:, 0]).to(args.device)
        second = torch.from_numpy(pair_features[:, 1]).to(args.device)
        targets_tensor = torch.from_numpy(targets).to(args.device)
        first_probs = model(first)
        second_probs = model(second)
        entropy = first_probs.sum() * 0.0
        entropy_effective = entropy
        if args.method == "bee2018":
            distance = bee2018_cosine_distance(first_probs, second_probs)
            estimated_yield = bee2018_approximate_yield(
                distance, slope=args.yield_slope, midpoint=args.yield_midpoint
            )
            pair_loss = F.binary_cross_entropy(estimated_yield, targets_tensor)
            total = pair_loss
        else:
            logits = predictor(first_probs, second_probs)
            pair_loss = F.binary_cross_entropy_with_logits(logits, targets_tensor)
            entropy = 0.5 * (
                entropy_regularizer(first_probs) + entropy_regularizer(second_probs)
            )
            entropy_effective = keras_activity_entropy_loss(first_probs, second_probs)
            total = pair_loss + args.entropy_strength * entropy_effective
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        values = {
            "loss": float(total.detach()),
            "pair_bce": float(pair_loss.detach()),
            "entropy": float(entropy.detach()),
            "entropy_effective": float(
                entropy_effective.detach() if args.method == "bee2021" else entropy.detach()
            ),
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
    return {key: value / steps for key, value in sums.items()}


def _train_koike(model, train_set, optimizer, args):
    model.train()
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # DATE/DAC uses drop_remainder=True; TCBB's TripletDataset keeps its
        # final partial batch.
        drop_last=args.method == "koike2024" and len(train_set) >= args.batch_size,
    )
    sums: Dict[str, float] = {}
    count = 0
    for step, batch in enumerate(loader):
        features = batch["img"].to(args.device)
        labels = batch["label"].to(args.device)
        probs = model(features)
        losses = koike_objective(
            probs,
            labels,
            method=args.method,
            margin=args.margin,
            entropy_strength=args.entropy_strength,
            max_run=args.paper_hp_max_run,
            hp_weight=args.hp_weight,
            gc_weight=args.gc_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_n = int(len(features))
        count += batch_n
        for key, value in losses.scalars().items():
            sums[key] = sums.get(key, 0.0) + value * batch_n
        if args.steps_per_epoch is not None and step + 1 >= args.steps_per_epoch:
            break
    return {key: value / max(count, 1) for key, value in sums.items()}


def _resolve_args(args):
    defaults = DEFAULTS[args.method]
    for key in ("batch_size", "epochs", "lr", "length"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.steps_per_epoch is None:
        args.steps_per_epoch = defaults["steps_per_epoch"]
    if args.bee_pair_sampling is None:
        args.bee_pair_sampling = defaults["pair_sampling"]
    if args.optimizer is None:
        args.optimizer = defaults["optimizer"]
    if args.hidden_dim is None:
        # Original PRIMO/Koike uses D/2.  Pass 2048 explicitly for a literal
        # 4096->2048 faithful run; matched cached-feature runs default to D/2.
        args.hidden_dim = None
    if args.map_at_r is None:
        if args.dataset not in MAP_AT_R_BY_DATASET:
            raise ValueError(f"--map_at_r is required for {args.dataset}")
        args.map_at_r = MAP_AT_R_BY_DATASET[args.dataset]
    if args.method == "bee2021" and not args.weights:
        if not args.primo_predictor_npz:
            raise ValueError(
                "bee2021 requires --primo_predictor_npz. Convert the official "
                "NUPACK-trained model; a random/analytic substitute is not PRIMO."
            )
        if not Path(args.primo_predictor_npz).is_file():
            raise FileNotFoundError(args.primo_predictor_npz)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[native-dna] CUDA unavailable; falling back to CPU", flush=True)
        args.device = "cpu"
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.cache_dir = str(Path(args.cache_dir).resolve())
    args.out = str(Path(args.out).resolve())
    if args.primo_predictor_npz:
        args.primo_predictor_npz = str(Path(args.primo_predictor_npz).resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=list(NUM_CLASS))
    parser.add_argument("--setting", default="setting1")
    parser.add_argument("--dataset_root", default=str(REPO / "dataset"))
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=None, help="Evaluate/extract only from this checkpoint")
    parser.add_argument("--allow_nonempty_out", action="store_true",
                        help="Allow mixing artifacts in an existing output directory")
    parser.add_argument("--allow_checkpoint_context_mismatch", action="store_true",
                        help="Allow checkpoint evaluation with a different split/cache path")
    parser.add_argument("--length", type=int, default=None,
                        help="18 for matched comparison; method-specific original default otherwise")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--bee_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--extract_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--bee_pair_sampling", choices=("random", "balanced"), default=None,
                        help="DNA24 uses random; PRIMO uses balanced similar/dissimilar pairs")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "adagrad"), default=None)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--entropy_strength", type=float, default=1e-2)
    parser.add_argument("--hp_weight", type=float, default=1.0)
    parser.add_argument("--gc_weight", type=float, default=1.0)
    parser.add_argument("--paper_hp_max_run", type=int, default=4,
                        help="TCBB faithful loss/heuristic target; common DP still uses --max_run")
    parser.add_argument("--paper_hp_postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yield_slope", type=float, default=12.1)
    parser.add_argument("--yield_midpoint", type=float, default=0.5)
    parser.add_argument("--feature_distance_threshold", type=float, default=None)
    parser.add_argument("--feature_positive_quantile", type=float, default=0.1)
    parser.add_argument("--threshold_pairs", type=int, default=100_000)
    parser.add_argument("--primo_predictor_npz", default=None)
    parser.add_argument("--val_split_ratio", type=float, default=0.0)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--eval_period", type=int, default=5)
    parser.add_argument(
        "--selection_metric", choices=("neural_raw", "projected"),
        default="neural_raw",
        help=("P0 default is raw 18-base Hamming, matching GroundedDNA and "
              "binary-to-base controls. Projected selection is an explicit "
              "alternative protocol and must not be mixed into the main table."),
    )
    parser.add_argument("--map_at_r", type=int, default=None)
    parser.add_argument("--gc_min", type=float, default=0.4)
    parser.add_argument("--gc_max", type=float, default=0.6)
    parser.add_argument("--max_run", type=int, default=3)
    parser.add_argument("--query_chunk", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help=(
            "Global gradient-norm clipping; disabled by default because none "
            "of the audited predecessor training protocols specifies it."
        ),
    )
    parser.add_argument("--save_train_extract", action="store_true")
    parser.add_argument("--save_soft_probs", action="store_true")
    args = parser.parse_args()
    _resolve_args(args)
    _seed_everything(args.seed)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.allow_nonempty_out:
        raise FileExistsError(
            f"refusing non-empty --out {output}; choose a fresh directory or "
            "pass --allow_nonempty_out explicitly"
        )
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(vars(args), output / "config.json")

    full_train = _make_dataset(args, "train")
    input_dim = int(full_train.visual_global.shape[1])
    val_set = None
    split_info = None
    if args.val_split_ratio > 0:
        opt_indices, val_indices, strategy = carve_val_indices(
            np.asarray(full_train.labels), args.val_split_ratio, args.val_split_seed
        )
        train_set = _make_dataset(args, "train")
        train_set.restrict_to(opt_indices)
        val_set = _make_dataset(args, "train")
        val_set.restrict_to(val_indices)
        split_info = {
            "ratio": args.val_split_ratio,
            "seed": args.val_split_seed,
            "strategy": strategy,
            "n_full": len(full_train),
            "n_opt": len(train_set),
            "n_val": len(val_set),
            "val_indices": val_indices.tolist(),
        }
        _json_dump(split_info, output / "val_split.json")
    else:
        train_set = full_train

    predictor = None
    threshold = args.feature_distance_threshold
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        model = _load_model_from_checkpoint(args, checkpoint, input_dim)
        model.to(args.device)
        threshold = checkpoint.get("config", {}).get(
            "feature_distance_threshold_resolved", threshold
        )
        optimizer = None
    else:
        train_features = _feature_matrix(train_set)
        model = _build_model(args, input_dim, train_features=train_features)
        # Adagrad initializes its accumulator tensors eagerly.  Move the model
        # first so optimizer state is created on the same device as parameters.
        model.to(args.device)
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            # Match tf.keras Adagrad defaults used by PRIMO and both Koike
            # releases, rather than PyTorch's different defaults.
            optimizer = torch.optim.Adagrad(
                model.parameters(),
                lr=args.lr,
                initial_accumulator_value=0.1,
                eps=1e-7,
            )
        if args.method.startswith("bee") and threshold is None:
            metric_features = _pair_metric_features(model, train_features, args.method)
            threshold = calibrate_feature_threshold(
                metric_features,
                quantile=args.feature_positive_quantile,
                n_pairs=args.threshold_pairs,
                seed=args.seed,
            )
            print(f"[native-dna] train-only feature threshold={threshold:.8f}", flush=True)
        if args.method == "bee2021":
            predictor = PRIMOYieldPredictor()
            predictor.load_official_npz(args.primo_predictor_npz)
            predictor.to(args.device).eval()
            for parameter in predictor.parameters():
                parameter.requires_grad_(False)

    n_params = sum(parameter.numel() for parameter in model.parameters())
    resolved_config = dict(vars(args))
    resolved_config["feature_distance_threshold_resolved"] = threshold
    resolved_config["input_dim"] = input_dim
    resolved_config["parameter_count"] = n_params
    _json_dump(resolved_config, output / "config.json")
    print(
        f"[native-dna] method={args.method} dataset={args.dataset} D={input_dim} "
        f"L={args.length} params={n_params:,} device={args.device}",
        flush=True,
    )

    if not args.weights:
        best = None
        started = time.time()
        for epoch in range(args.epochs):
            if args.method.startswith("bee"):
                losses = _train_bee(
                    model, train_set, optimizer, predictor, float(threshold), args, epoch
                )
            else:
                losses = _train_koike(model, train_set, optimizer, args)
            row = {"epoch": epoch, **losses, "seconds": time.time() - started}
            _append_log(output / "log.csv", row)
            print(
                f"[native-dna] epoch={epoch:03d} "
                + " ".join(f"{key}={value:.5f}" for key, value in losses.items()),
                flush=True,
            )
            if (epoch + 1) % args.eval_period == 0 or epoch == args.epochs - 1:
                checkpoint_path = _save_checkpoint(model, optimizer, args, epoch, threshold)
                if val_set is not None:
                    metrics = _evaluate_pair(model, val_set, train_set, args)
                    metrics.update({"epoch": epoch, "checkpoint": str(checkpoint_path)})
                    _json_dump(metrics, output / f"eval_val_epoch_{epoch:03d}.json")
                    score = metrics[args.selection_metric]["mAP_at_R"]
                    print(
                        f"[native-dna] val epoch={epoch:03d} "
                        f"neural_raw={metrics['neural_raw']['mAP_at_R']:.5f} "
                        f"projected={metrics['projected']['mAP_at_R']:.5f}",
                        flush=True,
                    )
                    if best is None or score > best["score"]:
                        best = {"epoch": epoch, "score": score, "checkpoint": str(checkpoint_path)}
        if val_set is not None:
            _json_dump(
                {
                    "selection_metric": f"val_{args.selection_metric}_mAP_at_R",
                    "best_epoch_zero_based": best["epoch"],
                    "refit_epochs": best["epoch"] + 1,
                    "best_score": best["score"],
                    "checkpoint": best["checkpoint"],
                    "split": split_info,
                    "test_evaluated": False,
                },
                output / "selection.json",
            )
            print(
                f"[native-dna] selected E*={best['epoch']} (run refit with "
                f"--epochs {best['epoch'] + 1}); test was not evaluated",
                flush=True,
            )
            return 0

    # Full-train refit or checkpoint-only evaluation: test is touched here only.
    query_set = _make_dataset(args, "test")
    database_set = _make_dataset(args, "database")
    metrics = _evaluate_pair(model, query_set, database_set, args, save_prefix="test")
    metrics.update(
        {
            "method": args.method,
            "dataset": args.dataset,
            "length": args.length,
            "parameter_count": n_params,
            # The label follows the CONFIGURED matched length, not a literal 18.
            # 18 was the 6-slot / 36-bit budget; the paper is 15 bases since
            # 2026-08-11. GDNA_NATIVE_DNA_BASES must hold the same value here as
            # in run_native_dna_p0{,_matrix}.py or the aggregator rejects the
            # cell for a protocol mismatch.
            "protocol": (
                f"matched_{_MATCHED_LENGTH}nt_adaptation"
                if args.length == _MATCHED_LENGTH
                else "original_length_only_adaptation"
            ),
            "supervision": "ground-truth labels" if args.method.startswith("koike") else "feature-distance pairs",
            "paper_hp_postprocess": bool(
                args.method == "koike2026" and args.paper_hp_postprocess
            ),
            "standard_extract_pre_dp_stage": (
                "paper_hp_postprocessed"
                if args.method == "koike2026" and args.paper_hp_postprocess
                else "neural_raw"
            ),
        }
    )
    if args.method == "bee2021":
        metrics["method_variant"] = (
            "frozen_predictor_no_alternating_refit"
            if args.length == 80
            else "frozen_predictor_length_transfer"
        )
    _json_dump(metrics, output / "evaluation_native_dna.json")
    # Common filenames make downstream bio/compositional scripts drop-in compatible.
    for source, target in (
        ("extract_test_query_neural_raw.npz", "extract_query_neural_raw.npz"),
        ("extract_test_db_neural_raw.npz", "extract_db_neural_raw.npz"),
        ("extract_test_query.npz", "extract_query.npz"),
        ("extract_test_db.npz", "extract_db.npz"),
        ("extract_test_query_bioproj.npz", "extract_query_bioproj.npz"),
        ("extract_test_db_bioproj.npz", "extract_db_bioproj.npz"),
    ):
        os.replace(output / source, output / target)
    if args.save_train_extract:
        train_neural_raw, train_codes, train_labels, train_probs = _extract_dataset(
            model, full_train, args
        )
        np.savez(
            output / "extract_train.npz",
            **extraction_payload(
                train_codes,
                train_labels,
                _paths(full_train, args.dataset_root),
                soft_probs=train_probs if args.save_soft_probs else None,
                neural_raw_base_indices=train_neural_raw,
            ),
        )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
