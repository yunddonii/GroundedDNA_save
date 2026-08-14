"""One validator for an extraction run, shared by every reader (re-audit §17.9).

Each consumer previously carried its own partial idea of "complete", and every
one of them was satisfiable by artefacts that do not exist:

  * `aggregate_phase2_f01._require_manifests` checked that two JSON files were
    present and that four fields agreed. A probe with no NPZ, no checkpoint, no
    config, `schema_version=999` and `dataset=WRONG` was accepted.
  * `phase2_f01_reinference.sh` skipped a cell whenever the two manifest
    *filenames* existed; two files containing `not-json` counted as complete.
  * `_write_completion_marker` accepted whatever split set it was handed, so a
    marker naming only `db` was written and looked authoritative.
  * the manifest writer checked shapes but not dtypes, not that `hash_2bit` is
    the 2-bit encoding of `base_indices`, and not that codebook indices fall
    inside the codebook.

The contract here is deliberately strict and fail-closed: it opens every file it
names, recomputes every digest it compares, and refuses on the first
disagreement. `allow_backfilled` has to be passed explicitly, so a
retrospectively bound diagnostic can never be mistaken for an extraction that
recorded itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Iterable, Mapping, Sequence

import numpy as np

from dna_utils.runtime_state import sha256_file

#: `base_indices` -> `hash_2bit`, the encoding every consumer assumes.
_BASE_TO_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)

MARKER_NAME = "extraction_complete.json"
DEFAULT_SPLITS = ("db", "query")

#: Fields that must be identical across the splits of one run: they describe the
#: run, not the split. A disagreement means the splits came from different
#: forward passes, which is the failure F01 exists to make visible.
_COMMON_IDENTITY = (
    "checkpoint_sha256",
    "config_sha256",
    "inference_epoch",
    "inference_epoch_source",
    "effective_sinkhorn_epsilon",
    "sinkhorn_schedule_horizon",
    "sinkhorn_annealing_enabled",
    "lr_schedule_horizon",
    "training_epoch_budget",
    "training_stop_epoch",
    "num_slots",
    "bases_per_slot",
    "total_bases",
    "total_bits",
    "dataset",
    "random_seed",
    "codebook_size",
    "backfilled",
)


class ExtractionInvalid(RuntimeError):
    """An extraction run cannot be admitted."""


@dataclass
class ValidatedRun:
    run_dir: str
    splits: dict
    backfilled: bool
    common: dict = field(default_factory=dict)


def base_indices_to_2bit(codes: np.ndarray) -> np.ndarray:
    """The canonical encoding, so writer and reader cannot drift apart."""
    return _BASE_TO_BITS[np.asarray(codes)].reshape(len(codes), -1)


def _load_json(path: str, what: str) -> dict:
    if not os.path.isfile(path):
        raise ExtractionInvalid(f"{what} is missing: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionInvalid(f"{what} is not readable JSON: {error}") from None
    if not isinstance(payload, dict):
        raise ExtractionInvalid(f"{what} is not an object: {path}")
    return payload


#: Every manifest field that must be present, non-null and of this type. The
#: first version only compared splits against each other, so DELETING a field
#: from both manifests passed as `None == None`, and `dataset=WRONG` with
#: `inference_epoch=999` was admitted because nothing said what the cell should
#: be.
_MANIFEST_SCHEMA = {
    "schema_version": int,
    "split": str,
    "n_rows": int,
    "checkpoint_path": str,
    "checkpoint_sha256": str,
    "config_path": str,
    "config_sha256": str,
    "inference_epoch": int,
    "inference_epoch_source": str,
    "lr_schedule_horizon": int,
    "training_epoch_budget": int,
    "training_stop_epoch": int,
    "num_slots": int,
    "bases_per_slot": int,
    "total_bases": int,
    "total_bits": int,
    "dataset": str,
    "random_seed": int,
    "codebook_size": int,
    "npz_path": str,
    "npz_sha256": str,
    "backfilled": bool,
    # Whether an epsilon SCHEDULE was in force during the forward pass. Stated
    # outright because it cannot be inferred from `inference_epoch_source`.
    "sinkhorn_annealing_enabled": bool,
}

MANIFEST_SCHEMA_VERSION = 2

#: Sources the resolver can legitimately report -- where the EPOCH came from,
#: which is a different question from whether epsilon was annealed. `bogus` is
#: not one of them.
_EPOCH_SOURCES = frozenset(
    {"explicit_flag", "checkpoint_metadata", "no_annealing",
     # The F01 defect itself: the epoch was never restored, so the router ran
     # at the initial epsilon. Used only when binding preserved legacy runs,
     # where the epoch is a property of the code path and not of any artefact.
     "f01_unrestored"})


@dataclass(frozen=True)
class ExpectedIdentity:
    """What the caller already knows the cell must be.

    Split-vs-split equality cannot catch a cell that is consistently wrong; the
    launcher and the aggregator both know the dataset, seed and geometry from
    the directory name and the protocol, so they must say so.
    """
    dataset: str | None = None
    random_seed: int | None = None
    inference_epoch: int | None = None
    inference_epoch_source: str | None = None
    num_slots: int | None = None
    bases_per_slot: int | None = None
    codebook_size: int | None = None
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    n_rows: "Mapping[str, int] | None" = None


def _check_schema(manifest: Mapping, *, split: str) -> None:
    for field_name, field_type in _MANIFEST_SCHEMA.items():
        if field_name not in manifest:
            raise ExtractionInvalid(
                f"{split}: manifest is missing {field_name!r}")
        value = manifest[field_name]
        if value is None:
            raise ExtractionInvalid(f"{split}: manifest {field_name} is null")
        if field_type is int and isinstance(value, bool):
            raise ExtractionInvalid(
                f"{split}: manifest {field_name} is a bool, expected int")
        if not isinstance(value, field_type):
            raise ExtractionInvalid(
                f"{split}: manifest {field_name} is {type(value).__name__}, "
                f"expected {field_type.__name__}")

    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ExtractionInvalid(
            f"{split}: unknown manifest schema_version "
            f"{manifest['schema_version']!r}")
    source = manifest["inference_epoch_source"]
    if source not in _EPOCH_SOURCES:
        raise ExtractionInvalid(
            f"{split}: inference_epoch_source {source!r} is not one of "
            f"{sorted(_EPOCH_SOURCES)}")

    # The forward pass ran at SOME epsilon whether or not a schedule was in
    # force, so this is required either way. The previous schema branched on
    # `source == "no_annealing"` -- but `source` says only where the epoch was
    # recovered from, and a fresh no-anneal run writes a sidecar and so resolves
    # as `checkpoint_metadata`. Its null epsilon was then rejected as an
    # annealed run with a missing field, and the whole producer-to-consumer path
    # was incompatible for a configuration the trainer supports (§20.9).
    epsilon = manifest.get("effective_sinkhorn_epsilon")
    if epsilon is None:
        raise ExtractionInvalid(
            f"{split}: effective_sinkhorn_epsilon is null; the router ran at "
            f"some epsilon, and a static one is a value, not an absence")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ExtractionInvalid(
            f"{split}: effective_sinkhorn_epsilon is "
            f"{type(epsilon).__name__}, expected a number")
    if not 0.0 < float(epsilon) <= 10.0:
        raise ExtractionInvalid(
            f"{split}: effective_sinkhorn_epsilon {epsilon} is outside (0, 10]")

    horizon = manifest.get("sinkhorn_schedule_horizon")
    if manifest["sinkhorn_annealing_enabled"]:
        if horizon is None:
            raise ExtractionInvalid(
                f"{split}: sinkhorn_schedule_horizon is null but annealing is "
                f"enabled; the epsilon cannot be reproduced without it")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ExtractionInvalid(
                f"{split}: sinkhorn_schedule_horizon is "
                f"{type(horizon).__name__}, expected int")
        if horizon <= 0:
            raise ExtractionInvalid(
                f"{split}: sinkhorn_schedule_horizon {horizon} is not positive")
    elif horizon is not None:
        raise ExtractionInvalid(
            f"{split}: sinkhorn_schedule_horizon is {horizon!r} but annealing "
            f"is disabled; there is no schedule to record")

    # Each source token carries an invariant. Without them a token is only a
    # self-assertion: `f01_unrestored` with epoch 4, and `no_annealing` with
    # annealing switched on, were both admitted (§23.4).
    if source == "f01_unrestored":
        # The token means one specific history: a run whose epoch was never
        # restored, so the annealed router stayed at its INITIAL epsilon, and
        # whose manifest was necessarily written after the fact -- those runs
        # recorded nothing themselves. Checking only the epoch left the rest of
        # that history free to be anything.
        if manifest["inference_epoch"] != 0:
            raise ExtractionInvalid(
                f"{split}: inference_epoch_source is f01_unrestored but the "
                f"epoch is {manifest['inference_epoch']}; the defect IS that "
                f"nothing restored the epoch, so it can only be 0")
        if not manifest["sinkhorn_annealing_enabled"]:
            raise ExtractionInvalid(
                f"{split}: inference_epoch_source is f01_unrestored but "
                f"annealing is disabled; with no schedule there is no initial "
                f"epsilon to have been stuck at, and nothing for F01 to cost")
        if not manifest["backfilled"]:
            raise ExtractionInvalid(
                f"{split}: inference_epoch_source is f01_unrestored but the "
                f"manifest claims the run recorded itself; those runs did not, "
                f"which is why the epoch has to be inferred at all")
    if source == "no_annealing" and manifest["sinkhorn_annealing_enabled"]:
        raise ExtractionInvalid(
            f"{split}: inference_epoch_source is no_annealing but annealing is "
            f"enabled; the epoch was skipped precisely because there was no "
            f"schedule to place it on")

    slots, per_slot = manifest["num_slots"], manifest["bases_per_slot"]
    if slots <= 0 or per_slot <= 0:
        raise ExtractionInvalid(
            f"{split}: geometry {slots}x{per_slot} is not positive")
    if manifest["total_bases"] != slots * per_slot:
        raise ExtractionInvalid(
            f"{split}: total_bases {manifest['total_bases']} != "
            f"{slots}x{per_slot}")
    if manifest["total_bits"] != 2 * slots * per_slot:
        raise ExtractionInvalid(
            f"{split}: total_bits {manifest['total_bits']} != "
            f"2x{slots}x{per_slot}")
    if manifest["lr_schedule_horizon"] <= 0:
        raise ExtractionInvalid(
            f"{split}: lr_schedule_horizon "
            f"{manifest['lr_schedule_horizon']} is not positive")
    if manifest["training_epoch_budget"] <= 0:
        raise ExtractionInvalid(
            f"{split}: training_epoch_budget "
            f"{manifest['training_epoch_budget']} is not positive")
    if manifest["inference_epoch"] < 0:
        raise ExtractionInvalid(
            f"{split}: inference_epoch {manifest['inference_epoch']} is negative")
    if manifest["random_seed"] < 0:
        raise ExtractionInvalid(
            f"{split}: random_seed {manifest['random_seed']} is negative")
    if manifest["n_rows"] <= 0:
        raise ExtractionInvalid(f"{split}: n_rows {manifest['n_rows']} <= 0")
    if manifest["codebook_size"] <= 0:
        raise ExtractionInvalid(
            f"{split}: codebook_size {manifest['codebook_size']} <= 0")

    budget = manifest["training_epoch_budget"]
    stop = manifest["training_stop_epoch"]
    if not 0 <= stop < budget:
        raise ExtractionInvalid(
            f"{split}: training_stop_epoch {stop} is not inside "
            f"[0, {budget})")
    if manifest["inference_epoch"] > stop:
        raise ExtractionInvalid(
            f"{split}: inference_epoch {manifest['inference_epoch']} is after "
            f"the training stop {stop}")


def _check_expected(manifest: Mapping, expected: "ExpectedIdentity", *,
                    split: str) -> None:
    for field_name in ("dataset", "random_seed", "inference_epoch",
                       "inference_epoch_source", "num_slots",
                       "bases_per_slot", "codebook_size", "checkpoint_sha256",
                       "config_sha256"):
        want = getattr(expected, field_name)
        if want is None:
            continue
        got = manifest.get(field_name)
        if got != want:
            raise ExtractionInvalid(
                f"{split}: manifest {field_name}={got!r} but the caller "
                f"expects {want!r}")
    if expected.n_rows is not None:
        want_rows = expected.n_rows.get(split)
        if want_rows is not None and manifest.get("n_rows") != want_rows:
            raise ExtractionInvalid(
                f"{split}: manifest n_rows={manifest.get('n_rows')!r} but the "
                f"caller expects {want_rows!r}")


def validate_code_arrays(base, hashed, codebook, *, rows: int, slots: int,
                         per_slot: int, codebook_size, split: str) -> None:
    """The array contract, callable before anything is published.

    Extracted so the producer and the backfill tool run the SAME checks instead
    of each carrying a partial copy: the backfill compared
    `hash_2bit.astype(uint8)`, which lets 256 and 257 pass as 0 and 1, and
    checked neither the hash range nor the codebook range, while the producer
    checked shapes but not dtypes or the encoding itself. A producer could
    therefore exit 0 with a marker over semantically invalid arrays and be
    refused only later, by a consumer.
    """
    base = np.asarray(base)
    hashed = np.asarray(hashed)
    codebook = np.asarray(codebook)
    bases = slots * per_slot

    for name, arr in (("base_indices", base), ("hash_2bit", hashed),
                      ("codebook_indices", codebook)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ExtractionInvalid(
                f"{split}: {name} has dtype {arr.dtype}, expected an integer "
                f"type -- a float code is not a code")
    if base.shape != (rows, bases):
        raise ExtractionInvalid(
            f"{split}: base_indices is {base.shape}, expected {(rows, bases)}")
    if hashed.shape != (rows, 2 * bases):
        raise ExtractionInvalid(
            f"{split}: hash_2bit is {hashed.shape}, expected "
            f"{(rows, 2 * bases)}")
    if codebook.shape != (rows, slots):
        raise ExtractionInvalid(
            f"{split}: codebook_indices is {codebook.shape}, expected "
            f"{(rows, slots)}")
    if base.size and (base.min() < 0 or base.max() > 3):
        raise ExtractionInvalid(
            f"{split}: base_indices outside 0..3 [{base.min()}, {base.max()}]")
    # Checked BEFORE any cast: `astype(uint8)` silently folds 256 to 0.
    if hashed.size and (hashed.min() < 0 or hashed.max() > 1):
        raise ExtractionInvalid(
            f"{split}: hash_2bit outside 0..1 [{hashed.min()}, {hashed.max()}]")
    if base.size and not np.array_equal(base_indices_to_2bit(base), hashed):
        raise ExtractionInvalid(
            f"{split}: hash_2bit is not the 2-bit encoding of base_indices; "
            f"the two arrays describe different codes")
    if not isinstance(codebook_size, int) or isinstance(codebook_size, bool):
        raise ExtractionInvalid(
            f"{split}: codebook_size {codebook_size!r} is not an int, so the "
            f"codebook range cannot be checked")
    if codebook.size and (codebook.min() < 0 or codebook.max() >= codebook_size):
        raise ExtractionInvalid(
            f"{split}: codebook_indices outside 0..{codebook_size - 1} "
            f"[{codebook.min()}, {codebook.max()}]")


def _validate_npz(manifest: Mapping, *, split: str) -> None:
    """Open the NPZ the manifest names and hold it to every declared field."""
    npz_path = str(manifest.get("npz_path") or "")
    if not os.path.isfile(npz_path):
        raise ExtractionInvalid(
            f"{split}: manifest names {npz_path!r}, which does not exist")
    digest = sha256_file(npz_path)
    if digest != manifest.get("npz_sha256"):
        raise ExtractionInvalid(
            f"{split}: {npz_path} hashes to {digest[:12]}... but the manifest "
            f"declares {str(manifest.get('npz_sha256'))[:12]}...")

    slots = int(manifest["num_slots"])
    per_slot = int(manifest["bases_per_slot"])
    rows = int(manifest["n_rows"])

    with np.load(npz_path, allow_pickle=False) as stored:
        for name in ("base_indices", "hash_2bit", "codebook_indices"):
            if name not in stored:
                raise ExtractionInvalid(f"{split}: {npz_path} has no {name}")
        base = np.asarray(stored["base_indices"])
        hashed = np.asarray(stored["hash_2bit"])
        codebook = np.asarray(stored["codebook_indices"])

    validate_code_arrays(base, hashed, codebook, rows=rows, slots=slots,
                         per_slot=per_slot,
                         codebook_size=manifest.get("codebook_size"),
                         split=split)


def validate_extraction_run(
    run_dir: str,
    *,
    required_splits: Sequence[str] = DEFAULT_SPLITS,
    allow_backfilled: bool = False,
    verify_npz: bool = True,
    expected: "ExpectedIdentity | None" = None,
) -> ValidatedRun:
    """Admit a run only if every artefact it names exists and agrees.

    `allow_backfilled` is explicit on purpose: a manifest written after the fact
    binds a cell to its inputs but is not evidence that the extraction recorded
    itself, so a paper path must opt in to accepting one.
    """
    required = tuple(required_splits)
    marker_path = os.path.join(run_dir, MARKER_NAME)
    marker = _load_json(marker_path, "completion marker")
    if marker.get("schema_version") != 1:
        raise ExtractionInvalid(
            f"{run_dir}: unknown completion marker schema_version "
            f"{marker.get('schema_version')!r}")
    declared = tuple(marker.get("splits") or ())
    if len(set(declared)) != len(declared):
        raise ExtractionInvalid(
            f"{run_dir}: completion marker repeats a split: {list(declared)}")
    if set(declared) != set(required):
        raise ExtractionInvalid(
            f"{run_dir}: completion marker declares splits {list(declared)}, "
            f"expected exactly {list(required)}")
    digests = marker.get("manifest_sha256")
    if not isinstance(digests, Mapping) or set(digests) != set(required):
        raise ExtractionInvalid(
            f"{run_dir}: marker manifest_sha256 keys "
            f"{sorted(digests) if isinstance(digests, Mapping) else digests} "
            f"do not match {sorted(required)}")

    manifests, backfilled = {}, False
    for split in required:
        path = os.path.join(run_dir, f"extraction_manifest_{split}.json")
        manifest = _load_json(path, f"{split} extraction manifest")
        recorded = (marker.get("manifest_sha256") or {}).get(split)
        actual = sha256_file(path)
        if recorded != actual:
            raise ExtractionInvalid(
                f"{split}: the marker records manifest digest "
                f"{str(recorded)[:12]}... but the file hashes to "
                f"{actual[:12]}...")
        _check_schema(manifest, split=split)
        if manifest["split"] != split:
            raise ExtractionInvalid(
                f"{split}: manifest declares split {manifest['split']!r}")
        if expected is not None:
            _check_expected(manifest, expected, split=split)
        for name, key in (("checkpoint", "checkpoint_path"),
                          ("config", "config_path")):
            file_path = str(manifest.get(key) or "")
            if not os.path.isfile(file_path):
                raise ExtractionInvalid(
                    f"{split}: manifest names a {name} that does not exist: "
                    f"{file_path!r}")
            digest = sha256_file(file_path)
            if digest != manifest.get(f"{name}_sha256"):
                raise ExtractionInvalid(
                    f"{split}: {name} {file_path} hashes to {digest[:12]}... "
                    f"but the manifest declares "
                    f"{str(manifest.get(f'{name}_sha256'))[:12]}...")
        if verify_npz:
            _validate_npz(manifest, split=split)
        manifests[split] = manifest

    states = {split: bool(m.get("backfilled", False))
              for split, m in manifests.items()}
    if len(set(states.values())) != 1:
        raise ExtractionInvalid(
            f"{run_dir}: mixed provenance across splits {states}; one split was "
            f"recorded by its extraction and another was bound after the fact")
    backfilled = next(iter(states.values()))

    first = manifests[required[0]]
    common = {}
    for key in _COMMON_IDENTITY:
        values = {split: manifests[split].get(key) for split in required}
        if len(set(map(repr, values.values()))) != 1:
            raise ExtractionInvalid(
                f"{run_dir}: splits disagree on {key}: {values}. They were not "
                f"produced by the same runtime state.")
        common[key] = first.get(key)

    if backfilled and not allow_backfilled:
        raise ExtractionInvalid(
            f"{run_dir}: the manifests were backfilled after the fact. They "
            f"bind the cell to its inputs but are not evidence the extraction "
            f"recorded itself; pass allow_backfilled=True to admit it as a "
            f"diagnostic.")

    return ValidatedRun(run_dir=run_dir, splits=manifests,
                        backfilled=backfilled, common=common)


def describe_failure(run_dir: str, **kwargs) -> str | None:
    """`None` when the run is admissible, else the reason. For shell callers."""
    try:
        validate_extraction_run(run_dir, **kwargs)
    except ExtractionInvalid as error:
        return str(error)
    except (KeyError, TypeError, ValueError) as error:
        # A malformed manifest must read as a refusal, not as a crash the shell
        # cannot interpret.
        return f"{run_dir}: malformed extraction metadata ({error!r})"
    return None


def metric_input_binding(run_dir: str, *,
                         required_splits: Sequence[str] = DEFAULT_SPLITS,
                         allow_backfilled: bool = False,
                         expected: "ExpectedIdentity | None" = None) -> dict:
    """The block every metric JSON must carry to be traceable to its input.

    A metric file that names no inputs can sit beside a valid extraction while
    describing a different one entirely, and nothing downstream can tell. This
    records the digests of the NPZs the numbers were computed from and of the
    manifests that vouch for them, so a later reader can re-check the binding
    instead of trusting file adjacency.
    """
    run = validate_extraction_run(run_dir, required_splits=required_splits,
                                  allow_backfilled=allow_backfilled,
                                  expected=expected)
    return {
        "schema_version": 1,
        "run_dir": os.path.abspath(run_dir),
        "npz_sha256": {
            split: manifest["npz_sha256"]
            for split, manifest in sorted(run.splits.items())
        },
        "manifest_sha256": {
            split: sha256_file(
                os.path.join(run_dir, f"extraction_manifest_{split}.json"))
            for split in sorted(run.splits)
        },
        "checkpoint_sha256": run.common["checkpoint_sha256"],
        "config_sha256": run.common["config_sha256"],
        "inference_epoch": run.common["inference_epoch"],
        "inference_epoch_source": run.common["inference_epoch_source"],
        "dataset": run.common["dataset"],
        "random_seed": run.common["random_seed"],
        "codebook_size": run.common["codebook_size"],
        "backfilled_inputs": run.backfilled,
        "validator_sha256": sha256_file(__file__),
    }


def check_metric_input_binding(run_dir: str, metric: Mapping, *,
                               what: str, allow_backfilled: bool) -> None:
    """Refuse a metric whose declared inputs are not this run's.

    `allow_backfilled` is the CALLER's trust policy and has no default. It used
    to be read off the artefact -- `allow_backfilled=bool(declared["backfilled_
    inputs"])` -- so a marker that declared itself backfilled authorised its own
    admission, and a reader that had explicitly asked for `allow_backfilled=
    False` still got it (§20.3). An artefact must never decide how much it is
    trusted.
    """
    declared = metric.get("input_binding")
    if not isinstance(declared, Mapping):
        raise ExtractionInvalid(
            f"{what} records no input_binding; it cannot be tied to the "
            f"extraction it claims to describe")
    if declared.get("schema_version") != 1:
        raise ExtractionInvalid(
            f"{what}: unknown input_binding schema_version "
            f"{declared.get('schema_version')!r}")
    if os.path.abspath(str(declared.get("run_dir") or "")) != os.path.abspath(
            run_dir):
        raise ExtractionInvalid(
            f"{what}: input_binding names run_dir "
            f"{declared.get('run_dir')!r}, not {run_dir}")
    if declared.get("backfilled_inputs") and not allow_backfilled:
        raise ExtractionInvalid(
            f"{what}: the metric declares retrospectively bound inputs, and "
            f"this reader did not admit backfilled runs")
    current = metric_input_binding(run_dir, allow_backfilled=allow_backfilled)
    # What the numbers were computed FROM. `validator_sha256` is deliberately
    # not here: this module checks artefacts, it does not produce them, so
    # requiring equality made every improvement to the checker invalidate every
    # measurement ever taken -- three full GPU recomputes were spent on exactly
    # that. It is still recorded, so a reader can see which checker admitted the
    # numbers; it is simply not a reason to reject them.
    for key in ("npz_sha256", "manifest_sha256", "checkpoint_sha256",
                "config_sha256", "inference_epoch", "inference_epoch_source",
                "dataset", "random_seed", "codebook_size",
                "backfilled_inputs"):
        if declared.get(key) != current[key]:
            raise ExtractionInvalid(
                f"{what}: input_binding.{key} does not match the extraction "
                f"in {run_dir}; the metric was computed from different files")
    if "validator_sha256" not in declared:
        raise ExtractionInvalid(
            f"{what}: input_binding does not name the validator that admitted "
            f"it")


ANALYSIS_MARKER_NAME = "analysis_complete.json"

#: Every number a Phase 2 cell must carry, exactly. The marker used to be
#: sealed by the bio-evaluation alone, so `mean_off_diag_nmi` was simply absent;
#: the aggregator read it with `.get()` and still counted the cell as paired, so
#: "15 paired" with a blank NMI column throughout was reachable (§20.4). A
#: missing metric is now a refusal, not a `None`.
REQUIRED_METRICS = (
    "map_at_R_bioproj",
    "full_map_bioproj",
    "full_map_pre_projection",
    "dna_unique_db",
    "mean_off_diag_nmi",
)

#: Constants that decide what the numbers above MEAN. A GC window or an R
#: cutoff silently changed between two cells makes their delta meaningless, so
#: the caller states what it expects and the marker is compared against it.
REQUIRED_PROTOCOL = (
    "dataset",
    "codebook_size",
    "total_bases",
    "map_r_cutoff",
    "gc_policy_version",
    "gc_count_min_inclusive",
    "gc_count_max_inclusive",
    "gc_min_frac",
    "gc_max_frac",
    "nmi_average_method",
    "sklearn_version",
)

#: The code whose bytes decide the numbers, keyed by the field that records it.
#: Recomputed and compared on read, so a marker cannot outlive the evaluator
#: that produced it. The validator itself is already covered by the binding.
ANALYSIS_SOURCES = {
    "eval_cell_bioproj_sha256": "scripts/eval_cell_bioproj.py",
    "evaluation_siglip2_sha256": "evaluation_siglip2.py",
    "pairwise_nmi_sha256": "scripts/pairwise_nmi.py",
    "bio_constraints_sha256": "dna_utils/bio_constraints.py",
    "gc_policy_sha256": "dna_utils/gc_policy.py",
    # The base/bit Hamming distances and the validity projection the retrieval
    # numbers are computed with. Omitting it left the distance implementation
    # free to change under a marker that still read as current.
    "dna_code_utils_sha256": "dna_utils/dna_code_utils.py",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analysis_source_digests() -> dict:
    """Digest every file that decides an analysis number, right now."""
    return {field_name: sha256_file(os.path.join(_REPO_ROOT, rel))
            for field_name, rel in sorted(ANALYSIS_SOURCES.items())}


def _check_metrics(metrics: Mapping, *, what: str) -> None:
    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        raise ExtractionInvalid(
            f"{what}: analysis is not complete, it has no {', '.join(missing)}")
    extra = sorted(set(metrics) - set(REQUIRED_METRICS))
    if extra:
        raise ExtractionInvalid(
            f"{what}: analysis carries unexpected metric(s) "
            f"{', '.join(extra)}; the metric set is fixed so two cells are "
            f"always comparable")
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExtractionInvalid(
                f"{what}: metric {key} is {value!r}, not a number")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ExtractionInvalid(f"{what}: metric {key} is not finite")
        # mAP, a unique-code ratio and an NMI are all proportions. A probe
        # marker holding 999 was previously accepted and aggregated.
        if not 0.0 <= value <= 1.0:
            raise ExtractionInvalid(
                f"{what}: metric {key}={value} is outside [0, 1]")


def _check_protocol(protocol: Mapping, *, what: str,
                    expected: "Mapping | None") -> None:
    missing = [k for k in REQUIRED_PROTOCOL if k not in protocol]
    if missing:
        raise ExtractionInvalid(
            f"{what}: analysis protocol has no {', '.join(missing)}")
    extra = sorted(set(protocol) - set(REQUIRED_PROTOCOL))
    if extra:
        raise ExtractionInvalid(
            f"{what}: analysis protocol carries unexpected key(s) "
            f"{', '.join(extra)}")
    if expected is None:
        return
    for key, want in sorted(expected.items()):
        if key not in REQUIRED_PROTOCOL:
            raise ExtractionInvalid(
                f"{what}: caller expects unknown protocol key {key!r}")
        if protocol[key] != want:
            raise ExtractionInvalid(
                f"{what}: protocol.{key} is {protocol[key]!r}, but this reader "
                f"requires {want!r}; the numbers do not mean what it assumes")


def _check_sources(sources: Mapping, *, what: str) -> None:
    current = analysis_source_digests()
    if set(sources) != set(current):
        raise ExtractionInvalid(
            f"{what}: analysis_sources names {sorted(sources)}, expected "
            f"{sorted(current)}")
    stale = [k for k, digest in sorted(current.items())
             if sources[k] != digest]
    if stale:
        raise ExtractionInvalid(
            f"{what}: {', '.join(stale)} has changed since these numbers were "
            f"computed; recompute them or read them from the commit that "
            f"produced them")


def write_analysis_marker(run_dir: str, *, metrics: Mapping,
                          binding: Mapping, protocol: Mapping,
                          sources: Mapping) -> str:
    """Seal one canonical analysis result, atomically, at the very end.

    The aggregator used to VALIDATE `cell_result.json` and `pairwise_nmi.json`
    and then READ its numbers from an unsigned
    `evaluation_siglip2_base_bioproj.json`. A probe with `mAP@R=999` in that
    third file was reported as 999, so the binding gate proved nothing. There is
    exactly one file to read now, it carries the numbers and their binding
    together, and it is written last.

    The contract is checked HERE as well as on read, so an incomplete or
    out-of-range marker cannot be created in the first place. A stage that has
    only half the numbers -- the bio-evaluation without the NMI -- therefore
    cannot seal a file named `analysis_complete`.
    """
    _check_metrics(metrics, what=f"{run_dir}/analysis")
    _check_protocol(protocol, what=f"{run_dir}/analysis", expected=None)
    _check_sources(sources, what=f"{run_dir}/analysis")
    payload = {
        "schema_version": 1,
        "metrics": dict(metrics),
        "input_binding": dict(binding),
        "protocol": dict(protocol),
        "analysis_sources": dict(sources),
    }
    path = os.path.join(run_dir, ANALYSIS_MARKER_NAME)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def read_analysis_marker(run_dir: str, *, allow_backfilled: bool,
                         expected: "ExpectedIdentity | None" = None,
                         expected_protocol: "Mapping | None" = None) -> dict:
    """The only admissible source of a cell's numbers.

    It used to check that four keys were mappings and stop there. A probe
    marker carrying `map_at_R_bioproj: 999`, `dataset: "WRONG"` and
    `analysis_sources: {"sha": "garbage"}` was admitted and aggregated (§20.2).
    The contents are now checked as strictly as the binding: an exact metric set
    of finite proportions, an exact protocol schema compared against what the
    caller expects, and source digests recomputed from the working tree.

    `allow_backfilled` has no default -- the reader states its trust policy,
    and the artefact does not get to state it for them.
    """
    path = os.path.join(run_dir, ANALYSIS_MARKER_NAME)
    payload = _load_json(path, "analysis marker")
    if payload.get("schema_version") != 1:
        raise ExtractionInvalid(
            f"{run_dir}: unknown analysis marker schema_version "
            f"{payload.get('schema_version')!r}")
    for key in ("metrics", "input_binding", "protocol", "analysis_sources"):
        if not isinstance(payload.get(key), Mapping):
            raise ExtractionInvalid(f"{run_dir}: analysis marker has no {key}")
    what = f"{run_dir}/analysis"
    _check_metrics(payload["metrics"], what=what)
    _check_protocol(payload["protocol"], what=what, expected=expected_protocol)
    _check_sources(payload["analysis_sources"], what=what)
    check_metric_input_binding(run_dir, payload, what=what,
                               allow_backfilled=allow_backfilled)
    if expected is not None:
        validate_extraction_run(run_dir, allow_backfilled=allow_backfilled,
                                expected=expected)
    return payload
