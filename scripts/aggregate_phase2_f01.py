"""Phase 2: what the epoch-0 extraction bug (F01) actually cost.

Pairs each seed-42 N candidate's LEGACY artefacts -- produced by an extraction
that ran the router at the initial Sinkhorn epsilon because `_current_epoch` is
a plain int absent from the state dict -- against a re-inference of the SAME
checkpoint with the epoch restored.

Only cells whose two sides used the same GC window are compared. The legacy P0
cell runner passed `--gc_min 0.416 --gc_max 0.584`, which is count [7, 8] at 15
bases against the central policy's [6, 9]; a pair straddling those windows would
measure the window change, not F01, so it is reported as unpaired rather than
quietly differenced.

This is a diagnostic of the defect's impact. It does NOT select N -- that is
F02/F03's train-only protocol in Phase 3.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import sklearn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402
from dna_utils.extraction_validation import ExtractionInvalid  # noqa: E402
from dna_utils.runtime_state import sha256_file  # noqa: E402

#: dataset/N -> legacy run directory, as audited against args.txt before launch.
LEGACY = {
    ("cifar10", 4): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("cifar10", 9): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("cifar10", 19): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("cifar10", 39): "result/260812+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001",
    ("flickr25k", 4): "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("flickr25k", 9): "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("flickr25k", 19): "result/260812+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("nuswide", 4): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("nuswide", 9): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("nuswide", 19): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("nuswide", 39): "result/260812+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001",
    ("mscoco", 4): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("mscoco", 9): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("mscoco", 19): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("mscoco", 39): "result/260812+mscoco_setting1_promptAblA_mscoco_A_v5b_s42_P0refit_e39+bs+64+e+40+proj_lr+0.001",
}
ORDER = ("cifar10", "flickr25k", "nuswide", "mscoco")
TOTAL_BASES = 15


#: What each dataset's cells must declare. The aggregator knows these from the
#: protocol, so it says so rather than accepting whatever the artefact claims:
#: fifteen copies of one CIFAR cell under fifteen names previously produced a
#: `complete: true` report of 15/15 (§22.4).
_DATASET_TOKEN = {"cifar10": "CIFAR10", "flickr25k": "Flickr25k",
                  "nuswide": "NUSWIDE", "mscoco": "MSCOCO"}
_CODEBOOK_SIZE = {"cifar10": 64, "flickr25k": 128, "nuswide": 128,
                  "mscoco": 128}
_MAP_R_CUTOFF = {"cifar10": 1000, "flickr25k": 5000, "nuswide": 5000,
                 "mscoco": 5000}
SEED = 42
SLOTS, BASES_PER_SLOT = 5, 3
INPUT_PAIR_RECEIPT_SCHEMA = 1
REPORT_RECEIPT_SCHEMA = 1
EXPECTED_CELLS = tuple(
    f"{dataset}_N{n}" for dataset, n in sorted(LEGACY))


def expected_identity(dataset: str, n: int, *, epoch: int,
                      epoch_source: str) -> "ExpectedIdentity":
    from dna_utils.extraction_validation import ExpectedIdentity
    return ExpectedIdentity(
        dataset=dataset, random_seed=SEED, inference_epoch=epoch,
        inference_epoch_source=epoch_source, num_slots=SLOTS,
        bases_per_slot=BASES_PER_SLOT,
        codebook_size=_CODEBOOK_SIZE[dataset])


def expected_protocol(dataset: str, policy) -> dict:
    return {
        "dataset": _DATASET_TOKEN[dataset],
        "codebook_size": _CODEBOOK_SIZE[dataset],
        "total_bases": TOTAL_BASES,
        "map_r_cutoff": _MAP_R_CUTOFF[dataset],
        "gc_policy_version": policy.policy_version,
        "gc_count_min_inclusive": policy.gc_min_count,
        "gc_count_max_inclusive": policy.gc_max_count,
        # Pinned too: comparing the two sides' GC FRACTIONS to each other only
        # proves they agree, not that either matches the central policy, and
        # the NMI is not interpretable without its averaging convention.
        "gc_min_frac": policy.gc_min_frac,
        "gc_max_frac": policy.gc_max_frac,
        "bio_max_homopolymer_run": policy.max_run,
        "nmi_average_method": "arithmetic",
        "sklearn_version": sklearn.__version__,
    }


def _publish_text_exclusive(path: Path, text: str) -> None:
    """Atomically publish one fresh evidence file without following its leaf."""
    path = _leaf_path(path)
    if path.exists() or path.is_symlink():
        raise ManifestMissing(f"refusing to overwrite {path}")
    from scripts.bind_legacy_phase2 import BindRefused, _publish_bytes
    try:
        _publish_bytes(path, text.encode("utf-8"))
    except BindRefused as error:
        raise ManifestMissing(str(error)) from None


def _leaf_path(value: os.PathLike | str) -> Path:
    raw = Path(os.path.abspath(os.fspath(value)))
    try:
        return raw.parent.resolve(strict=True) / raw.name
    except OSError as error:
        raise ManifestMissing(f"output parent is not canonical: {error}") from None


def _bound_bytes(path: Path, what: str) -> tuple[bytes, str]:
    from scripts.bind_legacy_phase2 import BindRefused, _stable_bytes
    try:
        return _stable_bytes(path, what)
    except BindRefused as error:
        raise ManifestMissing(str(error)) from None


def _publish_json_exclusive(path: Path, payload: dict) -> None:
    _publish_text_exclusive(
        path, json.dumps(payload, indent=2, sort_keys=True,
                         allow_nan=False) + "\n")


def _git_head() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO,
        capture_output=True, text=True, timeout=30, check=True).stdout
    if status.strip():
        raise ManifestMissing("Phase 2 aggregation source tree is dirty")
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True, timeout=30, check=True)
    return process.stdout.strip()


def _pair_authority_payload(fixed_root: Path, legacy_root: Path) -> dict:
    from scripts.bind_legacy_phase2 import verify_input_root
    fixed, fixed_sha = verify_input_root(fixed_root, "fixed")
    legacy, legacy_sha = verify_input_root(legacy_root, "legacy")
    if set(fixed["cells"]) != set(EXPECTED_CELLS) \
            or set(legacy["cells"]) != set(EXPECTED_CELLS):
        raise ManifestMissing("Phase 2 input receipts are not exact-15")

    if fixed.get("pair_evidence") != legacy.get("pair_evidence"):
        raise ManifestMissing(
            "fixed and legacy input receipts bind different sample/order evidence")
    if fixed.get("limitations") != legacy.get("limitations"):
        raise ManifestMissing("fixed and legacy input limitations differ")
    return {
        "schema_version": INPUT_PAIR_RECEIPT_SCHEMA,
        "receipt_type": "phase2_input_pair",
        "eligibility": "diagnostic_only_never_paper_main",
        "git_head": _git_head(),
        "roots": {
            "fixed": str(fixed_root.resolve()),
            "legacy": str(legacy_root.resolve()),
        },
        "input_receipt_sha256": {
            "fixed": fixed_sha, "legacy": legacy_sha},
        "source_sha256": {
            "aggregator": sha256_file(__file__),
            "materializer": sha256_file(
                str(REPO / "scripts" / "bind_legacy_phase2.py")),
            "validator": sha256_file(
                str(REPO / "dna_utils" / "extraction_validation.py")),
        },
        "pair_evidence": fixed["pair_evidence"],
        "limitations": fixed["limitations"],
    }


def _load_pair_receipt(path: Path, fixed_root: Path,
                       legacy_root: Path) -> tuple[dict, str]:
    from dna_utils.extraction_validation import _load_json_bound
    payload, digest = _load_json_bound(str(path), "Phase 2 input-pair receipt")
    expected = _pair_authority_payload(fixed_root, legacy_root)
    if payload != expected:
        raise ManifestMissing(
            "Phase 2 input-pair receipt differs from reopened input roots")
    return payload, digest


def _report_source_sha256() -> dict:
    from dna_utils.extraction_validation import analysis_source_digests
    sources = analysis_source_digests()
    for name, relative in {
        "aggregate_phase2_f01": "scripts/aggregate_phase2_f01.py",
        "bind_legacy_phase2": "scripts/bind_legacy_phase2.py",
        "extraction_validation": "dna_utils/extraction_validation.py",
        "phase2_cell_state": "scripts/_phase2_cell_state.py",
        "phase2_recompute_metrics": "scripts/phase2_recompute_metrics.sh",
        "runtime_state": "dna_utils/runtime_state.py",
        "seal_cell_analysis": "scripts/seal_cell_analysis.py",
    }.items():
        sources[f"{name}_sha256"] = sha256_file(str(REPO / relative))
    return sources


def verify_report_receipt(
        path: Path, *, expected_fixed_root: Path, expected_legacy_root: Path,
        expected_pair_receipt: Path, expected_report_json: Path,
        expected_report_md: Path) -> tuple[dict, str]:
    from dna_utils.extraction_validation import ANALYSIS_MARKER_NAME, _load_json_bound
    path = _leaf_path(path)
    receipt, digest = _load_json_bound(str(path), "Phase-2 report receipt")
    expected_keys = {
        "schema_version", "receipt_type", "eligibility",
        "paper_result_eligible", "git_head", "paired_cells",
        "expected_cells", "unpaired_cells", "complete",
        "input_pair_receipt", "input_receipt_sha256",
        "analysis_marker_sha256", "source_sha256", "reports"}
    if set(receipt) != expected_keys or receipt.get("schema_version") != 1 \
            or receipt.get("receipt_type") != "phase2_f01_report" \
            or receipt.get("eligibility") != "diagnostic_only_never_paper_main" \
            or receipt.get("paper_result_eligible") is not False \
            or receipt.get("paired_cells") != len(EXPECTED_CELLS) \
            or receipt.get("expected_cells") != len(EXPECTED_CELLS) \
            or receipt.get("unpaired_cells") != 0 \
            or receipt.get("complete") is not True \
            or receipt.get("git_head") != _git_head() \
            or receipt.get("source_sha256") != _report_source_sha256():
        raise ManifestMissing("Phase-2 final receipt envelope/source is invalid")

    pair_ref = receipt.get("input_pair_receipt") or {}
    pair_path = _leaf_path(pair_ref.get("path"))
    pair, pair_sha = _load_json_bound(str(pair_path), "Phase-2 input-pair receipt")
    roots = pair.get("roots") or {}
    fixed_root, legacy_root = Path(roots.get("fixed", "")), Path(roots.get("legacy", ""))
    expected_paths = (
        Path(expected_fixed_root).resolve(), Path(expected_legacy_root).resolve(),
        _leaf_path(expected_pair_receipt), _leaf_path(expected_report_json),
        _leaf_path(expected_report_md))
    if (fixed_root.resolve(), legacy_root.resolve(), pair_path,
            _leaf_path((receipt.get("reports") or {}).get("json", {}).get("path")),
            _leaf_path((receipt.get("reports") or {}).get("markdown", {}).get("path"))) \
            != expected_paths:
        raise ManifestMissing("Phase-2 receipt paths differ from this invocation")
    reopened_pair, reopened_sha = _load_pair_receipt(
        pair_path, fixed_root, legacy_root)
    if pair != reopened_pair or pair_sha != reopened_sha \
            or pair_ref.get("sha256") != pair_sha \
            or receipt.get("input_receipt_sha256") \
            != pair.get("input_receipt_sha256"):
        raise ManifestMissing("Phase-2 final receipt input authority differs")

    marker_shas = receipt.get("analysis_marker_sha256")
    if not isinstance(marker_shas, dict) or set(marker_shas) != {"fixed", "legacy"}:
        raise ManifestMissing("Phase-2 final receipt marker map is invalid")
    if any(set(marker_shas.get(side, {})) != set(EXPECTED_CELLS)
           for side in ("fixed", "legacy")):
        raise ManifestMissing("Phase-2 final receipt marker set is not exact")

    reports = receipt.get("reports") or {}
    if set(reports) != {"json", "markdown"}:
        raise ManifestMissing("Phase-2 final receipt report map is invalid")
    report_path = _leaf_path((reports["json"] or {}).get("path"))
    report, report_sha = _load_json_bound(str(report_path), "Phase-2 JSON report")
    markdown_path = _leaf_path((reports["markdown"] or {}).get("path"))
    markdown, markdown_sha = _bound_bytes(markdown_path, "Phase-2 Markdown report")
    if (reports["json"] or {}).get("sha256") != report_sha \
            or (reports["markdown"] or {}).get("sha256") != markdown_sha \
            or report.get("complete") is not True \
            or report.get("paired_cells") != len(EXPECTED_CELLS) \
            or report.get("unpaired") != [] \
            or (report.get("input_authority") or {}).get(
                "pair_receipt_sha256") != pair_sha:
        raise ManifestMissing("Phase-2 report bytes/authority differ from receipt")

    canonical, canonical_md, canonical_marker_shas, _ = _reduce_report(
        fixed_root, legacy_root, pair_path, pair, pair_sha)
    if canonical_marker_shas != marker_shas:
        raise ManifestMissing("Phase-2 report marker evidence differs from receipt")
    if canonical != report or canonical_md.encode("utf-8") != markdown:
        raise ManifestMissing("Phase-2 report semantics differ from sealed markers")

    # Final bookend: receipt, roots, markers, reports, and source must still be
    # the same evidence after the long semantic reconstruction.
    final_pair, final_pair_sha = _load_pair_receipt(
        pair_path, fixed_root, legacy_root)
    if final_pair != pair or final_pair_sha != pair_sha:
        raise ManifestMissing("Phase-2 input authority changed during verification")
    for side, root in (("fixed", fixed_root), ("legacy", legacy_root)):
        for cell, marker_sha in marker_shas[side].items():
            _, current_sha = _load_json_bound(
                str(root / cell / ANALYSIS_MARKER_NAME),
                "Phase-2 analysis marker final reopen")
            if current_sha != marker_sha:
                raise ManifestMissing(f"Phase-2 {side}/{cell} marker changed")
    final_report, final_report_sha = _load_json_bound(
        str(report_path), "Phase-2 JSON report final reopen")
    final_md, final_md_sha = _bound_bytes(
        markdown_path, "Phase-2 Markdown report final reopen")
    if final_report != report or final_report_sha != report_sha \
            or final_md != markdown or final_md_sha != markdown_sha \
            or receipt.get("git_head") != _git_head() \
            or receipt.get("source_sha256") != _report_source_sha256():
        raise ManifestMissing("Phase-2 evidence changed during final verification")
    # The completion authority is the literal last external read. In
    # particular, no source hashing window follows this bookend.
    final_receipt, final_digest = _load_json_bound(
        str(path), "Phase-2 report receipt final reopen")
    if final_receipt != receipt or final_digest != digest:
        raise ManifestMissing("Phase-2 evidence changed during final verification")
    return receipt, digest


def _require_manifests(run_dir: Path, *, identity=None,
                       protocol: dict | None = None) -> dict:
    """Admit a cell only if the shared strict validator admits it.

    The first version checked that two JSON files existed and that four fields
    agreed. A probe with no NPZ, no checkpoint, no config, `schema_version=999`
    and `dataset=WRONG` was accepted, so `15 paired / 0 unpaired` said nothing
    about provenance. The validator opens every file the manifests name,
    recomputes every digest, and checks the code arrays themselves.

    `allow_backfilled=True` because Phase 2 IS the retrospectively bound
    diagnostic; a paper path must not pass it.
    """
    from dna_utils.extraction_validation import (
        ExtractionInvalid, validate_extraction_run)
    try:
        validated = validate_extraction_run(
            str(run_dir), required_splits=("db", "query"),
            allow_backfilled=True, expected=identity)
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None

    # §17.3, second half: the aggregator reads metric JSONs that live beside the
    # NPZs but were never tied to them. A stale metric next to a valid
    # extraction was admitted, so the reported numbers had no end-to-end
    # binding. Each metric file must now name the inputs it was computed from.
    from dna_utils.extraction_validation import read_analysis_marker
    try:
        marker = read_analysis_marker(str(run_dir), allow_backfilled=True,
                                      expected_protocol=protocol)
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None
    return validated.splits, marker, _marker_evidence(run_dir, marker)


class ManifestMissing(RuntimeError):
    """A cell has metrics but no admissible provenance."""


def _marker_evidence(run_dir: Path, expected: dict) -> str:
    from dna_utils.extraction_validation import (
        ANALYSIS_MARKER_NAME, ExtractionInvalid, _load_json_bound)
    try:
        payload, digest = _load_json_bound(
            str(run_dir / ANALYSIS_MARKER_NAME), "Phase-2 analysis marker")
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None
    if payload != expected:
        raise ManifestMissing("analysis marker changed after validation")
    return digest


def _side(run_dir: Path, sealed: dict):
    """Metrics plus the GC window they were projected under.

    BOTH sides come from a sealed marker now. The legacy side used to be read
    unsigned, from whatever `evaluation_siglip2_base_bioproj.json` happened to
    sit in the August run directory -- computed by an older bio-projection and
    an older NMI. A difference between two evaluators is not a measurement of
    F01 (§19.4), so the legacy extraction is re-scored by today's evaluator via
    `scripts/bind_legacy_phase2.py`, which binds the legacy NPZs in a separate
    root and never writes into the legacy run.
    """
    metrics = sealed["metrics"]
    protocol = sealed["protocol"]
    return {
        "dir": str(run_dir),
        "sealed": True,
        "map_at_R_bioproj": metrics["map_at_R_bioproj"],
        "full_map_bioproj": metrics["full_map_bioproj"],
        # Validated and sealed, then dropped from every delta and mean. A
        # metric the contract requires and the report omits is a metric nobody
        # checks.
        "full_map_pre_projection": metrics["full_map_pre_projection"],
        "dna_unique_db": metrics["dna_unique_db"],
        "nmi": metrics["mean_off_diag_nmi"],
        "gc_min_frac": protocol["gc_min_frac"],
        "gc_max_frac": protocol["gc_max_frac"],
        "analysis_sources": sealed["analysis_sources"],
    }


#: Everything the two sides of one pair must share. F01 changes the epoch, the
#: epoch's source and the epsilon that follows from it -- and nothing else. Any
#: other disagreement means the pair is not measuring F01.
_PAIR_INVARIANT = (
    "checkpoint_sha256", "config_sha256", "dataset", "random_seed",
    "codebook_size", "num_slots", "bases_per_slot", "total_bases",
    "total_bits", "lr_schedule_horizon", "training_epoch_budget",
    "training_stop_epoch", "sinkhorn_schedule_horizon",
    "sinkhorn_annealing_enabled",
)


def _pair_identity(fixed: dict, legacy: dict) -> tuple[bool, list]:
    """Prove the two sides really are one run, differing only along F01."""
    differing = []
    for split in sorted(set(fixed) & set(legacy)):
        for key in _PAIR_INVARIANT:
            if fixed[split].get(key) != legacy[split].get(key):
                differing.append(
                    f"{split}.{key} ({fixed[split].get(key)!r} vs "
                    f"{legacy[split].get(key)!r})")
        if fixed[split].get("n_rows") != legacy[split].get("n_rows"):
            differing.append(
                f"{split}.n_rows ({fixed[split].get('n_rows')} vs "
                f"{legacy[split].get('n_rows')})")
    return not differing, differing


#: The metrics the report carries. Kept as one list so a metric cannot be
#: validated, sealed and then quietly dropped from the deltas and the means.
_REPORTED_METRICS = ("map_at_R_bioproj", "full_map_bioproj",
                     "full_map_pre_projection", "dna_unique_db", "nmi")

#: The axis F01 IS. Recorded per side so the reader can see it rather than
#: infer it from the directory names.
_PROVENANCE_FIELDS = ("inference_epoch", "inference_epoch_source",
                      "effective_sinkhorn_epsilon",
                      "sinkhorn_annealing_enabled", "backfilled")


def _provenance(manifest: dict) -> dict:
    return {key: manifest.get(key) for key in _PROVENANCE_FIELDS}


def _delta(new, old):
    if new is None or old is None:
        return None
    return float(new) - float(old)


def _reduce_report(root: Path, legacy_root: Path, pair_receipt_path: Path,
                   pair_authority: dict, pair_receipt_sha256: str
                   ) -> tuple[dict, str, dict, dict]:
    """Validate the 30 markers once and deterministically render the report."""
    policy = resolve_gc_policy(TOTAL_BASES)
    cells, unpaired = [], []
    marker_sha256 = {"fixed": {}, "legacy": {}}
    markers = {"fixed": {}, "legacy": {}}
    for dataset in ORDER:
        for (ds, n), _legacy_rel in sorted(LEGACY.items()):
            if ds != dataset:
                continue
            cell_dir = root / f"{ds}_N{n}"
            legacy_dir = legacy_root / f"{ds}_N{n}"
            protocol = expected_protocol(ds, policy)
            try:
                manifests, marker, marker_sha = _require_manifests(
                    cell_dir, protocol=protocol,
                    identity=expected_identity(
                        ds, n, epoch=n, epoch_source="explicit_flag"))
                legacy_manifests, legacy_marker, legacy_marker_sha = \
                    _require_manifests(
                        legacy_dir, protocol=protocol,
                        identity=expected_identity(
                            ds, n, epoch=0,
                            epoch_source="f01_unrestored"))
            except ManifestMissing as error:
                unpaired.append({"cell": f"{ds}/N{n}", "reason": str(error)})
                continue
            new = _side(cell_dir, marker)
            old = _side(legacy_dir, legacy_marker)
            same, differing = _pair_identity(manifests, legacy_manifests)
            if not same:
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": (f"the two sides are not the same run: "
                               f"{', '.join(differing)} differ, so the "
                               f"difference is not F01"),
                })
                continue
            if new["analysis_sources"] != old["analysis_sources"]:
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": ("the two sides were scored by different "
                               "analysis sources; the delta would measure the "
                               "evaluator change, not F01"),
                })
                continue
            if (new["gc_min_frac"], new["gc_max_frac"]) != (
                    old["gc_min_frac"], old["gc_max_frac"]):
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": "GC windows differ; the difference would measure "
                              "the window change, not F01",
                    "phase2_gc": [new["gc_min_frac"], new["gc_max_frac"]],
                    "legacy_gc": [old["gc_min_frac"], old["gc_max_frac"]],
                })
                continue
            cells.append({
                "cell": f"{ds}/N{n}", "dataset": ds, "N": n,
                "legacy": old, "phase2": new,
                "fixed_side_sealed": bool(new["sealed"]),
                "legacy_side_sealed": bool(old["sealed"]),
                "same_evaluator_both_sides": True,
                "provenance": {
                    "checkpoint_sha256": manifests["db"]["checkpoint_sha256"],
                    "config_sha256": manifests["db"]["config_sha256"],
                    "pair_identity_verified": True,
                    "phase2": _provenance(manifests["db"]),
                    "legacy": _provenance(legacy_manifests["db"]),
                },
                "delta": {
                    key: _delta(new[key], old[key])
                    for key in _REPORTED_METRICS
                },
            })
            name = f"{ds}_N{n}"
            marker_sha256["fixed"][name] = marker_sha
            marker_sha256["legacy"][name] = legacy_marker_sha
            markers["fixed"][name] = marker
            markers["legacy"][name] = legacy_marker

    def mean_delta(key):
        values = [c["delta"][key] for c in cells if c["delta"][key] is not None]
        return statistics.fmean(values) if values else None

    payload = {
        "what_this_measures": (
            "F01 only: the same checkpoint re-inferred with the training epoch "
            "restored, so the router runs at the annealed Sinkhorn epsilon "
            "instead of the initial one. Not an N selection."),
        "total_bases": TOTAL_BASES,
        "gc_policy": policy.as_manifest_record(),
        "paired_cells": len(cells),
        "unpaired": unpaired,
        "mean_delta": {key: mean_delta(key) for key in _REPORTED_METRICS},
        "cells": cells,
        "expected_cells": len(LEGACY),
        "complete": len(cells) == len(LEGACY),
        "eligibility": "diagnostic_only_never_paper_main",
        "input_authority": {
            "pair_receipt": str(pair_receipt_path),
            "pair_receipt_sha256": pair_receipt_sha256,
            "input_receipt_sha256": pair_authority["input_receipt_sha256"],
            "limitations": pair_authority["limitations"],
        },
    }

    def fmt(value, places=4):
        return "-" if value is None else f"{float(value):+.{places}f}"

    def raw(value, places=4):
        return "-" if value is None else f"{float(value):.{places}f}"

    lines = [
        "# Phase 2 — what the epoch-0 extraction bug (F01) cost", "",
        payload["what_this_measures"], "",
        "- Ranking below is **post-bio diagnostic**. D1's selection metric is "
        "raw base-Hamming mAP@R, which these artefacts do not store, so this "
        "does not show what the train-only selection would have chosen.",
        "- Both sides are scored by the SAME committed evaluator, and the "
        "aggregation refuses any pair whose recorded analysis sources differ.",
        "- Extraction manifests were **backfilled** after the fact and carry "
        "`backfilled: true`; the cells' inputs were verified byte-identical to "
        "their canonical legacy sources, but the manifests are not evidence "
        "that the extraction recorded itself.",
        "- CIFAR stores class-label bytes but no image-path identity; exact "
        "label order is paired, while within-class sample permutation cannot "
        "be excluded. This is one reason the result remains diagnostic-only.",
        f"- Paired cells: **{len(cells)} / {len(LEGACY)}**",
        f"- GC window (both sides): count "
        f"[{policy.gc_min_count}, {policy.gc_max_count}] at {TOTAL_BASES} bases "
        f"(`{policy.policy_version}`)", "",
        "| cell | mAP@R legacy | mAP@R fixed | Δ | DNA-uniq legacy | DNA-uniq fixed | Δ | NMI legacy | NMI fixed | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['cell']} "
            f"| {raw(cell['legacy']['map_at_R_bioproj'])} "
            f"| {raw(cell['phase2']['map_at_R_bioproj'])} "
            f"| {fmt(cell['delta']['map_at_R_bioproj'])} "
            f"| {raw(cell['legacy']['dna_unique_db'])} "
            f"| {raw(cell['phase2']['dna_unique_db'])} "
            f"| {fmt(cell['delta']['dna_unique_db'])} "
            f"| {raw(cell['legacy']['nmi'])} "
            f"| {raw(cell['phase2']['nmi'])} "
            f"| {fmt(cell['delta']['nmi'])} |")
    lines += [
        "",
        "Mean Δ (fixed − legacy): "
        f"mAP@R {fmt(payload['mean_delta']['map_at_R_bioproj'])}, "
        f"DNA-uniq {fmt(payload['mean_delta']['dna_unique_db'])}, "
        f"NMI {fmt(payload['mean_delta']['nmi'])}.",
    ]
    if unpaired:
        lines += ["", "## Unpaired", ""]
        lines += [f"- `{entry['cell']}`: {entry['reason']}" for entry in unpaired]
    return payload, "\n".join(lines) + "\n", marker_sha256, markers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root")
    parser.add_argument(
        "--legacy-root",
        help=("The LEGACY extraction re-scored by the same evaluator as the "
              "fixed side, bound by scripts/bind_legacy_phase2.py. Never "
              "written back into the legacy run directory."))
    parser.add_argument("--pair-receipt")
    parser.add_argument("--verify-report-receipt")
    parser.add_argument("--expected-fixed-root")
    parser.add_argument("--expected-legacy-root")
    parser.add_argument("--expected-pair-receipt")
    parser.add_argument("--expected-report-json")
    parser.add_argument("--expected-report-md")
    parser.add_argument("--seal-input-pair", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    parser.add_argument("--out-receipt")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help=("Write a `.partial` report beside the official paths when not "
              "every cell is admissible. The official files are never written "
              "from an incomplete run, and the exit status stays nonzero."))
    args = parser.parse_args()

    if args.verify_report_receipt:
        if any((args.phase2_root, args.legacy_root, args.pair_receipt,
                args.seal_input_pair, args.out_json, args.out_md,
                args.out_receipt, args.allow_incomplete)):
            parser.error("--verify-report-receipt is exclusive")
        expected = (args.expected_fixed_root, args.expected_legacy_root,
                    args.expected_pair_receipt, args.expected_report_json,
                    args.expected_report_md)
        if not all(expected):
            parser.error("report verification requires all expected paths")
        try:
            _, digest = verify_report_receipt(
                Path(args.verify_report_receipt),
                expected_fixed_root=Path(args.expected_fixed_root),
                expected_legacy_root=Path(args.expected_legacy_root),
                expected_pair_receipt=Path(args.expected_pair_receipt),
                expected_report_json=Path(args.expected_report_json),
                expected_report_md=Path(args.expected_report_md))
        except (ExtractionInvalid, ManifestMissing, OSError,
                TypeError, ValueError) as error:
            print(f"REFUSED: {error}")
            return 2
        print(f"VERIFIED Phase-2 report receipt {digest}")
        return 0
    if not all((args.phase2_root, args.legacy_root, args.pair_receipt)):
        parser.error("--phase2-root, --legacy-root and --pair-receipt are required")
    if any((args.expected_fixed_root, args.expected_legacy_root,
            args.expected_pair_receipt, args.expected_report_json,
            args.expected_report_md)):
        parser.error("--expected-* paths are only for report verification")

    root = Path(args.phase2_root).resolve()
    legacy_root = Path(args.legacy_root).resolve()
    pair_receipt_path = _leaf_path(args.pair_receipt)
    if args.seal_input_pair:
        if any((args.out_json, args.out_md, args.out_receipt,
                args.allow_incomplete)):
            parser.error("--seal-input-pair accepts no report-output option")
        published = False
        try:
            payload = _pair_authority_payload(root, legacy_root)
            _publish_json_exclusive(pair_receipt_path, payload)
            published = True
            _load_pair_receipt(pair_receipt_path, root, legacy_root)
        except (ExtractionInvalid, ManifestMissing, OSError, ValueError) as error:
            if published:
                pair_receipt_path.unlink(missing_ok=True)
            print(f"REFUSED: {error}")
            return 2
        print(f"sealed Phase 2 input pair -> {pair_receipt_path}")
        return 0

    if not all((args.out_json, args.out_md, args.out_receipt)):
        parser.error(
            "final aggregation requires --out-json, --out-md and --out-receipt")
    try:
        pair_authority, pair_receipt_sha256 = _load_pair_receipt(
            pair_receipt_path, root, legacy_root)
        source_sha256 = _report_source_sha256()
    except (ExtractionInvalid, ManifestMissing, OSError, ValueError) as error:
        print(f"REFUSED: {error}")
        return 2

    try:
        requested_outputs = [_leaf_path(value) for value in (
            args.out_json, args.out_md, args.out_receipt)]
    except ManifestMissing as error:
        print(f"REFUSED: {error}")
        return 2
    if any(path.exists() or path.is_symlink() for path in requested_outputs):
        print("REFUSED: report output paths must all be fresh and absent")
        return 2
    if any(not path.parent.is_dir() for path in requested_outputs):
        print("REFUSED: every report output parent must already exist")
        return 2

    payload, markdown, analysis_marker_sha256, analysis_markers = \
        _reduce_report(
            root, legacy_root, pair_receipt_path, pair_authority,
            pair_receipt_sha256)
    cells, unpaired = payload["cells"], payload["unpaired"]

    # A report covering 1 of 15 cells is not a smaller version of the answer,
    # it is a different claim -- and it used to be written straight over the
    # official file with rc0 (§20.8). The official paths are only ever written
    # by a complete run.
    complete = len(cells) == len(LEGACY)
    out_json, out_md = requested_outputs[:2]
    if not complete:
        if not args.allow_incomplete:
            print(f"REFUSED: {len(cells)} of {len(LEGACY)} cells admissible; "
                  f"{len(unpaired)} unpaired. Not writing {out_json} or "
                  f"{out_md}. Pass --allow-incomplete to write a .partial "
                  f"report instead.")
            for entry in unpaired:
                print(f"  - {entry['cell']}: {entry['reason']}")
            return 1
        out_json = out_json.with_suffix(f".partial{out_json.suffix}")
        out_md = out_md.with_suffix(f".partial{out_md.suffix}")

    _publish_json_exclusive(out_json, payload)
    _publish_text_exclusive(out_md, markdown)
    try:
        from dna_utils.extraction_validation import _load_json_bound
        reopened_report, report_json_sha256 = _load_json_bound(
            str(out_json), "Phase-2 JSON report")
        markdown_bytes, report_md_sha256 = _bound_bytes(
            out_md, "Phase-2 Markdown report")
    except (ExtractionInvalid, ManifestMissing, OSError, ValueError) as error:
        print(f"REFUSED: report publication could not be reopened: {error}")
        return 2
    if reopened_report != payload or markdown_bytes.decode("utf-8") != markdown:
        print("REFUSED: published Phase-2 report differs from validated content")
        return 2
    if not complete:
        print(f"wrote incomplete diagnostic {out_json} and {out_md}; "
              "no completion receipt was issued")
        return 1

    # The receipt is last. If any evidence moved while the two report files
    # were built, leave those files unreceipted and fail closed.
    try:
        _, reopened_pair_sha256 = _load_pair_receipt(
            pair_receipt_path, root, legacy_root)
    except (ExtractionInvalid, ManifestMissing, OSError, ValueError) as error:
        print(f"REFUSED: input authority changed during aggregation: {error}")
        return 2
    if reopened_pair_sha256 != pair_receipt_sha256:
        print("REFUSED: input-pair receipt changed during aggregation")
        return 2
    for side, side_root in (("fixed", root), ("legacy", legacy_root)):
        for cell, digest in analysis_marker_sha256[side].items():
            try:
                current = _marker_evidence(
                    side_root / cell, analysis_markers[side][cell])
            except ManifestMissing as error:
                print(f"REFUSED: {side}/{cell} analysis marker: {error}")
                return 2
            if current != digest:
                print(f"REFUSED: {side}/{cell} analysis marker changed")
                return 2

    if _report_source_sha256() != source_sha256:
        print("REFUSED: Phase-2 source changed during aggregation")
        return 2
    report_receipt = {
        "schema_version": REPORT_RECEIPT_SCHEMA,
        "receipt_type": "phase2_f01_report",
        "eligibility": "diagnostic_only_never_paper_main",
        "paper_result_eligible": False,
        "git_head": pair_authority["git_head"],
        "paired_cells": len(cells),
        "expected_cells": len(LEGACY),
        "unpaired_cells": len(unpaired),
        "complete": complete,
        "input_pair_receipt": {
            "path": str(pair_receipt_path),
            "sha256": pair_receipt_sha256,
        },
        "input_receipt_sha256": pair_authority["input_receipt_sha256"],
        "analysis_marker_sha256": analysis_marker_sha256,
        "source_sha256": source_sha256,
        "reports": {
            "json": {"path": str(out_json),
                     "sha256": report_json_sha256},
            "markdown": {"path": str(out_md),
                         "sha256": report_md_sha256},
        },
    }
    if _bound_bytes(out_json, "Phase-2 JSON report final check")[1] \
            != report_json_sha256 \
            or _bound_bytes(out_md, "Phase-2 Markdown report final check")[1] \
            != report_md_sha256:
        print("REFUSED: report bytes changed before receipt publication")
        return 2
    out_receipt = requested_outputs[2]
    _publish_json_exclusive(out_receipt, report_receipt)
    try:
        reopened_receipt, _ = _load_json_bound(
            str(out_receipt), "Phase-2 final report receipt")
    except (ExtractionInvalid, OSError, ValueError) as error:
        out_receipt.unlink(missing_ok=True)
        print(f"REFUSED: final receipt could not be reopened: {error}")
        return 2
    if reopened_receipt != report_receipt:
        out_receipt.unlink(missing_ok=True)
        print("REFUSED: final report receipt changed during publication")
        return 2
    print(f"wrote {out_json} and {out_md}: "
          f"{len(cells)} paired, {len(unpaired)} unpaired")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
