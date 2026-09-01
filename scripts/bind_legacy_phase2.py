"""Materialize an exact inventory-bound side of the Phase-2 F01 diagnostic."""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, stat, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from dna_utils import extraction_validation as validation  # noqa: E402
from dna_utils import runtime_state  # noqa: E402
from dna_utils.extraction_validation import (  # noqa: E402
    ExpectedIdentity, ExtractionInvalid, validate_code_arrays,
    validate_extraction_run,
)
from dna_utils.runtime_state import (  # noqa: E402
    ResolvedEpoch, annealed_epsilon, sha256_file, write_extraction_manifest,
)

INPUT_RECEIPT_NAME = "phase2_input_receipt.json"
_SPLITS = (("db", "extract_db.npz"), ("query", "extract_query.npz"))
_PAIR_ARRAYS = ("labels", "multi_hot_labels", "image_paths")
_INVENTORIES = {
    "fixed": (REPO / "docs/phase2_extraction_inventory.json", "9d8292f21e592a7761c0ee47264c15ec27041d99ee5d59d892905f77c52125e0"),
    "legacy": (REPO / "docs/phase2_legacy_bound_inventory.json", "160ce08612076c752d231417397deeec356072dd7d43800a942237072d49bc82")}
_DATASETS = {"cifar10": (64, (4, 9, 19, 39)),
             "flickr25k": (128, (4, 9, 19)),
             "nuswide": (128, (4, 9, 19, 39)),
             "mscoco": (128, (4, 9, 19, 39))}
_EXPECTED_CELLS = frozenset(f"{dataset}_N{n}"
    for dataset, (_, values) in _DATASETS.items() for n in values)
_AXIS = {"effective_sinkhorn_epsilon", "inference_epoch",
         "inference_epoch_source"}
_STAT = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
_LIMITATIONS = ["CIFAR labels bind preserved bytes but cannot exclude a within-class permutation"]


class BindRefused(RuntimeError):
    """An input is not the hard-bound Phase-2 snapshot."""

def _sha(path: Path) -> str:
    return sha256_file(str(path))

def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise BindRefused(f"cannot establish source state: {error}") from None

def _source_state() -> dict:
    status = _git("status", "--porcelain")
    if status:
        raise BindRefused(f"source tree is dirty:\n{status}")
    paths = (Path(__file__).resolve(), Path(validation.__file__).resolve(),
             Path(runtime_state.__file__).resolve())
    return {"git_head": _git("rev-parse", "HEAD"),
            "source_sha256": {str(path.relative_to(REPO)): _sha(path)
                               for path in paths}}

def _bound_json(path: Path, what: str) -> tuple[dict, str]:
    try:
        return validation._load_json_bound(str(path), what)
    except ExtractionInvalid as error:
        raise BindRefused(str(error)) from None

def _stable_bytes(path: Path, what: str) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise BindRefused(f"{what} is not a regular file")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
    except OSError as error:
        raise BindRefused(f"cannot read {what}: {error}") from None
    finally:
        if "fd" in locals():
            os.close(fd)
    current = path.stat(follow_symlinks=False)
    if any(getattr(before, key) != getattr(after, key)
           or getattr(after, key) != getattr(current, key) for key in _STAT):
        raise BindRefused(f"{what} changed while read")
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()

def _support_link_sha(link: Path, target: Path, what: str) -> str:
    try:
        before = link.lstat()
        text = os.readlink(link)
        resolved = (link.parent / text).resolve(strict=True)
        _, digest = _stable_bytes(resolved, what)
        after = link.lstat()
    except OSError as error:
        raise BindRefused(f"cannot bind {what}: {error}") from None
    if not stat.S_ISLNK(before.st_mode) or resolved != target \
            or text != os.readlink(link) \
            or any(getattr(before, key) != getattr(after, key) for key in _STAT):
        raise BindRefused(f"{what} symlink target changed")
    return digest

def _arg(text: str, field: str) -> str:
    pattern = re.compile(rf"^{re.escape(field)}-+(.*)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    raise BindRefused(f"args.txt has no field {field!r}")

def _source(source_root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise BindRefused(f"source path is not a string: {relative!r}")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise BindRefused(f"unsafe source path: {relative!r}")
    try:
        path = (source_root / rel).resolve(strict=True)
    except OSError as error:
        raise BindRefused(f"source is absent: {relative}: {error}") from None
    if not path.is_relative_to(source_root):
        raise BindRefused(f"source escapes --source-root: {relative}")
    return path

def _array_evidence(value) -> dict:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise BindRefused("pair evidence cannot contain object arrays")
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(
                np.ascontiguousarray(array).tobytes()).hexdigest()}

def _inspect_npz(path: Path, *, rows: int, runtime: dict,
                 split: str) -> tuple[str, dict]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise BindRefused(f"{split}: NPZ is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            first = hashlib.sha256()
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                first.update(block)
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as stored:
                names = ("base_indices", "hash_2bit", "codebook_indices")
                if any(name not in stored.files for name in names):
                    raise BindRefused(f"{split}: missing canonical code array")
                base, hashed, codebook = (np.asarray(stored[name]) for name in names)
                present = [name for name in _PAIR_ARRAYS if name in stored.files]
                if not {"labels", "multi_hot_labels"}.intersection(present):
                    raise BindRefused(f"{split}: no retrieval labels")
                evidence = {name: _array_evidence(stored[name]) for name in present}
            handle.seek(0)
            second = hashlib.sha256()
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                second.update(block)
        after = os.fstat(fd)
    except (OSError, ValueError) as error:
        raise BindRefused(f"{split}: unreadable NPZ {path}: {error}") from None
    finally:
        if "fd" in locals():
            os.close(fd)
    current = path.stat(follow_symlinks=False)
    if first.digest() != second.digest() or any(
            getattr(before, key) != getattr(after, key)
            or getattr(after, key) != getattr(current, key) for key in _STAT):
        raise BindRefused(f"{split}: NPZ changed while inspected")
    try:
        validate_code_arrays(
            base, hashed, codebook, rows=rows, slots=runtime["num_slots"],
            per_slot=runtime["bases_per_slot"],
            codebook_size=runtime["codebook_size"], split=split)
    except ExtractionInvalid as error:
        raise BindRefused(str(error)) from None
    return second.hexdigest(), evidence

def _load_inventory(side: str) -> tuple[dict[str, dict], dict]:
    path, pinned = _INVENTORIES[side]
    path = Path(path).resolve()
    payload, digest = _bound_json(path, f"{side} tracked inventory")
    if digest != pinned:
        raise BindRefused(f"{side} inventory SHA {digest} != pinned {pinned}")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not all(
            isinstance(cell, dict) for cell in raw_cells):
        raise BindRefused(f"{side} inventory has no cell list")
    names = [cell.get("cell") for cell in raw_cells]
    if len(names) != len(set(names)) or set(names) != set(_EXPECTED_CELLS) \
            or payload.get("schema_version") != 2 \
            or payload.get("cells_validated") != len(_EXPECTED_CELLS) \
            or payload.get("cells_refused") != [] \
            or payload.get("eligibility") \
            != "diagnostic_only_never_promoted_to_paper_main":
        raise BindRefused(f"{side} inventory is not exact-{len(_EXPECTED_CELLS)}")
    return ({cell["cell"]: cell for cell in raw_cells},
            {"path": str(path), "sha256": digest})

def _preflight_cell(side: str, entry: dict, source_root: Path) -> dict:
    name = entry["cell"]
    try:
        dataset, raw_n = name.rsplit("_N", 1)
        n, runtime = int(raw_n), entry["runtime_identity"]
        codebook_size = _DATASETS[dataset][0]
    except (KeyError, TypeError, ValueError) as error:
        raise BindRefused(f"malformed {side} cell {name!r}: {error}") from None
    if not isinstance(runtime, dict) \
            or set(runtime) != set(validation._COMMON_IDENTITY):
        raise BindRefused(f"{side}/{name}: runtime identity is not exact-18")
    budget = n + 1
    expected = {
        "backfilled": True, "dataset": dataset, "random_seed": 42,
        "num_slots": 5, "bases_per_slot": 3, "total_bases": 15,
        "total_bits": 30, "codebook_size": codebook_size,
        "lr_schedule_horizon": budget, "training_epoch_budget": budget,
        "training_stop_epoch": n, "sinkhorn_schedule_horizon": budget,
        "sinkhorn_annealing_enabled": True,
        "inference_epoch": n if side == "fixed" else 0,
        "inference_epoch_source": (
            "explicit_flag" if side == "fixed" else "f01_unrestored")}
    if entry.get("dataset") != dataset or entry.get("stop_epoch_zero_based") != n \
            or any(runtime.get(key) != value for key, value in expected.items()):
        raise BindRefused(f"{side}/{name}: runtime identity drift")

    canonical = entry.get("canonical_legacy_source") or {}
    support = _source(source_root, canonical.get("dir"))
    checkpoint, config = (support / "model_state_dict.pth", support / "config.pt")
    checkpoint_sha, config_sha = _sha(checkpoint), _sha(config)
    if len({checkpoint_sha, canonical.get("model_state_dict_sha256"),
            entry.get("checkpoint_sha256"), runtime["checkpoint_sha256"]}) != 1 \
            or len({config_sha, canonical.get("config_sha256"),
                    entry.get("config_sha256"), runtime["config_sha256"]}) != 1:
        raise BindRefused(f"{side}/{name}: checkpoint/config SHA drift")
    splits = entry.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"db", "query"}:
        raise BindRefused(f"{side}/{name}: split set is not exact")
    paths = {split: _source(source_root, splits[split].get("npz_path_relative"))
             for split, _ in _SPLITS}
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1 or any(paths[split].name != filename
                                for split, filename in _SPLITS):
        raise BindRefused(f"{side}/{name}: noncanonical split paths")
    data_dir = next(iter(parents))
    if (side == "legacy") != (data_dir == support):
        raise BindRefused(f"{side}/{name}: data/support source relationship drift")

    args, args_sha = _stable_bytes(data_dir / "args.txt", f"{side}/{name}/args")
    if args_sha != entry.get("args_txt_sha256"):
        raise BindRefused(f"{side}/{name}: args.txt SHA drift")
    text = args.decode("utf-8")
    arg_fields = {"num_semantic_parts": 5, "num_codons_per_codebook": 3,
                  "codebook_size": codebook_size, "epoch": budget,
                  "stop_after_epoch": n, "random_seed": 42}
    try:
        if any(int(_arg(text, key)) != value for key, value in arg_fields.items()):
            raise BindRefused(f"{side}/{name}: args.txt identity drift")
        epsilon = annealed_epsilon(
            runtime["inference_epoch"], budget,
            float(_arg(text, "sinkhorn_epsilon_init")),
            float(_arg(text, "sinkhorn_epsilon_final")))
    except (UnicodeDecodeError, TypeError, ValueError) as error:
        raise BindRefused(f"{side}/{name}: malformed args.txt: {error}") from None
    if not math.isclose(epsilon, runtime["effective_sinkhorn_epsilon"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise BindRefused(f"{side}/{name}: effective epsilon drift")
    checked, pair = {}, {}
    for split, _ in _SPLITS:
        rows = splits[split].get("n_rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise BindRefused(f"{side}/{name}/{split}: invalid n_rows")
        digest, evidence = _inspect_npz(
            paths[split], rows=rows, runtime=runtime,
            split=f"{side}/{name}/{split}")
        if digest != splits[split].get("npz_sha256"):
            raise BindRefused(f"{side}/{name}/{split}: NPZ SHA drift")
        checked[split] = {"path": paths[split], "rows": rows, "sha256": digest}
        pair[split] = evidence
    log = None
    log_sha = None
    if side == "fixed":
        log, log_sha = _stable_bytes(
            data_dir / "extract.log", f"fixed/{name}/extract.log")
    if (side == "legacy" and entry.get("extract_log_sha256") is not None) \
            or (log is not None and log_sha != entry.get("extract_log_sha256")):
        raise BindRefused(f"{side}/{name}: extract.log authority drift")
    return {"entry": entry, "runtime": runtime, "support": support,
            "args": args, "log": log, "splits": checked, "pair": pair}

def _preflight(source_root: Path) -> tuple[dict, dict, dict]:
    references, plans = {}, {}
    for side in ("fixed", "legacy"):
        entries, references[side] = _load_inventory(side)
        plans[side] = {name: _preflight_cell(side, entries[name], source_root)
                       for name in sorted(_EXPECTED_CELLS)}
    pair = {}
    for name in sorted(_EXPECTED_CELLS):
        fixed, legacy = plans["fixed"][name], plans["legacy"][name]
        if any(fixed["runtime"][field] != legacy["runtime"][field]
               for field in set(validation._COMMON_IDENTITY) - _AXIS):
            raise BindRefused(f"{name}: fixed/legacy identity differs")
        pair[name] = {}
        for split, _ in _SPLITS:
            if fixed["splits"][split]["rows"] != legacy["splits"][split]["rows"] \
                    or fixed["pair"][split] != legacy["pair"][split]:
                raise BindRefused(f"{name}/{split}: sample/order evidence differs")
            pair[name][split] = fixed["pair"][split]
    return references, plans, pair

def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def _publish_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    created = published = False
    try:
        _write_new(tmp, data)
        created = True
        os.link(tmp, path, follow_symlinks=False)
        published = True
        tmp.unlink()
        created = False
        _fsync_dir(path.parent)
    except FileExistsError:
        raise BindRefused(f"refusing to overwrite {path}") from None
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
            try:
                _fsync_dir(path.parent)
            except OSError:
                pass
        raise
    finally:
        if created:
            tmp.unlink(missing_ok=True)

def _json_new(path: Path, payload: dict) -> None:
    _publish_bytes(path, (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())

def _copy_npz(source: Path, destination: Path, *, plan: dict,
              split: str) -> str:
    info, runtime = plan["splits"][split], plan["runtime"]
    tmp = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    digest = hashlib.sha256()
    created = False
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            os.close(fd)
            raise BindRefused(f"source NPZ is not regular: {source}")
        try:
            writer_fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except BaseException:
            os.close(fd)
            raise
        created = True
        with os.fdopen(fd, "rb") as reader, os.fdopen(writer_fd, "wb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
        current = source.stat(follow_symlinks=False)
        if any(getattr(before, key) != getattr(after, key)
               or getattr(after, key) != getattr(current, key) for key in _STAT) \
                or digest.hexdigest() != info["sha256"]:
            raise BindRefused(f"source NPZ changed while copied: {source}")
        tmp_sha, tmp_pair = _inspect_npz(
            tmp, rows=info["rows"], runtime=runtime, split=split)
        if tmp_sha != info["sha256"] or tmp_pair != plan["pair"][split]:
            raise BindRefused(f"temporary NPZ recheck failed: {tmp}")
        os.link(tmp, destination, follow_symlinks=False)
        tmp.unlink()
        created = False
        _fsync_dir(destination.parent)
        final_sha, final_pair = _inspect_npz(
            destination, rows=info["rows"], runtime=runtime, split=split)
        final = destination.stat(follow_symlinks=False)
        if final_sha != info["sha256"] or final_pair != plan["pair"][split] \
                or not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
            raise BindRefused(f"published NPZ recheck failed: {destination}")
        return final_sha
    except FileExistsError:
        raise BindRefused(f"refusing to overwrite {destination}") from None
    finally:
        if created:
            tmp.unlink(missing_ok=True)

def _expected(entry: dict) -> ExpectedIdentity:
    runtime, splits = entry["runtime_identity"], entry["splits"]
    return ExpectedIdentity(
        dataset=runtime["dataset"], random_seed=runtime["random_seed"],
        inference_epoch=runtime["inference_epoch"],
        inference_epoch_source=runtime["inference_epoch_source"],
        num_slots=runtime["num_slots"], bases_per_slot=runtime["bases_per_slot"],
        codebook_size=runtime["codebook_size"],
        checkpoint_sha256=runtime["checkpoint_sha256"],
        config_sha256=runtime["config_sha256"],
        n_rows={split: splits[split]["n_rows"] for split, _ in _SPLITS})

def _materialize_cell(cell: Path, plan: dict, side: str) -> dict:
    cell.mkdir(exist_ok=False)
    runtime = plan["runtime"]
    npz_sha = {split: _copy_npz(plan["splits"][split]["path"], cell / filename,
                                plan=plan, split=split)
               for split, filename in _SPLITS}
    for filename in ("model_state_dict.pth", "config.pt"):
        (cell / filename).symlink_to(plan["support"] / filename)
    _write_new(cell / "args.txt", plan["args"])
    if plan["log"] is not None:
        _write_new(cell / "extract.log", plan["log"])
    resolved = ResolvedEpoch(
        epoch=runtime["inference_epoch"], source=runtime["inference_epoch_source"],
        effective_sinkhorn_epsilon=runtime["effective_sinkhorn_epsilon"],
        sinkhorn_schedule_horizon=runtime["sinkhorn_schedule_horizon"],
        checkpoint_sha256=runtime["checkpoint_sha256"],
        sinkhorn_annealing_enabled=runtime["sinkhorn_annealing_enabled"])
    for split, filename in _SPLITS:
        write_extraction_manifest(
            str(cell / f"extraction_manifest_{split}.json"),
            checkpoint_path=str(cell / "model_state_dict.pth"), resolved=resolved,
            num_slots=runtime["num_slots"], bases_per_slot=runtime["bases_per_slot"],
            split=split, n_rows=plan["splits"][split]["rows"],
            lr_schedule_horizon=runtime["lr_schedule_horizon"],
            training_epoch_budget=runtime["training_epoch_budget"],
            training_stop_epoch=runtime["training_stop_epoch"], extra={
                "npz_path": str(cell / filename), "npz_sha256": npz_sha[split],
                "config_path": str(cell / "config.pt"),
                "config_sha256": runtime["config_sha256"],
                "dataset": runtime["dataset"], "random_seed": 42,
                "codebook_size": runtime["codebook_size"], "backfilled": True,
                "backfill_reason": "inventory-pinned Phase-2 F01 diagnostic",
                "bind_tool_sha256": _sha(Path(__file__))})
    completion = cell / "extraction_complete.json"
    _json_new(completion, {
        "schema_version": 1, "splits": ["db", "query"],
        "manifest_sha256": {
            split: _sha(cell / f"extraction_manifest_{split}.json")
            for split, _ in _SPLITS}, "backfilled": True})
    try:
        run = validate_extraction_run(
            str(cell), allow_backfilled=True, expected=_expected(plan["entry"]))
        if run.common != runtime:
            raise BindRefused(
                f"{side}/{cell.name}: published full identity differs")
    except BaseException:
        completion.unlink(missing_ok=True)
        raise
    return {"runtime_identity": runtime,
            "completion_marker_sha256": run.completion_marker_sha256,
            "npz_sha256": npz_sha}

def materialize(source_root: Path, out_root: Path, side: str) -> tuple[dict, str]:
    if side not in _INVENTORIES:
        raise BindRefused(f"unknown side {side!r}")
    source_root, out_root = Path(source_root).resolve(strict=True), Path(out_root).absolute()
    if not source_root.is_dir() or out_root.exists() or out_root.is_symlink():
        raise BindRefused("source must exist and --out-root must be absent")
    start_state = _source_state()
    references, plans, pair = _preflight(source_root)
    out_root.mkdir(parents=True, exist_ok=False)
    try:
        cells = {name: _materialize_cell(
            out_root / name, plans[side][name], side)
            for name in sorted(_EXPECTED_CELLS)}
        if {path.name for path in out_root.iterdir()} != set(_EXPECTED_CELLS):
            raise BindRefused("output membership changed before receipt")
        end_state = _source_state()
        if end_state != start_state:
            raise BindRefused("source HEAD/digests changed during materialization")
    except BaseException:
        for marker in out_root.glob("*/extraction_complete.json"):
            marker.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": 1, "artifact_kind": "phase2_inventory_bound_inputs",
        "eligibility": "diagnostic_only_never_promoted_to_paper_main",
        "side": side, "output_root": str(out_root),
        "source_root": str(source_root), "inventories": references,
        "validator_sha256": _sha(Path(validation.__file__).resolve()),
        "bind_tool_sha256": _sha(Path(__file__)), "cells": cells,
        "pair_evidence": pair, "limitations": _LIMITATIONS, **start_state}
    receipt_path = out_root / INPUT_RECEIPT_NAME
    _json_new(receipt_path, receipt)
    try:
        return verify_input_root(out_root, side)
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        raise

def verify_input_root(root: Path, side: str) -> tuple[dict, str]:
    """Return the strictly reopened receipt payload and its exact SHA."""
    raw_root = Path(root)
    if side not in _INVENTORIES or raw_root.is_symlink():
        raise BindRefused("unknown side or symlink input root")
    root = raw_root.resolve(strict=True)
    receipt, receipt_sha = _bound_json(
        root / INPUT_RECEIPT_NAME, "Phase-2 input receipt")
    references, entries = {}, {}
    for candidate in ("fixed", "legacy"):
        entries[candidate], references[candidate] = _load_inventory(candidate)
    closure = (receipt.get("inventories") == references
               and receipt.get("validator_sha256")
               == _sha(Path(validation.__file__).resolve())
               and receipt.get("bind_tool_sha256") == _sha(Path(__file__))
               and {key: receipt.get(key) for key in ("git_head", "source_sha256")}
               == _source_state())
    if set(receipt) != {"schema_version", "artifact_kind", "eligibility", "side", "output_root", "source_root", "inventories", "validator_sha256", "bind_tool_sha256", "cells", "pair_evidence", "limitations", "git_head", "source_sha256"} \
            or receipt.get("schema_version") != 1 \
            or receipt.get("artifact_kind") != "phase2_inventory_bound_inputs" \
            or receipt.get("eligibility") \
            != "diagnostic_only_never_promoted_to_paper_main" \
            or receipt.get("side") != side or receipt.get("output_root") != str(root) \
            or receipt.get("limitations") != _LIMITATIONS or not closure:
        raise BindRefused(f"{root}: receipt envelope/source closure is invalid")
    if {path.name for path in root.iterdir()} \
            != set(_EXPECTED_CELLS) | {INPUT_RECEIPT_NAME} \
            or not isinstance(receipt.get("cells"), dict) \
            or set(receipt["cells"]) != set(_EXPECTED_CELLS) \
            or not isinstance(receipt.get("pair_evidence"), dict) \
            or set(receipt["pair_evidence"]) != set(_EXPECTED_CELLS):
        raise BindRefused(f"{root}: input membership is not exact")
    for name in sorted(_EXPECTED_CELLS):
        cell, entry = root / name, entries[side][name]
        if cell.is_symlink() or not cell.is_dir():
            raise BindRefused(f"{cell}: not a real directory")
        run = validate_extraction_run(
            str(cell), allow_backfilled=True, expected=_expected(entry))
        _, args_sha = _stable_bytes(
            cell / "args.txt", f"{side}/{name}/args.txt")
        if args_sha != entry.get("args_txt_sha256"):
            raise BindRefused(f"{side}/{name}: args.txt changed")
        log = cell / "extract.log"
        log_sha = (_stable_bytes(log, f"fixed/{name}/extract.log")[1]
                   if side == "fixed" else None)
        if (side == "fixed" and log_sha != entry.get("extract_log_sha256")) \
                or (side == "legacy" and (log.exists() or log.is_symlink())):
            raise BindRefused(f"{side}/{name}: extract.log changed")
        support = _source(
            Path(receipt["source_root"]),
            (entry.get("canonical_legacy_source") or {}).get("dir"))
        for filename, digest_key in (
                ("model_state_dict.pth", "checkpoint_sha256"),
                ("config.pt", "config_sha256")):
            linked = cell / filename
            target = support / filename
            if _support_link_sha(
                    linked, target, f"{side}/{name}/{filename}") \
                    != entry[digest_key]:
                raise BindRefused(f"{side}/{name}: {filename} target changed")
        actual_npz, actual_pair = {}, {}
        for split, filename in _SPLITS:
            path = cell / filename
            digest, evidence = _inspect_npz(
                path, rows=entry["splits"][split]["n_rows"],
                runtime=entry["runtime_identity"], split=f"{side}/{name}/{split}")
            state = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISREG(state.st_mode) \
                    or state.st_nlink != 1 \
                    or digest != entry["splits"][split]["npz_sha256"]:
                raise BindRefused(f"{side}/{name}/{split}: output NPZ changed")
            actual_npz[split], actual_pair[split] = digest, evidence
        expected_cell = {"runtime_identity": entry["runtime_identity"],
                         "completion_marker_sha256": run.completion_marker_sha256,
                         "npz_sha256": actual_npz}
        if run.common != entry["runtime_identity"] \
                or receipt["cells"][name] != expected_cell \
                or receipt["pair_evidence"][name] != actual_pair:
            raise BindRefused(f"{side}/{name}: receipt evidence differs")
    reopened, reopened_sha = _bound_json(
        root / INPUT_RECEIPT_NAME, "Phase-2 input receipt final reopen")
    if reopened != receipt or reopened_sha != receipt_sha:
        raise BindRefused(f"{root}: input receipt changed during verification")
    return receipt, receipt_sha

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=sorted(_INVENTORIES))
    parser.add_argument("--source-root")
    parser.add_argument("--out-root")
    parser.add_argument("--verify-root")
    args = parser.parse_args()
    try:
        if args.verify_root:
            if args.source_root or args.out_root:
                raise BindRefused("--verify-root is exclusive")
            _, digest = verify_input_root(Path(args.verify_root), args.side)
            print(f"VERIFIED {args.side} input receipt {digest}")
        else:
            if not args.source_root or not args.out_root:
                raise BindRefused("--source-root and --out-root are required")
            payload, digest = materialize(
                Path(args.source_root), Path(args.out_root), args.side)
            print(f"MATERIALIZED {len(payload['cells'])} {args.side} cells; {digest}")
        return 0
    except (BindRefused, ExtractionInvalid, OSError, KeyError, TypeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
