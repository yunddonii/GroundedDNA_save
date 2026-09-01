"""Focused contract tests for the Phase-2 legacy binding transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import validate_extraction_run  # noqa: E402
from dna_utils.runtime_state import sha256_file  # noqa: E402
from scripts import bind_legacy_phase2 as binder  # noqa: E402


_KEY = ("cifar10", 4)
_RELATIVE_RUN = Path("result") / "preserved_cifar10_N4"
_SOURCE_FILES = (
    "extract_db.npz", "extract_query.npz", "model_state_dict.pth",
    "config.pt", "args.txt",
)
_STABLE_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size",
                       "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _two_bit(base: np.ndarray) -> np.ndarray:
    return np.stack((base // 2, base % 2), axis=-1).reshape(base.shape[0], -1)


def _write_npz(path: Path, rows: int) -> None:
    base = (np.arange(rows * 15, dtype=np.int64).reshape(rows, 15) % 4)
    codebook = np.tile(np.arange(5, dtype=np.int64), (rows, 1))
    np.savez(
        path,
        base_indices=base,
        hash_2bit=_two_bit(base),
        codebook_indices=codebook,
        labels=np.arange(rows, dtype=np.int64) % 2,
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source_root = tmp_path / "preserved"
    run = source_root / _RELATIVE_RUN
    run.mkdir(parents=True)
    _write_npz(run / "extract_db.npz", rows=4)
    _write_npz(run / "extract_query.npz", rows=2)
    (run / "model_state_dict.pth").write_bytes(b"preserved-checkpoint\n")
    (run / "config.pt").write_bytes(b"preserved-config\n")
    (run / "args.txt").write_text(
        "num_semantic_parts----------------5\n"
        "num_codons_per_codebook-----------3\n"
        "codebook_size----------------------64\n"
        "epoch------------------------------5\n"
        "stop_after_epoch-------------------4\n"
        "sinkhorn_epsilon_init--------------1.0\n"
        "sinkhorn_epsilon_final-------------0.1\n"
        "random_seed------------------------42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(binder, "LEGACY", {_KEY: str(_RELATIVE_RUN)})
    return source_root, run


def _stable_stat(path: Path) -> tuple[int, ...]:
    value = path.stat(follow_symlinks=False)
    return tuple(getattr(value, field) for field in _STABLE_STAT_FIELDS)


def test_cli_binds_into_fresh_out_root_with_independent_canonical_npz(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, source = _fixture(tmp_path, monkeypatch)
    out_root = tmp_path / "new-diagnostic" / "legacy"
    before = {name: _stable_stat(source / name) for name in _SOURCE_FILES}

    monkeypatch.setattr(
        sys, "argv",
        [str(Path(binder.__file__)), "--source-root", str(source_root),
         "--out-root", str(out_root)],
    )
    assert binder.main() == 0

    cell = out_root / "cifar10_N4"
    run = validate_extraction_run(str(cell), allow_backfilled=True)
    assert run.backfilled is True
    assert run.common["inference_epoch"] == 0
    assert run.common["inference_epoch_source"] == "f01_unrestored"

    for split in ("db", "query"):
        copied = cell / f"extract_{split}.npz"
        original = source / f"extract_{split}.npz"
        copied_stat = copied.stat(follow_symlinks=False)
        original_stat = original.stat(follow_symlinks=False)
        assert stat.S_ISREG(copied_stat.st_mode)
        assert not copied.is_symlink()
        assert copied_stat.st_nlink == 1
        assert (copied_stat.st_dev, copied_stat.st_ino) != (
            original_stat.st_dev, original_stat.st_ino)
        assert sha256_file(str(copied)) == sha256_file(str(original))

        manifest = json.loads(
            (cell / f"extraction_manifest_{split}.json").read_text())
        assert manifest["npz_path"] == str(copied.resolve())
        assert Path(manifest["npz_path"]).parent == cell.resolve()

    # Current validation permits these two read-only identity aliases. They do
    # not alter the historical inode the way a hard link would.
    assert (cell / "model_state_dict.pth").is_symlink()
    assert (cell / "config.pt").is_symlink()
    assert before == {
        name: _stable_stat(source / name) for name in _SOURCE_FILES}
    assert not (source_root / "result_diagnostic").exists()
    assert not list(cell.glob(".*.tmp"))


def test_existing_out_root_and_cell_are_never_overwritten(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, source = _fixture(tmp_path, monkeypatch)
    out_root = tmp_path / "already-there"
    out_root.mkdir()
    sentinel = out_root / "sentinel"
    sentinel.write_bytes(b"keep-me")
    before = {name: _stable_stat(source / name) for name in _SOURCE_FILES}

    monkeypatch.setattr(
        sys, "argv",
        [str(Path(binder.__file__)), "--source-root", str(source_root),
         "--out-root", str(out_root)],
    )
    assert binder.main() == 2
    assert sentinel.read_bytes() == b"keep-me"
    assert not (out_root / "cifar10_N4").exists()

    direct_root = tmp_path / "direct"
    occupied = direct_root / "cifar10_N4"
    occupied.mkdir(parents=True)
    existing = occupied / "extract_db.npz"
    existing.write_bytes(b"do-not-replace")
    with pytest.raises(binder.BindRefused, match="refusing to overwrite"):
        binder.bind_cell(
            *_KEY, source_root=source_root, out_root=direct_root)
    assert existing.read_bytes() == b"do-not-replace"
    assert not (occupied / "extraction_complete.json").exists()
    assert before == {
        name: _stable_stat(source / name) for name in _SOURCE_FILES}
