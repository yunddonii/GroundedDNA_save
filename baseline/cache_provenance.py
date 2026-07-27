"""Content binding for the large immutable feature-cache inputs.

The feature arrays are large enough that hashing them in every short-lived
pipeline process is material.  ``memoized_sha256_file`` therefore maintains a
best-effort, stat-attested, cross-process memo file.  Set
``GROUNDEDDNA_SHA256_MEMO_PATH`` to override its default UID/repository-scoped
location under ``/tmp``.

The memo is only a performance cache: malformed, unsafe, or unwritable memo
state is ignored and the artifact is hashed normally.  Correctness never
depends on the memo being available.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable, Mapping

try:  # Linux is the supported training platform; retain a safe fallback.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    fcntl = None  # type: ignore[assignment]


_HASH_CACHE: dict[tuple, str] = {}
_MEMO_ENV = 'GROUNDEDDNA_SHA256_MEMO_PATH'
_MEMO_SCHEMA_VERSION = 1
_MAX_MEMO_BYTES = 16 * 1024 * 1024
_MAX_MEMO_ENTRIES = 4096


class _PersistentMemoUnavailable(RuntimeError):
    """The optional persistent memo cannot be used safely."""


def persistent_sha256_memo_path(
        configured: str | Path | None = None) -> Path:
    """Return the configured UID/repository-scoped persistent memo path."""
    if configured is not None:
        # Keep the final component unresolved so O_NOFOLLOW can reject a memo
        # symlink instead of silently following it.
        return Path(os.path.abspath(
            os.fspath(Path(configured).expanduser())))
    from_environment = os.environ.get(_MEMO_ENV)
    if from_environment:
        return Path(os.path.abspath(
            os.fspath(Path(from_environment).expanduser())))
    repository = Path(__file__).resolve().parents[1]
    repository_token = hashlib.sha256(
        os.fsencode(str(repository))).hexdigest()[:16]
    uid = os.getuid() if hasattr(os, 'getuid') else 'nouid'
    return Path('/tmp') / (
        f'groundeddna-sha256-memo-{uid}-{repository_token}.json')


def consumed_cache_artifact_names(
        *, paired_aug: bool, visual_tokens: bool) -> tuple[str, ...]:
    names = ['visual_global.f16.npy']
    if paired_aug:
        names += ['visual_global_aug0.f16.npy', 'visual_global_aug1.f16.npy']
    if visual_tokens:
        names.append('visual_tokens.f16.npy')
        if paired_aug:
            names += [
                'visual_tokens_aug0.f16.npy',
                'visual_tokens_aug1.f16.npy',
            ]
    return tuple(names)


def _stat_identity(path: Path) -> tuple:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return (
        str(resolved), int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
        int(stat.st_mtime_ns), int(stat.st_ctime_ns),
    )


def _identity_key(identity: tuple) -> str:
    return json.dumps(identity, ensure_ascii=True, separators=(',', ':'))


def _entries_sha256(entries: dict) -> str:
    encoded = json.dumps(
        entries, ensure_ascii=True, sort_keys=True,
        separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _remember_hash(identity: tuple, value: str) -> None:
    for stale in [
            key for key in _HASH_CACHE
            if key[0] == identity[0] and key != identity]:
        del _HASH_CACHE[stale]
    _HASH_CACHE[identity] = value


def _secure_regular_file(fd: int, path: Path) -> None:
    metadata = os.fstat(fd)
    if not stat_module.S_ISREG(metadata.st_mode):
        raise _PersistentMemoUnavailable(
            f'persistent SHA-256 memo is not a regular file: {path}')
    if hasattr(os, 'getuid') and metadata.st_uid != os.getuid():
        raise _PersistentMemoUnavailable(
            f'persistent SHA-256 memo has a different owner: {path}')
    if metadata.st_mode & 0o022:
        raise _PersistentMemoUnavailable(
            f'persistent SHA-256 memo is group/world writable: {path}')


@contextmanager
def _memo_lock(memo_path: Path) -> Iterator[None]:
    """Take an exclusive advisory lock without following a lock symlink."""
    if fcntl is None:
        raise _PersistentMemoUnavailable(
            'interprocess file locking is unavailable')
    try:
        memo_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = Path(f'{memo_path}.lock')
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, 'O_CLOEXEC', 0)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            _secure_regular_file(fd, lock_path)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
    except _PersistentMemoUnavailable:
        raise
    except OSError as error:
        raise _PersistentMemoUnavailable(
            f'cannot lock persistent SHA-256 memo {memo_path}: {error}') \
            from error
    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _empty_memo() -> dict:
    entries: dict = {}
    return {
        'schema_version': _MEMO_SCHEMA_VERSION,
        'entries_sha256': _entries_sha256(entries),
        'entries': entries,
    }


def _read_memo(memo_path: Path) -> dict:
    flags = os.O_RDONLY
    flags |= getattr(os, 'O_CLOEXEC', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(memo_path, flags)
    except FileNotFoundError:
        return _empty_memo()
    except OSError as error:
        raise _PersistentMemoUnavailable(
            f'cannot read persistent SHA-256 memo {memo_path}: {error}') \
            from error
    try:
        _secure_regular_file(fd, memo_path)
        metadata = os.fstat(fd)
        if metadata.st_size > _MAX_MEMO_BYTES:
            return _empty_memo()
        with os.fdopen(fd, 'r', encoding='utf-8') as handle:
            fd = -1
            payload = json.load(handle)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return _empty_memo()
    except OSError as error:
        raise _PersistentMemoUnavailable(
            f'cannot read persistent SHA-256 memo {memo_path}: {error}') \
            from error
    finally:
        if fd >= 0:
            os.close(fd)

    if (not isinstance(payload, dict)
            or payload.get('schema_version') != _MEMO_SCHEMA_VERSION
            or not isinstance(payload.get('entries'), dict)):
        return _empty_memo()
    entries = payload['entries']
    if (len(entries) > _MAX_MEMO_ENTRIES
            or payload.get('entries_sha256') != _entries_sha256(entries)):
        return _empty_memo()
    valid: dict[str, dict] = {}
    for key, record in entries.items():
        if (not isinstance(key, str) or not isinstance(record, dict)
                or not isinstance(record.get('identity'), list)
                or len(record['identity']) != 6
                or _identity_key(tuple(record['identity'])) != key
                or not _is_sha256(record.get('sha256'))):
            return _empty_memo()
        valid[key] = {
            'identity': record['identity'],
            'sha256': record['sha256'],
        }
    return {
        'schema_version': _MEMO_SCHEMA_VERSION,
        'entries_sha256': _entries_sha256(valid),
        'entries': valid,
    }


def _atomic_write_memo(memo_path: Path, payload: dict) -> None:
    temporary_fd = -1
    temporary_path: str | None = None
    try:
        payload['entries_sha256'] = _entries_sha256(payload['entries'])
        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix=f'.{memo_path.name}.', suffix='.tmp',
            dir=str(memo_path.parent))
        os.fchmod(temporary_fd, 0o600)
        with os.fdopen(temporary_fd, 'w', encoding='utf-8') as handle:
            temporary_fd = -1
            json.dump(
                payload, handle, ensure_ascii=True, sort_keys=True,
                separators=(',', ':'))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, memo_path)
        temporary_path = None
        try:
            directory_fd = os.open(
                memo_path.parent,
                os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file itself is already complete and atomically installed.
            pass
    except OSError as error:
        raise _PersistentMemoUnavailable(
            f'cannot update persistent SHA-256 memo {memo_path}: {error}') \
            from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _hash_stable_file(path: Path, before: tuple | None = None) -> str:
    """Hash ``path`` while attesting both its descriptor and pathname."""
    before = _stat_identity(path) if before is None else before
    digest = hashlib.sha256()
    resolved = Path(before[0])
    with resolved.open('rb') as handle:
        descriptor_before = os.fstat(handle.fileno())
        descriptor_identity = (
            str(resolved), int(descriptor_before.st_dev),
            int(descriptor_before.st_ino), int(descriptor_before.st_size),
            int(descriptor_before.st_mtime_ns),
            int(descriptor_before.st_ctime_ns),
        )
        if descriptor_identity != before:
            raise RuntimeError(
                f'cache artifact changed before hashing: {path}')
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        descriptor_after = os.fstat(handle.fileno())
        descriptor_after_identity = (
            str(resolved), int(descriptor_after.st_dev),
            int(descriptor_after.st_ino), int(descriptor_after.st_size),
            int(descriptor_after.st_mtime_ns),
            int(descriptor_after.st_ctime_ns),
        )
    after = _stat_identity(path)
    if descriptor_after_identity != before or after != before:
        raise RuntimeError(f'cache artifact changed while hashing: {path}')
    return digest.hexdigest()


def _memoized_under_lock(path: Path, memo_path: Path) -> str:
    before = _stat_identity(path)
    cached = _HASH_CACHE.get(before)
    if cached is not None:
        if _stat_identity(path) != before:
            raise RuntimeError(
                f'cache artifact changed during memo lookup: {path}')
        return cached

    try:
        memo = _read_memo(memo_path)
    except _PersistentMemoUnavailable:
        value = _hash_stable_file(path, before)
        _remember_hash(before, value)
        return value

    key = _identity_key(before)
    record = memo['entries'].get(key)
    if record is not None and tuple(record['identity']) == before:
        if _stat_identity(path) != before:
            raise RuntimeError(
                f'cache artifact changed during persistent memo lookup: {path}')
        value = record['sha256']
        _remember_hash(before, value)
        return value

    value = _hash_stable_file(path, before)
    # A stat change invalidates all older records for this resolved pathname.
    entries = {
        existing_key: existing_record
        for existing_key, existing_record in memo['entries'].items()
        if existing_record['identity'][0] != before[0]
    }
    if len(entries) >= _MAX_MEMO_ENTRIES:
        # Deterministic bounded eviction; cache misses remain correctness-safe.
        entries.pop(sorted(entries)[0])
    entries[key] = {'identity': list(before), 'sha256': value}
    memo['entries'] = entries
    try:
        _atomic_write_memo(memo_path, memo)
    except _PersistentMemoUnavailable:
        pass
    _remember_hash(before, value)
    return value


def memoized_sha256_file(
        path: str | Path, *, memo_path: str | Path | None = None) -> str:
    """Hash once per stable stat identity, across processes when possible."""
    path = Path(path)
    before = _stat_identity(path)
    cached = _HASH_CACHE.get(before)
    if cached is not None:
        if _stat_identity(path) != before:
            raise RuntimeError(
                f'cache artifact changed during memo lookup: {path}')
        return cached
    persistent_path = persistent_sha256_memo_path(memo_path)
    try:
        with _memo_lock(persistent_path):
            return _memoized_under_lock(path, persistent_path)
    except _PersistentMemoUnavailable:
        before = _stat_identity(path)
        value = _hash_stable_file(path, before)
        _remember_hash(before, value)
        return value


def hash_cache_artifacts(
        cache_dir: str | Path, names: Iterable[str]) -> dict[str, str]:
    cache_dir = Path(cache_dir).resolve()
    result: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f'invalid cache artifact basename: {name!r}')
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'required cache artifact is missing: {path}')
        result[name] = memoized_sha256_file(path)
    if not result:
        raise ValueError('no consumed cache artifacts were specified')
    return dict(sorted(result.items()))


def verify_cache_artifact_hashes(
        cache_dir: str | Path, expected: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError('cache artifact SHA-256 mapping is missing or empty')
    normalized: dict[str, str] = {}
    for name, digest in expected.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(digest, str) or len(digest) != 64):
            raise ValueError(f'invalid cache artifact hash entry: {name!r}')
        normalized[name] = digest
    actual = hash_cache_artifacts(cache_dir, sorted(normalized))
    mismatches = {
        name: (normalized[name], actual[name])
        for name in sorted(normalized)
        if normalized[name] != actual[name]
    }
    if mismatches:
        raise ValueError(
            f'checkpoint/cache consumed-array SHA-256 mismatch: {mismatches}')
    return actual
