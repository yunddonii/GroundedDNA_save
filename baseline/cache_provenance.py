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
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable, Mapping, Sequence

try:  # Linux is the supported training platform; retain a safe fallback.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    fcntl = None  # type: ignore[assignment]


_HASH_CACHE: dict[tuple, str] = {}
_MEMO_ENV = 'GROUNDEDDNA_SHA256_MEMO_PATH'
_MEMO_SCHEMA_VERSION = 1
_MAX_MEMO_BYTES = 16 * 1024 * 1024
_MAX_MEMO_ENTRIES = 4096

DECODE_FAILURE_AUDIT_SCHEMA = 'groundeddna.baseline-cache-decode-audit'
DECODE_FAILURE_AUDIT_SCHEMA_VERSION = 1
_ROW_VECTOR_ENCODING = 'sha256-int64-le-c-order-v1'
_FAILED_IMAGE_ID_VECTOR_ENCODING = (
    'groundeddna.decode-failed-image-id-vector-v1')
_DATASET_DIRECTORIES = {
    'Flickr25k': 'Flickr25k',
    'MSCOCO': 'MSCOCO',
    'NUSWIDE': 'NUSWIDE',
    'CIFAR10': 'CIFAR10',
}

# The only admitted nonzero decode-failure set in the paper cache.  These rows
# are absent from designated train and query and occur only in the retrieval
# database.  Binding both cache-row coordinates and image IDs prevents a cache
# reorder or a rewritten ``failed_image_indices`` list from preserving a mere
# count of sixteen while changing which examples are zero-feature rows.
MSCOCO_EXPECTED_FAILED_IMAGE_INDICES = (
    19898, 21672, 22022, 27072, 51770, 55405, 58687, 87701,
    88554, 92202, 95533, 98296, 101470, 107178, 112104, 121600,
)
MSCOCO_EXPECTED_FAILED_IMAGE_IDS = (
    'images/val2014/COCO_val2014_000000502557.jpg',
    'images/val2014/COCO_val2014_000000502671.jpg',
    'images/val2014/COCO_val2014_000000502599.jpg',
    'images/val2014/COCO_val2014_000000502732.jpg',
    'images/val2014/COCO_val2014_000000502749.jpg',
    'images/val2014/COCO_val2014_000000502570.jpg',
    'images/val2014/COCO_val2014_000000502715.jpg',
    'images/val2014/COCO_val2014_000000502558.jpg',
    'images/val2014/COCO_val2014_000000502630.jpg',
    'images/val2014/COCO_val2014_000000502582.jpg',
    'images/val2014/COCO_val2014_000000502698.jpg',
    'images/val2014/COCO_val2014_000000502663.jpg',
    'images/val2014/COCO_val2014_000000502456.jpg',
    'images/val2014/COCO_val2014_000000502510.jpg',
    'images/val2014/COCO_val2014_000000502737.jpg',
    'images/val2014/COCO_val2014_000000502644.jpg',
)
MSCOCO_EXPECTED_FAILED_INDICES_SHA256 = (
    'e0ae47a665ba4d39e978ca364e9c4a02f57142752dfa48e28cef519f2311eb98')
MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256 = (
    'ba88a2efcdae52f5bda789bd9796c7970b3305d1e6250447f7a852cacb7a178e')


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


def _read_stable_json(path: Path, *, description: str) -> tuple[object, str]:
    """Read JSON while proving that its pathname and inode did not change."""
    before = _stat_identity(path)
    resolved = Path(before[0])
    try:
        with resolved.open('rb') as handle:
            descriptor_before = os.fstat(handle.fileno())
            descriptor_identity = (
                str(resolved), int(descriptor_before.st_dev),
                int(descriptor_before.st_ino), int(descriptor_before.st_size),
                int(descriptor_before.st_mtime_ns),
                int(descriptor_before.st_ctime_ns),
            )
            if descriptor_identity != before:
                raise ValueError(f'{description} changed before reading: {path}')
            raw = handle.read()
            descriptor_after = os.fstat(handle.fileno())
            after_descriptor_identity = (
                str(resolved), int(descriptor_after.st_dev),
                int(descriptor_after.st_ino), int(descriptor_after.st_size),
                int(descriptor_after.st_mtime_ns),
                int(descriptor_after.st_ctime_ns),
            )
    except OSError as error:
        raise ValueError(f'cannot read {description} {path}: {error}') from error
    if after_descriptor_identity != before or _stat_identity(path) != before:
        raise ValueError(f'{description} changed while reading: {path}')
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'{description} is not valid UTF-8 JSON: {path}') from error
    return payload, hashlib.sha256(raw).hexdigest()


def _row_vector_sha256(rows: Sequence[int]) -> str:
    values = tuple(int(value) for value in rows)
    digest = hashlib.sha256()
    digest.update(_ROW_VECTOR_ENCODING.encode('ascii'))
    digest.update(b'\0')
    digest.update(len(values).to_bytes(8, 'little', signed=False))
    if values:
        digest.update(struct.pack('<' + 'q' * len(values), *values))
    return digest.hexdigest()


def _image_id_vector_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_FAILED_IMAGE_ID_VECTOR_ENCODING.encode('ascii'))
    digest.update(b'\0')
    digest.update(len(values).to_bytes(8, 'little', signed=False))
    for value in values:
        encoded = value.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little', signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _official_split_image_ids(
        dataset_root: Path, *, dataset: str, setting: str,
        ) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Reopen the exact non-CIFAR split rows used by the baseline loader."""
    if setting != 'setting1':
        raise ValueError(
            f'decode-failure audit supports only setting1, got {setting!r}')
    dataset_directory = dataset_root / _DATASET_DIRECTORIES[dataset]
    memberships: dict[str, set[str]] = {}
    source_sha256: dict[str, str] = {}
    for source_name, semantic in (
            ('train', 'train'), ('test', 'query'), ('database', 'database')):
        relative_source = (
            f'{_DATASET_DIRECTORIES[dataset]}/{setting}/{source_name}.txt')
        source = dataset_root / relative_source
        before = _stat_identity(source)
        try:
            raw = Path(before[0]).read_bytes()
        except OSError as error:
            raise ValueError(
                f'cannot read official {semantic} split {source}: {error}') from error
        if _stat_identity(source) != before:
            raise ValueError(
                f'official {semantic} split changed while reading: {source}')
        try:
            text = raw.decode('utf-8')
        except UnicodeError as error:
            raise ValueError(
                f'official {semantic} split is not UTF-8: {source}') from error
        image_ids: list[str] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            fields = line.split()
            if not fields:
                raise ValueError(
                    f'official {semantic} split row {line_number} is empty')
            raw_image_id = fields[0]
            if os.path.isabs(raw_image_id):
                raise ValueError(
                    f'official {semantic} split row {line_number} has an '
                    f'absolute image path: {raw_image_id!r}')
            image_id = os.path.normpath(raw_image_id)
            if image_id == '..' or image_id.startswith('../'):
                raise ValueError(
                    f'official {semantic} split row {line_number} escapes the '
                    f'dataset directory: {raw_image_id!r}')
            if image_id != raw_image_id:
                raise ValueError(
                    f'official {semantic} split row {line_number} is not a '
                    f'canonical cache image ID: {raw_image_id!r}')
            image_ids.append(raw_image_id)
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise ValueError(
                f'official {semantic} split is empty or contains duplicate paths')
        memberships[semantic] = set(image_ids)
        source_sha256[relative_source] = hashlib.sha256(raw).hexdigest()
    return memberships, source_sha256


def audit_cache_decode_failures(
        cache_dir: str | Path, *, dataset: str, dataset_root: str | Path,
        setting: str = 'setting1') -> dict[str, object]:
    """Recompute the paper cache's exact failed-decode row/path contract.

    All datasets except MSCOCO must have zero failed decodes.  MSCOCO's known
    sixteen corrupt catalog entries are retained only because they are official
    database-only rows; none may be in designated train or query.  The exact
    row coordinates and image IDs are pinned, rather than accepting any sixteen
    failures with the same count.
    """
    if dataset not in _DATASET_DIRECTORIES:
        raise ValueError(f'unsupported decode-audit dataset {dataset!r}')
    if setting != 'setting1':
        raise ValueError(
            f'decode-failure audit supports only setting1, got {setting!r}')
    cache = Path(cache_dir).expanduser().resolve(strict=True)
    root = Path(dataset_root).expanduser().resolve(strict=True)
    meta_raw, meta_sha256 = _read_stable_json(
        cache / 'meta.json', description='feature-cache metadata')
    image_ids_raw, image_ids_sha256 = _read_stable_json(
        cache / 'image_ids.json', description='feature-cache image IDs')
    if not isinstance(meta_raw, dict):
        raise ValueError('feature-cache metadata root is not an object')
    if (not isinstance(image_ids_raw, list) or not image_ids_raw
            or any(not isinstance(value, str) or not value
                   for value in image_ids_raw)
            or len(image_ids_raw) != len(set(image_ids_raw))):
        raise ValueError(
            'feature-cache image_ids must be a non-empty unique string list')
    image_ids = tuple(image_ids_raw)

    has_count = 'n_failed_image_decode' in meta_raw
    has_indices = 'failed_image_indices' in meta_raw
    if has_count != has_indices:
        raise ValueError(
            'feature-cache metadata must declare failed decode count and '
            'indices together')
    if not has_count:
        if dataset != 'CIFAR10':
            raise ValueError(
                'non-CIFAR cache metadata omits failed-decode provenance')
        count: object = 0
        indices_raw: object = []
        source_encoding = 'absent_means_zero_legacy_cifar_extractor'
    else:
        count = meta_raw.get('n_failed_image_decode')
        indices_raw = meta_raw.get('failed_image_indices')
        source_encoding = 'explicit_meta_fields'
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f'invalid n_failed_image_decode={count!r}')
    if (not isinstance(indices_raw, list)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in indices_raw)):
        raise ValueError('failed_image_indices must be a JSON list[int]')
    indices = tuple(indices_raw)
    if len(indices) != count:
        raise ValueError(
            'failed-decode count does not equal failed_image_indices length')
    if (indices != tuple(sorted(set(indices)))
            or any(value < 0 or value >= len(image_ids) for value in indices)):
        raise ValueError(
            'failed_image_indices must be increasing unique cache rows within '
            'image_ids bounds')
    failed_ids = tuple(image_ids[index] for index in indices)
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError('failed decode indices do not identify unique image paths')
    indices_sha256 = _row_vector_sha256(indices)
    failed_ids_sha256 = _image_id_vector_sha256(failed_ids)

    memberships = {'train': set(), 'query': set(), 'database': set()}
    membership_source_sha256: dict[str, str] = {}
    if dataset != 'CIFAR10':
        memberships, membership_source_sha256 = _official_split_image_ids(
            root, dataset=dataset, setting=setting)
        train_query_split_overlap = memberships['train'] & memberships['query']
        if train_query_split_overlap:
            raise ValueError(
                'official train/query split overlap is nonzero: '
                f'{sorted(train_query_split_overlap)!r}')
        cache_id_set = set(image_ids)
        for semantic, official_ids in memberships.items():
            missing = official_ids - cache_id_set
            if missing:
                raise ValueError(
                    f'official {semantic} split has {len(missing)} paths absent '
                    'from feature-cache image_ids')
    failed_set = set(failed_ids)
    train_failed = failed_set & memberships['train']
    query_failed = failed_set & memberships['query']
    database_failed = failed_set & memberships['database']
    assigned = set().union(*memberships.values())
    unassigned = failed_set - assigned
    database_only = database_failed - train_failed - query_failed
    train_query_overlap = train_failed | query_failed
    if train_query_overlap:
        raise ValueError(
            'failed decode paths overlap designated train/query rows: '
            f'{sorted(train_query_overlap)!r}')

    if dataset == 'MSCOCO':
        if indices != MSCOCO_EXPECTED_FAILED_IMAGE_INDICES:
            raise ValueError(
                'MSCOCO failed decode row set differs from the audited exact 16')
        if failed_ids != MSCOCO_EXPECTED_FAILED_IMAGE_IDS:
            raise ValueError(
                'MSCOCO failed decode image-path set differs from the audited '
                'exact 16')
        if indices_sha256 != MSCOCO_EXPECTED_FAILED_INDICES_SHA256 \
                or failed_ids_sha256 != MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256:
            raise ValueError('MSCOCO failed decode set digest is not canonical')
        if (unassigned or database_only != failed_set
                or len(database_failed) != len(failed_set)):
            raise ValueError(
                'MSCOCO failed decode paths are not exclusively official '
                'database-only rows')
        policy = 'kept_and_disclosed_db_only'
        sensitivity = 'later exclusion sensitivity; main inputs keep these rows'
    else:
        if indices or failed_ids:
            raise ValueError(
                f'{dataset} is required to have zero image decode failures')
        policy = 'no_decode_failures'
        sensitivity = 'not_applicable'

    return {
        'schema': DECODE_FAILURE_AUDIT_SCHEMA,
        'schema_version': DECODE_FAILURE_AUDIT_SCHEMA_VERSION,
        'status': 'verified',
        'dataset': dataset,
        'setting': setting,
        'cache_dir': str(cache),
        'cache_meta_sha256': meta_sha256,
        'cache_image_ids_sha256': image_ids_sha256,
        'source_encoding': source_encoding,
        'policy': policy,
        'sensitivity_analysis': sensitivity,
        'count': int(count),
        'failed_image_indices': list(indices),
        'failed_indices_sha256': indices_sha256,
        'failed_image_ids': list(failed_ids),
        'failed_image_ids_sha256': failed_ids_sha256,
        'official_membership': {
            'train_count': len(train_failed),
            'query_count': len(query_failed),
            'database_count': len(database_failed),
            'database_only_count': len(database_only),
            'unassigned_count': len(unassigned),
        },
        'official_train_query_overlap_count': 0,
        'membership_source_sha256': dict(sorted(
            membership_source_sha256.items())),
    }


def verify_cache_decode_failure_binding(
        identity: Mapping[str, object], *, dataset: str,
        setting: str = 'setting1') -> dict[str, object]:
    """Reopen a protocol identity's cache and exact decode-failure evidence."""
    cache_dir = identity.get('cache_dir')
    dataset_root = identity.get('dataset_root')
    declared = identity.get('cache_decode_failure_audit')
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError('protocol_identity.cache_dir is missing')
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError('protocol_identity.dataset_root is missing')
    if not isinstance(declared, Mapping):
        raise ValueError(
            'protocol_identity.cache_decode_failure_audit is missing')
    current = audit_cache_decode_failures(
        cache_dir, dataset=dataset, dataset_root=dataset_root, setting=setting)
    if dict(declared) != current:
        raise ValueError(
            'protocol_identity cache decode-failure audit differs from current '
            'cache metadata/image IDs/official split mapping')
    if identity.get('cache_meta_sha256') != current['cache_meta_sha256']:
        raise ValueError(
            'protocol_identity.cache_meta_sha256 differs from decode audit')
    if identity.get('cache_image_ids_sha256') != current[
            'cache_image_ids_sha256']:
        raise ValueError(
            'protocol_identity.cache_image_ids_sha256 differs from decode audit')
    split_sha256 = identity.get('split_sha256')
    if not isinstance(split_sha256, Mapping):
        raise ValueError('protocol_identity.split_sha256 is missing')
    for path, digest in current['membership_source_sha256'].items():
        if split_sha256.get(path) != digest:
            raise ValueError(
                f'protocol_identity split digest differs from decode audit for '
                f'{path!r}')
    return current
