import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import baseline.cache_provenance as cache_provenance
from baseline.cache_provenance import (
    consumed_cache_artifact_names,
    hash_cache_artifacts,
    memoized_sha256_file,
    persistent_sha256_memo_path,
    verify_cache_artifact_hashes,
)


class CacheProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        cache_provenance._HASH_CACHE.clear()

    def test_consumed_artifact_set_matches_loader_flags(self) -> None:
        self.assertEqual(
            consumed_cache_artifact_names(
                paired_aug=False, visual_tokens=False),
            ('visual_global.f16.npy',),
        )
        self.assertEqual(len(consumed_cache_artifact_names(
            paired_aug=True, visual_tokens=True)), 6)

    def test_content_and_symlink_target_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.npy'
            second = root / 'second.npy'
            first.write_bytes(b'first-content')
            second.write_bytes(b'second-content')
            link = root / 'visual_global.f16.npy'
            link.symlink_to(first.name)
            expected = hash_cache_artifacts(
                root, ['visual_global.f16.npy'])
            verify_cache_artifact_hashes(root, expected)
            link.unlink()
            link.symlink_to(second.name)
            with self.assertRaisesRegex(ValueError, 'consumed-array SHA-256'):
                verify_cache_artifact_hashes(root, expected)

    def test_content_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'visual_global.f16.npy'
            path.write_bytes(b'original')
            expected = hash_cache_artifacts(root, [path.name])
            path.write_bytes(b'mutated-content')
            os.utime(path, None)
            with self.assertRaisesRegex(ValueError, 'consumed-array SHA-256'):
                verify_cache_artifact_hashes(root, expected)

    def test_default_memo_path_is_uid_and_repository_scoped(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GROUNDEDDNA_SHA256_MEMO_PATH', None)
            path = persistent_sha256_memo_path()
        self.assertEqual(path.parent, Path('/tmp'))
        if hasattr(os, 'getuid'):
            self.assertIn(f'-{os.getuid()}-', path.name)
        self.assertRegex(path.name, r'-[0-9a-f]{16}\.json$')

    @staticmethod
    def _worker_code() -> str:
        return r'''
import os
import sys
import time
import baseline.cache_provenance as provenance

artifact, memo, marker = sys.argv[1:]
original = provenance._hash_stable_file
def observed_hash(*args, **kwargs):
    fd = os.open(marker, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(fd, (str(os.getpid()) + "\n").encode("ascii"))
    finally:
        os.close(fd)
    time.sleep(0.05)
    return original(*args, **kwargs)
provenance._hash_stable_file = observed_hash
print(provenance.memoized_sha256_file(artifact, memo_path=memo))
'''

    def _run_worker(
            self, artifact: Path, memo: Path, marker: Path
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, '-c', self._worker_code(), str(artifact),
             str(memo), str(marker)],
            cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        )

    def test_cross_process_lock_hashes_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'visual_global.f16.npy'
            artifact.write_bytes(b'large-array-content' * 1024)
            memo = root / 'memo.json'
            marker = root / 'hash-calls.txt'
            command = [
                sys.executable, '-c', self._worker_code(), str(artifact),
                str(memo), str(marker),
            ]
            workers = [subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) for _ in range(4)]
            outputs = [worker.communicate(timeout=30) for worker in workers]
            for worker, (_, stderr) in zip(workers, outputs):
                self.assertEqual(worker.returncode, 0, stderr)
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(
                {stdout.strip() for stdout, _ in outputs}, {expected})
            self.assertEqual(marker.read_text(encoding='utf-8').count('\n'), 1)
            json.loads(memo.read_text(encoding='utf-8'))

    def test_cross_process_stat_change_invalidates_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'visual_global.f16.npy'
            memo = root / 'memo.json'
            first_marker = root / 'first-hash.txt'
            second_marker = root / 'second-hash.txt'
            artifact.write_bytes(b'first')
            first = self._run_worker(artifact, memo, first_marker)
            artifact.write_bytes(b'second-content-with-a-new-size')
            second = self._run_worker(artifact, memo, second_marker)
            self.assertNotEqual(first.stdout.strip(), second.stdout.strip())
            self.assertEqual(
                second.stdout.strip(),
                hashlib.sha256(artifact.read_bytes()).hexdigest())
            self.assertEqual(second_marker.read_text(encoding='utf-8').count('\n'), 1)
            payload = json.loads(memo.read_text(encoding='utf-8'))
            records = [
                record for record in payload['entries'].values()
                if record['identity'][0] == str(artifact.resolve())
            ]
            self.assertEqual(len(records), 1)

    def test_corrupt_persistent_memo_fails_safe_and_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'visual_global.f16.npy'
            artifact.write_bytes(b'authoritative-content')
            memo = root / 'memo.json'
            memo.write_text('{truncated', encoding='utf-8')
            os.chmod(memo, 0o600)
            marker = root / 'hash-calls.txt'
            result = self._run_worker(artifact, memo, marker)
            self.assertEqual(
                result.stdout.strip(),
                hashlib.sha256(artifact.read_bytes()).hexdigest())
            self.assertEqual(marker.read_text(encoding='utf-8').count('\n'), 1)
            rebuilt = json.loads(memo.read_text(encoding='utf-8'))
            self.assertEqual(rebuilt['schema_version'], 1)
            self.assertEqual(len(rebuilt['entries']), 1)

    def test_well_formed_memo_corruption_is_detected_by_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'visual_global.f16.npy'
            artifact.write_bytes(b'authoritative-content')
            memo = root / 'memo.json'
            first_marker = root / 'first-hash.txt'
            self._run_worker(artifact, memo, first_marker)
            payload = json.loads(memo.read_text(encoding='utf-8'))
            record = next(iter(payload['entries'].values()))
            record['sha256'] = '0' * 64
            memo.write_text(json.dumps(payload), encoding='utf-8')
            second_marker = root / 'second-hash.txt'
            result = self._run_worker(artifact, memo, second_marker)
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(result.stdout.strip(), expected)
            self.assertEqual(second_marker.read_text(
                encoding='utf-8').count('\n'), 1)
            rebuilt = json.loads(memo.read_text(encoding='utf-8'))
            rebuilt_record = next(iter(rebuilt['entries'].values()))
            self.assertEqual(rebuilt_record['sha256'], expected)

    def test_environment_override_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'artifact.npy'
            artifact.write_bytes(b'content')
            configured = root / 'configured' / 'memo.json'
            with mock.patch.dict(
                    os.environ,
                    {'GROUNDEDDNA_SHA256_MEMO_PATH': str(configured)}):
                self.assertEqual(
                    memoized_sha256_file(artifact),
                    hashlib.sha256(b'content').hexdigest())
            self.assertTrue(configured.is_file())

    def test_replacement_detected_across_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'artifact.npy'
            artifact.write_bytes(b'content')
            before = cache_provenance._stat_identity(artifact)
            replaced = (
                before[0], before[1], before[2] + 1, before[3],
                before[4], before[5],
            )
            with mock.patch.object(
                    cache_provenance, '_stat_identity',
                    side_effect=[before, replaced]):
                with self.assertRaisesRegex(
                        RuntimeError, 'changed while hashing'):
                    cache_provenance._hash_stable_file(artifact)


if __name__ == '__main__':
    unittest.main()
