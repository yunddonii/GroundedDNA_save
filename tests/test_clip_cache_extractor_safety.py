import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_clip_features import (
    _assert_fresh_full_cache_targets,
    _canonical_transform_record,
    _validate_aug_only_cache,
)
from extract_clip_features_cifar10 import _assert_fresh_cache_targets


class ClipCacheExtractorSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ids = ['a', 'b']
        (self.root / 'image_ids.json').write_text(
            json.dumps(self.ids), encoding='utf-8')
        (self.root / 'meta.json').write_text(json.dumps({
            'N': 2, 'D_proj': 3, 'num_tokens': 4, 'H_v': 5,
            'dtype': 'float16', 'backbone': 'clip-test',
            'canonical_transform': _canonical_transform_record(224),
        }), encoding='utf-8')
        np.save(
            self.root / 'visual_global.f16.npy',
            np.zeros((2, 3), dtype=np.float16))
        np.save(
            self.root / 'visual_tokens.f16.npy',
            np.zeros((2, 4, 5), dtype=np.float16))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_aug_only_accepts_exact_canonical_context(self) -> None:
        meta = _validate_aug_only_cache(
            str(self.root), self.ids, clip_backbone='clip-test',
            image_size=224, dtype='float16', save_aug_views=2,
            overwrite_aug_views=False)
        self.assertEqual(meta['N'], 2)

    def test_aug_only_rejects_order_drift_and_implicit_overwrite(self) -> None:
        with self.assertRaisesRegex(ValueError, 'image_ids'):
            _validate_aug_only_cache(
                str(self.root), list(reversed(self.ids)),
                clip_backbone='clip-test', image_size=224, dtype='float16',
                save_aug_views=2, overwrite_aug_views=False)
        np.save(
            self.root / 'visual_global_aug0.f16.npy',
            np.zeros((2, 3), dtype=np.float16))
        with self.assertRaisesRegex(FileExistsError, 'overwrite_aug_views'):
            _validate_aug_only_cache(
                str(self.root), self.ids, clip_backbone='clip-test',
                image_size=224, dtype='float16', save_aug_views=2,
                overwrite_aug_views=False)

    def test_full_extractors_require_explicit_overwrite(self) -> None:
        with self.assertRaisesRegex(FileExistsError, 'overwrite_cache'):
            _assert_fresh_full_cache_targets(
                str(self.root), 0, overwrite_cache=False)
        with self.assertRaisesRegex(FileExistsError, 'overwrite_cache'):
            _assert_fresh_cache_targets(
                str(self.root), 0, overwrite_cache=False)
        _assert_fresh_full_cache_targets(
            str(self.root), 0, overwrite_cache=True)
        _assert_fresh_cache_targets(
            str(self.root), 0, overwrite_cache=True)


if __name__ == '__main__':
    unittest.main()
