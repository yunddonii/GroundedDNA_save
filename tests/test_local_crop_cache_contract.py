from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from extract_clip_local_crops import (
    _install_donor_global_links,
    main as local_crop_main,
    _remove_stale_augmented_visual_files,
    _sync_donor_text_links,
    _validated_donor_global_sources,
)
from dataloaders import _SigLIP2FeatureCache


class LocalCropGlobalFeatureContractTest(unittest.TestCase):
    @staticmethod
    def _write_reader_fixture(
        root: Path,
        meta: dict,
        *,
        donor: Path | None = None,
        aug_views: int = 0,
    ) -> None:
        (root / "meta.json").write_text(
            __import__("json").dumps(meta), encoding="utf-8",
        )
        np.save(
            root / "visual_tokens.f16.npy",
            np.zeros((3, 6, 5), dtype=np.float16),
        )
        if donor is None:
            (root / "image_ids.json").write_text(
                '["a", "b", "c"]', encoding="utf-8",
            )
            np.save(
                root / "visual_global.f16.npy",
                np.zeros((3, 4), dtype=np.float16),
            )
            np.save(
                root / "text_part.f16.npy",
                np.zeros((3, 6, 4), dtype=np.float16),
            )
            np.save(
                root / "has_text.bool.npy",
                np.ones(3, dtype=np.bool_),
            )
        else:
            donor.mkdir(exist_ok=True)
            (donor / "image_ids.json").write_text(
                '["a", "b", "c"]', encoding="utf-8",
            )
            np.save(
                donor / "text_part.f16.npy",
                np.zeros((3, 6, 4), dtype=np.float16),
            )
            np.save(
                donor / "has_text.bool.npy",
                np.ones(3, dtype=np.bool_),
            )
            for name in (
                "image_ids.json",
                "text_part.f16.npy",
                "has_text.bool.npy",
            ):
                (root / name).symlink_to(donor / name)
            for view in range(-1, aug_views):
                suffix = "" if view < 0 else f"_aug{view}"
                donor_global = donor / f"visual_global{suffix}.f16.npy"
                np.save(
                    donor_global,
                    np.full((3, 4), view + 2, dtype=np.float16),
                )
                (root / donor_global.name).symlink_to(donor_global)
            for view in range(aug_views):
                np.save(
                    root / f"visual_tokens_aug{view}.f16.npy",
                    np.zeros((3, 6, 5), dtype=np.float16),
                )

    def test_crop_cache_preserves_every_donor_full_image_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor = root / "donor"
            output = root / "output"
            donor.mkdir()
            output.mkdir()
            expected: dict[str, np.ndarray] = {}
            for index, name in enumerate((
                "visual_global.f16.npy",
                "visual_global_aug0.f16.npy",
                "visual_global_aug1.f16.npy",
            )):
                value = np.full((3, 4), index + 1, dtype=np.float16)
                expected[name] = value
                np.save(donor / name, value)
                # Re-running the fixed generator must replace globals emitted
                # by the historical crop-average implementation.
                np.save(output / name, np.full((3, 4), -9, dtype=np.float16))

            sources = _validated_donor_global_sources(
                str(donor), num_examples=3, projection_dim=4,
                num_aug_views=2,
            )
            _install_donor_global_links(sources, str(output))

            for name, value in expected.items():
                path = output / name
                self.assertTrue(path.is_symlink(), name)
                np.testing.assert_array_equal(np.load(path), value)

    def test_missing_or_shape_mismatched_donor_global_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            donor = Path(directory)
            np.save(
                donor / "visual_global.f16.npy",
                np.zeros((2, 4), dtype=np.float16),
            )
            with self.assertRaisesRegex(ValueError, "must be float16"):
                _validated_donor_global_sources(
                    str(donor), num_examples=3, projection_dim=4,
                    num_aug_views=0,
                )
            with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                _validated_donor_global_sources(
                    str(donor), num_examples=2, projection_dim=4,
                    num_aug_views=1,
                )

    def test_generator_rejects_negative_aug_views_before_touching_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            argv = [
                "extract_clip_local_crops.py",
                "--donor_dir", str(root / "missing-donor"),
                "--out_dir", str(output),
                "--pathlist_root", str(root / "missing-pathlist"),
                "--num_aug_views", "-1",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    local_crop_main()
            self.assertFalse(output.exists())

    def test_reader_rejects_legacy_confounded_local_crop_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_reader_fixture(root, {
                "N": 3, "num_tokens": 6, "H_v": 5, "D_proj": 4,
                "local_crops_L": 8, "local_crops_K": 3,
                "save_aug_views": 0,
            })
            with self.assertRaisesRegex(ValueError, "refusing legacy"):
                _SigLIP2FeatureCache(str(root))

    def test_reader_accepts_bound_token_only_contract_and_checks_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor = root / "donor"
            cache_root = root / "cache"
            cache_root.mkdir()
            meta = {
                "N": 3, "num_tokens": 6, "H_v": 5, "D_proj": 4,
                "local_crops_L": 8, "local_crops_K": 3,
                "save_aug_views": 2,
                "visual_global_source": "donor_full_image",
                "visual_global_files": [
                    "visual_global.f16.npy",
                    "visual_global_aug0.f16.npy",
                    "visual_global_aug1.f16.npy",
                ],
                "donor_cache": str(donor),
            }
            self._write_reader_fixture(
                cache_root, meta, donor=donor, aug_views=2,
            )
            cache = _SigLIP2FeatureCache(str(cache_root))
            self.assertEqual(cache.visual_tokens.shape, (3, 6, 5))
            self.assertEqual(len(cache.visual_tokens_aug), 2)
            self.assertEqual(len(cache.visual_global_aug), 2)

            meta["num_tokens"] = 7
            (cache_root / "meta.json").write_text(
                __import__("json").dumps(meta), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "geometry differs"):
                _SigLIP2FeatureCache(str(cache_root))

    def test_reader_rejects_spoofed_donor_metadata_without_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor = root / "donor"
            cache_root = root / "cache"
            cache_root.mkdir()
            meta = {
                "N": 3, "num_tokens": 6, "H_v": 5, "D_proj": 4,
                "local_crops_L": 8, "local_crops_K": 3,
                "save_aug_views": 0,
                "visual_global_source": "donor_full_image",
                "visual_global_files": ["visual_global.f16.npy"],
                "donor_cache": str(donor),
            }
            self._write_reader_fixture(cache_root, meta, donor=donor)
            (cache_root / "visual_global.f16.npy").unlink()
            np.save(
                cache_root / "visual_global.f16.npy",
                np.load(donor / "visual_global.f16.npy"),
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                _SigLIP2FeatureCache(str(cache_root))

    def test_reader_rejects_text_sidecar_from_a_different_donor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor = root / "donor"
            cache_root = root / "cache"
            cache_root.mkdir()
            meta = {
                "N": 3, "num_tokens": 6, "H_v": 5, "D_proj": 4,
                "local_crops_L": 8, "local_crops_K": 3,
                "save_aug_views": 0,
                "visual_global_source": "donor_full_image",
                "visual_global_files": ["visual_global.f16.npy"],
                "donor_cache": str(donor),
            }
            self._write_reader_fixture(cache_root, meta, donor=donor)
            (cache_root / "text_part.f16.npy").unlink()
            np.save(
                cache_root / "text_part.f16.npy",
                np.load(donor / "text_part.f16.npy"),
            )
            with self.assertRaisesRegex(ValueError, "text/row sidecar"):
                _SigLIP2FeatureCache(str(cache_root))

    def test_text_sidecars_rebind_to_new_donor_and_remove_stale_whiten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor_old = root / "donor-old"
            donor_new = root / "donor-new"
            output = root / "output"
            donor_old.mkdir()
            donor_new.mkdir()
            output.mkdir()
            for index, donor in enumerate((donor_old, donor_new)):
                (donor / "image_ids.json").write_text(
                    f'["donor-{index}"]', encoding="utf-8",
                )
                np.save(
                    donor / "text_part.f16.npy",
                    np.full((1, 2, 3), index, dtype=np.float16),
                )
                np.save(
                    donor / "has_text.bool.npy",
                    np.array([bool(index)], dtype=np.bool_),
                )
            (donor_old / "text_whiten.npz").write_bytes(b"old-whiten")
            (donor_old / "text_whiten.npz.meta.json").write_text(
                "{}", encoding="utf-8",
            )

            _sync_donor_text_links(str(donor_old), str(output))
            linked, removed = _sync_donor_text_links(
                str(donor_new), str(output),
            )

            self.assertEqual(linked, [
                "text_part.f16.npy",
                "has_text.bool.npy",
                "image_ids.json",
            ])
            self.assertEqual(removed, [
                "text_whiten.npz",
                "text_whiten.npz.meta.json",
            ])
            for name in linked:
                self.assertTrue(os.path.samefile(
                    output / name, donor_new / name,
                ))
            self.assertFalse((output / "text_whiten.npz").exists())
            self.assertFalse(
                (output / "text_whiten.npz.meta.json").exists()
            )

    def test_reader_rejects_missing_stale_and_shape_mismatched_aug_views(self) -> None:
        base_meta = {
            "N": 3, "num_tokens": 6, "H_v": 5, "D_proj": 4,
            "local_crops_L": 8, "local_crops_K": 3,
            "visual_global_source": "donor_full_image",
        }
        for case in ("missing", "stale", "shape"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                donor = root / "donor"
                cache_root = root / "cache"
                cache_root.mkdir()
                aug_views = 1 if case != "stale" else 0
                meta = dict(base_meta)
                meta.update({
                    "save_aug_views": aug_views,
                    "visual_global_files": ["visual_global.f16.npy"] + [
                        f"visual_global_aug{view}.f16.npy"
                        for view in range(aug_views)
                    ],
                    "donor_cache": str(donor),
                })
                self._write_reader_fixture(
                    cache_root, meta, donor=donor, aug_views=aug_views,
                )
                if case == "missing":
                    (cache_root / "visual_tokens_aug0.f16.npy").unlink()
                elif case == "stale":
                    np.save(
                        cache_root / "visual_tokens_aug1.f16.npy",
                        np.zeros((3, 6, 5), dtype=np.float16),
                    )
                    np.save(
                        cache_root / "visual_global_aug1.f16.npy",
                        np.zeros((3, 4), dtype=np.float16),
                    )
                else:
                    np.save(
                        cache_root / "visual_tokens_aug0.f16.npy",
                        np.zeros((3, 7, 5), dtype=np.float16),
                    )
                with self.assertRaisesRegex(
                    ValueError, "augmented visual files|visual_tokens_aug0",
                ):
                    _SigLIP2FeatureCache(str(cache_root))

    def test_stale_cleanup_removes_only_exact_out_of_range_aug_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "visual_tokens_aug0.f16.npy",
                "visual_global_aug0.f16.npy",
                "visual_tokens_aug2.f16.npy",
                "visual_global_aug2.f16.npy",
                "visual_tokens_augx.f16.npy",
                "keep.txt",
            ):
                (root / name).write_bytes(b"x")
            removed = _remove_stale_augmented_visual_files(str(root), 1)
            self.assertEqual(removed, [
                "visual_global_aug2.f16.npy",
                "visual_tokens_aug2.f16.npy",
            ])
            self.assertTrue((root / "visual_tokens_aug0.f16.npy").exists())
            self.assertTrue((root / "visual_global_aug0.f16.npy").exists())
            self.assertTrue((root / "visual_tokens_augx.f16.npy").exists())
            self.assertTrue((root / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
