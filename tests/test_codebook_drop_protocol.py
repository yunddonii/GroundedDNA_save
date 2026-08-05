from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.codebook_drop_ablation import main as slow_main
from scripts.codebook_drop_ablation_fast import main as fast_main


class CodebookDropProjectionProtocolTest(unittest.TestCase):
    def test_partial_gc_bounds_are_rejected_by_both_clis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codes = np.zeros((1, 18), dtype=np.int64)
            np.savez(root / "extract_query.npz", base_indices=codes)
            np.savez(root / "extract_db.npz", base_indices=codes)
            for entrypoint in (slow_main, fast_main):
                argv = [
                    "codebook_drop_ablation.py",
                    "--result_dir", str(root),
                    "--bio_project",
                    "--gc_min_frac", "0.3",
                ]
                with self.subTest(entrypoint=entrypoint.__module__):
                    with patch.object(sys, "argv", argv):
                        with self.assertRaisesRegex(
                            ValueError, "must be supplied together",
                        ):
                            entrypoint()


if __name__ == "__main__":
    unittest.main()
