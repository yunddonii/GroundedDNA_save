from __future__ import annotations

import unittest

from p0_protocol import (
    is_p0_refit,
    is_p0_stage1,
    resolve_eval_text_whiten_path,
    should_load_official_test_split,
    should_run_official_test_evaluation,
    should_run_test_consuming_visualization,
    should_run_training_time_evaluation,
)


class P0OfficialTestGuardTests(unittest.TestCase):
    def test_positive_val_ratio_identifies_stage1(self) -> None:
        self.assertTrue(is_p0_stage1(val_split_ratio=0.1))
        self.assertFalse(is_p0_stage1(val_split_ratio=0.0))

    def test_stage1_does_not_construct_test_dataset_or_loader(self) -> None:
        self.assertFalse(
            should_load_official_test_split(val_protocol_active=True)
        )
        self.assertTrue(
            should_load_official_test_split(val_protocol_active=False)
        )

    def test_fixed_epoch_full_train_refit_is_identified(self) -> None:
        self.assertTrue(
            is_p0_refit(stop_after_epoch=4, final_epoch_eval=True)
        )
        self.assertFalse(
            is_p0_refit(stop_after_epoch=None, final_epoch_eval=True)
        )
        self.assertFalse(
            is_p0_refit(stop_after_epoch=4, final_epoch_eval=False)
        )

    def test_refit_does_not_construct_test_during_training(self) -> None:
        self.assertFalse(
            should_load_official_test_split(
                val_protocol_active=False,
                p0_refit_active=True,
            )
        )

    def test_refit_blocks_per_epoch_validation_and_mid_eval(self) -> None:
        self.assertFalse(
            should_run_training_time_evaluation(p0_refit_active=True)
        )
        self.assertTrue(
            should_run_training_time_evaluation(p0_refit_active=False)
        )

    def test_stage1_blocks_test_even_when_eval_is_requested(self) -> None:
        self.assertFalse(
            should_run_official_test_evaluation(
                evaluation_requested=True,
                val_protocol_active=True,
            )
        )

    def test_full_train_refit_keeps_single_test_evaluation(self) -> None:
        self.assertTrue(
            should_run_official_test_evaluation(
                evaluation_requested=True,
                val_protocol_active=False,
            )
        )

    def test_eval_flag_still_controls_non_p0_runs(self) -> None:
        self.assertFalse(
            should_run_official_test_evaluation(
                evaluation_requested=False,
                val_protocol_active=False,
            )
        )

    def test_stage1_blocks_test_consuming_visualization(self) -> None:
        self.assertFalse(
            should_run_test_consuming_visualization(
                visualization_requested=True,
                val_protocol_active=True,
            )
        )
        self.assertTrue(
            should_run_test_consuming_visualization(
                visualization_requested=True,
                val_protocol_active=False,
            )
        )
        self.assertFalse(
            should_run_test_consuming_visualization(
                visualization_requested=False,
                val_protocol_active=False,
            )
        )
        self.assertFalse(
            should_run_test_consuming_visualization(
                visualization_requested=True,
                val_protocol_active=False,
                p0_refit_active=True,
            )
        )

    def test_eval_cache_switch_preserves_training_whitening_by_default(self) -> None:
        local_only = "/artifacts/text_whiten_trainOnly_localOnly.npz"
        self.assertEqual(
            resolve_eval_text_whiten_path(
                training_text_whiten_path=local_only,
                explicit_eval_text_whiten_path=None,
            ),
            local_only,
        )

    def test_eval_whitening_changes_only_through_explicit_override(self) -> None:
        self.assertEqual(
            resolve_eval_text_whiten_path(
                training_text_whiten_path="/train/local-only.npz",
                explicit_eval_text_whiten_path="/eval/intentional.npz",
            ),
            "/eval/intentional.npz",
        )


if __name__ == "__main__":
    unittest.main()
