import unittest

import torch

from model_siglip2 import (
    _cls_verified_consensus_mask,
    _mutual_dual_softmax_pruning,
    _slot_consensus_residual_importance,
    _topk_visual_keep,
)
from models.semantic_router import SemanticSinkhornRouter


class MutualDualSoftmaxPruningTest(unittest.TestCase):
    def test_cls_verification_prunes_only_common_global_low_patch(self) -> None:
        # Patches 0/1 are high for both text slots. Patch 0 is orthogonal to
        # the global embedding and must be pruned; patch 1 is globally strong
        # and must survive. Patches 2/3 are slot-specific, patch 4 is weak.
        visual = torch.tensor([[[1.0, 1.0, 0.0],
                                [1.0, 1.0, 2.0],
                                [1.0, 0.0, 1.0],
                                [0.0, 1.0, 1.0],
                                [-1.0, -1.0, 1.0]]])                # [B=1, N=5, D=3]
        text = torch.tensor([[[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0]]])                    # [B=1, M_local=5, D=3]
        global_visual = torch.tensor([[0.0, 0.0, 1.0]])             # [B=1, D=3]

        out = _cls_verified_consensus_mask(visual, text, global_visual)

        self.assertTrue(bool(out["common_candidate"][0, 0]))
        self.assertTrue(bool(out["common_candidate"][0, 1]))
        self.assertTrue(bool(out["prune"][0, 0]))
        self.assertFalse(bool(out["prune"][0, 1]))
        self.assertFalse(bool(out["keep"][0, 0]))
        self.assertTrue(bool(out["keep"][0, 1:].all()))
        self.assertAlmostEqual(float(out["mask_ratio"][0]), 0.2, places=6)
        self.assertEqual(float(out["fallback"][0]), 0.0)

    def test_consensus_residual_removes_only_all_slot_evidence(self) -> None:
        # Patch 0 is strong for every slot and should be suppressed. Patches
        # 1/2/3 are respectively specific to slots 0/1/2.
        importance = torch.tensor([[[10.0, 9.0, 1.0, 1.0],
                                    [10.0, 1.0, 9.0, 1.0],
                                    [10.0, 1.0, 1.0, 9.0]]])         # [1, 3, 4]
        residual = _slot_consensus_residual_importance(importance)
        keep = _topk_visual_keep(residual, visual_mask=None, keep_ratio=0.25)

        self.assertEqual(tuple(residual.shape), (1, 3, 4))
        self.assertTrue(torch.equal(keep.sum(dim=-1), torch.ones(1, 3, dtype=torch.long)))
        self.assertFalse(bool(keep[0, :, 0].any()))
        self.assertTrue(bool(keep[0, 0, 1]))
        self.assertTrue(bool(keep[0, 1, 2]))
        self.assertTrue(bool(keep[0, 2, 3]))

    def test_consensus_residual_preserves_subset_shared_evidence(self) -> None:
        # Patch 0 is shared by slots 0 and 1 but not slot 2. Geometric
        # consensus over all slots must not erase this legitimate overlap.
        importance = torch.tensor([[[9.0, 1.0, 1.0],
                                    [9.0, 1.0, 1.0],
                                    [1.0, 1.0, 9.0]]])               # [1, 3, 3]
        residual = _slot_consensus_residual_importance(importance)

        self.assertGreater(float(residual[0, 0, 0]), 0.0)
        self.assertGreater(float(residual[0, 1, 0]), 0.0)
        self.assertGreater(float(residual[0, 2, 2]), 0.0)

    def test_slot_specific_visual_and_text_masks(self) -> None:
        # [B=1, M=2, N=4, T=4]. Each slot has a disjoint set of strong
        # patch-token correspondences, so both pruning directions must differ.
        similarity = torch.full((1, 2, 4, 4), -10.0)
        similarity[0, 0, 0, 0] = 10.0
        similarity[0, 0, 1, 1] = 9.0
        similarity[0, 1, 2, 2] = 10.0
        similarity[0, 1, 3, 3] = 9.0
        text_mask = torch.ones(1, 2, 4, dtype=torch.bool)

        out = _mutual_dual_softmax_pruning(
            similarity=similarity,
            text_mask=text_mask,
            visual_mask=None,
            visual_keep_ratio=0.5,
            text_keep_ratio=0.5,
        )

        visual_keep = out["visual_keep"]
        text_keep = out["text_keep"]
        self.assertEqual(tuple(visual_keep.shape), (1, 2, 4))
        self.assertEqual(tuple(text_keep.shape), (1, 2, 4))
        self.assertTrue(torch.equal(visual_keep.sum(dim=-1), torch.tensor([[2, 2]])))
        self.assertTrue(torch.equal(text_keep.sum(dim=-1), torch.tensor([[2, 2]])))
        self.assertFalse(torch.equal(visual_keep[:, 0], visual_keep[:, 1]))
        self.assertFalse(torch.equal(text_keep[:, 0], text_keep[:, 1]))
        self.assertGreater(float(out["visual_importance"].std()), 0.0)
        self.assertGreater(float(out["text_importance"].std()), 0.0)

    def test_padding_and_variable_text_lengths(self) -> None:
        torch.manual_seed(0)
        similarity = torch.randn(2, 3, 8, 6)
        text_mask = torch.tensor(
            [
                [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        )                                                               # [B=2, M=3, T=6]
        visual_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]],
            dtype=torch.bool,
        )                                                               # [B=2, N=8]

        out = _mutual_dual_softmax_pruning(
            similarity=similarity,
            text_mask=text_mask,
            visual_mask=visual_mask,
            visual_keep_ratio=0.5,
            text_keep_ratio=0.5,
        )

        expected_visual = torch.tensor([[3, 3, 3], [4, 4, 4]])
        expected_text = torch.tensor([[2, 2, 1], [3, 2, 1]])
        self.assertTrue(torch.equal(out["visual_keep"].sum(dim=-1), expected_visual))
        self.assertTrue(torch.equal(out["text_keep"].sum(dim=-1), expected_text))
        self.assertFalse(bool((out["text_keep"] & ~text_mask).any()))
        self.assertFalse(
            bool((out["visual_keep"] & ~visual_mask[:, None, :]).any())
        )

    def test_slot_mask_blocks_sinkhorn_transport(self) -> None:
        torch.manual_seed(1)
        visual_tokens = torch.randn(1, 4, 6)                          # [B, N, D]
        text_tokens = torch.randn(1, 2, 6)                            # [B, M, D]
        route_keep = torch.tensor(
            [[[1, 0], [1, 0], [0, 1], [0, 1]]], dtype=torch.bool,
        )                                                             # [B, N, M]
        cost_bias = torch.zeros(1, 4, 2).masked_fill(~route_keep, -1e4)
        router = SemanticSinkhornRouter(epsilon=0.1, num_iters=20)

        out = router(
            visual_tokens=visual_tokens,
            text_part_tokens=text_tokens,
            cost_bias=cost_bias,
        )
        routing = out["routing_matrix"]                              # [B, N, M]
        self.assertEqual(tuple(routing.shape), (1, 4, 2))
        self.assertLess(float(routing.masked_select(~route_keep).max()), 1e-8)
        self.assertGreater(float(routing.masked_select(route_keep).min()), 0.0)

    def test_specificity_marginal_downweights_slot_common_patch(self) -> None:
        # Patch 2 is equally similar to both slots, while patches 0/1 are
        # selective. It should receive almost no visual marginal mass.
        visual_tokens = torch.tensor([[[1.0, 0.0],
                                       [0.0, 1.0],
                                       [1.0, 1.0]]])                  # [B=1, N=3, D=2]
        text_tokens = torch.tensor([[[1.0, 0.0],
                                     [0.0, 1.0]]])                   # [B=1, M=2, D=2]
        router = SemanticSinkhornRouter(epsilon=0.1, num_iters=50)

        out = router(
            visual_tokens=visual_tokens,
            text_part_tokens=text_tokens,
            specificity_weighted_marginal=True,
        )
        row_mass = out["routing_matrix"].sum(dim=-1)                 # [1, 3]

        self.assertLess(float(row_mass[0, 2]), 1e-6)
        self.assertGreater(float(row_mass[0, 0]), 0.49)
        self.assertGreater(float(row_mass[0, 1]), 0.49)
        self.assertGreater(float(out["visual_specificity_mean"].mean()), 0.0)
        self.assertLess(float(out["visual_marginal_effective_ratio"].mean()), 1.0)

    def test_specificity_marginal_default_is_legacy_uniform(self) -> None:
        torch.manual_seed(2)
        visual_tokens = torch.randn(2, 5, 4)
        text_tokens = torch.randn(2, 3, 4)
        router = SemanticSinkhornRouter(epsilon=0.2, num_iters=20)

        legacy = router(visual_tokens, text_tokens)
        explicit_off = router(
            visual_tokens,
            text_tokens,
            specificity_weighted_marginal=False,
        )

        self.assertTrue(torch.equal(
            legacy["routing_matrix"], explicit_off["routing_matrix"],
        ))
        self.assertIsNone(legacy["visual_specificity_mean"])
        self.assertIsNone(legacy["visual_marginal_effective_ratio"])

    def test_centered_consensus_mask_removes_only_all_slot_high_patch(self) -> None:
        # Patch 0 is above the per-slot visual mean for both slots. Patches
        # 1/2 are slot-specific and patch 3 is uniformly weak.
        visual_tokens = torch.tensor([[[1.0, 1.0],
                                       [1.0, 0.0],
                                       [0.0, 1.0],
                                       [-1.0, -1.0]]])                # [B=1, N=4, D=2]
        text_tokens = torch.tensor([[[1.0, 0.0],
                                     [0.0, 1.0]]])                   # [B=1, M=2, D=2]
        router = SemanticSinkhornRouter(epsilon=0.1, num_iters=50)

        out = router(
            visual_tokens=visual_tokens,
            text_part_tokens=text_tokens,
            centered_consensus_mask=True,
        )
        row_mass = out["routing_matrix"].sum(dim=-1)                 # [B=1, N=4]

        self.assertEqual(float(row_mass[0, 0]), 0.0)
        self.assertTrue(bool((row_mass[0, 1:] > 0.0).all()))
        self.assertAlmostEqual(
            float(out["visual_consensus_mask_ratio"][0]), 0.25, places=6,
        )
        self.assertAlmostEqual(
            float(out["visual_consensus_remaining_ratio"][0]), 0.75, places=6,
        )
        self.assertEqual(float(out["visual_consensus_fallback"][0]), 0.0)

    def test_centered_consensus_mask_default_is_legacy_uniform(self) -> None:
        torch.manual_seed(3)
        visual_tokens = torch.randn(2, 5, 4)
        text_tokens = torch.randn(2, 3, 4)
        router = SemanticSinkhornRouter(epsilon=0.2, num_iters=20)

        legacy = router(visual_tokens, text_tokens)
        explicit_off = router(
            visual_tokens,
            text_tokens,
            centered_consensus_mask=False,
        )

        self.assertTrue(torch.equal(
            legacy["routing_matrix"], explicit_off["routing_matrix"],
        ))
        self.assertIsNone(legacy["visual_consensus_mask_ratio"])
        self.assertIsNone(legacy["visual_consensus_remaining_ratio"])
        self.assertIsNone(legacy["visual_consensus_fallback"])


if __name__ == "__main__":
    unittest.main()
