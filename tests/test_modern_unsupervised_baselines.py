"""Golden and invariance tests for clean-room modern hashing equations."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from baseline.modern_unsupervised import (
    DUHEGFeatureHash,
    SDCFeatureHash,
    UMRCHFeatureHash,
    duheg_external_guidance,
    duheg_multi_positive_nce,
    sdc_loss,
    simclr_nt_xent,
    umrch_loss,
    umrch_semantic_similarity,
)


class ModernUnsupervisedMathTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_sdc_release_and_paper_heads_are_not_silently_hybridized(self) -> None:
        release = SDCFeatureHash(8, 6, hidden_dim=8)
        paper = SDCFeatureHash(8, 6, hidden_dim=4096)
        self.assertEqual(tuple(release.encoder_layers[0].weight.shape), (8, 8))
        self.assertEqual(tuple(paper.encoder_layers[0].weight.shape), (4096, 8))
        self.assertEqual(tuple(release(torch.randn(4, 8))['continuous_code'].shape), (4, 6))

    def test_sdc_loss_sorts_hash_pairs_by_frozen_feature_order(self) -> None:
        # Pair feature similarities are [-1, +1], so their sorted order is
        # already pair 0 then pair 1.  Compute the published Beta(5,5)
        # quantile target independently for the two midpoint probabilities.
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0],
                                 [-1.0, 0.0], [1.0, 0.0]])
        codes = torch.tensor([[1.0, 0.0], [1.0, 0.0],
                              [-1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        total, parts = sdc_loss(features, codes, beta=5.0)
        from scipy.stats import beta as beta_distribution
        target = torch.tensor(
            2 * beta_distribution.ppf([0.25, 0.75], 5, 5) - 1,
            dtype=torch.float32,
        )
        expected_rec = F.l1_loss(torch.tensor([-1.0, 0.0]), target)
        self.assertTrue(torch.allclose(parts['reconstruction'], expected_rec, atol=1e-6))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(codes.grad)

    def test_simclr_matches_direct_cross_entropy_construction(self) -> None:
        first = torch.randn(5, 7)
        second = torch.randn(5, 7)
        actual = simclr_nt_xent(first, second, temperature=0.3)
        z = F.normalize(torch.cat((first, second)), dim=1)
        logits = z @ z.T / 0.3
        logits.fill_diagonal_(-torch.inf)
        target = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
        expected = F.cross_entropy(logits, target)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_sdc_two_view_layout_uses_within_view_half_pairs(self) -> None:
        # Stored order is [v1_B; v2_B].  The official contrastive SDC release
        # pairs v1[:B/2]↔v1[B/2:] and v2[:B/2]↔v2[B/2:], not v1↔v2.
        features = F.normalize(torch.tensor([
            [1., 0.], [0., 1.], [-1., 0.], [0., -1.],
            [1., 1.], [1., -1.], [-1., 1.], [-1., -1.],
        ]), dim=1)
        codes = F.normalize(torch.tensor([
            [1., .1], [.2, 1.], [-.8, .4], [.3, -.7],
            [.9, .2], [.6, -.5], [-.4, .8], [-.9, -.3],
        ]), dim=1)
        _, actual = sdc_loss(
            features, codes, contrastive_pair_layout=True,
            quantization_weight=0.0)
        a = torch.tensor([0, 1, 4, 5])
        b = torch.tensor([2, 3, 6, 7])
        feature_similarity = F.cosine_similarity(features[a], features[b])
        code_similarity = F.cosine_similarity(codes[a], codes[b])
        ordered = code_similarity[torch.argsort(feature_similarity)]
        from scipy.stats import beta as beta_distribution
        grid = np.arange(1., 8., 2., dtype=np.float32) / 8.
        target = torch.tensor(
            2 * beta_distribution.ppf(grid, 5, 5) - 1,
            dtype=ordered.dtype)
        expected = F.l1_loss(ordered, target)
        self.assertTrue(torch.allclose(
            actual['reconstruction'], expected, atol=1e-7))

        # The old erroneous v1↔v2 layout must not accidentally be equivalent.
        _, wrong = sdc_loss(
            features, codes, contrastive_pair_layout=False,
            quantization_weight=0.0)
        self.assertFalse(torch.allclose(
            actual['reconstruction'], wrong['reconstruction'], atol=1e-6))

    def test_duheg_guidance_and_multi_positive_probability_mass(self) -> None:
        images = F.normalize(torch.tensor([[2.0, 0.0], [0.0, 3.0]]), dim=1)
        nouns = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0],
                                          [1.0, 1.0]]), dim=1)
        actual = duheg_external_guidance(images, nouns, temperature=0.5)
        weights = torch.softmax(images @ nouns.T / 0.5, dim=1)
        expected = F.normalize(weights @ nouns, dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

        release_half = duheg_external_guidance(
            images, nouns, temperature=0.5, compute_dtype=torch.float16)
        half_images = F.normalize(images.float(), dim=1).half()
        half_nouns = F.normalize(nouns.float(), dim=1).half()
        half_weights = torch.softmax(half_images @ half_nouns.T / 0.5, dim=1)
        expected_half = F.normalize(half_weights @ half_nouns, dim=1)
        expected_half = F.normalize(expected_half, dim=1).float()
        self.assertTrue(torch.equal(release_half, expected_half))

        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = first.clone()
        mask = torch.eye(2)
        loss = duheg_multi_positive_nce(first, second, mask, temperature=1.0)
        probability = torch.softmax(first @ second.T, dim=1).diagonal()
        self.assertTrue(torch.allclose(loss, -probability.log().mean(), atol=1e-7))

    def test_duheg_two_heads_share_architecture_but_not_parameters(self) -> None:
        model = DUHEGFeatureHash(4, 4, 6)
        self.assertEqual(tuple(model(torch.randn(3, 4))['continuous_code'].shape), (3, 6))
        self.assertEqual(tuple(model.encode_guidance(torch.randn(3, 4)).shape), (3, 6))
        self.assertIsNot(model.encoder_layers[0].weight, model.text_encoder[0].weight)

    def test_duheg_and_umrch_hash_paths_normalize_cached_clip_inputs(self) -> None:
        features = torch.randn(4, 5)
        positive_scales = torch.tensor([[0.2], [1.0], [3.0], [11.0]])

        duheg = DUHEGFeatureHash(5, 5, 6).eval()
        torch.testing.assert_close(
            duheg(features)['continuous_code'],
            duheg(features * positive_scales)['continuous_code'],
            rtol=1e-6, atol=1e-6)
        guidance = torch.randn(4, 5)
        torch.testing.assert_close(
            duheg.encode_guidance(guidance),
            duheg.encode_guidance(guidance * positive_scales),
            rtol=1e-6, atol=1e-6)

        umrch = UMRCHFeatureHash(5, 6).eval()
        torch.testing.assert_close(
            umrch(features)['continuous_code'],
            umrch(features * positive_scales)['continuous_code'],
            rtol=1e-6, atol=1e-6)

    def test_umrch_global_local_soft_iou_golden_case(self) -> None:
        # Near-zero temperature makes each image choose a distinct orthogonal
        # concept; their activated soft-IoU must therefore be zero off-diagonal.
        concepts = torch.eye(2)
        global_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        local_features = global_features[:, None, :]
        semantic = umrch_semantic_similarity(
            global_features, local_features, concepts,
            vl_temperature=0.01, concept_threshold=0.3,
            negative_threshold=0.0,
        )
        self.assertTrue(torch.allclose(semantic.diagonal(), torch.ones(2), atol=1e-6))
        self.assertLess(float(semantic[0, 1]), 1e-6)

    def test_umrch_loss_adds_cross_view_positive_without_mutating_input(self) -> None:
        codes = torch.randn(6, 8, requires_grad=True)
        semantic = torch.eye(6)
        original = semantic.clone()
        loss, parts = umrch_loss(codes, semantic)
        self.assertTrue(torch.equal(semantic, original))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(codes.grad).all())


if __name__ == '__main__':
    unittest.main()
