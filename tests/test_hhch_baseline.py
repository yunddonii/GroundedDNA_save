"""Focused equation and interface tests for the clean-room HHCH baseline."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from baseline.HHCH import HHCH, _parse_cluster_counts
from baseline.modern_unsupervised import (
    HHCHFeatureHash,
    hhch_hierarchical_instance_loss,
    hhch_hierarchical_kmeans,
    hhch_hierarchical_prototype_loss,
    hhch_logcosh_quantization_loss,
    hhch_loss,
    hyperbolic_kmeans_plusplus,
    poincare_einstein_midpoint,
    poincare_expmap0,
    poincare_pairwise_distance,
)


def _paper_mobius_distance(first: torch.Tensor, second: torch.Tensor,
                            curvature: float) -> torch.Tensor:
    """Literal HHCH Eqs. (1)--(2), used only as an independent oracle."""
    x = first[:, None, :]
    y = second[None, :, :]
    c = torch.as_tensor(curvature, dtype=first.dtype)
    x_sq = x.square().sum(dim=-1, keepdim=True)
    y_sq = y.square().sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = ((1.0 - 2.0 * c * xy + c * y_sq) * (-x)
                 + (1.0 - c * x_sq) * y)
    denominator = (1.0 - 2.0 * c * xy
                   + c.square() * x_sq * y_sq)
    mobius = numerator / denominator
    argument = (torch.sqrt(c) * mobius.norm(dim=-1)).clamp(
        max=1.0 - 1e-12)
    return 2.0 / torch.sqrt(c) * torch.atanh(argument)


def _check_poincare_expmap_and_distance_match_paper_equations():
    curvature = 0.01
    tangent = torch.tensor([[0.2, -0.4], [0.5, 0.1]], dtype=torch.float64)
    mapped = poincare_expmap0(tangent, curvature, clip_radius=None)
    norm = tangent.norm(dim=1, keepdim=True)
    expected = (torch.tanh(math.sqrt(curvature) * norm)
                * tangent / (math.sqrt(curvature) * norm))
    torch.testing.assert_close(mapped, expected, rtol=1e-10, atol=1e-12)

    other = poincare_expmap0(
        torch.tensor([[-0.1, 0.3], [0.7, -0.2]], dtype=torch.float64),
        curvature, clip_radius=None)
    actual_distance = poincare_pairwise_distance(mapped, other, curvature)
    expected_distance = _paper_mobius_distance(mapped, other, curvature)
    torch.testing.assert_close(
        actual_distance, expected_distance, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(
        actual_distance,
        poincare_pairwise_distance(other, mapped, curvature).T,
        rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        poincare_pairwise_distance(mapped, mapped, curvature).diagonal(),
        torch.zeros(mapped.shape[0], dtype=mapped.dtype),
        rtol=0.0, atol=0.0)


def _check_einstein_midpoint_of_symmetric_points_is_origin():
    points = poincare_expmap0(
        torch.tensor([[0.5, -0.2], [-0.5, 0.2]], dtype=torch.float64),
        0.01, clip_radius=None)
    midpoint = poincare_einstein_midpoint(points, 0.01)
    torch.testing.assert_close(midpoint, torch.zeros_like(midpoint),
                               rtol=0.0, atol=1e-12)


def _check_hyperbolic_kmeans_plus_plus_is_deterministic_and_separates_clouds():
    torch.manual_seed(7)
    left = torch.randn(12, 2, dtype=torch.float64) * 0.01 + torch.tensor([-0.4, 0.0])
    right = torch.randn(12, 2, dtype=torch.float64) * 0.01 + torch.tensor([0.4, 0.0])
    points = poincare_expmap0(torch.cat((left, right)), 0.01, clip_radius=None)
    centers_a, assignment_a = hyperbolic_kmeans_plusplus(
        points, 2, curvature=0.01, max_iter=20, seed=11)
    centers_b, assignment_b = hyperbolic_kmeans_plusplus(
        points, 2, curvature=0.01, max_iter=20, seed=11)
    torch.testing.assert_close(centers_a, centers_b)
    assert torch.equal(assignment_a, assignment_b)
    assert assignment_a[:12].unique().numel() == 1
    assert assignment_a[12:].unique().numel() == 1
    assert int(assignment_a[0]) != int(assignment_a[-1])


def _check_bottom_up_hierarchy_returns_original_sample_ancestors():
    torch.manual_seed(3)
    cloud_centers = torch.tensor(
        [[-0.5, -0.3], [-0.5, 0.3], [0.5, -0.3], [0.5, 0.3]],
        dtype=torch.float64)
    points = torch.cat([
        center + torch.randn(5, 2, dtype=torch.float64) * 0.005
        for center in cloud_centers
    ])
    points = poincare_expmap0(points, 0.01, clip_radius=None)
    hierarchy = hhch_hierarchical_kmeans(
        points, (4, 2), curvature=0.01, max_iter=20, seed=5)
    ancestors = hierarchy['assignments']
    centers = hierarchy['centers']
    assert isinstance(ancestors, torch.Tensor)
    assert ancestors.shape == (20, 2)
    assert [level.shape for level in centers] == [(4, 2), (2, 2)]
    assert int(ancestors[:, 0].min()) >= 0
    assert int(ancestors[:, 0].max()) < 4
    assert int(ancestors[:, 1].min()) >= 0
    assert int(ancestors[:, 1].max()) < 2


def _manual_hic_one_level(view_one: torch.Tensor, view_two: torch.Tensor,
                          ancestors: torch.Tensor, curvature: float,
                          temperature: float) -> torch.Tensor:
    distance_11 = _paper_mobius_distance(view_one, view_one, curvature)
    distance_12 = _paper_mobius_distance(view_one, view_two, curvature)
    distance_22 = _paper_mobius_distance(view_two, view_two, curvature)
    losses = []
    batch_size = view_one.shape[0]
    for anchor_view in (0, 1):
        for anchor in range(batch_size):
            positive_distance = (distance_12[anchor, anchor]
                                 if anchor_view == 0
                                 else distance_12[anchor, anchor])
            scores = [-positive_distance / temperature]
            for negative in range(batch_size):
                if ancestors[negative] == ancestors[anchor]:
                    continue
                if anchor_view == 0:
                    scores.extend((
                        -distance_12[anchor, negative] / temperature,
                        -distance_11[anchor, negative] / temperature,
                    ))
                else:
                    scores.extend((
                        -distance_12[negative, anchor] / temperature,
                        -distance_22[anchor, negative] / temperature,
                    ))
            scores = torch.stack(scores)
            losses.append(-(scores[0] - torch.logsumexp(scores, dim=0)))
    # Eq. (11): 1/B times the sum over the two anchor views.
    return torch.stack(losses).sum() / batch_size


def _check_hierarchical_instance_and_prototype_losses_match_equations():
    curvature = 0.01
    temperature = 0.2
    view_one = poincare_expmap0(torch.tensor(
        [[0.2, 0.0], [0.22, 0.01], [-0.3, 0.1]], dtype=torch.float64),
        curvature, clip_radius=None)
    view_two = poincare_expmap0(torch.tensor(
        [[0.19, -0.01], [0.23, 0.02], [-0.28, 0.08]], dtype=torch.float64),
        curvature, clip_radius=None)
    ancestors = torch.tensor([[0], [0], [1]])

    actual_instance = hhch_hierarchical_instance_loss(
        view_one, view_two, ancestors,
        temperature=temperature, curvature=curvature)
    expected_instance = _manual_hic_one_level(
        view_one, view_two, ancestors[:, 0], curvature, temperature)
    torch.testing.assert_close(
        actual_instance, expected_instance, rtol=1e-8, atol=1e-9)

    centers = [torch.stack((
        poincare_einstein_midpoint(torch.cat((view_one[:2], view_two[:2])), curvature),
        poincare_einstein_midpoint(torch.cat((view_one[2:], view_two[2:])), curvature),
    ))]
    actual_prototype = hhch_hierarchical_prototype_loss(
        view_one, view_two, ancestors, centers,
        temperature=temperature, curvature=curvature)
    logits_one = -_paper_mobius_distance(
        view_one, centers[0], curvature) / temperature
    logits_two = -_paper_mobius_distance(
        view_two, centers[0], curvature) / temperature
    expected_prototype = (F.cross_entropy(logits_one, ancestors[:, 0])
                          + F.cross_entropy(logits_two, ancestors[:, 0]))
    torch.testing.assert_close(
        actual_prototype, expected_prototype, rtol=1e-8, atol=1e-9)


def _check_paper_logcosh_quantization_and_total_loss_have_finite_gradients():
    torch.manual_seed(13)
    first_codes = torch.tanh(torch.randn(4, 6, requires_grad=True))
    second_codes = torch.tanh(torch.randn(4, 6, requires_grad=True))
    expected_quantization = 0.5 * (
        torch.log(torch.cosh(first_codes.abs() - 1.0)).sum()
        + torch.log(torch.cosh(second_codes.abs() - 1.0)).sum())
    actual_quantization = hhch_logcosh_quantization_loss(
        first_codes, second_codes)
    torch.testing.assert_close(actual_quantization, expected_quantization)

    projection = torch.nn.Linear(6, 3)
    first_hyperbolic = poincare_expmap0(projection(first_codes), 0.01,
                                        clip_radius=2.3)
    second_hyperbolic = poincare_expmap0(projection(second_codes), 0.01,
                                         clip_radius=2.3)
    ancestors = torch.tensor([[0], [0], [1], [1]])
    centers = [torch.stack((
        poincare_einstein_midpoint(first_hyperbolic[:2].detach(), 0.01),
        poincare_einstein_midpoint(first_hyperbolic[2:].detach(), 0.01),
    ))]
    total, parts = hhch_loss(
        first_codes, second_codes, first_hyperbolic, second_hyperbolic,
        ancestors, centers, temperature=0.2, curvature=0.01,
        quantization_weight=0.01)
    assert set(parts) == {
        'hierarchical_instance', 'hierarchical_prototype', 'quantization'}
    assert torch.isfinite(total)
    total.backward()
    assert all(parameter.grad is not None
               and bool(torch.isfinite(parameter.grad).all())
               for parameter in projection.parameters())


def _check_hhch_model_and_protocol_do_not_request_training_labels():
    model = HHCHFeatureHash(7, 36, hyper_dim=128, curvature=0.01,
                            dropout=0.0, clip_radius=2.3)
    features = torch.randn(5, 7)
    output = model(features)
    assert output['continuous_code'].shape == (5, 36)
    assert bool((output['continuous_code'].abs() <= 1.0).all())
    projected = model.project_features(features)
    assert projected.shape == (5, 128)
    assert bool((projected.norm(dim=1) < 10.0).all())

    fixed = HHCH()._get_fixed_config_dict()
    assert fixed['dataset_return_index'] is True
    assert fixed['dataset_return_paired_aug_img'] is True
    assert fixed['dataset_return_visual_tokens'] is False
    assert 'label' not in fixed
    assert _parse_cluster_counts('200,150,80') == (200, 150, 80)
    with unittest.TestCase().assertRaises(ValueError):
        _parse_cluster_counts('80,100')


def _check_release_lr_schedule_boundaries():
    assert HHCH._release_lr_multiplier(0) == 1.0
    assert HHCH._release_lr_multiplier(29) == 1.0
    assert HHCH._release_lr_multiplier(30) == 0.9
    assert HHCH._release_lr_multiplier(39) == 0.9
    assert HHCH._release_lr_multiplier(40) == 0.9 ** 2
    with unittest.TestCase().assertRaises(ValueError):
        HHCH._release_lr_multiplier(-1)


class HHCHBaselineTest(unittest.TestCase):
    def test_poincare_expmap_and_distance_match_paper_equations(self):
        _check_poincare_expmap_and_distance_match_paper_equations()

    def test_einstein_midpoint_of_symmetric_points_is_origin(self):
        _check_einstein_midpoint_of_symmetric_points_is_origin()

    def test_hyperbolic_kmeans_is_deterministic_and_separates_clouds(self):
        _check_hyperbolic_kmeans_plus_plus_is_deterministic_and_separates_clouds()

    def test_bottom_up_hierarchy_returns_original_sample_ancestors(self):
        _check_bottom_up_hierarchy_returns_original_sample_ancestors()

    def test_hierarchical_losses_match_paper_equations(self):
        _check_hierarchical_instance_and_prototype_losses_match_equations()

    def test_logcosh_total_loss_has_finite_gradients(self):
        _check_paper_logcosh_quantization_and_total_loss_have_finite_gradients()

    def test_model_and_protocol_do_not_request_training_labels(self):
        _check_hhch_model_and_protocol_do_not_request_training_labels()

    def test_release_lr_schedule_boundaries(self):
        _check_release_lr_schedule_boundaries()


if __name__ == '__main__':
    unittest.main()
