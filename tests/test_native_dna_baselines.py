import numpy as np
import torch
import torch.nn.functional as F

from baseline.native_dna import (
    Bee2018Encoder,
    DenseDNAEncoder,
    PRIMOYieldPredictor,
    bee2018_approximate_yield,
    bee2018_cosine_distance,
    canonical_indices_to_bits,
    entropy_regularizer,
    evaluate_base_retrieval,
    extraction_payload,
    keras_activity_entropy_loss,
    koike_break_homopolymers,
    koike_gc_balance_loss,
    koike_homopolymer_loss,
    koike_objective,
    pairwise_normalized_dna_distance,
    paper_indices_to_canonical,
    project_codes_memoized,
    sample_balanced_feature_pairs,
    sample_random_feature_pairs,
    semi_hard_triplet_loss,
)
from dna_utils.bio_constraints import is_valid_batch
from scripts.convert_primo_predictor import _to_pytorch_arrays


def test_dense_and_dna24_encoders_emit_simplex():
    dense = DenseDNAEncoder(input_dim=8, length=18, hidden_dim=4)
    dense_probs = dense(torch.randn(3, 8))
    assert dense_probs.shape == (3, 18, 4)
    assert torch.allclose(dense_probs.sum(-1), torch.ones(3, 18), atol=1e-6)

    dna24 = Bee2018Encoder(input_dim=8, length=18, pca_dim=3, channels=5)
    dna24_probs = dna24(torch.randn(3, 8))
    assert dna24_probs.shape == (3, 18, 4)
    assert torch.allclose(dna24_probs.sum(-1), torch.ones(3, 18), atol=1e-6)


def test_keras_activity_entropy_scaling_matches_one_and_two_encoder_calls():
    first = torch.full((5, 3, 4), 0.25)
    second = torch.full((5, 3, 4), 0.25)
    raw = entropy_regularizer(first)
    assert torch.allclose(keras_activity_entropy_loss(first), raw / 5)
    assert torch.allclose(keras_activity_entropy_loss(first, second), 2 * raw / 5)


def test_hard_koike_distance_equals_normalized_base_hamming():
    ids = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1], [3, 3, 3, 3]])
    onehot = F.one_hot(ids, num_classes=4).float()
    distance = pairwise_normalized_dna_distance(onehot)
    expected = (ids[:, None, :] != ids[None, :, :]).float().mean(dim=-1)
    assert torch.allclose(distance, expected)


def test_bee_cosine_and_sigmoid_golden_cases():
    first = F.one_hot(torch.tensor([[0, 1, 2, 3]]), num_classes=4).float()
    same = first.clone()
    different = F.one_hot(torch.tensor([[1, 2, 3, 0]]), num_classes=4).float()
    distance = bee2018_cosine_distance(
        torch.cat((first, first)), torch.cat((same, different))
    )
    assert torch.allclose(distance, torch.tensor([0.0, 1.0]))
    yields = bee2018_approximate_yield(distance)
    assert torch.allclose(yields.sum(), torch.tensor(1.0), atol=1e-6)
    assert yields[0] > 0.99 and yields[1] < 0.01


def test_semihard_triplet_prefers_near_positive_and_far_negative():
    labels = torch.tensor([0, 0, 1])
    good = torch.tensor(
        [[0.0, 0.1, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]],
        requires_grad=True,
    )
    bad = torch.tensor(
        [[0.0, 0.7, 0.2], [0.7, 0.0, 0.3], [0.2, 0.3, 0.0]],
        requires_grad=True,
    )
    assert semi_hard_triplet_loss(good, labels, margin=0.2) < semi_hard_triplet_loss(
        bad, labels, margin=0.2
    )


def test_vectorized_semihard_matches_reference_selection():
    torch.manual_seed(7)
    points = torch.randn(9, 5)
    reference_distance = torch.cdist(points, points).requires_grad_(True)
    actual_distance = reference_distance.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3])
    margin = 0.37

    adjacent = labels[:, None].eq(labels[None, :])
    positive = adjacent & ~torch.eye(len(labels), dtype=torch.bool)
    negative = ~adjacent
    reference_terms = []
    for anchor in range(len(labels)):
        negative_distance = reference_distance[anchor][negative[anchor]]
        if not positive[anchor].any() or not negative[anchor].any():
            continue
        for positive_id in torch.where(positive[anchor])[0]:
            positive_distance = reference_distance[anchor, positive_id]
            outside = negative_distance[negative_distance > positive_distance]
            selected = (
                outside.min() if outside.numel() else negative_distance.max()
            )
            reference_terms.append(
                F.relu(positive_distance - selected + margin)
            )
    reference = torch.stack(reference_terms).mean()
    actual = semi_hard_triplet_loss(actual_distance, labels, margin=margin)
    assert torch.allclose(actual, reference)
    reference.backward()
    actual.backward()
    assert torch.allclose(actual_distance.grad, reference_distance.grad)


def test_multilabel_semihard_handles_overlap_and_empty_anchor():
    labels = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    distance = torch.tensor(
        [[0.0, 0.1, 0.4, 0.8], [0.1, 0.0, 0.2, 0.7],
         [0.4, 0.2, 0.0, 0.6], [0.8, 0.7, 0.6, 0.0]],
        requires_grad=True,
    )
    loss = semi_hard_triplet_loss(distance, labels, margin=0.2)
    assert torch.isfinite(loss)
    loss.backward()


def test_tcbb_bio_losses_and_inference_heuristic():
    alternating = F.one_hot(torch.tensor([[0, 2, 0, 2, 0, 2]]), 4).float()
    repeated = F.one_hot(torch.tensor([[0, 0, 0, 0, 0, 2]]), 4).float()
    assert koike_homopolymer_loss(repeated, max_run=3) > koike_homopolymer_loss(
        alternating, max_run=3
    )
    assert torch.allclose(koike_gc_balance_loss(alternating), torch.tensor(0.0))

    probs = repeated.numpy() * 0.9 + 0.1 / 4
    fixed = koike_break_homopolymers(probs, max_run=3)
    run = 1
    longest = 1
    for position in range(1, fixed.shape[1]):
        run = run + 1 if fixed[0, position] == fixed[0, position - 1] else 1
        longest = max(longest, run)
    assert longest <= 3


def test_koike_objective_combines_all_source_terms():
    probs = torch.softmax(torch.randn(4, 6, 4), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    losses = koike_objective(
        probs, labels, method="koike2026", entropy_strength=0.01,
        max_run=3, hp_weight=1.3, gc_weight=0.7,
    )
    expected = (
        losses.triplet
        + 0.01 * losses.entropy / probs.shape[0]
        + 1.3 * (losses.homopolymer + 0.01 * losses.probability)
        + 0.7 * losses.gc
    )
    assert torch.allclose(losses.total, expected)


def test_base_order_serialization_and_common_projection():
    paper = np.array([[0, 1, 2, 3]])  # A,T,C,G
    canonical = paper_indices_to_canonical(paper)
    assert canonical.tolist() == [[0, 3, 1, 2]]  # A,T,C,G under A,C,G,T ids
    assert canonical_indices_to_bits(canonical).tolist() == [[0, 0, 1, 1, 0, 1, 1, 0]]

    invalid = np.array([[0] * 18, [2] * 18], dtype=np.int64)
    result = project_codes_memoized(invalid, gc_min=0.4, gc_max=0.6, max_run=3)
    assert result["post_compliance"] == 1.0
    assert is_valid_batch(result["projected_codes"], 0.4, 0.6, 3).all()

    # Fractional bounds are converted with ceil/floor: an 18-mer therefore
    # admits exactly 8--10 GC bases under the shared [0.4, 0.6] rule.
    gc7 = np.array([[1, 0] * 7 + [0, 3, 0, 3]], dtype=np.int64)
    gc8 = np.array([[1, 0] * 8 + [0, 3]], dtype=np.int64)
    assert not is_valid_batch(gc7, 0.4, 0.6, 3).item()
    assert is_valid_batch(gc8, 0.4, 0.6, 3).item()

    six_as = np.array([[0] * 6], dtype=np.int64)
    minimum = project_codes_memoized(six_as, gc_min=0.5, gc_max=0.5, max_run=3)
    assert minimum["edit_distances"].tolist() == [3]


def test_primo_predictor_accepts_18_and_80_nt():
    predictor = PRIMOYieldPredictor()
    for length in (18, 80):
        first = torch.softmax(torch.randn(2, length, 4), dim=-1)
        second = torch.softmax(torch.randn(2, length, 4), dim=-1)
        logits = predictor(first, second)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()


def test_primo_converter_transposes_keras_weights_to_pytorch_layout():
    conv = np.arange(3 * 36 * 36, dtype=np.float64).reshape(3, 36, 36)
    dense = np.arange(36, dtype=np.float64).reshape(36, 1)
    arrays = {
        "model_weights/conv1d/conv1d/kernel:0": conv,
        "model_weights/conv1d/conv1d/bias:0": np.arange(36),
        "model_weights/logit/logit/kernel:0": dense,
        "model_weights/logit/logit/bias:0": np.array([7.0]),
    }
    converted, names = _to_pytorch_arrays(arrays)
    assert converted["conv_weight"].shape == (36, 36, 3)
    assert np.array_equal(converted["conv_weight"], conv.transpose(2, 1, 0))
    assert converted["dense_weight"].shape == (1, 36)
    assert np.array_equal(converted["dense_weight"], dense.T)
    assert all(value.dtype == np.float32 for value in converted.values())
    assert names["conv_kernel"].endswith("kernel:0")


def test_primo_local_interaction_axis_order_matches_manual_outer_products():
    first = torch.arange(44, dtype=torch.float32).reshape(1, 11, 4)
    second = torch.arange(44, 88, dtype=torch.float32).reshape(1, 11, 4)
    actual = PRIMOYieldPredictor.local_interactions(first, second)
    expected_positions = []
    for center in range(1, 10):
        channels = []
        for base in range(4):
            left = first[0, center - 1:center + 2, base]
            right = second[0, center - 1:center + 2, base]
            channels.append(torch.outer(left, right).reshape(-1))
        expected_positions.append(torch.cat(channels))
    expected = torch.stack(expected_positions).unsqueeze(0)
    assert torch.equal(actual, expected)


def test_bee_pair_samplers_match_primo_pooling_and_avoid_within_chunk_reuse():
    features = np.arange(200, dtype=np.float32).reshape(100, 2)
    rng = np.random.RandomState(7)
    pairs, _, _ = sample_random_feature_pairs(
        features, threshold=10.0, batch_size=20, rng=rng
    )
    assert len(np.unique(pairs.reshape(-1))) == 40
    assert np.all(pairs[:, 0] != pairs[:, 1])

    # Literal reference to PRIMO Dataset.balanced_pairs: each random chunk
    # contributes all positives and an equal number of negatives, then the
    # possibly oversized pool is shuffled and truncated.  Seed 2 deliberately
    # overshoots, demonstrating that the final batch need not be exactly 50/50.
    threshold = 20.0
    batch_size = 10
    reference_rng = np.random.RandomState(2)
    reference_pool = []
    balanced_count = 0
    while balanced_count < batch_size:
        candidate = reference_rng.permutation(len(features))[
            : 2 * batch_size
        ].reshape(-1, 2)
        delta = features[candidate[:, 0]] - features[candidate[:, 1]]
        similar = np.sqrt(np.square(delta).sum(axis=1)) <= threshold
        n_similar = int(similar.sum())
        reference_pool.extend(
            (candidate[similar], candidate[~similar][:n_similar])
        )
        balanced_count += 2 * n_similar
    reference_pool = np.concatenate(reference_pool, axis=0)
    expected_pairs = reference_pool[
        reference_rng.permutation(len(reference_pool))[:batch_size]
    ]
    expected_delta = (
        features[expected_pairs[:, 0]] - features[expected_pairs[:, 1]]
    )
    expected_targets = (
        np.sqrt(np.square(expected_delta).sum(axis=1)) <= threshold
    ).astype(np.float32)

    rng = np.random.RandomState(2)
    pairs, targets, _ = sample_balanced_feature_pairs(
        features, threshold=threshold, batch_size=batch_size, rng=rng
    )
    assert np.array_equal(pairs, expected_pairs)
    assert np.array_equal(targets, expected_targets)
    assert targets.sum() == 6
    assert np.all(pairs[:, 0] != pairs[:, 1])


def test_base_hamming_map_and_extraction_raw_semantics():
    query = np.array([[0, 0]], dtype=np.int64)
    database = np.array([[0, 0], [0, 1], [3, 3]], dtype=np.int64)
    query_labels = np.array([[1, 0]], dtype=np.int64)
    database_labels = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int64)
    metrics = evaluate_base_retrieval(
        query, database, query_labels, database_labels, map_at_r=3
    )
    assert np.isclose(metrics["mAP_at_R"], (1.0 + 2.0 / 3.0) / 2.0)

    deployment = np.array([[0, 1, 2]], dtype=np.int64)
    neural_raw = np.array([[0, 0, 2]], dtype=np.int64)
    payload = extraction_payload(
        deployment, np.array([[1]]), ["x"],
        soft_probs=np.zeros((1, 3, 4)), neural_raw_base_indices=neural_raw,
    )
    assert np.array_equal(payload["base_indices"], deployment)
    assert np.array_equal(payload["neural_raw_base_indices"], neural_raw)
