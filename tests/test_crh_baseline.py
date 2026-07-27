import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from baseline.CRH import (
    CRH,
    CRHFeatureHash,
    adaptive_head_dim,
    compose_class_centers,
    crh_margin_cross_entropy,
    crh_objective,
    crh_should_reassign,
    crh_tanh_quantization,
    crh_weighted_reassignment_cost,
    generate_multi_head_codebook,
    greedy_distinct_assignments,
    prepare_crh_supervised_training_dataset,
)
from baseline.base_model import Logger, build_model_from_checkpoint_payload
from scripts.run_modern_baseline_p0 import (
    VARIANTS,
    _audit_semantic_condition,
    _source_horizon,
)


class _TinyLabelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "img": torch.from_numpy(self.features[index]),
            "label": torch.from_numpy(self.labels[index]),
            "image_path": f"tiny:{index}",
        }


class _ForbiddenEvaluationDataset:
    @property
    def labels(self):
        raise AssertionError("CRH optimization accessed evaluation labels")


class CRHMathTest(unittest.TestCase):
    def test_adaptive_head_dimension_matches_capacity_constraint(self) -> None:
        # Flickr25k: C=24, M=48.  MSCOCO: C=80, M=160.
        self.assertEqual(adaptive_head_dim(36, 48), 6)
        self.assertEqual(adaptive_head_dim(48, 48), 6)
        self.assertEqual(adaptive_head_dim(36, 160), 9)
        self.assertEqual(adaptive_head_dim(48, 160), 8)
        with self.assertRaisesRegex(ValueError, "positive"):
            adaptive_head_dim(0, 48)

    def test_generated_subcodebooks_are_seeded_unique_and_binary(self) -> None:
        first = generate_multi_head_codebook(36, 48, seed=7)
        second = generate_multi_head_codebook(36, 48, seed=7)
        other = generate_multi_head_codebook(36, 48, seed=8)
        self.assertEqual(first.shape, (6, 48, 6))
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(set(first.unique().tolist()), {-1.0, 1.0})
        for head in first:
            self.assertEqual(torch.unique(head, dim=0).shape[0], 48)

    def test_multi_head_center_composition_is_exact(self) -> None:
        codebook = torch.tensor([
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]],
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]],
        ])
        assignments = torch.tensor([[2, 0], [1, 2]])
        actual = compose_class_centers(codebook, assignments)
        expected = torch.tensor([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
        ])
        torch.testing.assert_close(actual, expected)

    def test_margin_cross_entropy_matches_scalar_equations(self) -> None:
        logits = torch.tensor([[2.0, 1.0], [-1.0, 2.0]])
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        scale = 2.25
        margin = 0.2
        actual = crh_margin_cross_entropy(
            logits, centers, labels, scale=scale, margin=margin
        )

        # Independent scalar transcription of paper Eqs. (1)--(2).
        total = 0.0
        for vector, target in zip(logits.tolist(), labels.tolist()):
            norm = math.sqrt(sum(value * value for value in vector))
            scores = []
            for class_index, center in enumerate(centers.tolist()):
                center_norm = math.sqrt(
                    sum(value * value for value in center)
                )
                cosine = sum(
                    left * right for left, right in zip(vector, center)
                ) / (norm * center_norm)
                scores.append(
                    scale * (cosine - target[class_index] * margin)
                )
            denominator = sum(math.exp(score) for score in scores)
            cardinality = sum(target)
            sample = 0.0
            for class_index, positive in enumerate(target):
                if positive:
                    sample -= (
                        positive
                        / cardinality
                        * math.log(math.exp(scores[class_index]) / denominator)
                    )
            total += sample
        expected = total / len(logits)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_quantization_and_total_objective_match_equations(self) -> None:
        logits = torch.tensor([[0.0, 0.5], [-1.0, 2.0]])
        expected_q = sum(
            (abs(math.tanh(value)) - 1.0) ** 2
            for value in logits.flatten().tolist()
        ) / logits.numel()
        actual_q = crh_tanh_quantization(logits)
        self.assertAlmostEqual(actual_q.item(), expected_q, places=7)

        centers = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        total, parts = crh_objective(
            logits,
            centers,
            labels,
            scale=3.0,
            margin=0.2,
            quantization_weight=0.1,
        )
        torch.testing.assert_close(
            total,
            parts["margin_cross_entropy"] + 0.1 * parts["quantization"],
        )

    def test_weighted_multilabel_cost_matches_direct_eq7_loops(self) -> None:
        codes = torch.tensor([
            [1.0, -2.0, 3.0, -4.0],
            [-1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, -3.0, -4.0],
        ])
        labels = torch.tensor([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        codebook = torch.tensor([
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
        ])
        actual = crh_weighted_reassignment_cost(codes, labels, codebook)

        # Direct Eq. (7): weighted average of squared distances for each
        # (head, class, candidate), without the vectorized identity used by
        # the implementation.
        binary = torch.where(codes >= 0, 1.0, -1.0).reshape(3, 2, 2)
        expected = torch.empty_like(actual)
        for head in range(2):
            for class_index in range(2):
                members = [
                    row for row in range(3)
                    if labels[row, class_index].item() == 1.0
                ]
                denominator = sum(
                    1.0 / labels[row].sum().item() for row in members
                )
                for candidate in range(3):
                    numerator = 0.0
                    for row in members:
                        weight = 1.0 / labels[row].sum().item()
                        squared_distance = sum(
                            (
                                binary[row, head, dimension].item()
                                - codebook[
                                    head, candidate, dimension
                                ].item()
                            ) ** 2
                            for dimension in range(2)
                        )
                        numerator += weight * squared_distance
                    expected[head, class_index, candidate] = (
                        numerator / denominator
                    )
        torch.testing.assert_close(actual, expected)

    def test_greedy_assignment_is_seeded_distinct_and_cost_consistent(self) -> None:
        costs = torch.tensor([
            [[0.0, 4.0, 8.0, 9.0],
             [0.0, 1.0, 7.0, 8.0],
             [0.0, 1.0, 2.0, 3.0]],
            [[2.0, 0.0, 7.0, 9.0],
             [1.0, 0.0, 3.0, 8.0],
             [0.0, 1.0, 2.0, 7.0]],
        ])
        first, first_cost = greedy_distinct_assignments(costs, seed=19)
        second, second_cost = greedy_distinct_assignments(costs, seed=19)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_cost, second_cost)
        for head in range(costs.shape[0]):
            self.assertEqual(torch.unique(first[head]).numel(), 3)
            direct = sum(
                costs[head, class_index, first[head, class_index]].item()
                for class_index in range(3)
            )
            self.assertAlmostEqual(first_cost[head].item(), direct)

    def test_update_schedule_is_first_twenty_then_every_five(self) -> None:
        self.assertTrue(all(crh_should_reassign(epoch) for epoch in range(20)))
        self.assertFalse(crh_should_reassign(20))
        self.assertFalse(crh_should_reassign(23))
        self.assertTrue(crh_should_reassign(24))
        self.assertTrue(crh_should_reassign(29))

    def test_zero_label_filter_is_training_only_and_preserves_source(self) -> None:
        dataset = _TinyLabelDataset(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float32),
        )
        original_labels = dataset.labels.copy()
        filtered, audit = prepare_crh_supervised_training_dataset(
            dataset, protocol_stage="P0_stage1_val_selection"
        )
        self.assertIsInstance(filtered, Subset)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(len(dataset), 4)
        np.testing.assert_array_equal(dataset.labels, original_labels)
        self.assertEqual(audit["crh_training_rows_total"], 4)
        self.assertEqual(audit["crh_training_rows_retained"], 3)
        self.assertEqual(
            audit["crh_training_zero_label_rows_dropped"], 1
        )
        self.assertEqual(
            audit["crh_training_partition_stage"],
            "P0_stage1_val_selection",
        )
        retained_labels = torch.stack([
            filtered[index]["label"] for index in range(len(filtered))
        ])
        self.assertTrue(torch.all(retained_labels.sum(dim=1) > 0))

    def test_zero_label_filter_rejects_missing_training_class(self) -> None:
        dataset = _TinyLabelDataset(
            np.zeros((3, 4), dtype=np.float32),
            np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "classes absent"):
            prepare_crh_supervised_training_dataset(
                dataset, protocol_stage="P0_stage2_refit_test"
            )


class CRHIntegrationTest(unittest.TestCase):
    def test_dynamic_state_is_registered_and_restored_from_checkpoint(self) -> None:
        model = CRHFeatureHash(
            d_in=5, bit=36, num_classes=10, seed=11, codebook_scale=2
        )
        replacement = torch.arange(9, -1, -1).expand(
            model.num_heads, -1
        ).clone()
        model.install_assignments(replacement)
        payload = {
            "config": {
                "method": "crh",
                "dataset": "CIFAR10",
                "setting": "setting1",
                "bit": 36,
                "seed": 11,
                "crh_codebook_scale": 2,
                "resolved_projection_dim": 5,
            },
            "model_state_dict": model.state_dict(),
        }
        restored = build_model_from_checkpoint_payload(payload, device="cpu")
        self.assertIsInstance(restored, CRHFeatureHash)
        torch.testing.assert_close(restored.codebook, model.codebook)
        torch.testing.assert_close(restored.assignments, replacement)
        torch.testing.assert_close(
            restored.current_class_centers, model.current_class_centers
        )

    def test_optimizer_and_scheduler_match_audited_source_settings(self) -> None:
        method = CRH()
        model = CRHFeatureHash(
            d_in=4, bit=36, num_classes=10, seed=3, codebook_scale=2
        )
        optimizer = method._init_optimizer(
            model,
            "adam",
            1e-4,
            adam_weight_decay=1e-5,
        )
        self.assertEqual(optimizer.defaults["betas"], (0.5, 0.999))
        self.assertEqual(optimizer.defaults["weight_decay"], 1e-5)
        scheduler = method._init_scheduler(
            optimizer,
            "coslr",
            schedule_horizon=300,
            max_epoch=12,
        )
        self.assertEqual(scheduler.T_max, 300)
        self.assertEqual(scheduler.eta_min, 1e-7)

    def test_driver_registers_crh_only_as_supervised_s_tier(self) -> None:
        specification = VARIANTS["crh-supervised"]
        self.assertEqual(specification["method"], "crh")
        self.assertEqual(specification["tier"], "S")
        self.assertEqual(specification["source_batch_size"], 128)
        self.assertFalse(specification["needs_aug"])
        self.assertEqual(
            _source_horizon(specification, "MSCOCO"), 30
        )
        self.assertEqual(
            _source_horizon(specification, "CIFAR10"), 300
        )
        condition, blockers = _audit_semantic_condition(
            type("Args", (), {
                "variant": "crh-supervised",
                "dataset": "Flickr25k",
            })(),
            cache_dir=Path("."),  # not consulted for CRH
        )
        self.assertEqual(condition["information_tier"], "S")
        self.assertEqual(condition["comparison_panel"], "supervised")
        self.assertTrue(condition["uses_training_labels_in_objective"])
        self.assertEqual(blockers, [])

    def test_training_drops_zero_rows_without_touching_eval_datasets(self) -> None:
        trainset = _TinyLabelDataset(
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32),
            np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float32),
        )
        forbidden = _ForbiddenEvaluationDataset()
        model = CRHFeatureHash(
            d_in=4, bit=6, num_classes=3, seed=5, codebook_scale=2
        )
        method = CRH()
        optimizer = method._init_optimizer(
            model, "adam", 1e-4, adam_weight_decay=1e-5
        )
        scheduler = method._init_scheduler(
            optimizer, "coslr", schedule_horizon=30, max_epoch=1
        )
        config = {
            "batch_size": 2,
            "num_workers": 0,
            "max_epoch": 1,
            "crh_margin": 0.2,
            "crh_scale": math.sqrt(2.0) * math.log(2),
            "crh_quantization_weight": 0.0,
            "crh_gradient_clip_norm": 1.0,
            "seed": 5,
            "protocol_stage": "P0_stage1_val_selection",
        }
        saved_payloads = []
        eval_calls = []
        with tempfile.TemporaryDirectory() as directory:
            method.result_dir = directory
            method.logger = Logger(directory, config)
            method.valset = forbidden
            method.testset = forbidden
            method.dbset = forbidden
            method.start_eval_process = (
                lambda model_id, id_type: eval_calls.append(
                    (model_id, id_type)
                )
            )
            method._save_train_model_params = (
                lambda model_id, id_type, etc_info=None:
                saved_payloads.append((model_id, id_type, etc_info))
            )
            method._train_model(
                model,
                optimizer,
                scheduler,
                trainset,
                forbidden,
                forbidden,
                directory,
                directory,
                "cpu",
                2,
                1,
                config,
            )
            log_text = (Path(directory) / "log.csv").read_text(
                encoding="utf-8"
            )
            config_snapshot = (
                Path(directory) / "config.json"
            ).read_text(encoding="utf-8")

        # Source partition and all evaluation sentinels remain untouched.
        self.assertEqual(len(trainset), 4)
        self.assertEqual(eval_calls, [("000", "epoch")])
        self.assertEqual(config["crh_training_rows_total"], 4)
        self.assertEqual(config["crh_training_rows_retained"], 3)
        self.assertEqual(
            config["crh_training_zero_label_rows_dropped"], 1
        )
        self.assertIn("train_crh_training_rows_total", log_text)
        self.assertIn("train_crh_training_zero_label_rows_dropped", log_text)
        self.assertIn('"crh_training_rows_retained": 3', config_snapshot)
        self.assertEqual(len(saved_payloads), 1)
        checkpoint_audit = saved_payloads[0][2][
            "crh_training_partition_audit"
        ]
        self.assertEqual(
            checkpoint_audit["crh_training_zero_label_rows_dropped"], 1
        )


if __name__ == "__main__":
    unittest.main()
