import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.eval_heldout_caption_foils import (
    FIXED_MARGIN,
    LOCAL_SLOTS,
    N_SLOTS,
    PAIR_METRICS,
    CacheBundle,
    FoilEntry,
    FrozenEvaluationGuard,
    PairSpec,
    _hash_model_state,
    aggregate_records,
    build_pair_specs,
    compare_run_records,
    compute_pair_records,
    encode_text_minimal_pairs,
    exact_surface_context_signature,
    generator_rule_edit_template_signature,
    pair_set_sha256,
    _validate_edit_span,
)


SLOT_KEYS = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)


def _foil_entry(
    image_id: str,
    *,
    context: str = "a dog on grass",
    source_atom: str = "cat",
    target_atom: str = "dog",
    family: str = "object",
    contrast_set: str = "pet",
) -> FoilEntry:
    start = context.index(target_atom)
    edits = [
        {"valid": False, "reason": "global_slot_excluded", "role": "global"}
    ]
    edits.append(
        {
            "valid": True,
            "operation": "single_span_contrast_substitution",
            "role": "primary",
            "family": family,
            "contrast_set": contrast_set,
            "source_atom": source_atom,
            "target_atom": target_atom,
            "source_span": [start, start + len(source_atom)],
            "target_span": [start, start + len(target_atom)],
            "target_absent_from_all_factual_captions": True,
        }
    )
    edits.extend(
        {"valid": False, "reason": "unsupported", "role": "other"}
        for _ in range(4)
    )
    return FoilEntry(
        image_id=image_id,
        slot_keys=SLOT_KEYS,
        texts=("", context, "", "", "", ""),
        valid=(False, True, False, False, False, False),
        edits=tuple(edits),
    )


def _cache_for_entries(entries) -> CacheBundle:
    image_ids = [entry.image_id for entry in entries]
    n_rows = len(entries)
    valid = np.asarray([entry.valid for entry in entries], dtype=bool)
    token_mask = np.zeros((n_rows, N_SLOTS, 1), dtype=bool)
    token_mask[:, 1, 0] = valid[:, 1]
    return CacheBundle(
        root=Path("/synthetic/cache"),
        image_ids=image_ids,
        visual_tokens=np.zeros((n_rows, 2, 4), dtype=np.float16),
        visual_global=np.zeros((n_rows, 4), dtype=np.float16),
        text_part=np.zeros((n_rows, N_SLOTS, 4), dtype=np.float16),
        has_text=np.ones(n_rows, dtype=bool),
        text_tokens=np.zeros((n_rows, N_SLOTS, 1, 4), dtype=np.float16),
        text_token_mask=np.ones((n_rows, N_SLOTS, 1), dtype=bool),
        text_foil_part=np.zeros((n_rows, N_SLOTS, 4), dtype=np.float16),
        text_foil_valid=valid,
        text_foil_tokens=np.zeros(
            (n_rows, N_SLOTS, 1, 4), dtype=np.float16
        ),
        text_foil_token_mask=token_mask,
        opt_train_rows=np.asarray([0], dtype=np.int64),
        train_all_rows=np.arange(n_rows, dtype=np.int64),
        heldout_rows=np.arange(1, n_rows, dtype=np.int64),
    )


class _FakeTextDNAModel:
    def __init__(self):
        self.adapt_inputs = []
        self.encode_kwargs = []

    def _pool_local_foil_tokens_like_factual(
        self, tokens, token_mask, visual_tokens_raw, visual_attention_mask
    ):
        del token_mask, visual_tokens_raw, visual_attention_mask
        return tokens[:, 1:, 0, :]

    def _adapt_pooled_text_for_loss(self, raw):
        self.adapt_inputs.append(raw.detach().clone())
        return raw

    def _encode_text_tokens_to_dna(
        self, text_tokens, *, allow_mm_ema, deterministic_codon
    ):
        self.encode_kwargs.append((allow_mm_ema, deterministic_codon))
        logits = text_tokens[..., :4]
        continuous = torch.softmax(logits, dim=-1).unsqueeze(2).repeat(1, 1, 2, 1)
        return {
            "continuous_code": continuous.reshape(
                text_tokens.shape[0], N_SLOTS * 2, 4
            ),
            "codebook_indices": torch.argmax(logits, dim=-1),
        }


class HeldoutCaptionFoilEvalTest(unittest.TestCase):
    def test_surface_and_generator_rule_overlap_are_separate(self):
        train = _foil_entry("train-image")
        heldout_overlap = _foil_entry("heldout-overlap")
        heldout_new_surface_same_rule = _foil_entry(
            "heldout-disjoint", context="the dog beside water"
        )
        heldout_new_rule = _foil_entry(
            "heldout-new-rule",
            context="a truck beside water",
            source_atom="car",
            target_atom="truck",
            contrast_set="road_vehicle",
        )
        all_entries = (
            train,
            heldout_overlap,
            heldout_new_surface_same_rule,
            heldout_new_rule,
        )
        cache = _cache_for_entries(all_entries)
        entries = {
            entry.image_id: entry
            for entry in all_entries
        }

        pairs = build_pair_specs(cache, entries)

        self.assertEqual(len(pairs), 3)
        self.assertTrue(pairs[0].exact_surface_context_overlap_opt_train)
        self.assertTrue(
            pairs[0].generator_rule_edit_template_overlap_opt_train
        )
        self.assertFalse(pairs[1].exact_surface_context_overlap_opt_train)
        self.assertTrue(
            pairs[1].generator_rule_edit_template_overlap_opt_train
        )
        self.assertFalse(pairs[2].exact_surface_context_overlap_opt_train)
        self.assertFalse(
            pairs[2].generator_rule_edit_template_overlap_opt_train
        )
        self.assertEqual(
            exact_surface_context_signature(train, 1),
            exact_surface_context_signature(heldout_overlap, 1),
        )
        self.assertNotEqual(
            exact_surface_context_signature(train, 1),
            exact_surface_context_signature(heldout_new_surface_same_rule, 1),
        )
        self.assertEqual(
            generator_rule_edit_template_signature(train, 1),
            generator_rule_edit_template_signature(
                heldout_new_surface_same_rule, 1
            ),
        )
        self.assertNotEqual(
            generator_rule_edit_template_signature(train, 1),
            generator_rule_edit_template_signature(heldout_new_rule, 1),
        )
        self.assertEqual(pair_set_sha256(pairs), pair_set_sha256(list(pairs)))

    def test_valid_edit_requires_target_absence_contract(self):
        entry = _foil_entry("image")
        bad_edit = dict(entry.edits[1])
        bad_edit.pop("target_absent_from_all_factual_captions")
        with self.assertRaisesRegex(ValueError, "target-absence contract"):
            _validate_edit_span(
                Path("fixture.jsonl"),
                1,
                SLOT_KEYS[1],
                entry.texts[1],
                bad_edit,
            )

    def test_global_slot_never_enters_pair_metrics(self):
        pair = PairSpec(
            pair_id="global",
            cache_row=0,
            image_id="image",
            slot=0,
            slot_name="C_global",
            family="none",
            contrast_set="none",
            source_atom="",
            target_atom="",
            exact_surface_context_sha256="0" * 64,
            exact_surface_context_overlap_opt_train=False,
            generator_rule_edit_template_sha256="1" * 64,
            generator_rule_edit_template_overlap_opt_train=False,
        )
        continuous = np.zeros((1, N_SLOTS, 2, 4), dtype=np.float32)
        continuous[..., 0] = 1.0
        codeword = np.zeros((1, N_SLOTS), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "C_global"):
            compute_pair_records(
                run_label="A",
                row_order=[0],
                pair_specs=[pair],
                visual_continuous=continuous,
                visual_codeword=codeword,
                factual_continuous=continuous,
                factual_codeword=codeword,
                foil_continuous=continuous,
                foil_codeword=codeword,
            )

    def test_one_slot_world_keeps_factual_global_and_other_slots(self):
        model = _FakeTextDNAModel()
        batch = 1
        factual_raw = torch.zeros(batch, N_SLOTS, 4)
        factual_raw[:, 0, 0] = 9.0
        factual_tokens = torch.zeros(batch, N_SLOTS, 1, 4)
        foil_tokens = torch.zeros_like(factual_tokens)
        for slot in LOCAL_SLOTS:
            factual_tokens[:, slot, 0, slot % 4] = float(slot)
            foil_tokens[:, slot, 0, (slot + 1) % 4] = float(slot + 10)
        factual_mask = torch.ones(batch, N_SLOTS, 1, dtype=torch.bool)
        foil_mask = torch.ones_like(factual_mask)
        foil_mask[:, 0] = False
        visual = torch.zeros(batch, 2, 4)

        output = encode_text_minimal_pairs(
            model,
            factual_raw=factual_raw,
            factual_tokens=factual_tokens,
            factual_token_mask=factual_mask,
            foil_tokens=foil_tokens,
            foil_token_mask=foil_mask,
            visual_tokens_raw=visual,
        )

        factual_pooled = output["factual_pooled"]
        worlds = output["foil_worlds"]
        self.assertEqual(tuple(worlds.shape), (1, 5, 6, 4))
        for world_index, target_slot in enumerate(LOCAL_SLOTS):
            self.assertTrue(
                torch.equal(worlds[:, world_index, 0], factual_pooled[:, 0])
            )
            for slot in LOCAL_SLOTS:
                expected = (
                    foil_tokens[:, slot, 0]
                    if slot == target_slot
                    else factual_pooled[:, slot]
                )
                self.assertTrue(
                    torch.equal(worlds[:, world_index, slot], expected),
                    msg=f"world={world_index}, slot={slot}",
                )
        self.assertEqual(model.encode_kwargs, [(False, True), (False, True)])
        self.assertEqual(
            tuple(output["foil_target_continuous"].shape), (1, 6, 2, 4)
        )

    def test_pair_metrics_have_expected_direction_and_hamming(self):
        visual = np.zeros((1, N_SLOTS, 2, 4), dtype=np.float32)
        factual = np.zeros_like(visual)
        foil = np.zeros_like(visual)
        visual[..., 0] = 1.0
        factual[..., 0] = 1.0
        foil[..., 0] = 1.0
        foil[0, 1, 0] = np.asarray([0.0, 1.0, 0.0, 0.0])
        visual_cw = np.zeros((1, N_SLOTS), dtype=np.int64)
        factual_cw = np.zeros_like(visual_cw)
        foil_cw = np.zeros_like(visual_cw)
        visual_cw[0, 1] = factual_cw[0, 1] = 7
        foil_cw[0, 1] = 8
        pair = PairSpec(
            pair_id="pair",
            cache_row=11,
            image_id="image",
            slot=1,
            slot_name=SLOT_KEYS[1],
            family="object",
            contrast_set="pet",
            source_atom="cat",
            target_atom="dog",
            exact_surface_context_sha256="a" * 64,
            exact_surface_context_overlap_opt_train=True,
            generator_rule_edit_template_sha256="c" * 64,
            generator_rule_edit_template_overlap_opt_train=True,
        )

        records = compute_pair_records(
            run_label="A",
            row_order=[11],
            pair_specs=[pair],
            visual_continuous=visual,
            visual_codeword=visual_cw,
            factual_continuous=factual,
            factual_codeword=factual_cw,
            foil_continuous=foil,
            foil_codeword=foil_cw,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertAlmostEqual(record["visual_factual_cosine"], 1.0)
        self.assertAlmostEqual(record["visual_foil_cosine"], 0.5)
        self.assertAlmostEqual(record["visual_cosine_gap"], 0.5)
        self.assertTrue(record["factual_preference"])
        self.assertFalse(record["margin_violation"])
        self.assertTrue(record["text_codeword_flip"])
        self.assertEqual(record["text_dna_base_hamming_count"], 1)
        self.assertAlmostEqual(record["text_dna_base_hamming_fraction"], 0.5)
        self.assertAlmostEqual(
            record["visual_text_base_distance_advantage"], 0.5
        )
        self.assertEqual(record["visual_codeword_match_advantage"], 1)
        self.assertNotIn("visual_codeword_flip", record)

    def test_margin_boundary_is_a_violation(self):
        pair = PairSpec(
            pair_id="pair",
            cache_row=0,
            image_id="image",
            slot=1,
            slot_name=SLOT_KEYS[1],
            family="object",
            contrast_set="pet",
            source_atom="cat",
            target_atom="dog",
            exact_surface_context_sha256="b" * 64,
            exact_surface_context_overlap_opt_train=False,
            generator_rule_edit_template_sha256="d" * 64,
            generator_rule_edit_template_overlap_opt_train=False,
        )
        continuous = np.zeros((1, N_SLOTS, 1, 4), dtype=np.float32)
        continuous[..., 0] = 1.0
        codeword = np.zeros((1, N_SLOTS), dtype=np.int64)
        record = compute_pair_records(
            run_label="A",
            row_order=[0],
            pair_specs=[pair],
            visual_continuous=continuous,
            visual_codeword=codeword,
            factual_continuous=continuous,
            factual_codeword=codeword,
            foil_continuous=continuous,
            foil_codeword=codeword,
            margin=FIXED_MARGIN,
        )[0]
        self.assertFalse(record["factual_preference"])
        self.assertTrue(record["margin_violation"])
        self.assertFalse(record["margin_satisfied"])

    def test_image_cluster_bootstrap_and_paired_delta_use_same_pairs(self):
        def record(pair_id, image_id, value):
            row = {
                "pair_id": pair_id,
                "image_id": image_id,
                "cache_row": int(pair_id[-1]),
                "slot": 1,
                "slot_name": SLOT_KEYS[1],
                "family": "object",
                "exact_surface_context_sha256": pair_id * 8,
                "exact_surface_context_overlap_opt_train": False,
                "generator_rule_edit_template_sha256": pair_id * 8,
                "generator_rule_edit_template_overlap_opt_train": True,
            }
            row.update({metric: float(value) for metric in PAIR_METRICS})
            return row

        left = [
            record("pair0", "image-a", 0.0),
            record("pair1", "image-a", 0.0),
            record("pair2", "image-b", 1.0),
        ]
        right = [
            record("pair0", "image-a", 0.5),
            record("pair1", "image-a", 0.5),
            record("pair2", "image-b", 1.5),
        ]

        summary = aggregate_records(
            left, n_bootstrap=100, seed=9, scope="test"
        )
        self.assertEqual(summary["n_pairs"], 3)
        self.assertEqual(summary["n_images"], 2)
        self.assertEqual(summary["bootstrap_unit"], "image_cluster")
        self.assertAlmostEqual(
            summary["metrics"]["visual_cosine_gap"]["mean"], 1.0 / 3.0
        )

        comparison = compare_run_records(
            "A", left, "AB", right, n_bootstrap=100, seed=9
        )
        delta = comparison["summary"]["overall"]["metrics"][
            "visual_cosine_gap"
        ]["mean"]
        self.assertAlmostEqual(delta, 0.5)
        self.assertEqual(comparison["direction"], "AB_minus_A")

        with self.assertRaisesRegex(ValueError, "pair universe differs"):
            compare_run_records(
                "A", left, "AB", right[:-1], n_bootstrap=10, seed=9
            )

    def test_frozen_guard_and_state_hash_detect_buffer_mutation(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("ema_probe", torch.zeros(2))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        state_before = _hash_model_state(model)
        clean_guard = FrozenEvaluationGuard(model, torch.device("cpu"))
        result = clean_guard.verify()
        self.assertTrue(result["buffers_and_ema_unchanged"])
        self.assertEqual(state_before, _hash_model_state(model))

        dirty_guard = FrozenEvaluationGuard(model, torch.device("cpu"))
        model.ema_probe.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "buffers/EMA"):
            dirty_guard.verify()
        self.assertNotEqual(state_before, _hash_model_state(model))


if __name__ == "__main__":
    unittest.main()
