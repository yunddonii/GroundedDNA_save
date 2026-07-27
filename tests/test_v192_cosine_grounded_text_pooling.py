import unittest

import torch
from torch import nn

from model_siglip2 import (
    NUM_LOCAL_PARTS,
    NUM_SEMANTIC_PARTS,
    SigLIP2SemanticOTModel,
)


def _minimal_model(d_model: int = 8) -> SigLIP2SemanticOTModel:
    model = SigLIP2SemanticOTModel.__new__(SigLIP2SemanticOTModel)
    nn.Module.__init__(model)
    model.d_model = d_model
    model.codebook_text_prompts = None
    model.text_embed_transform = "none"
    model.text_adapter = nn.ModuleList([
        nn.Identity() for _ in range(NUM_SEMANTIC_PARTS)
    ])
    model.cosine_grounded_text_gate = nn.Parameter(
        torch.zeros(NUM_LOCAL_PARTS)
    )
    return model


class V192CosineGroundedTextPoolingTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        batch, patches, text_len, dim = 2, 4, 5, 8
        visual = torch.randn(batch, patches, dim, requires_grad=True)
        routing = torch.rand(
            batch, patches, NUM_SEMANTIC_PARTS, requires_grad=True,
        )
        base_text = torch.randn(
            batch, NUM_SEMANTIC_PARTS, dim, requires_grad=True,
        )
        cached_text = torch.randn(
            batch, NUM_SEMANTIC_PARTS, text_len, dim, requires_grad=True,
        )
        text_mask = torch.ones(
            batch, NUM_SEMANTIC_PARTS, text_len, dtype=torch.bool,
        )
        text_mask[:, :, -1] = False
        return visual, routing, base_text, cached_text, text_mask

    def test_zero_gate_is_exact_baseline_and_receives_gradient(self):
        model = _minimal_model()
        visual, routing, base_text, cached_text, text_mask = self._inputs()

        refined, entropy, support = model._cosine_visual_grounded_text_pooling(
            visual_tokens=visual,
            routing_matrix=routing,
            base_text_tokens=base_text,
            cached_text_tokens=cached_text,
            cached_text_token_mask=text_mask,
        )

        self.assertTrue(torch.equal(refined, base_text))
        self.assertEqual(entropy.shape, (2, NUM_LOCAL_PARTS))
        self.assertEqual(support.shape, (2, NUM_LOCAL_PARTS))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(support).all())
        self.assertTrue(((entropy >= 0.0) & (entropy <= 1.0 + 1e-6)).all())
        self.assertTrue(((support > 0.0) & (support <= 1.0 + 1e-6)).all())

        refined[:, 1:].square().mean().backward()
        self.assertIsNotNone(model.cosine_grounded_text_gate.grad)
        self.assertTrue(torch.isfinite(model.cosine_grounded_text_gate.grad).all())
        self.assertIsNone(visual.grad)
        self.assertIsNone(routing.grad)

    def test_nonzero_gate_trains_all_token_text_pool_only(self):
        model = _minimal_model()
        with torch.no_grad():
            model.cosine_grounded_text_gate.fill_(0.25)
        visual, routing, base_text, cached_text, text_mask = self._inputs()

        refined, _, _ = model._cosine_visual_grounded_text_pooling(
            visual_tokens=visual,
            routing_matrix=routing,
            base_text_tokens=base_text,
            cached_text_tokens=cached_text,
            cached_text_token_mask=text_mask,
        )

        self.assertEqual(refined.shape, base_text.shape)
        self.assertTrue(torch.equal(refined[:, 0], base_text[:, 0]))
        self.assertFalse(torch.allclose(refined[:, 1:], base_text[:, 1:]))
        refined[:, 1:].sum().backward()
        self.assertIsNotNone(cached_text.grad)
        self.assertIsNotNone(base_text.grad)
        self.assertIsNone(visual.grad)
        self.assertIsNone(routing.grad)


if __name__ == "__main__":
    unittest.main()
