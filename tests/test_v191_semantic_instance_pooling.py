import unittest

import torch
from torch import nn

from model_siglip2 import (
    NUM_SEMANTIC_PARTS,
    SigLIP2SemanticOTModel,
)


def _minimal_model(d_model: int = 8) -> SigLIP2SemanticOTModel:
    model = SigLIP2SemanticOTModel.__new__(SigLIP2SemanticOTModel)
    nn.Module.__init__(model)
    model.d_model = d_model
    model.codebook_text_prompts = None
    model.text_adapter = nn.ModuleList([
        nn.Identity() for _ in range(NUM_SEMANTIC_PARTS)
    ])
    model.soft_grounded_text_attn = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=2,
        batch_first=True,
        dropout=0.0,
    )
    model.soft_grounded_text_ln = nn.LayerNorm(d_model)
    model.cibhash_visual_projectors = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for _ in range(NUM_SEMANTIC_PARTS)
    ])
    return model


class V191SemanticInstancePoolingTest(unittest.TestCase):
    def test_soft_visual_grounded_pool_shapes_and_stop_gradient(self):
        torch.manual_seed(0)
        model = _minimal_model()
        batch, patches, text_len, dim = 2, 4, 5, model.d_model
        visual = torch.randn(batch, patches, dim, requires_grad=True)
        routing = torch.rand(batch, patches, NUM_SEMANTIC_PARTS, requires_grad=True)
        base_text = torch.randn(batch, NUM_SEMANTIC_PARTS, dim, requires_grad=True)
        cached_text = torch.randn(
            batch, NUM_SEMANTIC_PARTS, text_len, dim, requires_grad=True,
        )
        text_mask = torch.ones(
            batch, NUM_SEMANTIC_PARTS, text_len, dtype=torch.bool,
        )
        text_mask[:, :, -1] = False

        refined, entropy = model._soft_visual_grounded_text_pooling(
            visual_tokens=visual,
            routing_matrix=routing,
            base_text_tokens=base_text,
            cached_text_tokens=cached_text,
            cached_text_token_mask=text_mask,
        )

        self.assertEqual(refined.shape, (batch, NUM_SEMANTIC_PARTS, dim))
        self.assertEqual(entropy.shape, (batch, NUM_SEMANTIC_PARTS - 1))
        self.assertTrue(torch.equal(refined[:, 0], base_text[:, 0]))
        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(((entropy >= 0.0) & (entropy <= 1.0 + 1e-6)).all())

        refined[:, 1:].sum().backward()
        self.assertIsNotNone(cached_text.grad)
        self.assertIsNotNone(base_text.grad)
        self.assertIsNone(visual.grad)
        self.assertIsNone(routing.grad)

    def test_per_slot_cibhash_projection_is_differentiable(self):
        torch.manual_seed(1)
        model = _minimal_model()
        semantic = torch.randn(
            3, NUM_SEMANTIC_PARTS, model.d_model, requires_grad=True,
        )

        projected = model._project_cibhash_visual_tokens(semantic)

        self.assertEqual(projected.shape, semantic.shape)
        self.assertFalse(torch.allclose(projected, semantic))
        projected.square().mean().backward()
        self.assertIsNotNone(semantic.grad)
        self.assertTrue(torch.isfinite(semantic.grad).all())


if __name__ == "__main__":
    unittest.main()
