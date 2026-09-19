"""(b-1) Masked entity completion, bound per slot -- a port of OVSegmentor's entity loss.

Off-protocol, branch ``arch-exp-2026-09`` (2026-09-20). Built only when
``--lambda_mec > 0``; the paper model never constructs it.

Source reviewed line by line: https://github.com/Jazzcharles/OVSegmentor @ cfcbc3a
(Xu et al., "Learning Open-vocabulary Semantic Segmentation Models From Natural
Language Supervision", CVPR 2023):

  models/multi_label_contrastive.py  MultimodalGroupingBlock, MultimodalGroupingNetwork,
                                     ProjectMLP, MultiLabelContrastive.loss / forward_train
  models/group_vit.py                Mlp, Attention, CrossAttnBlock, AttnBlock
  datasets/clip_dataset.py           build_question_and_answer (CLIP-tokenizer variant)
  configs/default.yml, stage-1 yml   entity_weight 1.0, cross_layers 1, proj_num_layers 2,
                                     contrast_temperature 0.07 (learnable, clamped at 100)

What OVSegmentor does, and what is kept unchanged here: every entity word of a
caption is masked; the masked caption ("question") is encoded token by token; the
question tokens cross-attend to the visual group tokens, then self-attend
(``MultimodalGroupingBlock``); the output at token 0 goes through a 2-layer
projector (``ProjectMLP``: Conv1d-BN-ReLU, hidden 4096); and that vector is
contrasted (symmetric InfoNCE, ``MultiLabelContrastive.loss``) against the text
embedding of the "answer", i.e. the prefix of a random ImageNet prompt template
followed by the masked entities joined with " and ".

Deliberate deviations, each for a stated reason:

1. Keys. OVSegmentor's question attends over all K group tokens. Here the axis-m
   question attends over slot m's code only, so the completion can succeed only
   if slot m itself carries axis m's content. That binding is the purpose of b-1.
2. Fixed targets. OVSegmentor trains its text encoder and text projector, so its
   answer targets move. Here the question tokens and answers are precomputed with
   the frozen CLIP text tower, and the bridge projects straight into CLIP's 512-d
   joint space. With fixed targets the loss has no incentive to shrink the set of
   codewords it is matched against -- the attractor recorded for the rejected
   ``L_text_codeword_contrastive`` (PROJECT_LOG, 2026-06-30).
3. Padding mask. OVSegmentor passes the question attention mask to AttnBlock but
   its ``Attention.forward`` never applies it; here it is applied to the keys.
4. Masking unit. OVSegmentor maps only the FIRST BPE id of an entity to id 332
   ('m</w>'), leaving later sub-tokens of a multi-token entity visible; the cache
   builder here replaces the whole word with "M", which yields the same id 332.
5. Modules that never enter OVSegmentor's forward (AssignAttention, the mlp_ratio
   token/channel widths) are not built. Dropout/drop-path are 0 there and omitted.
6. A sample whose axis caption contains no vocabulary entity is left out of that
   axis's loss (OVSegmentor would contrast an entity-less template prefix).
7. Code, not continuous feature. The key is the slot's straight-through
   quantised token, so the value seen is the codeword that the stored DNA code
   is decoded from; the gradient reaches the encoder through the STE.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MEC_AXES: Tuple[str, ...] = (
    "C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture",
)


class Mlp(nn.Module):
    """group_vit.Mlp (dropout is 0 in every OVSegmentor use, so omitted)."""

    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """group_vit.Attention, plus the key-padding mask OVSegmentor passes but drops."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, qkv_fuse: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key=None, mask: Optional[torch.Tensor] = None):
        B, N, C = query.shape
        h = self.num_heads
        if self.qkv_fuse:
            qkv = self.qkv(query).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                       # [B, h, N, C/h]
        else:
            key = query if key is None else key
            S = key.shape[1]
            q = self.q_proj(query).reshape(B, N, h, C // h).transpose(1, 2)
            k = self.k_proj(key).reshape(B, S, h, C // h).transpose(1, 2)
            v = self.v_proj(key).reshape(B, S, h, C // h).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale                 # [B, h, N, S]
        if mask is not None:                                          # deviation 3
            keep = mask.to(torch.bool)[:, None, None, :]
            attn = attn.masked_fill(~keep, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttnBlock(nn.Module):
    """group_vit.CrossAttnBlock with post_norm=True (as MultimodalGroupingBlock builds it)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = Attention(dim, num_heads, qkv_bias=True, qkv_fuse=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.norm_post = nn.LayerNorm(dim)

    def forward(self, query, key):
        x = query + self.attn(query, key)          # norm_q / norm_k are Identity under post_norm
        x = x + self.mlp(self.norm2(x))
        return self.norm_post(x)


class AttnBlock(nn.Module):
    """group_vit.AttnBlock (fused qkv self-attention)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=True, qkv_fuse=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.mlp(self.norm2(x))


class MultimodalGroupingBlock(nn.Module):
    """multi_label_contrastive.MultimodalGroupingBlock.forward, line for line."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)
        self.pre_assign_attn = CrossAttnBlock(dim, num_heads)
        self.post_attn = AttnBlock(dim, num_heads)
        self.norm_new_x = nn.LayerNorm(dim)

    def forward(self, ans_tokens, visual_tokens, text_tokens, question_masks=None):
        text_tokens = self.norm_tokens(text_tokens)
        visual_tokens = self.norm_x(visual_tokens)
        projected = self.pre_assign_attn(text_tokens, visual_tokens)
        ans_temp = projected if ans_tokens is None else ans_tokens + projected
        new_x = self.post_attn(ans_temp, mask=question_masks)
        new_x = new_x + projected
        return self.norm_new_x(new_x)


class ProjectMLP(nn.Module):
    """multi_label_contrastive.ProjectMLP (Conv1d k=1 + BN + ReLU hidden layers)."""

    def __init__(self, in_dim: int, inner_dim: int = 4096, out_dim: int = 256, num_layers: int = 2):
        super().__init__()
        hidden = []
        for i in range(num_layers - 1):
            hidden += [nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1),
                       nn.BatchNorm1d(inner_dim), nn.ReLU(inplace=True)]
        self.linear_hidden = nn.Sequential(*hidden)
        self.linear_out = (nn.Conv1d(in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1)
                           if num_layers >= 1 else nn.Identity())

    def forward(self, x):                           # [B, C] -> [B, out]
        x = x.unsqueeze(-1)                         # [B, C, 1]  ('b l c -> b c l' with l = 1)
        x = self.linear_out(self.linear_hidden(x))
        return x.squeeze(-1)


def _norm_image_id(path: str) -> str:
    """'.../Flickr25k/images/im15316.jpg' or 'images/im15316.jpg' -> 'images/im15316.jpg'."""
    return "images/" + os.path.basename(str(path))


class MaskedEntityCompletion(nn.Module):
    """Per-slot masked entity completion over the 4 local slots."""

    def __init__(self, slot_dim: int, text_dim: int = 512, num_heads: int = 8,
                 cross_layers: int = 1, proj_num_layers: int = 2, proj_inner_dim: int = 4096,
                 temperature: float = 0.07, seed: int = 0):
        super().__init__()
        width = min(slot_dim, text_dim)                       # OVSegmentor: min(img, text) width
        self.align_proj_img = nn.Linear(slot_dim, width) if slot_dim > text_dim else nn.Identity()
        self.align_proj_text = nn.Linear(text_dim, width) if text_dim > slot_dim else nn.Identity()
        self.blocks = nn.ModuleList([MultimodalGroupingBlock(width, num_heads) for _ in range(cross_layers)])
        self.bridge_projector = ProjectMLP(width, proj_inner_dim, text_dim, proj_num_layers)
        self.logit_scale = nn.Parameter(torch.ones([]) * float(np.log(1.0 / temperature)))
        self._seed = int(seed)
        self._gen: Optional[torch.Generator] = None
        self.row_of: dict = {}
        self.n_templates = 0

    # ------------------------------------------------------------------ cache
    def load_cache(self, cache_dir: str) -> dict:
        meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        if tuple(meta["axes"]) != MEC_AXES:
            raise ValueError(f"MEC cache axes {meta['axes']} != {MEC_AXES}")
        ids = json.load(open(os.path.join(cache_dir, "image_ids.json")))
        self.row_of = {str(i): r for r, i in enumerate(ids)}
        q = np.load(os.path.join(cache_dir, "q_tokens.f16.npy"))           # [R, 4, T, D]
        qm = np.load(os.path.join(cache_dir, "q_mask.bool.npy"))           # [R, 4, T]
        ans = np.load(os.path.join(cache_dir, "answers.f16.npy"))          # [R, 4, NT, D]
        val = np.load(os.path.join(cache_dir, "valid.bool.npy"))           # [R, 4]
        # Non-persistent: the cache is data, not weights, and must not bloat checkpoints.
        self.register_buffer("q_tokens", torch.from_numpy(q), persistent=False)
        self.register_buffer("q_mask", torch.from_numpy(qm), persistent=False)
        self.register_buffer("answers", torch.from_numpy(ans), persistent=False)
        self.register_buffer("valid", torch.from_numpy(val), persistent=False)
        self.n_templates = int(ans.shape[2])
        return meta

    # ---------------------------------------------------------------- forward
    def _clip_loss(self, x: torch.Tensor, y: torch.Tensor):
        """MultiLabelContrastive.loss on one device: symmetric InfoNCE, shared scale."""
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits = x @ y.t() * scale
        labels = torch.arange(x.shape[0], device=x.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc.detach()

    def forward(self, slot_codes: torch.Tensor, image_ids: Sequence[str]):
        """slot_codes: [B, 4, D_slot] straight-through codes of the 4 local slots."""
        B, M, _ = slot_codes.shape
        if M != len(MEC_AXES):
            raise ValueError(f"expected {len(MEC_AXES)} local slots, got {M}")
        dev = slot_codes.device
        rows = torch.tensor([self.row_of.get(_norm_image_id(i), -1) for i in image_ids],
                            device=dev, dtype=torch.long)
        known = rows >= 0
        rows_c = rows.clamp_min(0)
        if self._gen is None or self._gen.device != dev:
            # A private generator: template sampling must not shift the global RNG stream.
            self._gen = torch.Generator(device=dev)
            self._gen.manual_seed(self._seed)
        losses: List[torch.Tensor] = []
        accs: List[torch.Tensor] = []
        for m in range(M):
            valid = known & self.valid[rows_c, m]
            if int(valid.sum().item()) < 2:
                continue
            q = self.q_tokens[rows_c, m].float()                              # [B, T, D_t]
            qm = self.q_mask[rows_c, m]                                        # [B, T]
            t_idx = torch.randint(self.n_templates, (B,), device=dev, generator=self._gen)
            ans = self.answers[rows_c, m, t_idx].float()                      # [B, D_t]
            key = self.align_proj_img(slot_codes[:, m]).unsqueeze(1)          # [B, 1, W]
            txt = self.align_proj_text(q)                                     # [B, T, W]
            x = None
            for blk in self.blocks:
                x = blk(x, key, txt, question_masks=qm)
            pred = self.bridge_projector(x[:, 0])                            # [B, D_t]
            loss_m, acc_m = self._clip_loss(pred[valid], ans[valid])
            losses.append(loss_m)
            accs.append(acc_m)
        if not losses:
            z = slot_codes.sum() * 0.0
            return z, z.detach()
        return torch.stack(losses).mean(), torch.stack(accs).mean()
