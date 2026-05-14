"""Visual / text adapters placed on top of frozen SigLIP2 outputs.

VisualAdapter and TextAdapter are SEPARATE nn.Module instances. They never
share parameters — instantiate one for the visual branch and another for the
text branch. Each adapter is a small MLP:

    LayerNorm -> Linear -> GELU -> Dropout -> Linear  (+ optional residual)

This is the only place where trainable parameters live when the SigLIP2
backbone is frozen (which is the default).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class _MLPAdapter(nn.Module):
    """Internal block used to build VisualAdapter and TextAdapter.

    Two SEPARATE instances are created (one per branch). Parameters are NEVER
    shared between instances.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.in_dim  = int(in_dim)
        self.out_dim = int(out_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, out_dim)

        # residual is only valid when the input/output dims match
        self.residual = bool(residual and (in_dim == out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)                 # [..., out_dim]
        if self.residual:
            h = h + x
        return h


class VisualAdapter(_MLPAdapter):
    """Adapter for visual tokens.

    Expected input shape:  [B, N, in_dim]   (N = number of patch tokens, no [CLS])
    Output shape:          [B, N, out_dim]
    """
    pass


class TextAdapter(_MLPAdapter):
    """Adapter for per-part text features.

    Expected input shape:  [B, M, in_dim]   (M = number of structured parts)
    Output shape:          [B, M, out_dim]
    """
    pass
