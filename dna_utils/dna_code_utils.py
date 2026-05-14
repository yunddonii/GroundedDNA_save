"""DNA base ↔ 2-bit / one-hot / index utilities and Hamming distances.

Convention (matching `dataloaders.gen_hash_table`):
    base index     symbol     2-bit
        0            A         [0, 0]
        1            C         [0, 1]
        2            G         [1, 0]
        3            T         [1, 1]

The 2-bit form is the one that matches the project's existing DB hash table
(``dataloaders.gen_hash_table`` outputs concatenated 2-bit per base). Use the
``hash_2bit`` form for retrieval / storage compatibility.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


# Lookup table: base index -> 2-bit code. Long for safe indexing.
BASE_TO_BITS = torch.tensor(
    [
        [0, 0],  # A
        [0, 1],  # C
        [1, 0],  # G
        [1, 1],  # T
    ],
    dtype=torch.long,
)


# ----------------------------------------------------------------- conversions

def base_indices_to_2bit(base_indices: torch.Tensor) -> torch.Tensor:
    """Map base indices to 2-bit tensor.

    Input:
        base_indices : LongTensor of shape ``[..., L]`` (any leading dims).

    Output:
        bits : LongTensor of shape ``[..., L, 2]``.
    """
    if base_indices.dtype not in (torch.int32, torch.int64):
        base_indices = base_indices.long()
    table = BASE_TO_BITS.to(base_indices.device)
    return table[base_indices]   # [..., L, 2]


def base_indices_to_2bit_flat(base_indices: torch.Tensor) -> torch.Tensor:
    """Same as :func:`base_indices_to_2bit` but with the bit axis flattened
    into the position axis.

    Input:  ``[..., L]``     (e.g. ``[B, 18]``)
    Output: ``[..., L*2]``   (e.g. ``[B, 36]``)
    """
    bits = base_indices_to_2bit(base_indices)              # [..., L, 2]
    return bits.flatten(-2, -1)                            # [..., L*2]


def two_bit_flat_to_base_indices(bits_flat: torch.Tensor) -> torch.Tensor:
    """Invert :func:`base_indices_to_2bit_flat`.

    Input:  ``[..., L*2]``
    Output: ``[..., L]``  LongTensor in {0, 1, 2, 3}.
    """
    last = bits_flat.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"[two_bit_flat_to_base_indices] last dim must be even; got {last}."
        )
    bits = bits_flat.view(*bits_flat.shape[:-1], last // 2, 2)        # [..., L, 2]
    return (bits[..., 0].long() * 2 + bits[..., 1].long())            # [..., L]


def onehot_to_base_indices(onehot: torch.Tensor) -> torch.Tensor:
    """``[..., L, 4]`` one-hot → ``[..., L]`` LongTensor."""
    return onehot.argmax(dim=-1).long()


def base_indices_to_onehot(
    base_indices: torch.Tensor, num_classes: int = 4,
) -> torch.Tensor:
    """``[..., L]`` LongTensor → ``[..., L, 4]`` float one-hot."""
    return F.one_hot(base_indices.long(), num_classes=num_classes).to(torch.float32)


# ----------------------------------------------------------------- distances

def base_hamming_distance(
    query_base: torch.Tensor, db_base: torch.Tensor,
) -> torch.Tensor:
    """Base-level Hamming distance.

    Every base mismatch counts as 1 regardless of how the bases differ.

    Input:
        query_base : LongTensor [Nq, L]
        db_base    : LongTensor [Nd, L]

    Output:
        dist : LongTensor [Nq, Nd], values in [0, L].
    """
    if query_base.shape[-1] != db_base.shape[-1]:
        raise ValueError(
            f"[base_hamming_distance] L mismatch: query={query_base.shape[-1]} "
            f"vs db={db_base.shape[-1]}"
        )
    return (
        query_base.unsqueeze(1) != db_base.unsqueeze(0)
    ).sum(dim=-1)   # [Nq, Nd]


def bit_hamming_distance_2bit(
    query_bits: torch.Tensor, db_bits: torch.Tensor,
) -> torch.Tensor:
    """2-bit (or any binary) Hamming distance.

    Compatible with the DB hash table produced by `dataloaders.gen_hash_table`
    (which is 2 bits per base, flattened to ``[N, 2L]``).

    Input:
        query_bits : Tensor [Nq, 2L]   values in {0, 1}
        db_bits    : Tensor [Nd, 2L]   values in {0, 1}

    Output:
        dist : LongTensor [Nq, Nd], values in [0, 2L].
    """
    if query_bits.shape[-1] != db_bits.shape[-1]:
        raise ValueError(
            f"[bit_hamming_distance_2bit] last-dim mismatch: "
            f"query={query_bits.shape[-1]} vs db={db_bits.shape[-1]}"
        )
    return (
        query_bits.unsqueeze(1) != db_bits.unsqueeze(0)
    ).sum(dim=-1)   # [Nq, Nd]


__all__ = [
    "BASE_TO_BITS",
    "base_indices_to_2bit",
    "base_indices_to_2bit_flat",
    "two_bit_flat_to_base_indices",
    "onehot_to_base_indices",
    "base_indices_to_onehot",
    "base_hamming_distance",
    "bit_hamming_distance_2bit",
]
