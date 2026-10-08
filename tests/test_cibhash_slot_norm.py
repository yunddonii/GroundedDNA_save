"""(2026-10-08, plan 4a) --cibhash_ntxent_slot_norm {computed, all}: how the per-slot
visual-token NT-Xent terms are averaged when --cibhash_local_target none drops the local
terms. Drives the real parser and the real DNACodonHashLoss method on synthetic tokens."""
import pytest
import torch

from config import Config
from loss_siglip2 import DNACodonHashLoss

_B, _M, _D = 6, 5, 8


def _args(*argv):
    return Config.build_parser().parse_args(list(argv))


def _crit(*argv):
    return DNACodonHashLoss(_args(*argv))


def _views(seed=0):
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(_B, _M, _D, generator=g); v2 = v1 + 0.1 * torch.randn(_B, _M, _D, generator=g)
    return v1, v2


def _terms(crit, v1, v2, local_target):
    """Per-slot NT-Xent values the method would compute (recomputed here with M=1 calls)."""
    out = []
    for m in range(_M):
        if m > 0 and local_target == "none":
            continue
        t, _ = crit._loss_cibhash_visual_per_codebook(v1[:, m:m + 1], v2[:, m:m + 1], temperature=0.3,
                                                       local_target="instance")
        out.append(t)
    return torch.stack(out)


def test_parser_default_is_computed_and_rejects_other_values():
    assert _args().cibhash_ntxent_slot_norm == "computed"
    assert _args("--cibhash_ntxent_slot_norm", "all").cibhash_ntxent_slot_norm == "all"
    with pytest.raises(SystemExit):
        _args("--cibhash_ntxent_slot_norm", "half")


def test_default_norm_is_the_legacy_mean_over_computed_terms():
    v1, v2 = _views()
    crit = _crit()
    full, _ = crit._loss_cibhash_visual_per_codebook(v1, v2, temperature=0.3, local_target="instance")
    assert torch.allclose(full, _terms(crit, v1, v2, "instance").mean(), atol=1e-6)
    glob_only, _ = crit._loss_cibhash_visual_per_codebook(v1, v2, temperature=0.3, local_target="none")
    # legacy: the single remaining (global) term carries weight 1
    assert torch.allclose(glob_only, _terms(crit, v1, v2, "none").mean(), atol=1e-6)
    assert torch.allclose(glob_only, _terms(crit, v1, v2, "instance")[0], atol=1e-6)


def test_all_norm_divides_by_the_slot_count():
    v1, v2 = _views()
    crit = _crit("--cibhash_ntxent_slot_norm", "all")
    full, _ = crit._loss_cibhash_visual_per_codebook(v1, v2, temperature=0.3, local_target="instance")
    legacy = _crit()._loss_cibhash_visual_per_codebook(v1, v2, temperature=0.3, local_target="instance")[0]
    assert torch.allclose(full, legacy, atol=1e-6), "with every term present, sum/M == mean"
    glob_only, _ = crit._loss_cibhash_visual_per_codebook(v1, v2, temperature=0.3, local_target="none")
    expect = _terms(crit, v1, v2, "instance")[0] / _M
    assert torch.allclose(glob_only, expect, atol=1e-6)
    # and it is exactly the legacy full value minus the local half
    local_half = _terms(crit, v1, v2, "instance")[1:].sum() / _M
    assert torch.allclose(glob_only, full - local_half, atol=1e-6)


def test_constructor_rejects_a_bad_norm_value():
    a = _args(); a.cibhash_ntxent_slot_norm = "half"
    with pytest.raises(ValueError):
        DNACodonHashLoss(a)
