"""F09: validate every ablation command before it costs GPU hours.

The chain's three ablations were each wrong in a different way, and none would
have been caught by reading the shell:

  A2  zeroed three text lambdas but omitted --disable_text_supervision, so
      caption-derived routing and pruning stayed live. The cell would have run
      to completion and reported a "no text" number produced with text.
  A4  passed --num_codebooks 1 against M=5, which fails inside the quantizer.
      The correct switch is --share_codebook, and matched capacity has to be
      restored or the ablation confounds "shared" with "smaller".
  A5  passed --router_type mean, which argparse does not accept at all. Worse,
      the paper's A5 is not a router ablation: it is the L_joint x noGumbel
      factorial, four cells, whose point is that the two terms are additive and
      that noGumbel alone is harmful on some datasets.

So the failure modes span silent-wrong (A2), crash-late (A4) and
does-not-exist-and-is-the-wrong-experiment (A5). A preflight has to catch all
three, which means checking the flags against the REAL argparse rather than a
copy of the flag names.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.ablation_spec import (  # noqa: E402
    ABLATIONS,
    AblationInvalid,
    build_ablation_args,
    preflight_ablation,
    preflight_all,
)


def test_every_declared_ablation_preflights():
    """The whole point: nothing reaches a GPU without passing this."""
    preflight_all(dataset="Flickr25k")


def test_a2_disables_text_supervision():
    """Zeroing the lambdas is not enough; the routing path has its own gate."""
    flags = build_ablation_args("A2", dataset="Flickr25k")
    assert "--disable_text_supervision" in flags


def test_a2_also_zeroes_the_text_losses():
    flags = " ".join(build_ablation_args("A2", dataset="Flickr25k"))
    for lam in ("--lambda_text_code_kl", "--lambda_text_hash_ntxent",
                "--lambda_xmodal_commit"):
        assert f"{lam} 0.0" in flags


def test_a4_uses_share_codebook_not_num_codebooks():
    """--num_codebooks 1 against M=5 fails inside the quantizer."""
    flags = build_ablation_args("A4", dataset="Flickr25k")
    assert "--share_codebook" in flags
    assert "--num_codebooks" not in flags


@pytest.mark.parametrize("dataset,expected", [
    ("CIFAR10", 320),      # 5 slots x 64
    ("Flickr25k", 640),    # 5 slots x 128
    ("NUSWIDE", 640),
    ("MSCOCO", 640),
])
def test_a4_restores_matched_capacity(dataset, expected):
    """Without this the ablation confounds "shared codebook" with "5x smaller
    codebook", and the drop cannot be attributed."""
    flags = build_ablation_args("A4", dataset=dataset)
    i = flags.index("--codebook_size")
    assert int(flags[i + 1]) == expected


def test_a5_is_a_factorial_not_a_router_ablation():
    """The paper's A5 is L_joint x noGumbel over four cells."""
    cells = ABLATIONS["A5"].cells
    assert len(cells) == 4
    names = {c.name for c in cells}
    assert names == {"A5_none", "A5_joint", "A5_nogumbel", "A5_both"}


def test_a5_cells_toggle_exactly_the_two_terms():
    by = {c.name: " ".join(c.flags) for c in ABLATIONS["A5"].cells}
    assert "--lambda_codon_joint 0.0" in by["A5_none"]
    assert "--no_gumbel_softmax" not in by["A5_none"]
    assert "--lambda_codon_joint 0.0" in by["A5_nogumbel"]
    assert "--no_gumbel_softmax" in by["A5_nogumbel"]
    assert "--lambda_codon_joint 0.0" not in by["A5_joint"]
    assert "--no_gumbel_softmax" not in by["A5_joint"]
    assert "--no_gumbel_softmax" in by["A5_both"]


def test_router_type_mean_is_rejected():
    """The exact value the chain passed. argparse has no such choice."""
    with pytest.raises(AblationInvalid) as ex:
        preflight_ablation(["--router_type", "mean"], dataset="Flickr25k")
    assert "router_type" in str(ex.value)


def test_num_codebooks_one_is_rejected():
    with pytest.raises(AblationInvalid):
        preflight_ablation(["--num_codebooks", "1", "--num_semantic_parts", "5"],
                           dataset="Flickr25k")


def test_unknown_flag_is_rejected():
    with pytest.raises(AblationInvalid):
        preflight_ablation(["--not_a_real_flag", "1"], dataset="Flickr25k")


def test_preflight_uses_the_real_parser():
    """A hand-maintained list of valid flags would drift from config.py. This
    asserts we are asking argparse itself."""
    from dna_utils.ablation_spec import _known_options
    opts = _known_options()
    assert "--disable_text_supervision" in opts
    assert "--share_codebook" in opts
    assert "--router_type" in opts
    assert "--not_a_real_flag" not in opts


def test_every_ablation_carries_a_distinct_selection_mode():
    """F08: ablations must not resolve to the reference run's directory."""
    modes = set()
    for name, spec in ABLATIONS.items():
        for cell in spec.cells:
            f = " ".join(cell.flags)
            assert "--selection_mode" in f, f"{cell.name} has no selection_mode"
            i = cell.flags.index("--selection_mode")
            modes.add(cell.flags[i + 1])
    assert len(modes) == sum(len(s.cells) for s in ABLATIONS.values())


def test_a2_is_recorded_as_information_ablation_not_loss_ablation():
    """Reporting matters: A2 removes an INPUT, so its delta is not comparable
    with A5's, which removes loss terms at identical inputs."""
    assert ABLATIONS["A2"].removes_input is True
    assert ABLATIONS["A5"].removes_input is False
