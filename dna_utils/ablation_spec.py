"""Declared ablation cells, validated against the real parser (F09).

The chain's three ablations were each wrong in a different way, and the shell
would have reported all three as "step done":

  A2  zeroed `--lambda_text_code_kl`, `--lambda_text_hash_ntxent` and
      `--lambda_xmodal_commit` but omitted `--disable_text_supervision`. The
      caption-derived routing and pruning path has its own gate
      (`model_siglip2.py:3539`), so the cell would have finished and reported a
      "no text" number produced *with* text. Silent and wrong.
  A4  passed `--num_codebooks 1` while M=5. The quantizer rejects the mismatch,
      so the cell crashes late, after the data is loaded. The switch is
      `--share_codebook`; and sharing one codebook across five slots cuts total
      capacity fivefold, so `--codebook_size` has to be raised to keep the
      ablation about sharing rather than about size.
  A5  passed `--router_type mean`, which argparse does not accept at all. And
      the paper's A5 is not a router ablation: it is the L_joint x noGumbel
      factorial over four cells, whose finding is that the two are additive and
      that noGumbel ALONE is harmful on Flickr and CIFAR.

The preflight therefore checks flags against `Config`'s own parser rather than
against a copied list of names, which would drift from `config.py` exactly as
the chain drifted from the model.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

#: Codebook size per dataset in the reference recipe, used to restore matched
#: capacity under A4.
_CODEBOOK_SIZE = {"CIFAR10": 64, "Flickr25k": 128, "NUSWIDE": 128, "MSCOCO": 128}
_NUM_SLOTS = 5


class AblationInvalid(ValueError):
    """A cell would not run, or would not measure what it claims."""


@dataclass(frozen=True)
class AblationCell:
    name: str
    flags: List[str]
    describes: str


@dataclass(frozen=True)
class AblationSpec:
    name: str
    question: str
    #: True when the cell removes an INPUT rather than a loss term. A2's delta
    #: is not comparable with A5's, which changes only the objective at
    #: identical inputs, and the paper must not tabulate them as one column.
    removes_input: bool
    cells: List[AblationCell] = field(default_factory=list)


def _a2(dataset: str) -> List[AblationCell]:
    return [AblationCell(
        name="A2_no_text",
        describes="no text supervision anywhere: losses AND the routing path",
        flags=[
            "--disable_text_supervision",
            "--lambda_text_code_kl", "0.0",
            "--lambda_text_hash_ntxent", "0.0",
            "--lambda_xmodal_commit", "0.0",
            "--selection_mode", "A2_no_text",
        ])]


def _a4(dataset: str) -> List[AblationCell]:
    k = _CODEBOOK_SIZE.get(dataset, 128)
    return [AblationCell(
        name="A4_shared_codebook",
        describes=("one codebook shared across slots, at matched TOTAL capacity "
                   f"({_NUM_SLOTS} x {k} = {_NUM_SLOTS * k}) so the delta is "
                   "about sharing, not about size"),
        flags=[
            "--share_codebook",
            "--codebook_size", str(_NUM_SLOTS * k),
            "--selection_mode", "A4_shared_codebook",
        ])]


def _a5(dataset: str) -> List[AblationCell]:
    # The paper's A5: the two terms this work adds, as a 2x2 factorial. Reported
    # together because noGumbel alone is HARMFUL on Flickr and CIFAR and only
    # helps in combination -- a single-factor result would mislead.
    return [
        AblationCell("A5_none", ["--lambda_codon_joint", "0.0",
                                 "--selection_mode", "A5_none"],
                     "neither new term"),
        AblationCell("A5_joint", ["--selection_mode", "A5_joint"],
                     "L_joint only"),
        AblationCell("A5_nogumbel", ["--lambda_codon_joint", "0.0",
                                     "--no_gumbel_softmax",
                                     "--selection_mode", "A5_nogumbel"],
                     "deterministic straight-through only"),
        AblationCell("A5_both", ["--no_gumbel_softmax",
                                 "--selection_mode", "A5_both"],
                     "both, i.e. the unified recipe"),
    ]


ABLATIONS: Dict[str, AblationSpec] = {
    "A2": AblationSpec(
        "A2", "does the text path carry the compositional claim?",
        removes_input=True, cells=_a2("Flickr25k")),
    "A4": AblationSpec(
        "A4", "do per-slot codebooks matter, at matched capacity?",
        removes_input=False, cells=_a4("Flickr25k")),
    "A5": AblationSpec(
        "A5", "what do the two newly added loss terms contribute?",
        removes_input=False, cells=_a5("Flickr25k")),
}

_BUILDERS = {"A2": _a2, "A4": _a4, "A5": _a5}


def build_ablation_args(name: str, *, dataset: str) -> List[str]:
    """Flags for a single-cell ablation, resolved for this dataset."""
    if name not in _BUILDERS:
        raise AblationInvalid(f"unknown ablation {name!r}; known: {sorted(_BUILDERS)}")
    cells = _BUILDERS[name](dataset)
    if len(cells) != 1:
        raise AblationInvalid(
            f"{name} has {len(cells)} cells; iterate ABLATIONS[{name!r}].cells "
            f"instead of asking for one flag list")
    flags = list(cells[0].flags)
    preflight_ablation(flags, dataset=dataset)
    return flags


_PARSER_CACHE = None


def _parser():
    """The REAL parser from config.py. Cached because building it walks every
    argument group. Asking argparse rather than keeping a copy of the flag names
    is the point: a copy drifts from config.py exactly as the chain drifted from
    the model."""
    global _PARSER_CACHE
    if _PARSER_CACHE is None:
        from config import Config
        _PARSER_CACHE = Config.build_parser()
        if _PARSER_CACHE is None:
            raise AblationInvalid("could not build config.py's argparse parser")
    return _PARSER_CACHE


def _known_options() -> set:
    return {o for act in _parser()._actions for o in act.option_strings}


def _choices_for(option: str) -> Sequence[str] | None:
    for act in _parser()._actions:
        if option in act.option_strings:
            return act.choices
    return None


def preflight_ablation(flags: Sequence[str], *, dataset: str) -> None:
    """Refuse a cell that would not run, or would not measure what it claims."""
    known = _known_options()
    i = 0
    seen: List[Tuple[str, str | None]] = []
    while i < len(flags):
        tok = flags[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if tok not in known:
            raise AblationInvalid(
                f"{tok} is not accepted by config.py's parser. The chain passed "
                f"flags that argparse rejects; validate against the real parser, "
                f"not a copied list of names.")
        val = flags[i + 1] if i + 1 < len(flags) and not flags[i + 1].startswith("--") else None
        ch = _choices_for(tok)
        if ch is not None and val is not None and val not in [str(c) for c in ch]:
            raise AblationInvalid(
                f"{tok}={val!r} is not among the accepted choices {list(ch)}. "
                f"This is how `--router_type mean` reached a launcher.")
        seen.append((tok, val))
        i += 2 if val is not None else 1

    d = dict(seen)
    if d.get("--num_codebooks") not in (None,) and d.get("--num_codebooks") != str(_NUM_SLOTS):
        raise AblationInvalid(
            f"--num_codebooks {d['--num_codebooks']} disagrees with the "
            f"{_NUM_SLOTS}-slot architecture and fails inside the quantizer. To "
            f"ablate per-slot codebooks use --share_codebook and restore matched "
            f"capacity with --codebook_size.")
    if "--share_codebook" in d and "--codebook_size" not in d:
        raise AblationInvalid(
            "--share_codebook without --codebook_size confounds sharing with a "
            "fivefold capacity cut; pass the matched total.")


def preflight_all(*, dataset: str) -> None:
    for name, spec in ABLATIONS.items():
        for cell in _BUILDERS[name](dataset):
            try:
                preflight_ablation(cell.flags, dataset=dataset)
            except AblationInvalid as ex:
                raise AblationInvalid(f"{cell.name}: {ex}") from ex


def render_command(cell: AblationCell) -> str:
    return " ".join(shlex.quote(f) for f in cell.flags)
