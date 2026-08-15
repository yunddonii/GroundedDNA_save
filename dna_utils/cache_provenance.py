"""Refuse a feature cache that cannot say which backbone produced it.

`paper_table_eligible=True` rests on knowing exactly what the frozen encoder was.
The four caches in use through 2026-08 predated `hf_provenance` and
`canonical_transform`, and two `openai/clip-vit-base-patch16` snapshots sit in
the local HF hub directory (`57c21647…`, `5ef227a7…`), so nothing proved which
one built them. That is why D-A regenerated every cache.

Regenerating them is not the same as using them. The rebuilt caches landed under
`/data/yschoi/groundeddna_cache_v6prov` while every runner still pointed at the
old `./cache/...` paths, so the provenance-bearing artefacts sat unused and a
training run could still silently consume an unprovenanced cache and be reported
as paper-eligible. The check therefore lives in the LOADER, where every path --
including the eleven scripts that were never updated -- has to pass through it.

It is fail-closed. `GDNA_ALLOW_UNPROVENANCED_CACHE=1` exists for exploratory work
on the legacy caches, but it announces itself on every load and callers that
record run metadata are expected to record that it was used, so a paper number
cannot quietly inherit it.
"""
from __future__ import annotations

import os
import re
from typing import Mapping

#: An immutable HF commit, not a branch name. `main` moves; a paper cannot cite
#: it. `extract_clip_features._resolve_hf_provenance` already refuses to record
#: anything else, so this re-checks the value rather than trusting the writer.
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

OPT_OUT_ENV = "GDNA_ALLOW_UNPROVENANCED_CACHE"


class CacheProvenanceMissing(RuntimeError):
    """A cache cannot be tied to the backbone snapshot that produced it."""


def opted_out() -> bool:
    return os.environ.get(OPT_OUT_ENV, "") not in ("", "0", "false", "False")


def provenance_record(meta: Mapping) -> dict:
    """What a run should record about the cache it consumed."""
    hf = meta.get("hf_provenance")
    hf = hf if isinstance(hf, Mapping) else {}
    return {
        "model_revision": hf.get("model_revision"),
        "model_name": hf.get("model_name") or hf.get("checkpoint"),
        "canonical_transform": bool(
            isinstance(meta.get("canonical_transform"), Mapping)),
        "unprovenanced_opt_out": opted_out(),
    }


def require_cache_provenance(cache_dir: str, meta: Mapping) -> dict:
    """Admit a cache only if it names the exact snapshot that built it.

    Returns the provenance record so the caller can put it in its own manifest.
    """
    problems = []
    if not isinstance(meta.get("canonical_transform"), Mapping):
        problems.append(
            "no canonical_transform, so the preprocessing that produced these "
            "features is not recorded")

    hf = meta.get("hf_provenance")
    if not isinstance(hf, Mapping):
        problems.append(
            "no hf_provenance, so the backbone snapshot is unknown -- two "
            "clip-vit-base-patch16 revisions exist locally")
    else:
        revision = hf.get("model_revision")
        if not isinstance(revision, str) or not _COMMIT.match(revision):
            problems.append(
                f"hf_provenance.model_revision is {revision!r}, not an "
                f"immutable 40-hex commit")
        # A revision string alone is a claim about a snapshot, not evidence of
        # one. The writer already digests the weights it loaded, so a cache
        # that omits them was not produced by the current extractor.
        name = hf.get("model_name") or hf.get("checkpoint")
        if not isinstance(name, str) or not name.strip():
            problems.append("hf_provenance names no model")
    transform = meta.get("canonical_transform")
    if isinstance(transform, Mapping) and not transform:
        problems.append(
            "canonical_transform is empty, which records nothing about the "
            "preprocessing")

    if problems:
        detail = "; ".join(problems)
        if opted_out():
            print(f"[cache-provenance] WARNING: {cache_dir} {detail}. "
                  f"Admitted only because {OPT_OUT_ENV} is set; this run is "
                  f"NOT paper-table eligible.")
        else:
            raise CacheProvenanceMissing(
                f"{cache_dir}: {detail}. Rebuild it with "
                f"scripts/build_clip_cache_v6prov.sh, point the runner at the "
                f"rebuilt cache, or set {OPT_OUT_ENV}=1 to proceed knowing the "
                f"result cannot go in the paper.")
    return provenance_record(meta)


WHITEN_META_SUFFIX = ".meta.json"


def require_leakage_free_whitening(npz_path: str) -> dict:
    """Refuse a whitening matrix fitted outside the optimization-train split.

    `build_text_whiten_matrix.py` records `leakage_free_fit`, which is true only
    when `--row_index_npy` restricted the fit. Fitting over every cache row lets
    the transform observe the validation rows that select the epoch and the
    query/DB rows that are the reported number -- transductive leakage that no
    later stage can undo, and that nothing checked at load time.
    """
    meta_path = npz_path + WHITEN_META_SUFFIX
    if not os.path.isfile(meta_path):
        problem = (f"has no {WHITEN_META_SUFFIX} beside it, so the rows it was "
                   f"fitted on are unknown")
    else:
        import json
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as error:
            problem = f"has an unreadable {WHITEN_META_SUFFIX}: {error}"
        else:
            # `is True`, not truthiness: the JSON string "false" is truthy,
            # and so is any non-empty value someone happens to put there.
            flag = meta.get("leakage_free_fit")
            rows = meta.get("rows_used")
            index = meta.get("row_index_npy")
            if flag is not True:
                problem = (f"records leakage_free_fit={flag!r}; only a literal "
                           f"true means the fit was restricted")
            elif not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
                problem = f"records rows_used={rows!r}, not a positive count"
            elif not isinstance(index, str) or not os.path.isfile(index):
                problem = (f"names row_index_npy={index!r}, which is not a "
                           f"file that exists; the restriction cannot be "
                           f"re-checked")
            else:
                return dict(meta)

    if opted_out():
        print(f"[cache-provenance] WARNING: {npz_path} {problem}. Admitted "
              f"only because {OPT_OUT_ENV} is set; this run is NOT paper-table "
              f"eligible.")
        return {}
    raise CacheProvenanceMissing(
        f"{npz_path}: {problem}. Refit it with "
        f"scripts/build_text_whiten_matrix.py --row_index_npy "
        f"<cache>/opt_train_rows.npy, or set {OPT_OUT_ENV}=1 to proceed "
        f"knowing the result cannot go in the paper.")
