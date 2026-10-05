"""The one official-test extraction and raw evaluation of a scratch full-train refit.

This is the block that ran at the end of ``train_siglip2.main`` (moved here unchanged in generation
v9, audits 743-744), so the legacy refit path and the anchor model's separately approved stage-T entry
(``scripts/anchor_terminal_test.py``) execute the same code: the evaluation-cache / text-whitening
resolution, ``extraction_siglip2.extract_code`` (which reloads the saved terminal checkpoint and
encodes the official query and database splits), then the RAW ``evaluation_siglip2.evaluation``
(``bio_project`` left at its default False) that writes ``evaluation_siglip2_<distance_mode>.json``.
It mutates ``args`` exactly as the inline block did (the evaluation cache stays selected afterwards;
the text-whitening override is scoped to extraction).
"""
import os


def run_official_test(args, *, distance_mode: str, codebook_size: int, verified=None) -> None:
    """``verified``: the stage-T entry's dna_utils.runtime_state.VerifiedRuntime, handed to
    extract_code (audits 759-760); the trainer's legacy call passes none and runs unchanged."""
    from extraction_siglip2 import extract_code as _extract_code
    from evaluation_siglip2 import evaluation as _evaluation
    # Optionally swap cache for final evaluation (e.g., train on FAIRrank
    # multi-view cache, evaluate on whole-image cache for paper-claim
    # inference mode). Restores after eval so any post-hoc analysis still
    # has training cache pointer if needed.
    _eval_cache = getattr(args, "eval_cache_dir", None)
    _saved_cache = args.siglip2_feature_cache_dir
    _saved_whiten = getattr(args, "text_whiten_npz", None)
    from p0_protocol import resolve_eval_text_whiten_path

    _eval_whiten = resolve_eval_text_whiten_path(
        training_text_whiten_path=_saved_whiten,
        explicit_eval_text_whiten_path=getattr(
            args, "eval_text_whiten_npz", None
        ),
    )
    if _eval_whiten and not os.path.exists(_eval_whiten):
        raise FileNotFoundError(
            "FINAL extraction text-whitening matrix does not exist: "
            f"{_eval_whiten}"
        )
    # AUTO-DETECT: if user did not set --eval_cache_dir, try stripping known
    # multi-view/crop suffixes from the training cache. This ensures viz +
    # final eval consistently use whole-image inference on any dataset that
    # trained on a FAIRrank/localL8K3 cache with a matching whole-image
    # sibling. User-provided --eval_cache_dir takes precedence.
    if not _eval_cache:
        _train_cache = str(_saved_cache).rstrip("/")
        for _sfx in ("_FAIRrankL8K3", "_localL8K3"):
            if _train_cache.endswith(_sfx):
                _candidate = _train_cache[: -len(_sfx)]
                if os.path.exists(_candidate):
                    _eval_cache = _candidate
                    print(f"[final-eval] AUTO-DETECT whole-image cache: {_train_cache} (train) -> {_eval_cache} (eval)")
                    args.eval_cache_dir = _eval_cache
                    break
    if _eval_cache and os.path.exists(_eval_cache):
        print(f"[final-eval] OVERRIDE cache: {_saved_cache} -> {_eval_cache}")
        args.siglip2_feature_cache_dir = _eval_cache
    args.text_whiten_npz = _eval_whiten
    if _eval_whiten != _saved_whiten:
        print(f"[final-eval] EXPLICIT text_whiten override: {_eval_whiten}")
    elif _eval_cache and _saved_whiten:
        print(
            "[final-eval] PRESERVE training text_whiten across cache "
            f"override: {_saved_whiten}"
        )
    try:
        print("[final-eval] running extraction ...")
        if verified is None:
            _extract_code(args)
        else:
            _extract_code(args, verified=verified)
    except Exception as ex:
        # A paper run whose final extraction failed has no codes to report.
        # Swallowing this produced tables traced to a stale extraction.
        raise RuntimeError(
            f"[final-eval] extraction failed: {ex}. This is fatal for a "
            f"paper run; the reported codes would come from a previous "
            f"extraction in the same directory.") from ex
    finally:
        # The explicit override is scoped to extraction. Post-eval
        # diagnostics and the in-memory training model retain the
        # training transform contract.
        args.text_whiten_npz = _saved_whiten
    try:
        print(f"[final-eval] running evaluation (distance_mode={distance_mode}) ...")
        from evaluation_siglip2 import resolve_map_at_r as _resolve_map_at_r
        _map_r = _resolve_map_at_r(getattr(args, "dataset", None))
        if _map_r is not None:
            print(f"[final-eval] paper mAP@R cutoff for {args.dataset} = {_map_r}")
        _evaluation(
            args.save_result_path,
            distance_mode=distance_mode,
            codebook_size=codebook_size,
            map_at_r=_map_r,
            dataset_name=getattr(args, "dataset", None),
        )
    except Exception as ex:
        # The extraction above is fatal for exactly this reason, and the
        # evaluation is what turns those codes into the reported number.
        # Printing and continuing let a run finish with rc 0 and no metric
        # JSON at all, which downstream then reads as a completed cell.
        raise RuntimeError(
            f"[final-eval] evaluation failed: {ex}. This is fatal for a "
            f"paper run: the cell would exit successfully with no metrics, "
            f"and a reader cannot tell that from a cell that has them."
        ) from ex
