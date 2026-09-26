#!/usr/bin/env python
"""Code-to-own-axis probe, v2 (contract section 8.2; audit sections 665.2, 668.4, 672.2, 673).

Endpoint (contract v2 section 8.2, proposed for audit acceptance): WITHIN-IMAGE CODEWORD-TO-OWN-AXIS
TOP-1 ACCURACY (STRICT).
  rows      -- the run's own train-only validation split (the trainer's val_split.carve_val_indices
               with val_split_ratio 0.1, seed 42), ascending dataset index, the first 512; fewer than
               512 validation rows refuses (no smaller population is substituted). The rows' cache
               identities (`trainset._feat_cache_rows`, the admitted dataset-to-cache mapping) are
               hashed into `row_ids_sha256`; every probe of a dataset must report the same digest
               and the same admitted split identity;
  forward   -- deployment: eval mode, no caption reaches the model; the local slots' quantised
               codewords `quantized_tokens[:, 1:, :]` (4 local slots, in slot order);
  target    -- the same images' cached caption features for the four axes, through the same
               checkpoint's text adapter (`_adapt_pooled_text_for_loss`), local axes in the same
               order; captions are only the target;
  centring  -- in float64, per slot, over the whole selected set at once (not per loader batch),
               then L2 normalisation; a centred vector with norm <= 1e-12 refuses the measurement;
  hit       -- for image i and slot m, the own-axis cosine is STRICTLY greater than the three other
               axes' (a tie is a miss; the exploratory probe's first-index argmax is not used);
  output    -- integer hits, ties and total (= 4 x 512) and the unrounded float ratio hits / total.
The arithmetic differs from the exploratory quant_gap probe in the tie rule and the row source,
so its old values are not the same measurement.

Admission comes first (audit 673.1, 679, 681): the tree must be the reviewed generation manifest's;
the audit ledger must carry a `probe` approval line naming that manifest and the frozen N record;
the frozen N record must replay (JSON, logs, pinned bytes); the coordinate must be one of its
stage-D coordinates; its record must pass the reducer's JSON-level admission (an approved campaign's
receipt); and the config.pt and checkpoint bytes must be their pins -- all before the first
deserialisation. Then the trainer's own input admission (`_phase3_input_authority_from_args`:
seal stats and runtime paths) must reproduce the record's input authority before any model or
dataset is built. Every input is read once and parsed from its hashed bytes; the output is written
once.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.anchor_confirm_decision as D  # noqa: E402

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()
LOCAL_SLOTS = 4
ZERO_NORM = 1e-12


def strict_own_axis_hits(q, t) -> dict:
    """[B, 4, D] codewords and [B, 4, D] caption embeddings of the same images and slots."""
    import torch
    D.need(isinstance(q, torch.Tensor) and isinstance(t, torch.Tensor), "inputs must be tensors")
    D.need(q.dim() == 3 and q.shape == t.shape and q.shape[0] > 0 and q.shape[1] == LOCAL_SLOTS
           and q.shape[2] > 0, f"inputs must be matching nonempty [B, {LOCAL_SLOTS}, D], got "
                               f"{tuple(q.shape)} and {tuple(t.shape)}")
    q, t = q.double(), t.double()
    D.need(bool(torch.isfinite(q).all()) and bool(torch.isfinite(t).all()), "non-finite input features")
    qc, tc = q - q.mean(0, keepdim=True), t - t.mean(0, keepdim=True)
    norms = torch.cat([qc.norm(dim=-1).flatten(), tc.norm(dim=-1).flatten()])
    D.need(bool((norms > ZERO_NORM).all()), "a centred feature has zero norm; the measurement refuses")
    S = torch.einsum("bmd,bad->bma", qc / qc.norm(dim=-1, keepdim=True),
                     tc / tc.norm(dim=-1, keepdim=True))                     # [B, slot, axis]
    D.need(bool(torch.isfinite(S).all()), "non-finite similarities")
    own = S.diagonal(dim1=1, dim2=2)                                        # [B, slot]
    others = S.masked_fill(torch.eye(LOCAL_SLOTS, dtype=torch.bool), float("-inf")).amax(dim=2)
    hits = int((own > others).sum().item())
    ties = int((own == others).sum().item())
    total = int(own.numel())
    return {"hits": hits, "total": total, "ties_counted_as_misses": ties, "ratio": hits / total}


def validation_split(trainset, args):
    """(opt_idx, val_idx) exactly as the trainer carves them."""
    from val_split import carve_val_indices, get_train_labels
    D.need(float(args.val_split_ratio) == D.SPLIT["val_split_ratio"]
           and int(args.val_split_seed) == D.SPLIT["val_split_seed"], "not the contract's split")
    opt_idx, val_idx, _ = carve_val_indices(get_train_labels(trainset), ratio=float(args.val_split_ratio),
                                            seed=int(args.val_split_seed))
    return opt_idx, val_idx


def held_out_rows(trainset, args) -> list:
    _, val_idx = validation_split(trainset, args)
    ordered = sorted(int(i) for i in val_idx)
    D.need(len(ordered) >= D.PROBE_IMAGES, f"the validation split has {len(ordered)} rows, fewer "
                                           f"than the contract's {D.PROBE_IMAGES}")
    return ordered[:D.PROBE_IMAGES]


def probe(sources, sources_sha256, coordinate, *, manifest: dict, approval: dict, selection: dict,
          reuse_path=None, reuse_sha256=None, device: str = "cuda:0") -> dict:
    """`manifest`, `approval` (with its `request_sha256`) and `selection` are verified by the caller
    before this is called."""
    import torch
    from types import SimpleNamespace
    import scripts.phase3_selection_matrix as M
    consumed = D.Consumed()
    incumbent = M.anchor_incumbent()
    D.need(coordinate in D.expected_coordinates("decide", arms=M.ANCHOR_ARMS,
                                                 frozen=selection["n_selected"]),
           f"{coordinate} is not a stage-D coordinate of the approved frozen N record")
    reuse = D.load_reuse(consumed, reuse_path, reuse_sha256)
    listed = consumed.json(sources, sources_sha256)
    D.need(listed.get("version") == M.ANCHOR_CONFIRM_VERSION, f"{sources}: wrong sources version")
    entries = [e for e in listed.get("coordinates", [])
               if (e["dataset"], e["arm"], e["N"], e["seed"]) == coordinate]
    D.need(len(entries) == 1, f"{sources} lists {coordinate} {len(entries)} times")
    admitted = D.admit_metadata(consumed, coordinate, entries[0], incumbent=incumbent, reuse=reuse,
                                manifest_sha256=manifest["sha256"],
                                selection_sha256=selection["record"]["sha256"])
    if admitted["authority"]["kind"] == "receipt":
        D.need(admitted["authority"]["manifest_sha256"] == manifest["sha256"],
               f"{coordinate}: its campaign ran under another generation manifest")
    config_pin = D.verify_config_pin(consumed, admitted)                  # bytes only, no load
    record, run_dir = admitted["record"], admitted["run_dir"]
    completion = record["completion"]
    checkpoint = run_dir / completion["final_checkpoint"]
    ckpt_raw = consumed.read(checkpoint, completion["final_checkpoint_sha256"])
    sidecar = consumed.json(Path(f"{checkpoint}.runtime.json"), completion["checkpoint_runtime_sha256"])
    D.need(D.same(sidecar.get("checkpoint_epoch_zero_based"), coordinate[2]),
           f"{checkpoint}: the runtime sidecar is not the terminal epoch {coordinate[2]}")
    D.need(os.environ.get("GDNA_NUM_SEMANTIC_PARTS") == "5",
           "export GDNA_NUM_SEMANTIC_PARTS=5 before python starts")
    # first deserialisation: the pinned bytes, under the probe approval, in the admitted generation
    M.recheck_generation(manifest, "before the probe's first load")
    config = torch.load(io.BytesIO(consumed.read(run_dir / "config.pt", config_pin)), map_location="cpu",
                        weights_only=False)
    args = SimpleNamespace(**{k: v for k, v in config.items() if not k.startswith("__")})
    from train_siglip2 import _assert_phase3_runtime_rows, _phase3_input_authority_from_args
    try:
        inputs = _phase3_input_authority_from_args(args)            # seal stats + runtime paths
    except RuntimeError as error:
        raise D.NotReducible(f"{coordinate}: input admission refused: {error}") from None
    admitted_inputs = record.get("input_authority") or {}
    keys = ("seal_file_sha256", "aggregate_sha256", "split_identity_sha256")
    D.need(inputs is not None and all(inputs.get(k) == admitted_inputs.get(k) for k in keys),
           f"{coordinate}: the probe's inputs are not the record's admitted inputs")

    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    model = SigLIP2SemanticOTModel(args).to(device).eval()
    result = model.load_state_dict(torch.load(io.BytesIO(ckpt_raw), map_location="cpu",
                                              weights_only=False), strict=False)
    D.need(not result.missing_keys and not result.unexpected_keys,
           f"checkpoint keys do not match the model: missing {result.missing_keys[:4]}, "
           f"unexpected {result.unexpected_keys[:4]}")
    apply_inference_epoch(model, str(checkpoint), args)
    trainset, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset, setting=args.setting,
        train_transform=None, test_transform=None, load_train=True, load_database=False,
        load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=args.siglip2_feature_cache_dir)
    opt_idx, val_idx = validation_split(trainset, args)
    try:                                           # the admitted dataset-to-cache mapping and split
        _assert_phase3_runtime_rows(args, trainset, opt_idx, val_idx)
    except RuntimeError as error:
        raise D.NotReducible(f"{coordinate}: the rows are not the admitted split: {error}") from None
    rows = held_out_rows(trainset, args)
    cache_rows = [int(trainset._feat_cache_rows[i]) for i in rows]
    codes, captions = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            samples = [trainset[i] for i in rows[start:start + 64]]
            out = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                        return_routing=True,
                        cached_visual_tokens_raw=torch.stack(
                            [s["cached_visual_tokens_raw"] for s in samples]).to(device),
                        cached_visual_global=torch.stack(
                            [s["cached_visual_global"] for s in samples]).to(device),
                        cached_text_part_raw=None, cached_has_text=None)   # deployment forward
            raw_text = torch.stack([s["cached_text_part_raw"] for s in samples]).to(device)
            codes.append(out["quantized_tokens"][:, 1:, :].cpu())
            captions.append(model._adapt_pooled_text_for_loss(raw_text)[:, 1:, :].cpu())
    scores = strict_own_axis_hits(torch.cat(codes), torch.cat(captions))
    return {"artifact_kind": D.PROBE_KIND, "schema": D.PROBE_SCHEMA,
            "producer_sha256": hashlib.sha256(consumed.read(Path(__file__).resolve())).hexdigest(),
            "manifest_sha256": manifest["sha256"],
            "approval": {k: approval[k] for k in ("section", "scope", "line")},
            "request_sha256": approval["request_sha256"],
            "coordinate": list(coordinate), "record_sha256": admitted["entry"]["record_sha256"],
            "checkpoint_sha256": completion["final_checkpoint_sha256"],
            "config_pt_sha256": config_pin, "split": dict(D.SPLIT),
            "split_identity_sha256": inputs["split_identity_sha256"],
            "routing": "deployment_no_text", "n_images": len(rows),
            "row_ids_sha256": hashlib.sha256(json.dumps(cache_rows).encode()).hexdigest(),
            "caption_target": {**D.CAPTION_TARGET, "input_seal_sha256": inputs["seal_file_sha256"]},
            "hits": scores["hits"], "total": scores["total"],
            "ties_counted_as_misses": scores["ties_counted_as_misses"],
            "code_picks_own_axis": float(scores["ratio"]),
            "authority": admitted["authority"], "_consumed": consumed}


def probe_records(sources, sources_sha256, selection) -> dict:
    """{coordinate: record digest} for exactly the stage-D coordinates of the frozen N record."""
    import scripts.phase3_selection_matrix as M
    listed = D.Consumed().json(sources, sources_sha256)
    D.need(listed.get("version") == M.ANCHOR_CONFIRM_VERSION, f"{sources}: wrong sources version")
    coords = set(D.expected_coordinates("decide", arms=M.ANCHOR_ARMS, frozen=selection["n_selected"]))
    records = {}
    for e in listed.get("coordinates", []):
        key = (e["dataset"], e["arm"], D.exact_int(e["N"], "N"), D.exact_int(e["seed"], "seed"))
        D.need(key not in records, f"{sources} lists {key} twice")
        records[key] = e["record_sha256"]
    D.need(set(records) == coords, f"{sources} does not list exactly the stage-D coordinates")
    return records


def parse_coordinate(text: str):
    ds, arm, n, seed = text.split(":")
    return ds, arm, int(n), int(seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--manifest", required=True, help="the reviewed generation manifest")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--selection", required=True, help="the frozen N record")
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--approval-section", type=int, required=True,
                        help="the audit ledger section whose `probe` line approves this")
    parser.add_argument("--reuse-admission")
    parser.add_argument("--reuse-admission-sha256")
    parser.add_argument("--coordinate", required=True, help="dataset:arm:N:seed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        D.need(bool(args.reuse_admission) == bool(args.reuse_admission_sha256),
               "--reuse-admission and --reuse-admission-sha256 go together")
        manifest = D.M.recheck_generation({"path": args.manifest, "sha256": args.manifest_sha256},
                                          "at the probe's entry")
        D.need(type(args.approval_section) is int and args.approval_section > 0,
               "an approval reference is a positive ledger section number")
        selection = D.verify_selection(args.selection, args.selection_sha256)   # metadata only
        D.need(selection["anchor_manifest_sha256"] == manifest["sha256"],
               "the frozen N record comes from another generation")
        request = D.probe_request(manifest["sha256"], str(args.selection_sha256),
                                  probe_records(args.sources, args.sources_sha256, selection))
        approval = D.M.audit_approval(args.approval_section, "probe", manifest=manifest["sha256"],
                                      selection=str(args.selection_sha256),
                                      request=D.M._json_digest(request))
        approval["request_sha256"] = D.M._json_digest(request)
        payload = probe(args.sources, args.sources_sha256, parse_coordinate(args.coordinate),
                        manifest=manifest, approval=approval, selection=selection,
                        reuse_path=args.reuse_admission, reuse_sha256=args.reuse_admission_sha256,
                        device=args.device)
        D.M.recheck_generation(manifest, "before the probe publishes")
        digest = D.write_once(Path(args.out), payload)
    except (D.NotReducible, D.M.CellRefused, KeyError, ValueError, FileExistsError) as error:
        print(f"[anchor-probe] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-probe] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
