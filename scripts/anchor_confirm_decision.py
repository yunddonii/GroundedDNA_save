#!/usr/bin/env python
"""Anchor confirmation reducer, v3 (contract v3 sections 7, 8, 13; audit sections 668-673, 709).

The architecture is FIXED (audit 709, the user's decision): `axis_center=anchors` for all four
datasets. Nothing here chooses between anchors and a control; retrieval and alignment are reported
and never select the architecture, drop a dataset or keep an old refit. Missing or invalid evidence
still refuses.

select  -- evidence for every (dataset, anchors, N in the grid, seed 42) -> each dataset's N:
           argmax of the raw base-Hamming mAP@R at the cell's own terminal epoch, ties to the
           smallest N. Writes the frozen N record; `--anchor-confirm decide` and `decide` below
           re-verify it by replaying this reduction, never by trusting its digest alone.
decide  -- the frozen N record (replayed) + evidence for every (dataset, anchors, frozen N, seed
           42/43/44) + one probe per record -> per dataset the descriptive validation summary
           (retrieval and code-to-own-axis per seed, mean, sample SD) and the stage-R membership it
           implies: a scratch full-train anchor refit at the frozen N for seeds 42/43/44.

Evidence authority (audit 672.1, 679). A record counts for its coordinate only through a completed
anchor-confirmation campaign that ran under an audit approval: the pinned receipt lists the
coordinate's cell id (recomputed here), the record, a complete campaign binding and the completion
pins; the plan snapshot it names carries the same campaign nonce, the generation manifest, the
approval line -- re-verified NOW in the audit ledger for the stage's scope (seed 42: stage-S-run;
seeds 43/44: stage-D-run, naming the frozen N record) -- and the cell's sealed recipe, whose shape,
digest and protocol values for this coordinate are checked; the trainer evidence, the runtime
sidecar and the record's completed anchor check (the typed config.pt comparison that ran inside the
campaign, `assert_anchor_recipe`) agree with it. Contract v3 admits no historical reuse: every
stage-S/D record is a fresh anchor cell. JSON-level admission completes before any config.pt is
read (audit 673.1).

Generation (audit 671.2, 678.2, 697). The reducer runs only in the tree the reviewed generation
manifest pins -- re-verified at entry, with the modules' import digests, and again before it
publishes -- and every record must come from a campaign whose plan snapshot names that manifest:
one generation carries stage S, the frozen N and stage D.

Inputs (audit 694, 696). The approved request's seal pins, the campaign's admitted seals, each
record's input authority and its launch/cell bindings' input identities must all agree.

Probe population (audit 709.3). Every probe measures the first 500 rows (ascending dataset index)
of its dataset's train-only validation split, 4 local slots each: 2000 strict decisions. CIFAR-10 and
Flickr25K have exactly 500 validation rows; the same policy applies to all four datasets.

The reducer NEVER deserialises a binary (audit 679.2, 681): config.pt is checked by byte identity
against its admitted pin only. Every file is read once and parsed from its hashed bytes, re-verified
before the output is written once (O_EXCL). Membership is the contract's, never a count taken from
an input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M  # noqa: E402

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()
REDUCER_VERSION = "anchor-confirm-reducer/3"
N_SELECTION_KIND = "anchor_confirmation_n_selection"
DECISION_KIND = "anchor_confirmation_decision"
PROBE_KIND = "anchor_confirmation_code_axis"
PROBE_SCHEMA = "anchor-confirm-code-axis/3"
PROBE_SOURCE = REPO / "scripts" / "anchor_confirm_code_axis.py"
PROBE_IMAGES = 500
PROBE_LOCAL_SLOTS = 4
#: the preregistered probe population of contract v3 section 8.2 (audit 709.3)
PROBE_POPULATION = {"rows": "first 500 of the train-only validation split, ascending dataset index",
                    "n_images": PROBE_IMAGES, "local_slots": PROBE_LOCAL_SLOTS,
                    "decisions": PROBE_IMAGES * PROBE_LOCAL_SLOTS}
FIXED_ARCHITECTURE = {"axis_center": "anchors", "datasets": list(M.ANCHOR_DATASETS),
                      "authority": "the user's decision recorded in audit section 709"}
SPLIT = {"val_split_ratio": 0.1, "val_split_seed": 42}
#: the completion pins a receipt cell must carry (the launcher's receipt carries these and more)
RECEIPT_COMPLETION_KEYS = ("final_checkpoint_sha256", "log_csv_sha256", "checkpoint_runtime_sha256",
                           "phase3_campaign_evidence_sha256")
CAPTION_TARGET = {"features": "cached_text_part_raw local axes",
                  "adapter": "_adapt_pooled_text_for_loss of the same checkpoint"}


class NotReducible(RuntimeError):
    """The evidence cannot support the requested output."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise NotReducible(message)


class Consumed:
    """Every file read, at the digest of the bytes that were read (audit section 500 pattern)."""

    def __init__(self) -> None:
        self.digests: dict = {}

    def read(self, path, want: str | None = None) -> bytes:
        name = str(Path(path))
        try:
            data = Path(path).read_bytes()
        except OSError as error:
            raise NotReducible(f"{name} is unreadable: {error}") from None
        digest = hashlib.sha256(data).hexdigest()
        need(self.digests.get(name, digest) == digest, f"{name} changed between two reads")
        if want is not None:
            need(digest == want, f"{name} is not the pinned bytes {str(want)[:12]}...")
        self.digests[name] = digest
        return data

    def json(self, path, want: str | None = None):
        return json.loads(self.read(path, want).decode("utf-8"))

    def reverify(self) -> list:
        return [name for name, want in sorted(self.digests.items())
                if not Path(name).is_file()
                or hashlib.sha256(Path(name).read_bytes()).hexdigest() != want]


def proportion(value, what: str) -> float:
    need(not isinstance(value, bool) and isinstance(value, (int, float)), f"{what} is {value!r}")
    number = float(value)
    need(math.isfinite(number), f"{what} is {number!r}, not finite")
    need(0.0 <= number <= 1.0, f"{what} is {number!r}, outside [0, 1]")
    return number


def same(value, expected) -> bool:
    """Type-exact equality: 4.0 is not 4 and True is not 1 (audit 669.1, applied to the reducer)."""
    return type(value) is type(expected) and value == expected


def is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def same_cell(cell, coordinate) -> bool:
    return isinstance(cell, list) and len(cell) == 4 and all(same(a, b) for a, b in zip(cell, coordinate))


def exact_int(value, what: str) -> int:
    need(type(value) is int, f"{what} is {value!r}, not an integer")
    return value


def expected_coordinates(stage: str, *, arms, frozen=None) -> list:
    arms = tuple(arms)
    need(arms == M.ANCHOR_RUN_ARMS,
         f"arms {arms} are not the contract's {M.ANCHOR_RUN_ARMS} (the architecture is fixed)")
    if stage == "select":
        return [(ds, arm, n, M.SEED) for ds in M.ANCHOR_DATASETS for arm in arms
                for n in M.CANDIDATE_N]
    coords = []
    for ds in M.ANCHOR_DATASETS:
        for arm in arms:
            n = exact_int(frozen[ds][arm], f"frozen N of {ds}/{arm}")
            need(n in M.CANDIDATE_N, f"frozen N of {ds}/{arm} is {n}, off the grid {M.CANDIDATE_N}")
            coords += [(ds, arm, n, seed) for seed in (M.SEED, *M.ANCHOR_DECIDE_SEEDS)]
    return coords


def campaign_approval(snapshot: dict, scope: str, *, manifest_sha256: str, selection_sha256,
                      receipt: dict, record: dict, coordinate) -> dict:
    """The approval and execution request the campaign names in its plan snapshot, re-verified in
    the audit ledger now (audit 679.1, 689.2): the same line must still stand in the same section,
    for this scope, this generation, (stage D) this frozen N record and exactly this request; and
    the request must be the one this record ran under -- stage, mode, namespace, result root and a
    cell equal to the record's coordinate."""
    authorities = snapshot["plan"].get("authorities") or {}
    recorded = authorities.get("anchor_approval")
    need(isinstance(recorded, dict), "the campaign names no audit approval")
    recorded = recorded if isinstance(recorded, dict) else {}
    need(recorded.get("scope") == scope,
         f"the campaign ran under approval scope {recorded.get('scope')!r}, not {scope}")
    request = authorities.get("anchor_request")
    need(isinstance(request, dict), "the campaign names no execution request")
    request = request if isinstance(request, dict) else {}
    pins = {"manifest": manifest_sha256, "request": M._json_digest(request)}
    if "selection" in M.APPROVAL_SCOPES[scope]:
        need(is_sha256(selection_sha256), f"{scope} evidence needs the frozen N record's digest")
        pins["selection"] = selection_sha256
    live = M.audit_approval(recorded.get("section"), scope, **pins)
    need(live["line"] == recorded.get("line"), "the approval line the campaign recorded is not the ledger's")
    stage = "select" if scope.startswith("stage-S") else "decide"
    seals = snapshot.get("input_seals")
    admitted = {key: {"path": (a or {}).get("seal_path"), "sha256": (a or {}).get("seal_file_sha256")}
                for key, a in (seals or {}).items()} if isinstance(seals, dict) else None
    need(bool(admitted) and admitted == request.get("input_seals"),
         "the campaign's admitted input seals are not its approved request's")
    need(request.get("schema") == M.REQUEST_SCHEMA and request.get("stage") == stage
         and request.get("mode") == "run" and request.get("manifest") == manifest_sha256
         and request.get("selection") == (selection_sha256 if stage == "decide" else None)
         and request.get("namespace") == receipt.get("namespace")
         and any(same_cell(cell, coordinate) for cell in request.get("cells") or [])
         and Path(record["run_dir"]).parent == Path(str(request.get("result_root"))),
         f"{coordinate}: the campaign's approved execution request is not the one this record ran under")
    return {"section": live["section"], "scope": scope, "line": live["line"],
            "request_sha256": pins["request"]}


def probe_request(manifest_sha256: str, selection_sha256: str, records: dict) -> dict:
    """The canonical request of the probe operation one approval covers: this generation, this
    frozen N record, exactly these stage-D records ({coordinate: record digest}) and the contract's
    row population (audit 709.3: the image count and the four-slot denominator are part of what is
    approved)."""
    return {"schema": M.REQUEST_SCHEMA, "version": M.ANCHOR_CONFIRM_VERSION, "operation": "probe",
            "manifest": manifest_sha256, "selection": selection_sha256,
            "population": dict(PROBE_POPULATION),
            "records": sorted([*c, digest] for c, digest in records.items())}


def admit_metadata(consumed: Consumed, coordinate, entry: dict, *, incumbent: dict,
                   manifest_sha256: str, selection_sha256=None) -> dict:
    """JSON-level admission of one record; no binary artifact is opened here."""
    from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
    from dna_utils.scientific_recipe import RecipeMismatch, canonical, check_payload_shape
    from dna_utils.scientific_recipe import digest as recipe_digest
    ds, arm, n, seed = coordinate
    record = consumed.json(entry["record"], entry["record_sha256"])
    need(same(record.get("dataset"), ds) and same(record.get("N"), n) and same(record.get("seed"), seed),
         f"{entry['record']}: not the record of {coordinate}")
    need(record.get("stage") == "select" and record.get("selection_mode") == "select",
         f"{entry['record']}: not a stage-1 selection record (refit/test records are refused)")
    need(record.get("smoke") is False and record.get("is_candidate_cell") is True,
         f"{entry['record']}: a smoke or non-candidate record is not evidence")
    need(same(record.get("val_split_ratio"), SPLIT["val_split_ratio"])
         and same(record.get("val_split_seed"), SPLIT["val_split_seed"]),
         f"{entry['record']}: not the 90/10 train-only split with seed 42")
    completion, selection = record["completion"], record["selection"]
    need(same(completion.get("final_checkpoint_epoch_zero_based"), n)
         and same(selection.get("selection_epoch_zero_based"), n),
         f"{entry['record']}: terminal checkpoint/selection epoch is not N={n}")
    need(selection.get("selection_metric") == "eval_mAP_at_R"
         and selection.get("distance_mode") == "base", f"{entry['record']}: not the raw base-Hamming mAP@R")
    run_dir = Path(record["run_dir"])
    source = entry.get("source")
    if source == "receipt":
        receipt = consumed.json(entry["receipt"], entry["receipt_sha256"])
        need(receipt.get("campaign_kind") == M.ANCHOR_CAMPAIGN_KIND
             and (receipt.get("anchor_confirmation") or {}).get("version") == M.ANCHOR_CONFIRM_VERSION,
             f"{entry['receipt']}: not an {M.ANCHOR_CONFIRM_VERSION} campaign receipt")
        nonce = receipt.get("campaign_nonce")
        need(isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{64}", nonce) is not None
             and is_sha256(receipt.get("plan_snapshot_sha256")),
             f"{entry['receipt']}: no campaign nonce or plan snapshot digest")
        cell_id = M.campaign_cell_id(ds, n, topp=incumbent[ds]["topp"], joint=incumbent[ds]["joint"],
                                     stage="select", seed=seed, anchor_arm=arm)
        cell = (receipt.get("cells") or {}).get(cell_id)
        need(isinstance(cell, dict), f"{entry['receipt']}: no completed cell {cell_id}")
        cell = cell if isinstance(cell, dict) else {}
        need(Path(entry["record"]).name == cell.get("record") and entry["record_sha256"] == cell.get("record_sha256"),
             f"{entry['record']}: not the record the receipt lists for {cell_id}")
        campaign, pins = cell.get("campaign"), cell.get("completion")
        need(isinstance(campaign, dict) and campaign.get("cell_id") == cell_id
             and campaign.get("plan_snapshot_sha256") == receipt["plan_snapshot_sha256"]
             and campaign.get("campaign_nonce") == nonce,
             f"{entry['receipt']}: the campaign binding of {cell_id} is missing or another campaign's")
        need(isinstance(pins, dict) and all(is_sha256(pins.get(k)) for k in RECEIPT_COMPLETION_KEYS),
             f"{entry['receipt']}: the completion pins of {cell_id} are missing")
        need(record.get("campaign") == campaign and all(completion.get(k) == v for k, v in pins.items()),
             f"{entry['record']}: campaign or completion evidence differs from the receipt")
        snapshot = consumed.json(Path(entry["receipt"]).parent / receipt["plan_snapshot_file"])
        need(M._json_digest(snapshot) == receipt["plan_snapshot_sha256"],
             f"{receipt['plan_snapshot_file']}: not the receipt's plan snapshot")
        need(snapshot["plan"].get("campaign_nonce") == nonce,
             f"{receipt['plan_snapshot_file']}: the snapshot is another campaign's")
        generation = ((snapshot["plan"].get("authorities") or {}).get("anchor_manifest") or {})
        need(is_sha256(generation.get("sha256")),
             f"{receipt['plan_snapshot_file']}: the campaign names no generation manifest")
        approval = campaign_approval(snapshot, "stage-S-run" if seed == M.SEED else "stage-D-run",
                                     manifest_sha256=manifest_sha256, selection_sha256=selection_sha256,
                                     receipt=receipt, record=record, coordinate=coordinate)
        binding = snapshot["plan"]["cell_bindings"].get(cell_id)
        need(isinstance(binding, dict) and binding.get("anchor_arm") == arm,
             f"{cell_id}: the plan snapshot does not seal this arm's recipe")
        payload = binding.get("scientific_recipe")
        try:
            check_payload_shape(payload)
        except RecipeMismatch as error:
            raise NotReducible(f"{cell_id}: the sealed recipe is malformed: {error}") from None
        need(recipe_digest(payload) == binding.get("expected_scientific_recipe_sha256"),
             f"{cell_id}: the sealed recipe is not its declared digest")
        inputs = (snapshot.get("input_seals") or {}).get(f"{ds}:stage1")
        need(isinstance(inputs, dict) and record.get("input_authority") == inputs,
             f"{entry['record']}: its input authority is not the campaign's admitted {ds}:stage1 seal")
        for field, value in (("input_seal_sha256", inputs.get("seal_file_sha256")),
                             ("input_aggregate_sha256", inputs.get("aggregate_sha256")),
                             ("input_authority_sha256", inputs.get("authority_sha256")),
                             ("split_identity_sha256", inputs.get("split_identity_sha256")),
                             ("hf_identity_sha256", (inputs.get("hf_runtime") or {}).get("identity_sha256"))):
            need(is_sha256(value) and campaign.get(field) == value and binding.get(field) == value,
                 f"{cell_id}: its launch or cell binding's {field} is not the admitted seal's")
        want = M.anchor_protocol_fields(ds, n, seed=seed, arm=arm, incumbent=incumbent)
        wrong = sorted(k for k, v in want.items()
                       if k not in payload["fields"] or canonical(payload["fields"][k]) != canonical(v))
        need(not wrong, f"{cell_id}: the sealed recipe is not the contract's for {coordinate}: {wrong[:6]}")
        evidence = consumed.json(run_dir / PHASE3_CAMPAIGN_BINDING_NAME,
                                 completion["phase3_campaign_evidence_sha256"])
        need(evidence.get("cell_id") == cell_id
             and evidence.get("scientific_recipe_sha256") == binding["expected_scientific_recipe_sha256"],
             f"{run_dir}: trainer evidence does not name {cell_id}'s sealed recipe")
        sidecar = consumed.json(run_dir / f"{completion['final_checkpoint']}.runtime.json",
                                completion["checkpoint_runtime_sha256"])
        need(same(sidecar.get("checkpoint_epoch_zero_based"), n)
             and (sidecar.get("extra") or {}).get("phase3_campaign") == evidence,
             f"{run_dir}: the runtime sidecar is not this cell's terminal checkpoint witness")
        anchor = record.get("anchor_confirmation") or {}
        need(anchor.get("version") == M.ANCHOR_CONFIRM_VERSION and anchor.get("arm") == arm
             and anchor.get("scientific_recipe_sha256") == binding["expected_scientific_recipe_sha256"]
             and anchor.get("campaign_evidence_sha256") == completion["phase3_campaign_evidence_sha256"]
             and is_sha256(anchor.get("config_pt_sha256")),
             f"{entry['record']}: no anchor evidence for arm {arm} (the completed check of the sealed recipe)")
        authority = {"kind": "receipt", "receipt": str(entry["receipt"]), "cell_id": cell_id,
                     "manifest_sha256": generation.get("sha256"), "approval": approval}
        config_pin = anchor["config_pt_sha256"]
    else:
        raise NotReducible(f"{coordinate}: contract v3 admits campaign receipts only (no "
                           f"historical reuse), got source {source!r}")
    rows = list(csv.DictReader(io.StringIO(consumed.read(
        run_dir / "log.csv", completion["log_csv_sha256"]).decode("utf-8"))))
    need(bool(rows) and rows[-1].get("epoch", "").strip().isdigit()
         and int(rows[-1]["epoch"]) == n, f"{run_dir}: the terminal log row is not epoch {n}")
    try:
        raw_score = float(rows[-1]["eval_mAP_at_R"])
    except (KeyError, ValueError):
        raise NotReducible(f"{run_dir}: the terminal row has no mAP@R") from None
    score = proportion(raw_score, f"{run_dir} terminal mAP@R")
    need(score == proportion(selection["selection_value"], "record selection value"),
         f"{entry['record']}: the record's selection value is not the pinned log's")
    return {"record": record, "entry": entry, "run_dir": run_dir, "score": score,
            "authority": authority, "config_pt_sha256": config_pin}


def verify_config_pin(consumed: Consumed, admitted: dict) -> str:
    """Byte identity of the saved config.pt against its admitted pin -- and nothing more: the reducer
    never deserialises (audit 679.2, 681). The typed comparison with the sealed recipe ran inside
    the approved campaign (`assert_anchor_recipe`); the record's anchor block carries its result and
    the receipt pins the record."""
    want = admitted["config_pt_sha256"]
    consumed.read(admitted["run_dir"] / "config.pt", want)
    return want


def verify_probe(consumed: Consumed, coordinate, entry: dict, admitted: dict, config_digest: str, *,
                 manifest_sha256: str, selection_sha256: str, request_sha256: str) -> dict:
    """The typed measurement envelope (audit 672.2, 680.3): endpoint schema and producer; exact
    coordinate, record, checkpoint, config and execution authority; the generation and the probe's
    own approval, re-verified in the ledger; the admitted split identity and caption input seal;
    integer hits, ties and total under the strict rule; the unrounded ratio as a float."""
    probe = consumed.json(entry["probe"], entry["probe_sha256"])
    name = entry["probe"]
    need(probe.get("artifact_kind") == PROBE_KIND and probe.get("schema") == PROBE_SCHEMA,
         f"{name}: not a {PROBE_SCHEMA} probe")
    need(probe.get("producer_sha256") == hashlib.sha256(consumed.read(PROBE_SOURCE)).hexdigest(),
         f"{name}: produced by another probe source")
    stated = probe.get("coordinate")
    need(isinstance(stated, list) and len(stated) == 4
         and all(same(a, b) for a, b in zip(stated, coordinate)),
         f"{name}: measured another coordinate than {coordinate}")
    record = admitted["record"]
    completion = record["completion"]
    need(probe.get("record_sha256") == admitted["entry"]["record_sha256"]
         and probe.get("checkpoint_sha256") == completion["final_checkpoint_sha256"]
         and probe.get("config_pt_sha256") == config_digest,
         f"{name}: measured another record, checkpoint or configuration than {coordinate}")
    need(probe.get("authority") == admitted["authority"],
         f"{name}: not measured under this record's execution authority")
    need(probe.get("manifest_sha256") == manifest_sha256, f"{name}: produced under another generation")
    approval = probe.get("approval")
    need(isinstance(approval, dict), f"{name}: names no probe approval")
    live = M.audit_approval(approval.get("section"), "probe", manifest=manifest_sha256,
                            selection=selection_sha256, request=request_sha256)
    need(live["line"] == approval.get("line") and probe.get("request_sha256") == request_sha256,
         f"{name}: its approval line or request is not the ledger's")
    split = probe.get("split")
    need(isinstance(split, dict) and set(split) == set(SPLIT)
         and all(same(split[k], v) for k, v in SPLIT.items())
         and probe.get("routing") == "deployment_no_text",
         f"{name}: not the train-only validation rows under deployment routing")
    inputs = record.get("input_authority") or {}
    need(is_sha256(probe.get("split_identity_sha256"))
         and probe["split_identity_sha256"] == inputs.get("split_identity_sha256"),
         f"{name}: its rows were not drawn under the record's admitted split identity")
    caption = probe.get("caption_target")
    need(isinstance(caption, dict)
         and {k: caption.get(k) for k in CAPTION_TARGET} == CAPTION_TARGET
         and caption.get("input_seal_sha256") == inputs.get("seal_file_sha256"),
         f"{name}: its caption target is not the record's admitted input")
    need(probe.get("population") == PROBE_POPULATION,
         f"{name}: measured another population than the contract's {PROBE_POPULATION}")
    n_images = exact_int(probe.get("n_images"), "probe n_images")
    hits = exact_int(probe.get("hits"), "probe hits")
    total = exact_int(probe.get("total"), "probe total")
    need(n_images == PROBE_IMAGES and total == PROBE_LOCAL_SLOTS * n_images and 0 <= hits <= total,
         f"{name}: counts are not {PROBE_IMAGES} images x {PROBE_LOCAL_SLOTS} local slots")
    ties = exact_int(probe.get("ties_counted_as_misses"), "probe ties")
    need(0 <= ties <= total - hits, f"{name}: {ties} ties do not fit {total - hits} misses")
    ratio = probe.get("code_picks_own_axis")
    need(type(ratio) is float and ratio == hits / total, f"{name}: the reported ratio is not hits/total")
    need(is_sha256(probe.get("row_ids_sha256")), f"{name}: no row-identity digest")
    return {"ratio": hits / total, "hits": hits, "total": total,
            "rows": (probe["row_ids_sha256"], probe["split_identity_sha256"])}


def load_sources(consumed: Consumed, path, want: str, coordinates, *, with_probe: bool) -> dict:
    sources = consumed.json(path, want)
    need(sources.get("version") == M.ANCHOR_CONFIRM_VERSION, f"{path}: wrong sources version")
    entries = sources.get("coordinates")
    need(isinstance(entries, list), f"{path}: no coordinate list")
    keyed = {}
    for entry in entries:
        key = (entry["dataset"], entry["arm"], exact_int(entry["N"], "N"), exact_int(entry["seed"], "seed"))
        need(key not in keyed, f"duplicate evidence for {key}")
        keyed[key] = entry
    need(set(keyed) == set(coordinates),
         f"evidence membership differs from the contract: missing "
         f"{sorted(set(coordinates) - set(keyed))[:4]}, extra {sorted(set(keyed) - set(coordinates))[:4]}")
    for key in ("record_sha256",) + (("probe_sha256",) if with_probe else ()):
        digests = [entry[key] for entry in entries]
        need(len(set(digests)) == len(digests), f"one {key.split('_')[0]} is reused across coordinates")
    return keyed


def one_generation(admitted: dict, manifest_sha256: str) -> None:
    """Every receipt-backed record comes from a campaign that ran under this manifest."""
    named = sorted({a["authority"]["manifest_sha256"] for a in admitted.values()
                    if a["authority"]["kind"] == "receipt"})
    need(named == [manifest_sha256], f"the campaigns ran under generation manifests "
                                     f"{[d[:12] for d in named]}, not {manifest_sha256[:12]}...")


def generation_of(manifest: dict) -> dict:
    return {"anchor_manifest_sha256": manifest["sha256"], "commit": manifest.get("commit"),
            "contract_sha256": (manifest.get("contract") or {}).get("sha256")}


def select_n(scores: dict) -> int:
    """argmax of the score, ties to the smallest N."""
    best = max(scores.values())
    return min(n for n, value in scores.items() if value == best)


def reduce_select(sources, sources_sha256, *, manifest: dict, arms=M.ANCHOR_RUN_ARMS,
                  consumed=None) -> dict:
    consumed = consumed or Consumed()
    incumbent = M.anchor_incumbent()
    coords = expected_coordinates("select", arms=arms)
    keyed = load_sources(consumed, sources, sources_sha256, coords, with_probe=False)
    admitted = {c: admit_metadata(consumed, c, keyed[c], incumbent=incumbent,
                                  manifest_sha256=manifest["sha256"]) for c in coords}
    one_generation(admitted, manifest["sha256"])
    configs = {c: verify_config_pin(consumed, admitted[c]) for c in coords}
    n_selected, scores = {}, {}
    for ds in M.ANCHOR_DATASETS:
        for arm in arms:
            per_n = {c[2]: admitted[c]["score"] for c in coords if c[0] == ds and c[1] == arm}
            n_selected.setdefault(ds, {})[arm] = select_n(per_n)
            scores.setdefault(ds, {})[arm] = {str(n): v for n, v in sorted(per_n.items())}
    return {"artifact_kind": N_SELECTION_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "reducer": REDUCER_VERSION, "rule": "argmax raw base-Hamming mAP@R at own terminal "
            "epoch, ties to the smallest N, seed 42", "arms": list(arms),
            "fixed_architecture": dict(FIXED_ARCHITECTURE),
            "generation": generation_of(manifest),
            "sources": {"path": str(sources), "sha256": sources_sha256},
            "n_selected": n_selected, "scores": scores,
            "evidence": {"|".join(map(str, c)): {"authority": admitted[c]["authority"],
                                                 "record_sha256": admitted[c]["entry"]["record_sha256"],
                                                 "config_pt_sha256": configs[c],
                                                 "score": admitted[c]["score"]} for c in coords},
            "_consumed": consumed}


def verify_selection(path, want, *, consumed=None) -> dict:
    """Replay the stage-S reduction a frozen N record names and require the same choices
    (audit 671.2, 672.3): a caller's digest proves bytes, not that the selection happened."""
    consumed = consumed or Consumed()
    frozen = consumed.json(path, want)
    need(frozen.get("artifact_kind") == N_SELECTION_KIND
         and frozen.get("version") == M.ANCHOR_CONFIRM_VERSION
         and frozen.get("reducer") == REDUCER_VERSION, f"{path}: not a {REDUCER_VERSION} N record")
    need(frozen.get("arms") == list(M.ANCHOR_RUN_ARMS)
         and frozen.get("fixed_architecture") == FIXED_ARCHITECTURE,
         f"{path}: not a record of the fixed four-dataset anchor model")
    generation = frozen.get("generation") or {}
    need(isinstance(generation.get("anchor_manifest_sha256"), str),
         f"{path}: names no generation manifest")
    replay = reduce_select(frozen["sources"]["path"], frozen["sources"]["sha256"],
                           manifest={"sha256": generation["anchor_manifest_sha256"],
                                     "commit": generation.get("commit"),
                                     "contract": {"sha256": generation.get("contract_sha256")}},
                           arms=tuple(frozen["arms"]), consumed=consumed)
    replay.pop("_consumed")
    need(replay["n_selected"] == frozen["n_selected"] and replay["scores"] == frozen["scores"],
         f"{path}: its N choices are not what its own evidence reduces to")
    return {"n_selected": frozen["n_selected"], "record": {"path": str(path), "sha256": want},
            "anchor_manifest_sha256": generation["anchor_manifest_sha256"]}


def reduce_decide(args, manifest: dict) -> dict:
    """Descriptive stage-D summary of the fixed anchor model and the stage-R membership it implies.
    No score selects the architecture: a poor retrieval or alignment result is reported as it is."""
    consumed = Consumed()
    incumbent = M.anchor_incumbent()
    selection = verify_selection(args.selection, args.selection_sha256, consumed=consumed)
    need(selection["anchor_manifest_sha256"] == manifest["sha256"],
         "stage S ran under another generation manifest; one generation carries the whole chain")
    n_frozen = selection["n_selected"]
    need(set(n_frozen) == set(M.ANCHOR_DATASETS)
         and all(set(n_frozen[ds]) == set(M.ANCHOR_RUN_ARMS) for ds in n_frozen),
         f"the frozen N record does not cover the anchor arm of all of {M.ANCHOR_DATASETS}")
    coords = expected_coordinates("decide", arms=M.ANCHOR_RUN_ARMS, frozen=n_frozen)
    keyed = load_sources(consumed, args.sources, args.sources_sha256, coords, with_probe=True)
    admitted = {c: admit_metadata(consumed, c, keyed[c], incumbent=incumbent,
                                  manifest_sha256=manifest["sha256"],
                                  selection_sha256=args.selection_sha256) for c in coords}
    one_generation(admitted, manifest["sha256"])
    configs = {c: verify_config_pin(consumed, admitted[c]) for c in coords}
    request_sha256 = M._json_digest(probe_request(
        manifest["sha256"], args.selection_sha256, {c: keyed[c]["record_sha256"] for c in coords}))
    probes = {c: verify_probe(consumed, c, keyed[c], admitted[c], configs[c],
                              manifest_sha256=manifest["sha256"], selection_sha256=args.selection_sha256,
                              request_sha256=request_sha256)
              for c in coords}
    summary, refits = {}, []
    seeds = (M.SEED, *M.ANCHOR_DECIDE_SEEDS)
    arm = "anchors"
    for ds in M.ANCHOR_DATASETS:
        need(len({probes[c]["rows"] for c in coords if c[0] == ds}) == 1,
             f"{ds}: the probes did not measure one shared row population (rows and split identity) "
             "across seeds")
        n = n_frozen[ds][arm]
        val = [admitted[(ds, arm, n, s)]["score"] for s in seeds]
        axis = [probes[(ds, arm, n, s)]["ratio"] for s in seeds]
        summary[ds] = {
            "axis_center": arm, "frozen_N": n,
            "retrieval": {"per_seed": val, "mean": statistics.fmean(val),
                          "sample_sd": statistics.stdev(val),                  # ddof = 1
                          "metric": "raw base-Hamming mAP@R at the terminal epoch, train-only "
                                    f"validation, R={M.MAP_R_CUTOFF[ds]}"},
            "code_to_axis": {"per_seed": axis,
                             "hits_total": [[probes[(ds, arm, n, s)]["hits"],
                                             probes[(ds, arm, n, s)]["total"]] for s in seeds],
                             "mean": statistics.fmean(axis), "sample_sd": statistics.stdev(axis)}}
        refits += [{"dataset": ds, "axis_center": arm, "N": n, "seed": s,
                    "refit": "scratch, full designated train split"} for s in seeds]
    return {"artifact_kind": DECISION_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "reducer": REDUCER_VERSION, "selection": selection["record"],
            "generation": generation_of(manifest), "fixed_architecture": dict(FIXED_ARCHITECTURE),
            "summary": summary, "stage_R_membership": refits,
            "note": "train-only; per dataset; n = 3; descriptive -- no score selects the "
                    "architecture, no threshold, significance or equivalence test; stage R/T are "
                    "separate authorizations; the stage-R membership is provisional until the "
                    "final recipe freeze: a lambda the required checks (TODO 13-15) change "
                    "supersedes it for that dataset (contract v3 section 7.6)",
            "_consumed": consumed}


def write_once(path: Path, payload: dict) -> str:
    consumed = payload.pop("_consumed")
    changed = consumed.reverify()
    need(not changed, f"inputs changed before the output was written: {changed[:4]}")
    payload["consumed_sha256"] = dict(sorted(consumed.digests.items()))
    blob = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select", "decide"))
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--manifest", required=True, help="the reviewed generation manifest")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--selection", help="decide: the frozen N record")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--arms", default="anchors", help="select: must be `anchors` (contract v3)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = M.recheck_generation({"path": args.manifest, "sha256": args.manifest_sha256},
                                        "at the reducer's entry")
        if args.stage == "select":
            payload = reduce_select(args.sources, args.sources_sha256, manifest=manifest,
                                    arms=tuple(a for a in args.arms.split(",") if a))
        else:
            need(bool(args.selection) and bool(args.selection_sha256),
                 "decide needs --selection and --selection-sha256")
            payload = reduce_decide(args, manifest)
        M.recheck_generation(manifest, "before the reducer publishes")
        digest = write_once(Path(args.out), payload)
    except (NotReducible, M.CellRefused, KeyError, ValueError, TypeError, FileExistsError) as error:
        print(f"[anchor-reducer] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-reducer] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
