#!/usr/bin/env python
"""Candidate F record (audit 742.2): the four-dataset anchor-model freeze, PREPARATION ONLY.

Reads JSON/CSV metadata only (records, receipts, plan snapshots, the frozen N record, the stage-D
summary, the probe envelopes, the stage-L decision, the ledger text). It never opens config.pt,
a checkpoint, a criterion payload, an extraction or a cached array, and it does not import the
launcher or the reducer: the recipe digest helper is loaded by file path from the v7 tree's
`dna_utils/scientific_recipe.py` (stdlib imports only), which is the code that sealed those recipes.

  check  (default)  load every input, run every check, print the problems; write nothing
  --write           also write the candidate record once (O_EXCL, mode 0444) if no check fails
  --self-test       run the checks on deliberately broken in-memory copies; every mutant must be
                    caught by its declared check, and the unbroken inputs must pass
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

V7 = Path("/data/yschoi/gdna_anchor_confirm_v1")
V8 = Path("/data/yschoi/gdna_anchor_lambda_v8")
A7 = V7 / "artifacts/anchor_confirmation"
A8 = V8 / "artifacts/anchor_confirmation"
LEDGER = Path("/home/yschoi/GroundedDNA/docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md")
OUT = A8 / "ancF_candidate_v1.json"
DATASETS = ("cifar10", "flickr25k", "nuswide", "mscoco")
SEEDS = (42, 43, 44)

#: every input file, at the digest the audit has accepted (sections 725-742)
PINS = {
    "v7_manifest": (A7 / "authority_manifest_v7.json", "c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128"),
    "v7_contract": (V7 / "docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md", "e6c4978e6d30559e01c2ee34ae2b4d39c01717557ee16c9e96f85980aa8d3efc"),
    "todo_migration": (V7 / "docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md", "32e7b88feffa9da9e4f7d527e4819e3d617267f1220b6b1c1a7008910642833e"),
    "select_sources": (A7 / "ancS7_select_sources.json", "0cfab95106bfadb03e5e9750d8e5a17d7ff16523ff14b66bba1cab5519067656"),
    "selected_n": (A7 / "ancS7_selected_n.json", "5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff"),
    "s_receipt": (A7 / "ancS7_sweep_complete.json", "5915768767e3fd12b28c3f09e1a071c26fd5cbddbbac15662c495b80c4876f6b"),
    "s_snapshot": (A7 / "ancS7_snapshot_23e0763089474895.json", "6f2c03527b265078584fdfe49f57b3c9091080aeca5ab637b51a6858de4cea49"),
    "d_receipt": (A7 / "ancD7_sweep_complete.json", "4272eeac4759354f94a58af13d84ab62cf78ca3457fa0d07cd379bd0ad0aa6e8"),
    "d_snapshot": (A7 / "ancD7_snapshot_64f311500273a3ed.json", "e383ff68ec0bb50e58e04c52b7e4c7175660a5e9a0aaeb8b356c8840e78e0451"),
    "probe_sources": (A7 / "ancP7_probe_sources.json", "39b304e92a94f879a71bdbfdc8b2cf9bce3209e81e09bdf460d047a4ca2b8702"),
    "decide_sources": (A7 / "ancP7_decide_sources.json", "19678606b6ad530ec781b59197a284fd685b942823ec14280e6e1a6690179de2"),
    "decision": (A7 / "ancP7_decision.json", "28b10a4c850f508bacf4d9402cda8c458fe92d2bfc8b616e14a1a6ce9866cc32"),
    "l_proposal": (V7 / "docs/ANCHOR_LAMBDA_PROPOSAL_L_v1.md", "fe375fdfe24b27dd59da7d0c2fdeb2bf994a5cb2953542486724a62b7c1c3a8f"),
    "v8_manifest": (A8 / "authority_manifest_v8r2.json", "58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc"),
    "l_contract": (V8 / "docs/ANCHOR_LAMBDA_CONTRACT_v1.md", "34a62bfa05600b76f316b3ad62f0da42b8aae8ae2be4fdad490647d834907dfe"),
    "l_receipt": (A8 / "ancL8_sweep_complete.json", "0991f0a031eea624de9ff42cb96c034589e596307a0bf61c2d9710f58c24b838"),
    "l_snapshot": (A8 / "ancL8_snapshot_1eab97a081863d9f.json", "d7c8874f054753fa73af3633d753571b2714a66ed024883749bcb07ae9c5ff24"),
    "l_sources": (A8 / "ancL8_lambda_sources.json", "e0c32688fdb86ac13f3bf6ede3326f633df051b391b4f3592fd20f55e785b003"),
    "l_decision": (A8 / "ancL8_lambda_decision.json", "22014e6bc21dee508eb949a0272b2bf60b4a33ee706e8c301f716c114932f37e"),
    "p3_aggregate": (Path("/data/yschoi/gdna_p3exec/artifacts/phase3_selection/p3rfB_refit_aggregate.json"),
                     "b4f3b0dff467f7c1fd4ca134edba66085115939a4097e6bb201bd13ca83452a5"),
    "p3_selected_n": (Path("/data/yschoi/gdna_p3exec/artifacts/phase3_selection/selected_n.json"),
                      "2bf6133d8cdc7471e40a33d20f1bb19228aa770e67a999a436efb5abac6b2549"),
}
#: audit 742.2's cross-check table (selected fields, not the complete recipe)
TABLE = {
    "cifar10":   {"N": 4,  "lambda_wasserstein": 0.15, "lambda_bu": 0.02, "lambda_text_hash_ntxent": 0.05,
                  "routing_adaptive_topp_min": 0.3, "routing_adaptive_topp_max": 0.7, "lambda_codon_joint": 0.02},
    "flickr25k": {"N": 4,  "lambda_wasserstein": 0.15, "lambda_bu": 0.02, "lambda_text_hash_ntxent": 0.05,
                  "routing_adaptive_topp_min": 0.6, "routing_adaptive_topp_max": 0.95, "lambda_codon_joint": 0.02},
    "nuswide":   {"N": 4,  "lambda_wasserstein": 0.15, "lambda_bu": 0.02, "lambda_text_hash_ntxent": 0.05,
                  "routing_adaptive_topp_min": 0.4, "routing_adaptive_topp_max": 0.8, "lambda_codon_joint": 0.05},
    "mscoco":    {"N": 39, "lambda_wasserstein": 0.05, "lambda_bu": 0.02, "lambda_text_hash_ntxent": 0.10,
                  "routing_adaptive_topp_min": 0.6, "routing_adaptive_topp_max": 0.95, "lambda_codon_joint": 0.03},
}
FIXED = {"axis_center": "anchors", "use_gumbel_softmax": False, "hash_target_mode": "siglip_cos",
         "disable_text_supervision": False, "num_semantic_parts": 5, "num_codebooks": 5}
LAMBDAS = ("lambda_wasserstein", "lambda_bu", "lambda_text_hash_ntxent")
APPROVALS = {"s": (725, "stage-S-run"), "d": (727, "stage-D-run"), "probe": (730, "probe"), "l": (739, "stage-L-run")}

_spec = importlib.util.spec_from_file_location("v7_scientific_recipe", V7 / "dna_utils/scientific_recipe.py")
SR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SR)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jdig(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load() -> dict:
    """Every input parsed from the bytes whose digest is recorded with it."""
    ctx = {"sha": {}, "json": {}, "records": {}, "probes": {}}
    for key, (path, _want) in PINS.items():
        raw = Path(path).read_bytes()
        ctx["sha"][key] = hashlib.sha256(raw).hexdigest()
        if path.suffix == ".json":
            ctx["json"][key] = json.loads(raw)
    for c in ctx["json"]["decide_sources"]["coordinates"]:
        for kind, store in (("record", "records"), ("probe", "probes")):
            raw = Path(c[kind]).read_bytes()
            ctx[store][c[kind]] = {"sha": hashlib.sha256(raw).hexdigest(), "json": json.loads(raw)}
    ctx["ledger"] = LEDGER.read_text()
    ctx["ledger_sha256"] = hashlib.sha256(ctx["ledger"].encode()).hexdigest()
    return ctx


def ledger_line_stands(text: str, section: int, line: str) -> bool:
    lines = text.splitlines()
    start = [i for i, l in enumerate(lines) if l.startswith(f"## {section}. ")]
    if len(start) != 1:
        return False
    end = next((i for i in range(start[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return any(l.strip().strip("`").strip() == line for l in lines[start[0]:end])


def check(ctx: dict) -> list:
    """(tag, message) for every failed check; empty when the candidate is consistent."""
    P = []
    need = lambda ok, tag, msg: ok or P.append((tag, msg))
    J = ctx["json"]
    for key, (path, want) in PINS.items():
        need(ctx["sha"][key] == want, "pins", f"{key}: {path} is {ctx['sha'][key][:12]}, not {want[:12]}")
    sel, dec = J["selected_n"], J["decision"]
    coords = J["decide_sources"]["coordinates"]
    # membership: 4 datasets x seeds 42/43/44 at the frozen N, anchors only, nothing missing or repeated
    keys = [(c["dataset"], c["arm"], c["N"], c["seed"]) for c in coords]
    want_keys = sorted((ds, "anchors", sel["n_selected"][ds]["anchors"], s) for ds in DATASETS for s in SEEDS)
    need(sorted(keys) == want_keys and len(set(keys)) == len(keys), "membership",
         f"decide sources hold {sorted(keys)}, not {want_keys}")
    need(len({c["record_sha256"] for c in coords}) == len(coords) and len({c["probe_sha256"] for c in coords}) == len(coords),
         "membership", "a record or probe is reused across coordinates")
    need(sorted([r["dataset"], r["seed"], r["N"]] for r in dec["stage_R_membership"])
         == sorted([ds, s, sel["n_selected"][ds]["anchors"]] for ds in DATASETS for s in SEEDS)
         and all(r["axis_center"] == "anchors" for r in dec["stage_R_membership"]),
         "membership", "the stage-R membership is not 4 datasets x seeds 42/43/44 at the frozen N")
    # generation: one v7 manifest and contract for the N record, the D summary, every campaign and probe
    v7m = PINS["v7_manifest"][1]
    for name in ("selected_n", "decision"):
        g = J[name]["generation"]
        need(g["anchor_manifest_sha256"] == v7m and g["contract_sha256"] == PINS["v7_contract"][1],
             "generation", f"{name} names generation {g}")
    need(dec["selection"]["sha256"] == PINS["selected_n"][1], "generation", "the D summary names another N record")
    need(sel["fixed_architecture"] == dec["fixed_architecture"] == J["l_decision"]["fixed_architecture"]
         and sel["fixed_architecture"]["axis_center"] == "anchors"
         and sel["fixed_architecture"]["datasets"] == list(DATASETS),
         "generation", "the fixed-architecture authority differs between the N record, D summary and L decision")
    stages = {}
    for st in ("s", "d"):
        receipt, snap = J[f"{st}_receipt"], J[f"{st}_snapshot"]
        a = snap["plan"]["authorities"]
        need(jdig(snap) == receipt["plan_snapshot_sha256"] and receipt["plan_snapshot_file"] == PINS[f"{st}_snapshot"][0].name,
             "links", f"{st}: the receipt does not name this snapshot")
        need(a["anchor_manifest"]["sha256"] == v7m, "generation", f"{st}: campaign ran under another manifest")
        sec, scope = APPROVALS[st]
        ap = a["anchor_approval"]
        need(ap["section"] == sec and ap["scope"] == scope and f"request={jdig(a['anchor_request'])}" in ap["line"]
             and f"manifest={v7m}" in ap["line"] and ledger_line_stands(ctx["ledger"], sec, ap["line"]),
             "approval", f"{st}: approval line {sec}/{scope} not standing for this request")
        stages[st] = (receipt, snap, jdig(a["anchor_request"]))
    need("selection=" + PINS["selected_n"][1] in J["d_snapshot"]["plan"]["authorities"]["anchor_approval"]["line"],
         "approval", "the D approval does not name the frozen N record")
    # per coordinate: links, N, typed recipe, scores, probes
    per = {}
    for c in coords:
        ds, n, seed = c["dataset"], c["N"], c["seed"]
        st = "s" if seed == 42 else "d"
        receipt, snap, _req = stages[st]
        need(Path(c["receipt"]).name == PINS[f"{st}_receipt"][0].name and c["receipt_sha256"] == PINS[f"{st}_receipt"][1],
             "links", f"{ds}/{seed}: evidence names receipt {c['receipt']}")
        rec_entry, probe_entry = ctx["records"][c["record"]], ctx["probes"][c["probe"]]
        rec = rec_entry["json"]
        need(rec_entry["sha"] == c["record_sha256"], "links", f"{ds}/{seed}: record bytes are not {c['record_sha256'][:12]}")
        cid = rec["campaign"]["cell_id"]
        cell = receipt["cells"].get(cid) or {}
        need(cell.get("record") == Path(c["record"]).name and cell.get("record_sha256") == c["record_sha256"],
             "links", f"{ds}/{seed}: the receipt does not list this record for {cid}")
        need(rec["dataset"] == ds and rec["N"] == n and rec["seed"] == seed and n == sel["n_selected"][ds]["anchors"]
             and n == dec["summary"][ds]["frozen_N"] and n == TABLE[ds]["N"]
             and rec["completion"]["final_checkpoint_epoch_zero_based"] == n,
             "N", f"{ds}/{seed}: N differs across record, N record, D summary and table")
        b = snap["plan"]["cell_bindings"].get(cid) or {}
        payload = b.get("scientific_recipe") or {}
        try:
            SR.check_payload_shape(payload)
            recomputed = SR.digest(payload)
        except Exception as error:                       # a malformed payload is a finding, not a crash
            recomputed = f"malformed: {error}"
        need(b.get("N") == n and b.get("anchor_arm") == "anchors"
             and recomputed == b.get("expected_scientific_recipe_sha256")
             == rec["anchor_confirmation"]["scientific_recipe_sha256"],
             "recipe", f"{ds}/{seed}: sealed recipe digest {str(recomputed)[:12]} is not the binding's/record's")
        if seed == 42:
            ev = sel["evidence"].get(f"{ds}|anchors|{n}|42") or {}
            need(ev.get("record_sha256") == c["record_sha256"] and ev.get("score") == rec["selection"]["selection_value"],
                 "links", f"{ds}: the N record's evidence is not this seed-42 record")
        probe = probe_entry["json"]
        need(probe_entry["sha"] == c["probe_sha256"] and probe["coordinate"] == [ds, "anchors", n, seed]
             and probe["record_sha256"] == c["record_sha256"] and probe["manifest_sha256"] == v7m
             and probe["hits"] / probe["total"] == probe["code_picks_own_axis"] and probe["total"] == 2000,
             "probe", f"{ds}/{seed}: probe envelope does not bind this record")
        pa = probe["approval"]
        need(pa["section"] == APPROVALS["probe"][0] and ledger_line_stands(ctx["ledger"], pa["section"], pa["line"])
             and f"request={probe['request_sha256']}" in pa["line"], "approval", f"{ds}/{seed}: probe approval")
        need(c["probe"] in dec["consumed_sha256"] and dec["consumed_sha256"][c["probe"]] == c["probe_sha256"]
             and dec["consumed_sha256"].get(c["record"]) == c["record_sha256"],
             "links", f"{ds}/{seed}: the D summary did not consume this record and probe")
        per[(ds, seed)] = {"cid": cid, "fields": payload.get("fields") or {}, "recipe_sha256": recomputed,
                           "score": rec["selection"]["selection_value"], "ratio": probe["code_picks_own_axis"],
                           "hits": probe["hits"], "total": probe["total"]}
    # typed recipe equality across seeds, the cross-check table and the approved incumbent authorities
    agg, p3sel = J["p3_aggregate"]["datasets"], J["p3_selected_n"]
    for ds in DATASETS:
        if not all((ds, s) in per for s in SEEDS):
            continue
        f42 = per[(ds, 42)]["fields"]
        for s in (43, 44):
            moved = SR.field_differences(f42, per[(ds, s)]["fields"])
            need(moved == ["random_seed"] and per[(ds, s)]["fields"].get("random_seed") == s,
                 "recipe", f"{ds}: seed {s}'s sealed recipe differs from seed 42's in {moved}")
        need(f42.get("random_seed") == 42, "recipe", f"{ds}: seed-42 recipe random_seed")
        for k, v in {**TABLE[ds], **FIXED}.items():
            if k == "N":
                continue
            need(SR.canonical(f42.get(k)) == SR.canonical(v), "table", f"{ds}: {k} is {f42.get(k)!r}, table says {v!r}")
        recipe = agg[ds]["recipe"]
        need(float(recipe["routing_adaptive_topp_min"]) == f42["routing_adaptive_topp_min"]
             and float(recipe["routing_adaptive_topp_max"]) == f42["routing_adaptive_topp_max"]
             and float(recipe["lambda_codon_joint"]) == f42["lambda_codon_joint"],
             "table", f"{ds}: the sealed recipe's top-p/joint are not the approved aggregate's")
        # the aggregate records no lambdas: they come from each dataset's pinned wrapper, as sealed
        dsum = dec["summary"][ds]
        need(dsum["retrieval"]["per_seed"] == [per[(ds, s)]["score"] for s in SEEDS]
             and dsum["code_to_axis"]["per_seed"] == [per[(ds, s)]["ratio"] for s in SEEDS],
             "links", f"{ds}: the D summary's per-seed values are not these records' and probes'")
    # the stage-L decision: v8 generation, v7 history by digest, Flickr-first scope, no change
    L = J["l_decision"]
    lf = L["datasets"].get("flickr25k") or {}
    need(L["artifact_kind"] == "anchor_confirmation_lambda_decision" and L["scope"] == ["flickr25k"]
         and L["generation"]["anchor_manifest_sha256"] == PINS["v8_manifest"][1],
         "lambda", "the L decision is not the v8 Flickr-first decision")
    need(L["history"]["manifest"] == v7m and L["history"]["selection"]["sha256"] == PINS["selected_n"][1]
         and L["history"]["decision"]["sha256"] == PINS["decision"][1]
         and L["history"]["stage_s_receipt"]["sha256"] == PINS["s_receipt"][1],
         "lambda", "the L decision's v7 history pins are not this lineage")
    need(lf.get("changed_axes") == [] and all(v is None for v in (lf.get("winners") or {"x": 1}).values())
         and lf.get("control", {}).get("equal_to_incumbent_seed42") is True and lf.get("N") == 4
         and ("flickr25k", 42) in per
         and all(lf["recipe"][k] == per[("flickr25k", 42)]["fields"][k] for k in LAMBDAS),
         "lambda", "the L decision changed a Flickr lambda or does not keep the validated recipe")
    need(L["consumed_sha256"].get("artifacts/anchor_confirmation/ancL8_lambda_sources.json") == PINS["l_sources"][1],
         "lambda", "the L decision did not consume the submitted sources")
    la = J["l_snapshot"]["plan"]["authorities"]["anchor_approval"]
    need(la["section"] == APPROVALS["l"][0] and la["scope"] == APPROVALS["l"][1]
         and ledger_line_stands(ctx["ledger"], la["section"], la["line"]), "approval", "L approval line")
    ctx["_per"] = per
    return P


def build(ctx: dict) -> dict:
    J, per = ctx["json"], ctx["_per"]
    rel = lambda key: {"path": str(PINS[key][0]), "sha256": ctx["sha"][key]}
    datasets = {}
    for ds in DATASETS:
        f42 = per[(ds, 42)]["fields"]
        dsum = J["decision"]["summary"][ds]
        coords = {c["seed"]: c for c in J["decide_sources"]["coordinates"] if c["dataset"] == ds}
        datasets[ds] = {
            "axis_center": "anchors",
            "N": J["selected_n"]["n_selected"][ds]["anchors"],
            "lambdas": {k: f42[k] for k in LAMBDAS},
            "routing_adaptive_topp": [f42["routing_adaptive_topp_min"], f42["routing_adaptive_topp_max"]],
            "lambda_codon_joint": f42["lambda_codon_joint"],
            "use_gumbel_softmax": f42["use_gumbel_softmax"],
            "lambda_authority": ("stage-L decision (Flickr-first): no candidate qualified; the approved values are kept"
                                 if ds == "flickr25k" else
                                 "the dataset's own approved values, kept under the Flickr-first L scope; "
                                 "no Flickr candidate was run on or copied to this dataset"),
            "validated_recipe": {
                "source": "v7 stage-S plan snapshot, seed-42 cell binding",
                "snapshot": rel("s_snapshot"), "cell_id": per[(ds, 42)]["cid"],
                "scientific_recipe_sha256": per[(ds, 42)]["recipe_sha256"], "field_count": len(f42),
                "seeds_43_44": "sealed recipes equal this one except random_seed (D snapshot cell bindings)",
                "selection_protocol_fields": {k: f42[k] for k in (
                    "epoch", "lr_schedule_horizon", "sinkhorn_schedule_horizon", "stop_after_epoch",
                    "val_split_ratio", "val_split_seed", "selection_mode", "keep_final_checkpoint")},
            },
            "seed_records": {str(s): {"stage": "S" if s == 42 else "D", "record": coords[s]["record"],
                                     "record_sha256": coords[s]["record_sha256"], "receipt": coords[s]["receipt"],
                                     "receipt_sha256": coords[s]["receipt_sha256"], "cell_id": per[(ds, s)]["cid"],
                                     "scientific_recipe_sha256": per[(ds, s)]["recipe_sha256"],
                                     "validation_mAP_at_R": per[(ds, s)]["score"]} for s in SEEDS},
            "probes": {str(s): {"probe": coords[s]["probe"], "probe_sha256": coords[s]["probe_sha256"],
                                "hits": per[(ds, s)]["hits"], "total": per[(ds, s)]["total"],
                                "code_picks_own_axis": per[(ds, s)]["ratio"]} for s in SEEDS},
            "stage_d_summary": {"retrieval": dsum["retrieval"], "code_to_axis": dsum["code_to_axis"]},
        }
    s_auth = J["s_snapshot"]["plan"]["authorities"]
    d_auth = J["d_snapshot"]["plan"]["authorities"]
    probe0 = ctx["probes"][J["decide_sources"]["coordinates"][0]["probe"]]["json"]
    return {
        "artifact_kind": "anchor_confirmation_freeze_candidate",
        "schema": "anchor-freeze-candidate/1",
        "status": "PREPARATION ONLY: candidate F record for audit review; not accepted; it authorizes no "
                  "stage R or T, official-test access, source deployment or downstream work",
        "prepared_under": {"audit_section": 742, "ledger": str(LEDGER), "ledger_sha256": ctx["ledger_sha256"]},
        "fixed_architecture": J["selected_n"]["fixed_architecture"],
        "datasets": datasets,
        "validation_lineage": {
            "generation": "v7", "manifest": rel("v7_manifest"), "contract": rel("v7_contract"),
            "commit": J["selected_n"]["generation"]["commit"],
            "stage_s": {"receipt": rel("s_receipt"), "snapshot": rel("s_snapshot"),
                        "snapshot_semantic_sha256": J["s_receipt"]["plan_snapshot_sha256"],
                        "request_sha256": jdig(s_auth["anchor_request"]), "approval": s_auth["anchor_approval"]},
            "frozen_n_record": {**rel("selected_n"), "sources": rel("select_sources"), "rule": J["selected_n"]["rule"],
                                "reduction_permission": "audit section 726.3 (prose); accepted in section 727"},
            "stage_d": {"receipt": rel("d_receipt"), "snapshot": rel("d_snapshot"),
                        "snapshot_semantic_sha256": J["d_receipt"]["plan_snapshot_sha256"],
                        "request_sha256": jdig(d_auth["anchor_request"]), "approval": d_auth["anchor_approval"]},
            "probes": {"sources": rel("probe_sources"), "request_sha256": probe0["request_sha256"],
                       "approval": probe0["approval"]},
            "stage_d_summary": {**rel("decision"), "sources": rel("decide_sources"),
                                "reduction_permission": "audit section 731 (prose); accepted in section 731/732",
                                "stage_R_membership": J["decision"]["stage_R_membership"]},
        },
        "lambda_check": {
            "scope": "Flickr25K first (user decision answering audit 733.1): five one-axis candidates plus a "
                     "continuity control at N4 seed 42; expand to another dataset only if Flickr's choice moves",
            "stop_rule": "no candidate qualified, so no other dataset runs stage L and no N is reselected "
                         "(audit 742.1)",
            "generation": "v8", "manifest": rel("v8_manifest"), "contract": rel("l_contract"),
            "proposal": rel("l_proposal"), "receipt": rel("l_receipt"), "snapshot": rel("l_snapshot"),
            "approval": J["l_snapshot"]["plan"]["authorities"]["anchor_approval"],
            "sources": rel("l_sources"), "decision": rel("l_decision"),
            "reduction_permission": "audit section 741 (prose); verified in section 742",
            "rule": J["l_decision"]["rule"]["comparison"],
            "threshold": J["l_decision"]["datasets"]["flickr25k"]["threshold"],
            "result": {"changed_axes": [], "recipe": J["l_decision"]["datasets"]["flickr25k"]["recipe"]},
            "role": "the accepted lambda-check authority; linked separately, never merged into an S/D reduction",
        },
        "incumbent_recipe_authorities": {"approved_refit_aggregate": rel("p3_aggregate"),
                                         "approved_selected_n": rel("p3_selected_n")},
        "todo_migration": rel("todo_migration"),
        "not_claimed": [
            "global hyperparameter optimality, statistical equivalence or significance",
            "superiority of anchors over the incumbent model or over baselines",
            "completion of stage R or T, or of any downstream TODO item",
            "any old-model, smoke or Gumbel-ON exploratory result as evidence for this model",
        ],
    }


MUTANTS = {
    "duplicate seed": ("membership", lambda c: c["json"]["decide_sources"]["coordinates"].__setitem__(
        2, copy.deepcopy(c["json"]["decide_sources"]["coordinates"][1]))),
    "Flickr lambda copied to MS-COCO": ("table", lambda c: TABLE["mscoco"].__setitem__("lambda_wasserstein", 0.15)),
    "D summary at another N": ("N", lambda c: c["json"]["decision"]["summary"]["nuswide"].__setitem__("frozen_N", 9)),
    "seed-43 recipe drift": ("recipe", lambda c: c["json"]["d_snapshot"]["plan"]["cell_bindings"][
        next(k for k in c["json"]["d_snapshot"]["plan"]["cell_bindings"] if k.startswith("cifar10|") and "seed=43" in k)
    ]["scientific_recipe"]["fields"].__setitem__("lambda_bu", 0.0)),
    "L decision changed an axis": ("lambda", lambda c: c["json"]["l_decision"]["datasets"]["flickr25k"].__setitem__(
        "changed_axes", ["lambda_bu"])),
    "probe of another record": ("probe", lambda c: next(iter(c["probes"].values()))["json"].__setitem__(
        "record_sha256", "0" * 64)),
    "record bytes changed": ("links", lambda c: next(iter(c["records"].values())).__setitem__("sha", "f" * 64)),
    "approval line gone": ("approval", lambda c: c.__setitem__("ledger", c["ledger"].replace(
        "scope=stage-D-run", "scope=stage-D-gone"))),
    "pinned input replaced": ("pins", lambda c: c["sha"].__setitem__("decision", "e" * 64)),
    "fixed architecture dropped a dataset": ("generation", lambda c: c["json"]["decision"]["fixed_architecture"]
                                             .__setitem__("datasets", ["cifar10", "flickr25k", "nuswide"])),
}


def self_test(ctx: dict) -> int:
    base = check(copy.deepcopy(ctx))
    print(f"unbroken inputs: {len(base)} problems")
    caught = 0
    for name, (tag, mutate) in MUTANTS.items():
        saved = copy.deepcopy(TABLE)
        c = copy.deepcopy(ctx)
        mutate(c)
        tags = sorted({t for t, _ in check(c)})
        TABLE.clear(); TABLE.update(saved)
        ok = tag in tags
        caught += ok
        print(f"  {'CAUGHT' if ok else 'MISSED'}  {name:40s} declared={tag:10s} raised={tags}")
    print(f"{caught}/{len(MUTANTS)} mutants caught by their declared check")
    return 0 if not base and caught == len(MUTANTS) else 1


def main(argv) -> int:
    ctx = load()
    if "--self-test" in argv:
        return self_test(ctx)
    problems = check(ctx)
    print(f"inputs: {len(PINS)} pinned files, {len(ctx['records'])} records, {len(ctx['probes'])} probes; "
          f"ledger {ctx['ledger_sha256'][:12]}")
    if problems:
        print("PROBLEMS:", *[f"[{t}] {m}" for t, m in problems], sep="\n  ")
        return 1
    print("all checks pass: membership, generation, links, N, recipe, table, probes, approvals, lambda")
    record = build(ctx)
    blob = (json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    if "--write" in argv:
        fd = os.open(OUT, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
        print("wrote", OUT, sha(OUT))
    else:
        print(f"dry: candidate is {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
