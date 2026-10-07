"""Stage-2 caption pipeline tools (tools/stage2_*.py): CPU-only, no model, no dataset / cache access."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import stage2_captions as cap  # noqa: E402
import stage2_common as com  # noqa: E402
import stage2_concepts as con  # noqa: E402
import stage2_prompts as pr  # noqa: E402
import stage2_survey as sur  # noqa: E402
import stage2_validate as val  # noqa: E402

PINNED_SHA256 = {
    "survey": "3ecd7904a754852af844a14029f7c046fbc57e188f423e6820f02dc52e99b65e",
    "concepts_update": "63af48a3eba5c582ed91d16a44ac05ae8becac4dd7dc0d769f996e8d5932f7d0",   # kept, unused
    "concepts_batch_v1": "595b6e8f25a0a80e26f74d4c7714d1a3e23edfed129f68145ba059f696ca4377",      # kept, unused
    "concepts_consolidate_v1": "2667d9cb76ce5e97dfdd6162d09fbb910f109b774155c3e1eedf20870aa2c8e0",  # kept, unused
    "concepts_batch": "7a21a3ce6d5432af1d046cd5fa853197c0758f8d74227d42ceaa5596d71c937a",
    "concepts_consolidate": "18ff3bf7f0bd148acc4c261c10428ebad1dcacc50d87f5ac8410171a81678e49",
    "visual_check": "94750a24415d7ead6a60c7a345500469327ffb0b7cc18938baef692bbd51cc0a",
    "attributes": "842795fa73153fbc8c9a7326a55640269f693557f9dd822b63933a908a22e251",
    "attributes_merge": "8b9f24c40e5d8cbf9fc055487c1888ee00c2b9857f18610b7764bfee08520147",
    "captions_template": "c55fba3868ff863fc351a39f32a1e1a4c290f7d61db547bf700d757da8bb17b7",
}
ATTRS = [
    {"name": "Main subject", "key": "Main_subject", "definition": "The main thing.", "concepts": ["dog", "cat"]},
    {"name": "Setting", "key": "Setting", "definition": "Where it is.", "concepts": ["beach", "street"]},
    {"name": "Colour", "key": "Colour", "definition": "Dominant colours.", "concepts": ["red", "blue"]},
    {"name": "Action", "key": "Action", "definition": "What happens.", "concepts": ["running", "sitting"]},
]
KEY_MAP = {"Main_subject": "C_primary_object", "Setting": "C_secondary_object",
           "Colour": "C_activity_or_relation", "Action": "C_color_texture"}
ATTRIBUTES = {"attributes": ATTRS, "key_map": KEY_MAP, "note": ""}
KEYS = [a["key"] for a in ATTRS]


# --------------------------------------------------------------------------- prompts
def test_prompt_constants_are_pinned():
    for name, sha in PINNED_SHA256.items():
        text = pr.ALL_PROMPTS[name]
        assert pr.sha256_of(text) == sha == hashlib.sha256(text.encode("utf-8")).hexdigest(), name
    assert pr.PROMPT_SURVEY.startswith("You are describing one image from a photo collection.")
    assert pr.PROMPT_SURVEY.endswith("Do not guess what is outside the image.")
    assert "<PHRASES>" in pr.PROMPT_CONCEPTS_UPDATE and "<CONCEPTS>" in pr.PROMPT_CONCEPTS_UPDATE
    assert pr.PROMPT_CONCEPTS_BATCH.endswith("Phrases:\n<PHRASES>") and "<CONCEPTS>" not in pr.PROMPT_CONCEPTS_BATCH
    assert pr.PROMPT_CONCEPTS_CONSOLIDATE.endswith("Concepts:\n<CONCEPTS>") and "<PHRASES>" not in pr.PROMPT_CONCEPTS_CONSOLIDATE
    clar = ("A concept names something that can recur across many images in the collection, not a description "
            "of one image; keep each concept to one to three words.")
    for p in (pr.PROMPT_CONCEPTS_BATCH, pr.PROMPT_CONCEPTS_CONSOLIDATE):
        first_stop = p.index(". ") + 2
        assert p[first_stop:].startswith(clar)                                   # right after the first sentence
        assert '{"concepts": ["...", "..."]}' in p and "examples" not in p.split("Output ONLY")[1]
    assert pr.PROMPT_CAPTIONS_TEMPLATE.endswith("- <A4>: <D4>")
    assert set(PINNED_SHA256) == set(pr.ALL_PROMPTS)


def test_caption_prompt_fill_uses_keys_as_json_keys():
    p = pr.build_caption_prompt(ATTRS)
    assert '{"codebook_texts": {"Main_subject": "", "Setting": "", "Colour": "", "Action": "", "C_global": ""}}' in p
    assert "- Main_subject: The main thing." in p and "<A1>" not in p and "<D4>" not in p
    with pytest.raises(ValueError):
        pr.build_caption_prompt(ATTRS[:3])


# --------------------------------------------------------------------------- key map / attributes
def test_key_map_by_position():
    attrs = con.validate_attributes([{"name": a["name"], "definition": a["definition"]} for a in ATTRS])
    assert [a["key"] for a in attrs] == KEYS
    assert con.key_map_by_position(attrs) == KEY_MAP
    reversed_attrs = con.validate_attributes([{"name": a["name"], "definition": a["definition"]} for a in ATTRS[::-1]])
    assert con.key_map_by_position(reversed_attrs)["Action"] == "C_primary_object"
    with pytest.raises(con.ParseFailure):
        con.validate_attributes([{"name": "A", "definition": "d"}] * 4)        # duplicate names
    with pytest.raises(con.ParseFailure):
        con.validate_attributes([{"name": "A", "definition": ""}] + [{"name": f"B{i}", "definition": "d"} for i in range(3)])
    with pytest.raises(con.ParseFailure):
        con.key_map_by_position(attrs[:3])


# --------------------------------------------------------------------------- resume semantics
def test_parse_failure_rows_are_retried(tmp_path):
    p = tmp_path / "survey.shard0.jsonl"
    rows = [{"image_id": "images/a.jpg", "phrases": [], "_parse_error": "Expecting value"},
            {"image_id": "images/b.jpg", "phrases": ["red bike"]}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    assert com.done_ids([p]) == {"images/b.jpg"}
    assert set(com.failed_rows([p])) == {"images/a.jpg"}
    paths = ["/root/images/a.jpg", "/root/images/b.jpg", "/root/images/c.jpg"]
    todo = [x for x in paths if x[len("/root/"):] not in com.done_ids([p])]
    assert todo == ["/root/images/a.jpg", "/root/images/c.jpg"]
    with open(p, "a") as f:                                                    # the retry succeeds
        f.write(json.dumps({"image_id": "images/a.jpg", "phrases": ["dog"]}) + "\n")
    assert com.done_ids([p]) == {"images/a.jpg", "images/b.jpg"}
    assert com.failed_rows([p]) == {}
    assert com.read_rows([p])["images/a.jpg"]["phrases"] == ["dog"]           # last clean row wins


def test_concat_cli_reports_missing_ids(tmp_path, capsys):
    shard = tmp_path / "captions_pilot.shard0.jsonl"
    shard.write_text(json.dumps({"image_id": "x", "codebook_texts": {}}) + "\n"
                     + json.dumps({"image_id": "y", "_parse_error": "e"}) + "\n")
    sample = tmp_path / "s.json"
    sample.write_text(json.dumps({"image_ids": ["x", "y"]}))
    out = tmp_path / "out.jsonl"
    rc = com.main(["concat", "--out", str(out), "--expect", str(sample), "--max_missing_frac", "0.01", str(shard)])
    assert rc == 2 and out.read_text().count("\n") == 1
    rc = com.main(["concat", "--out", str(out), "--expect", str(sample), "--max_missing_frac", "0.5", str(shard)])
    assert rc == 0
    assert com.main(["check", "--expect", str(sample), str(shard)]) == 1


def test_write_json_once_first_writer_wins(tmp_path):
    p = tmp_path / "sample.json"
    a = com.write_json_once(p, {"image_ids": [1]})
    b = com.write_json_once(p, {"image_ids": [2]})
    assert a == b == {"image_ids": [1]}


# --------------------------------------------------------------------------- caption rows
def _attributes_with_sha():
    a = dict(ATTRIBUTES)
    a["_sha256"] = "deadbeef"
    return a


def test_caption_rows_carry_six_legacy_keys_with_empty_scene_type():
    spec = com.DatasetSpec("Flickr25k")
    make_row = cap.make_row_factory(spec, _attributes_with_sha(), "promptsha")
    raw = '```json\n{"codebook_texts": {"Main_subject": "A brown dog", "Setting": "sandy beach at noon", ' \
          '"Colour": "warm brown and blue", "Action": "dog runs after a ball", "C_global": "A brown dog runs on a beach."}}\n```'
    row = make_row(spec.root + "images/im1.jpg", raw)
    assert row["image_id"] == "images/im1.jpg" and "_parse_error" not in row
    assert tuple(row["codebook_texts"]) == com.LEGACY_KEYS
    assert row["codebook_texts"]["C_scene_type"] == ""
    assert row["codebook_texts"]["C_primary_object"] == "A brown dog"
    assert row["codebook_texts"]["C_color_texture"] == "dog runs after a ball"      # positional map, not semantic
    assert row["codebook_texts_v10"] == {"Main_subject": "A brown dog", "Setting": "sandy beach at noon",
                                         "Colour": "warm brown and blue", "Action": "dog runs after a ball",
                                         "C_global": "A brown dog runs on a beach."}
    assert row["prompt_version"] == "v10" and row["prompt_sha256"] == "promptsha" and row["attributes_sha256"] == "deadbeef"
    # extractor view: the six keys it reads are all present
    assert all(k in row["codebook_texts"] for k in ("C_global", "C_primary_object", "C_secondary_object",
                                                     "C_activity_or_relation", "C_color_texture", "C_scene_type"))


@pytest.mark.parametrize("raw", [
    '{"codebook_texts": {"Main_subject": "dog", "Setting": "beach", "Colour": "brown", "C_global": "x"}}',  # missing Action
    '{"codebook_texts": {"Main_subject": "dog", "Setting": "", "Colour": "brown", "Action": "runs", "C_global": "x"}}',  # empty
    '{"codebook_texts": {"Main_subject": "dog", "Setting": "beach", "Colour": "brown", "Action": "runs"',  # unclosed
    'The image shows a dog.',
])
def test_caption_row_missing_or_bad_is_a_parse_error(raw):
    spec = com.DatasetSpec("Flickr25k")
    row = cap.make_row_factory(spec, _attributes_with_sha(), "s")(spec.root + "images/im1.jpg", raw)
    assert row["_parse_error"] and row["codebook_texts"] == {}


def test_load_attributes_rejects_non_positional_key_map(tmp_path):
    bad = {"attributes": ATTRS, "key_map": {**KEY_MAP, "Main_subject": "C_color_texture", "Action": "C_primary_object"}}
    p = tmp_path / "attributes.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        cap.load_attributes(str(p))
    p.write_text(json.dumps(ATTRIBUTES))
    assert cap.load_attributes(str(p))["_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- survey sampling
def test_survey_sampling_uses_opt_rows_only_and_extra_is_disjoint():
    image_ids = [f"images/im{i}.jpg" for i in range(40)]
    opt_rows = np.array([i for i in range(30) if i % 3 != 0])               # train rows 0..29, val rows dropped
    opt = [image_ids[r] for r in opt_rows]
    main = com.sample_ids(opt, 10, 20261007)
    assert len(main) == 10 and len(set(main)) == 10
    assert all(m in set(opt) for m in main)
    assert all(int(re.search(r"im(\d+)", m).group(1)) % 3 != 0 for m in main)  # never a non-opt row
    assert main == com.sample_ids(opt, 10, 20261007)                        # deterministic
    extra = com.sample_ids(opt, 5, 20261008, exclude=main)
    assert len(extra) == 5 and not set(extra) & set(main) and set(extra) <= set(opt)
    with pytest.raises(SystemExit):
        com.sample_ids(opt, len(opt), 1, exclude=main)                      # not enough left
    assert com.shard_of(list(range(10)), 1, 4) == [1, 5, 9]
    assert sur.parse_survey('{"phrases": ["red bike", " ", "wet road"]}') == ["red bike", "wet road"]
    with pytest.raises(ValueError):
        sur.parse_survey('{"phrase": ["x"]}')


# --------------------------------------------------------------------------- concepts with a stub model
def _stub_gen(diverge_shuffle=False, merge_ok=True):
    """Answers each prompt type by sniffing its unique wording."""
    attr_calls = [0]

    def gen(prompt, **kw):
        assert "Concepts found so far" not in prompt                   # the cumulative prompt is never sent
        assert "examples" not in prompt.split("Output ONLY")[-1]        # flat lists only
        if prompt.startswith(pr.PROMPT_CONCEPTS_BATCH[:60]):            # per-batch extraction
            lines = prompt.split("Phrases:\n")[1].splitlines()
            idx = [re.search(r"phrase (\d+) a", ln).group(1) for ln in lines]
            return json.dumps({"concepts": [f"concept {i}" for i in idx] + ["Concept 0"]})
        if prompt.startswith(pr.PROMPT_CONCEPTS_CONSOLIDATE[:60]):     # consolidation: identity
            items = json.loads(prompt.split("Concepts:\n")[1])
            assert all(isinstance(c, str) for c in items)
            return json.dumps({"concepts": items})
        if prompt.startswith("For each concept below"):
            items = json.loads(prompt.split("Concepts:\n")[1])
            return json.dumps({"concepts": [{"name": c, "visible": c != "concept 1"} for c in items]})
        if prompt.startswith("Below is a list of visual concepts"):
            items = json.loads(prompt.split("Concepts:\n")[1])
            assert all(isinstance(c, str) for c in items)
            attr_calls[0] += 1
            names = ["Subject", "Place", "Colours", "Activity"]
            if diverge_shuffle and attr_calls[0] == 2:                 # shuffle 1 names its groups differently
                names = ["Thing", "Where", "Hue", "Motion"]
            groups = [sorted(items)[i::4] for i in range(4)]       # stable membership across shuffles
            return json.dumps({"attributes": [{"name": n, "definition": f"{n} definition.", "concepts": g}
                                              for n, g in zip(names, groups)], "note": ""})
        if prompt.startswith("Below are three groupings"):
            if not merge_ok:
                return '{"attributes": [{"name": "Only", "definition": "d", "concepts": []}], "note": "fewer"}'
            return json.dumps({"attributes": [{"name": n, "definition": f"{n} merged.", "concepts": [f"c{n}"]}
                                              for n in ["Subject", "Place", "Colours", "Activity"]], "note": "merged"})
        raise AssertionError("unknown prompt")
    return gen


def _survey(n=12):
    return {f"images/im{i}.jpg": [f"phrase {i} a", f"phrase {i} b"] for i in range(n)}


def test_concepts_pipeline_with_stub_model(tmp_path):
    logs = []
    attrs = con.run_all(_stub_gen(), _survey(), tmp_path, batch_images=5, shuffles=3, dataset="Flickr25k", log=logs.append)
    cj = json.load(open(tmp_path / "concepts.json"))
    assert len(cj["shuffles"]) == 3 and all(len(s["counts_per_batch"]) == 3 for s in cj["shuffles"])
    # per batch: its image concepts + "Concept 0" (merged with "concept 0" in the batch that holds image 0)
    assert all(sum(s["counts_per_batch"]) == 14 and max(s["counts_per_batch"]) <= 6 for s in cj["shuffles"])
    assert all(s["n_union"] == 12 for s in cj["shuffles"])                         # 12 images -> 12 concepts after dedupe
    assert cj["n_union_all"] == 12 and cj["n_consolidated"] == 12 and len(cj["working"]) == 12
    assert all(isinstance(c, str) for c in cj["working"])                           # names only
    assert len({c.lower() for c in cj["working"]}) == len(cj["working"])           # de-duplicated by lowercased name
    assert "concept 0" in {c.lower() for c in cj["working"]}
    assert cj["agreement"]["sizes"] == [12, 12, 12] and cj["agreement"]["mean_jaccard_names"] == 1.0
    assert "consolidat" in cj["working_list_rule"] and cj["consolidation"]["levels"][0]["n_chunks"] == 1
    assert cj["consolidation"]["retried_chunks"] == [] and cj["consolidation"]["skipped_chunks"] == []
    assert set(cj["prompt_sha256s"]) == {"concepts_batch", "concepts_consolidate"}
    for s in cj["shuffles"]:
        assert s["parse_mode_counts"] == {"strict": 3} and s["retried_batches"] == [] and s["skipped_batches"] == []
    sh0 = json.load(open(tmp_path / "concepts_shuffle0.json"))
    assert sh0["method"] == con.STEP2A_METHOD and len(sh0["batch_lists"]) == 3 and sh0["union_size"] == 12
    assert all(len(bl["image_ids"]) in (5, 2) and bl["parse_mode"] == "strict" for bl in sh0["batch_lists"])
    assert json.load(open(tmp_path / "concepts_consolidated.json"))["stopped"] == "single_call"
    vis = json.load(open(tmp_path / "concepts_visual.json"))
    assert vis["dropped_not_visible"] == ["concept 1"] and "concept 1" not in vis["concepts"]
    assert [a["key"] for a in attrs["attributes"]] == ["Subject", "Place", "Colours", "Activity"]
    assert attrs["source"] == "shuffle0" and attrs["agreement"]["agree"] is True
    assert attrs["key_map"] == {"Subject": "C_primary_object", "Place": "C_secondary_object",
                                "Colours": "C_activity_or_relation", "Activity": "C_color_texture"}
    aj = json.load(open(tmp_path / "attributes.json"))
    assert aj["key_map"] == attrs["key_map"] and set(aj["prompt_sha256s"]) >= {
        "attributes", "attributes_merge", "visual_check", "concepts_batch", "concepts_consolidate"}
    assert aj["prompt_sha256s"]["concepts_batch"] == PINNED_SHA256["concepts_batch"]
    assert all((tmp_path / f"attributes_shuffle{s}.json").exists() for s in range(3))
    assert any("ATTRIBUTES" in m for m in logs)
    assert not list(tmp_path.glob("*.tmp*"))                                        # atomic writes left no temp files
    # resume: a second run reuses every file and never calls the model
    calls = []
    def counting(prompt, **kw):
        calls.append(prompt); return _stub_gen()(prompt, **kw)
    con.run_all(counting, _survey(), tmp_path, batch_images=5, shuffles=3, log=lambda m: None)
    assert calls == []
    # stale downstream files (built from a different list) are recomputed, not reused
    vis_p = tmp_path / "concepts_visual.json"
    v = json.load(open(vis_p)); v["working_sha256"] = "stale"; vis_p.write_text(json.dumps(v))
    con.run_all(counting, _survey(), tmp_path, batch_images=5, shuffles=3, log=lambda m: None)
    assert [c[:20] for c in calls] == ["For each concept bel"]
    sh = tmp_path / "concepts_shuffle1.json"
    sh.write_text(json.dumps({"seed": 1, "concepts": [], "counts_per_batch": [69, 46, 81]}))   # old cumulative schema
    calls.clear()
    con.run_all(counting, _survey(), tmp_path, batch_images=5, shuffles=3, log=lambda m: None)
    assert sum(c.startswith(pr.PROMPT_CONCEPTS_BATCH[:60]) for c in calls) == 3           # only shuffle 1 redone
    assert json.load(open(sh))["method"] == con.STEP2A_METHOD
    # smoke mode: first N batches of one shuffle, stops after consolidation, own files (setting recorded)
    calls.clear()
    out = con.run_all(counting, _survey(), tmp_path / "smoke", batch_images=5, shuffles=1, smoke_batches=2, log=lambda m: None)
    assert out["n_consolidated"] == 10 and out["shuffles"][0]["n_batches"] == 2 and out["smoke_batches"] == 2
    assert sum(c.startswith(pr.PROMPT_CONCEPTS_BATCH[:60]) for c in calls) == 2
    assert not any(c.startswith("For each concept") for c in calls) and not (tmp_path / "smoke" / "attributes.json").exists()
    assert json.load(open(tmp_path / "smoke" / "concepts_shuffle0.json"))["smoke_batches"] == 2


def test_concepts_disagreement_triggers_merge(tmp_path):
    attrs = con.run_all(_stub_gen(diverge_shuffle=True), _survey(), tmp_path, batch_images=5, shuffles=3, log=lambda m: None)
    assert attrs["agreement"]["agree"] is False and attrs["source"] == "merge"
    assert attrs["note"] == "merged" and [a["key"] for a in attrs["attributes"]] == ["Subject", "Place", "Colours", "Activity"]
    with pytest.raises(con.ParseFailure):
        con.run_all(_stub_gen(diverge_shuffle=True, merge_ok=False), _survey(), tmp_path / "b", batch_images=5, shuffles=3, log=lambda m: None)


CUT_MID_STRING = '{"concepts": ["red car", "wet road", "wet cobbl'                       # NUS-WIDE failure mode


# --------------------------------------------------------------------------- flat-list parsing and degeneracy
def test_parse_flat_concepts_strict_and_regex_fallback():
    assert con.parse_flat_concepts('{"concepts": ["Red car", "red car", "wet road"]}') == (["Red car", "wet road"], "strict")
    assert con.parse_flat_concepts('```json\n{"concepts": ["a", "b"]}\n```')[1] == "strict"
    assert con.parse_flat_concepts('{"concepts": [{"name": "a", "examples": ["x"]}, "b"]}') == (["a", "b"], "strict")
    # MS-COCO failure mode: malformed JSON mid-object (missing comma) -> regex fallback, reply closed
    names, mode = con.parse_flat_concepts('{"concepts": ["red car", "wet road" "blue sky", "red car"]}')
    assert (names, mode) == (["red car", "wet road", "blue sky"], "regex")
    # unterminated last string: accepted only when the length cap was hit; the cut string is dropped
    assert con.parse_flat_concepts(CUT_MID_STRING, hit_cap=True) == (["red car", "wet road"], "regex")
    with pytest.raises(con.ParseFailure, match="did not close"):
        con.parse_flat_concepts(CUT_MID_STRING, hit_cap=False)
    with pytest.raises(con.ParseFailure, match='no "concepts" key'):
        con.parse_flat_concepts('{"items": ["a"]}')
    with pytest.raises(con.ParseFailure, match="no strings"):
        con.parse_flat_concepts('{"concepts": [')
    with pytest.raises(con.ParseFailure, match="empty"):
        con.parse_flat_concepts('{"concepts": []}')


def test_parse_flat_concepts_degenerate_detectors():
    with pytest.raises(con.ParseFailure, match="longer than 60"):                   # one 26k-character line (NUS-WIDE)
        con.parse_flat_concepts('{"concepts": ["' + "wet road, " * 2600 + '"]}')
    with pytest.raises(con.ParseFailure, match="longer than 60"):
        con.parse_flat_concepts('{"concepts": ["ok", "' + "x" * 61 + '"]}')
    assert con.parse_flat_concepts('{"concepts": ["' + "x" * 60 + '"]}')[1] == "strict"
    with pytest.raises(con.ParseFailure, match="repeated 3"):                       # greedy repetition loop
        con.parse_flat_concepts('{"concepts": ["sky", "tree", "Tree", "tree", "car"]}')
    assert con.parse_flat_concepts('{"concepts": ["tree", "tree", "car", "tree"]}')[0] == ["tree", "car"]  # 2 in a row is fine
    with pytest.raises(con.ParseFailure, match="repeated 3"):
        con.parse_flat_concepts('{"concepts": ["a", "b", "b", "b", "c', hit_cap=True)  # detector also runs on regex mode


def _failing_then(ok_reply, fail_batches, fail_twice=()):
    """Stub: batch prompts whose first phrase index is in fail_batches get a degenerate reply on attempt 1,
    and again on attempt 2 when in fail_twice; records the repetition_penalty of every call."""
    calls = []

    def gen(prompt, **kw):
        calls.append(kw.get("repetition_penalty"))
        if prompt.startswith(pr.PROMPT_CONCEPTS_BATCH[:60]):
            first = int(re.search(r"phrase (\d+) a", prompt.split("Phrases:\n")[1]).group(1))
            if first in fail_batches and (kw.get("repetition_penalty") is None or first in fail_twice):
                return '{"concepts": ["x", "x", "x", "x"]}'
        return ok_reply
    gen.calls = calls
    return gen


def _survey_ordered(n):
    """Survey whose seed-0 shuffle is identity-free for the test: we key failures on the first phrase of a batch."""
    return {f"images/im{i:03d}.jpg": [f"phrase {i} a"] for i in range(n)}


def test_batch_retry_with_repetition_penalty_then_skip(tmp_path):
    survey = _survey_ordered(50)                                                     # 10 batches of 5
    ids = sorted(survey)
    order = np.random.default_rng(0).permutation(len(ids))
    firsts = [int(re.search(r"im(\d+)", ids[order[b]]).group(1)) for b in range(0, 50, 5)]
    # batch 0 fails once and recovers on the retry; batch 1 fails twice and is skipped (1/10 = 10 %, not > 10 %)
    gen = _failing_then('{"concepts": ["sky", "tree"]}', fail_batches={firsts[0], firsts[1]}, fail_twice={firsts[1]})
    r = con.concepts_batch_pass(gen, survey, 0, 5, log=lambda m: None)
    assert r["retried_batches"] == [0, 1] and [s["batch"] for s in r["skipped_batches"]] == [1]
    assert len(r["skipped_batches"][0]["errors"]) == 2 and "repeated 3" in r["skipped_batches"][0]["errors"][0]
    assert len(r["batch_lists"]) == 9 and r["batch_lists"][0]["retried"] and not r["batch_lists"][1]["retried"]
    assert r["parse_mode_counts"] == {"strict": 9} and r["union"] == ["sky", "tree"] and r["n_batches"] == 10
    assert gen.calls.count(con.RETRY_REPETITION_PENALTY) == 2 and gen.calls.count(None) == 10
    assert gen.calls[0] is None and gen.calls[1] == 1.1                              # retry immediately follows
    # 2 skipped of 10 (20 %) -> fail hard
    gen = _failing_then('{"concepts": ["sky"]}', fail_batches={firsts[2], firsts[5]}, fail_twice={firsts[2], firsts[5]})
    with pytest.raises(con.ParseFailure, match=r"2/10 batches skipped"):
        con.concepts_batch_pass(gen, survey, 0, 5, log=lambda m: None)
    # a run where the retry fixes everything has no skips and reports which batches needed it
    gen = _failing_then('{"concepts": ["sky"]}', fail_batches=set(firsts))
    r = con.concepts_batch_pass(gen, survey, 0, 5, log=lambda m: None)
    assert r["retried_batches"] == list(range(10)) and r["skipped_batches"] == []
    # run_all surfaces the fail-hard and never writes a shuffle file for it
    gen = _failing_then('{"concepts": ["sky"]}', fail_batches=set(firsts), fail_twice=set(firsts))
    with pytest.raises(con.ParseFailure):
        con.run_all(gen, survey, tmp_path, batch_images=5, shuffles=1, log=lambda m: None)
    assert not (tmp_path / "concepts_shuffle0.json").exists()


def test_consolidation_skips_a_chunk_after_retry():
    bad = lambda p, **kw: CUT_MID_STRING  # noqa: E731
    r = con.consolidate(bad, ["a", "b", "c"], log=lambda m: None)
    assert r["concepts"] == ["a", "b", "c"] and len(r["skipped_chunks"]) == 1 and r["retried_chunks"] == ["consolidate L1 single"]
    assert not hasattr(con, "concepts_update_pass")                                # the cumulative step is gone
    # the ending-only repair still applies (a closed object with a bracket typo is accepted, strict mode)
    good = con.parse_flat_concepts('{"concepts": ["a", "b"]]')
    assert good == (["a", "b"], "strict")


def _n_concepts(n):
    return [f"c{i}" for i in range(n)]


def test_consolidation_is_chunked_above_150():
    calls = []

    def halving(prompt, **kw):                                                      # each call returns the first half
        items = json.loads(prompt.split("Concepts:\n")[1])
        assert len(items) <= 150 and all(isinstance(c, str) for c in items)
        calls.append(len(items))
        return json.dumps({"concepts": items[: max(1, len(items) // 2)]})

    r = con.consolidate(halving, _n_concepts(320), log=lambda m: None)
    # level 1: 150,150,20 -> 75,75,10 = 160 (> 150) ; level 2: 150,10 -> 75,5 = 80 ; level 3: single call -> 40
    assert calls == [150, 150, 20, 150, 10, 80]
    assert [(lv["n_in"], lv["n_chunks"], lv["n_out"]) for lv in r["levels"]] == [(320, 3, 160), (160, 2, 80), (80, 1, 40)]
    assert len(r["concepts"]) == 40 and r["stopped"] == "single_call" and r["levels"][-1]["final"]
    calls.clear()
    r = con.consolidate(halving, _n_concepts(100), log=lambda m: None)
    assert calls == [100] and len(r["concepts"]) == 50 and r["levels"][0]["n_chunks"] == 1
    # a level that does not shrink stops the recursion with the merged list (bounded calls)
    calls.clear()
    identity = lambda p, **kw: json.dumps({"concepts": json.loads(p.split("Concepts:\n")[1])})  # noqa: E731
    r = con.consolidate(identity, _n_concepts(200), log=lambda m: None)
    assert r["stopped"] == "not_shrinking" and len(r["concepts"]) == 200 and len(r["levels"]) == 1
    # chunk size is a parameter (run_all passes it through)
    calls.clear()
    r = con.consolidate(halving, _n_concepts(7), chunk=3, log=lambda m: None)
    assert calls[:3] == [3, 3, 1]


def test_names_only_helpers():
    assert json.loads(con.render_concepts(["a b", "c"])) == ["a b", "c"]
    assert con.dedupe_names(["Red  car", "red car", " ", "sky", "SKY"]) == ["Red car", "sky"]
    assert con.list_sha(["B", "a"]) == con.list_sha(["a", "b"]) != con.list_sha(["a"])   # order- and case-free


def test_grouping_agreement_rule():
    g = lambda names, sets: {"attributes": [{"name": n, "definition": "d", "concepts": s} for n, s in zip(names, sets)]}  # noqa: E731
    base = g(["A", "B", "C", "D"], [["x1", "x2"], ["y1", "y2"], ["z1", "z2"], ["w1", "w2"]])
    same = g(["a", "b", "c", "d"], [["x1", "x2"], ["y1", "y2"], ["z1", "z2"], ["w1", "w2"]])
    assert con.groupings_agree([base, same, same])["agree"]
    renamed = g(["A", "B", "C", "E"], [["x1", "x2"], ["y1", "y2"], ["z1", "z2"], ["w1", "w2"]])
    assert not con.groupings_agree([base, same, renamed])["agree"]
    reshuffled = g(["A", "B", "C", "D"], [["y1", "z1"], ["x1", "w1"], ["x2", "w2"], ["y2", "z2"]])
    r = con.groupings_agree([base, reshuffled, same])
    assert not r["agree"] and r["vs_shuffle0"][0]["mean_membership_jaccard"] < 0.6


# --------------------------------------------------------------------------- validator gates
def _stub_encoder(dim=256):
    cache = {}

    def vec(word):
        if word not in cache:
            cache[word] = np.random.default_rng(abs(hash(word)) % (2 ** 32)).standard_normal(dim)
        return cache[word]

    def enc(texts):
        out = []
        for t in texts:
            ws = re.findall(r"[a-z]+", t.lower()) or ["empty"]
            v = sum(vec(w) for w in ws)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out).astype(np.float32)
    return enc


def _words(rng, vocab, k):
    return " ".join(rng.choice(vocab, size=k, replace=False))


def _synthetic(n=80, shared_vocab_new=False, shared_vocab_ref=True, seed=0):
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    mk = lambda m: np.array(["".join(rng.choice(letters, 7)) for _ in range(m)])  # noqa: E731
    vocabs_new = [mk(60) for _ in range(4)]
    if shared_vocab_new:
        vocabs_new = [vocabs_new[0]] * 4
    vocabs_ref = [mk(60) for _ in range(4)]
    if shared_vocab_ref:
        vocabs_ref = [vocabs_ref[0]] * 4
    ok, ref, survey = {}, {}, {}
    for i in range(n):
        iid = f"images/im{i}.jpg"
        v10 = {k: _words(rng, vocabs_new[a], 6) for a, k in enumerate(KEYS)}
        v10["C_global"] = _words(rng, vocabs_new[0], 12)
        legacy = {KEY_MAP[k]: v10[k] for k in KEYS}
        legacy.update(C_global=v10["C_global"], C_scene_type="")
        ok[iid] = {"image_id": iid, "codebook_texts": {k: legacy[k] for k in com.LEGACY_KEYS}, "codebook_texts_v10": v10}
        ref[iid] = {k: _words(rng, vocabs_ref[a], 6) for a, k in enumerate(com.LEGACY_LOCAL_KEYS)}
        ref[iid]["C_global"] = _words(rng, vocabs_ref[0], 12)
        survey[iid] = [" ".join(v10[k].split()[:3]) for k in KEYS]
    return ok, ref, survey


def test_validator_passes_on_clean_synthetic_captions():
    ok, ref, survey = _synthetic()
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert m["all_pass"], m["gates"]
    assert m["parse_rate"] == 0 and m["none_like_rate"] == 0 and m["name_prefix_rate"] == 0
    assert m["new_on_common"]["overlap"]["mean_pairwise_shared"] == 0 < m["reference"]["overlap"]["mean_pairwise_shared"]
    assert 0 < m["new"]["unseen_word"]["overall"] < 1 and m["new"]["unseen_word"]["n_images_with_survey"] == 80
    for f, s in m["new"]["per_field"].items():
        assert s["words_p50"] == (12 if f == "C_global" else 6) and 0 < s["type_token_ratio"] <= 1
    md = val.render_md(m, "t")
    assert "overall: PASS" in md and "| parse |" in md


def test_validator_gate_parse_fails():
    ok, ref, survey = _synthetic()
    m = val.compute_metrics(ok, 2, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert not m["gates"]["parse"]["pass"] and not m["all_pass"]
    m2 = val.compute_metrics(ok, 0, 1, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert not m2["gates"]["parse"]["pass"] and m2["parse_rate"] == pytest.approx(1 / 81)


@pytest.mark.parametrize("text", ["none", "a chair, none visible", "not visible here", "no visible sign", "a dog without a leash",
                                  "cannot tell", "unclear shape", "n/a", "barely lit", "faint outline", "hard to see wall", "absent"])
def test_validator_gate_none_like_fails(text):
    ok, ref, survey = _synthetic()
    for i, iid in enumerate(list(ok)[:2]):
        ok[iid]["codebook_texts_v10"]["Setting"] = text
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert not m["gates"]["none_like"]["pass"] and m["new"]["per_field"]["Setting"]["none_like_rate"] == pytest.approx(2 / 80)
    assert val.NONE_LIKE.search("a phone on a table") is None


@pytest.mark.parametrize("prefix", ["Main subject: ", "main_subject - ", "MAIN SUBJECT ", "Setting: "])
def test_validator_gate_name_prefix_fails(prefix):
    ok, ref, survey = _synthetic()
    key = "Setting" if prefix.lower().startswith("setting") else "Main_subject"
    for iid in list(ok)[:2]:
        ok[iid]["codebook_texts_v10"][key] = prefix + ok[iid]["codebook_texts_v10"][key]
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert not m["gates"]["name_prefix"]["pass"] and m["new"]["per_field"][key]["name_prefix_rate"] == pytest.approx(2 / 80)


def test_validator_gate_overlap_fails_when_new_shares_more_than_reference():
    ok, ref, survey = _synthetic(shared_vocab_new=True, shared_vocab_ref=False)
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    g = m["gates"]["overlap_le_reference"]
    assert not g["pass"] and g["value"] > g["threshold"]


def test_validator_gate_within_image_cosine_fails():
    ok, ref, survey = _synthetic()
    for r in ok.values():
        for k in KEYS:
            r["codebook_texts_v10"][k] = r["codebook_texts_v10"]["Main_subject"]   # four identical fields
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    assert not m["gates"]["within_image_cos"]["pass"]
    assert m["new_on_common"]["cosine"]["within_image_mean"] == pytest.approx(1.0, abs=1e-5)


def test_validator_gate_cross_image_cosine_fails():
    ok, ref, survey = _synthetic()
    for r in ok.values():
        r["codebook_texts_v10"]["Colour"] = "same words every image here"
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, _stub_encoder(), n_pairs=500)
    g = m["gates"]["cross_image_cos"]
    assert not g["pass"] and g["per_attribute"]["Colour"] == pytest.approx(1.0, abs=1e-5)
    assert all(v < 0.75 for k, v in g["per_attribute"].items() if k != "Colour")
    assert m["new"]["per_field"]["Colour"]["duplicate_rate"] == pytest.approx(1 - 1 / 80)


def test_validator_without_encoder_or_reference_fails_closed():
    ok, ref, survey = _synthetic()
    m = val.compute_metrics(ok, 0, 0, ref, survey, ATTRIBUTES, None)
    assert not m["gates"]["within_image_cos"]["pass"] and not m["gates"]["cross_image_cos"]["pass"]
    m = val.compute_metrics(ok, 0, 0, {}, survey, ATTRIBUTES, _stub_encoder())
    assert not m["gates"]["overlap_le_reference"]["pass"] and not m["all_pass"]


def test_validator_load_captions_schema_check(tmp_path):
    good = {"image_id": "a", "codebook_texts": {k: ("" if k == "C_scene_type" else "t") for k in com.LEGACY_KEYS},
            "codebook_texts_v10": {**{k: "t" for k in KEYS}, "C_global": "t"}}
    bad = json.loads(json.dumps(good)); bad["image_id"] = "b"; bad["codebook_texts"]["C_scene_type"] = "indoor"
    bad2 = json.loads(json.dumps(good)); bad2["image_id"] = "c"; bad2["codebook_texts_v10"]["Extra"] = "x"
    fail = {"image_id": "d", "_parse_error": "x"}
    p = tmp_path / "c.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (good, bad, bad2, fail)))
    ok, failed, unmatched = val.load_captions([str(p)], ATTRIBUTES)
    assert list(ok) == ["a"] and failed == ["d"] and unmatched == ["b", "c"]


def test_parse_json_object_repairs_bracket_typos():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from stage2_common import parse_json_object
    good = '{"phrases": ["a", "b"]}'
    assert parse_json_object(good) == {"phrases": ["a", "b"]}
    assert parse_json_object('{"phrases": ["a", "b"]]') == {"phrases": ["a", "b"]}        # ']]' typo (seen on NUS-WIDE)
    assert parse_json_object('{"phrases": ["a", "b"]') == {"phrases": ["a", "b"]}         # missing '}'
    assert parse_json_object('```json\n{"phrases": ["a"]]\n```') == {"phrases": ["a"]}
    assert parse_json_object('{"phrases": ["a", "b"') == {"phrases": ["a", "b"]}          # closed after a complete string
    import pytest
    with pytest.raises(Exception):
        parse_json_object('{"phrases": ["a", "b')                                         # unrecoverable (cut mid-string)
    with pytest.raises(ValueError):
        parse_json_object('["a", "b"]')                                                   # not an object
