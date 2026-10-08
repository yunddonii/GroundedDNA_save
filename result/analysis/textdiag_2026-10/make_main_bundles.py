"""Mirror the text-path line's records into the main tree's result/ layout (user order 2026-10-09).

For every finished training cell of this line (tags td1_* / td3_*, excluding the live *_N19_* queue
cells, which record themselves via stageL/run_queue_main.sh):
  * with --move: the run directory is moved from the worktree's result/ to
    /home/yschoi/GroundedDNA/result/ (same filesystem; no copy, no symlink); every move is appended to
    result/analysis/textdiag_2026-10/RUN_DIR_MOVES.tsv (old path <TAB> new path) so the committed
    records that cite worktree paths stay traceable.
  * one bundle per stage in /home/yschoi/GroundedDNA/result/analysis/<stage folder>/, following the
    past stages (stage11_ancsoft_alldata, stage12_n_reselect): PREREGISTRATION.md, cells.txt
    (<rc>|<tag>|<command>), cells_failed.txt, eval_runs.txt (<dataset> <arm> <seed> <run dir>),
    <tag>.log (cell stdout; failed attempts as <tag>.failedN.log), scoring JSONs, summary.txt/json,
    RESULT.md (the MODI_PROJECT_LOG entry/entries of that stage), analysis_batch.sh.
Idempotent: re-running refreshes the bundles; moved runs are found in the main tree.
Usage: make_main_bundles.py [--move] [--only STAGE ...]
"""
import argparse, csv, glob, json, os, re, shutil, sys

WT = "/home/yschoi/gdna_textdiag"
TD = os.path.join(WT, "result/analysis/textdiag_2026-10")
MAIN = "/home/yschoi/GroundedDNA/result"
RUNS = "/data/yschoi/gdna_p3exec_authority/runs"
MODI = os.path.join(WT, "docs/MODI_PROJECT_LOG.md")
MOVES = os.path.join(TD, "RUN_DIR_MOVES.tsv")

STAGES = {
    "stage1": dict(folder="textdiag_stage1_baseline_off_20261006",
                   tag=r"^td1_(flickr|nus|coco)_(B0|OFF|H1|H2)_s\d+$",
                   prereg=[os.path.join(TD, "stage1/PREREGISTRATION.md")],
                   modi=["] Stage 1: baseline, text-OFF"],
                   scores=["stage1/a3v2", "stage1/a3v2_2000", "stage1/d4d6", "d4d6"],
                   extra=["stage1/A3V2_SUMMARY.md", "stage1/A3V2_2000_SUMMARY.md", "stage1/h1_bit_identity.json",
                          "stage1/d4_control_B0.json", "stage1/d4_control_OFF.json"]),
    "stage3": dict(folder="textdiag_stage3_deploy_path_20261007",
                   tag=r"^td3_(gate_H2|gate2_H2|TD|TDv2|XM0|TDXM|TDXM5|S|SN2|BN2|nus_H2|coco_H2)_s\d+$",
                   prereg_sections=["# Stage 3 pre-registration"],
                   modi=["] Stage 3 arm N1 (TD"],
                   scores=["stage3/a3v2_2000", "stage3/d4d6", "stage3/a3v2_nuscoco"],
                   extra=["stage3/A3V2_2000_SUMMARY.md", "stage3/gate_bit_identity.json", "stage3/gate2_bit_identity.json"]),
    "stage3c": dict(folder="textdiag_stage3c_image_conditioned_anchor_20261008",
                    tag=r"^td3_(PmemSS|PmemSS_smoke|Phead|PheadNoSS|MIX)_s\d+$",
                    prereg_sections=["## Stage 3-C"],
                    modi=["] Stage 3-C: image-conditioned"],
                    scores=["stage3/a3v2_2000", "stage3/d4d6", "stage3/a3v2_2000_pmem", "stage3/d4d6_pmem", "stage3/pmem"],
                    extra=[]),
    "stage4g": dict(folder="textdiag_stage4g_global_gate_20261008",
                    tag=r"^td3_(G0|G0p|G1|G1p)_s\d+$",
                    prereg_sections=["## Stage 4-G"], modi=["] Stage 4-G:"],
                    scores=["stage3/a3v2_2000", "stage3/d4d6"], extra=[]),
    "stage4a": dict(folder="textdiag_stage4a_local_ntxent_removal_20261008",
                    tag=r"^td3_(A0|A0p|A1)_s\d+$",
                    prereg_sections=["## Stage 4-A"], modi=["] Stage 4-A (plan 4a)"],
                    scores=["stage3/a3v2_2000", "stage3/d4d6"], extra=[]),
    "stage4d": dict(folder="textdiag_stage4d_text_term_reduction_20261008",
                    tag=r"^td3_(TH0|TH0p|XM0p)_s\d+$",
                    prereg_sections=["## Stage 4-D"], modi=["] Stage 4-D"],
                    scores=["stage3/a3v2_2000", "stage3/d4d6"], extra=[]),
    "stageL": dict(folder="textdiag_stageL_train_length_20261008",
                   tag=r"^td3_(L9|L19|L19p|PmemSSp1)_s\d+$",
                   prereg_sections=["## Stage L (", "## Stage L extension"], modi=["] Stage L"],
                   scores=["stage3/a3v2_2000", "stage3/d4d6", "stageL/a3v2_2000", "stageL/d4d6", "stageL/pmem"],
                   extra=[]),
}
RUN_RE = re.compile(r"^\d{6}\+(?P<ds>[a-z0-9]+)_setting1_(?P<tag>.+)\+bs\+64\+e\+\d+\+proj_lr\+[0-9.]+$")


def all_runs():
    out = {}
    for root in (os.path.join(WT, "result"), MAIN):
        for d in glob.glob(os.path.join(root, "*+*_setting1_td[13]_*")):
            m = RUN_RE.match(os.path.basename(d))
            if not m or os.path.islink(d):
                continue
            tag = m.group("tag")
            if "_N19_" in tag or "FAKE" in tag:
                continue
            if not os.path.exists(os.path.join(d, "model_state_dict.pth")):
                continue
            out.setdefault(tag, []).append(d)
    return out


def cmd_of(tag):
    for p in glob.glob(os.path.join(TD, "*/cmds", f"{tag}.cmd")):
        return open(p).read().strip()
    return None


def last_rc(path):
    rc = None
    for line in open(path, errors="replace"):
        m = re.match(r"^rc=(-?\d+)\s*$", line.strip())
        if m:
            rc = int(m.group(1))
    return rc


def logs_of(tag):
    """(final log, [failed attempt logs]) from the queue logs and the tmux run records."""
    cands = []
    for p in [os.path.join(TD, "stage1/logs", f"{tag}.log"), os.path.join(TD, "stage3/logs", f"{tag}.log"),
              os.path.join(RUNS, f"{tag}.log")]:
        if os.path.exists(p):
            cands.append(p)
    # relaunch names (2026-10-08: td3_G1r_*, td3_G1pr_* trained td3_G1_* / td3_G1p_*)
    m = re.match(r"^(td3_G1p?)_(s\d+)$", tag)
    if m:
        alt = os.path.join(RUNS, f"{m.group(1)}r_{m.group(2)}.log")
        if os.path.exists(alt):
            cands.append(alt)
    ok = [p for p in cands if last_rc(p) == 0]
    final = max(ok, key=os.path.getmtime) if ok else None
    failed = [p for p in cands if p != final and last_rc(p) not in (0,)]
    return final, failed, cands


def eval_row(run_dir):
    f = os.path.join(run_dir, "log.csv")
    rows = list(csv.DictReader(open(f)))
    ev = [r for r in rows if r.get("eval_mAP_at_R")]
    g = lambda r, k: (round(float(r[k]), 4) if r.get(k) not in (None, "") else None)
    return [{"epoch": int(float(r["epoch"])), "mAP_at_R": g(r, "eval_mAP_at_R"), "unique": g(r, "eval_unique_code_ratio"),
             "dead": g(r, "eval_dead_code_ratio_mean")} for r in ev]


def a3_of(tag, score_dirs):
    out = {}
    for sd in score_dirs:
        p = os.path.join(TD, sd, f"*_{tag}.json")
        for f in glob.glob(p):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if "codon" in d and "lexical" in d.get("codon", {}):
                out[sd] = {lvl: d[lvl]["lexical"]["S"]["value"] for lvl in ("codon", "codeword")}
    return out


def modi_entries(keys):
    text = open(MODI).read()
    parts = re.split(r"(?m)^(?=## 20)", text)
    return [p for p in parts if any(k in p.split("\n", 1)[0] for k in keys)]


def prereg_sections(heads):
    text = open(os.path.join(TD, "stage3/PREREGISTRATION.md")).read()
    parts = re.split(r"(?m)^(?=#{1,2} )", text)
    return [p for p in parts if any(p.startswith(h) or p.startswith(h.lstrip("# ")) for h in heads)]


def build(stage, spec, runs, move):
    dst = os.path.join(MAIN, "analysis", spec["folder"])
    os.makedirs(dst, exist_ok=True)
    tags = sorted(t for t in runs if re.match(spec["tag"], t))
    cells, failed, evals, summary = [], [], [], []
    for tag in tags:
        dirs = runs[tag]
        if len(dirs) != 1:
            print(f"  WARN {tag}: {len(dirs)} run dirs {dirs}"); continue
        rd = dirs[0]
        if move and rd.startswith(os.path.join(WT, "result") + os.sep):
            new = os.path.join(MAIN, os.path.basename(rd))
            if os.path.exists(new):
                print(f"  WARN {tag}: destination exists, not moved");
            else:
                shutil.move(rd, new)
                with open(MOVES, "a") as fh:
                    fh.write(f"{rd}\t{new}\n")
                rd = new
        ds = RUN_RE.match(os.path.basename(rd)).group("ds")
        arm = re.sub(r"^td[13]_((flickr|nus|coco)_)?", "", re.sub(r"_s\d+$", "", tag))
        seed = re.search(r"_s(\d+)$", tag).group(1)
        final, fails, cands = logs_of(tag)
        cmd = cmd_of(tag) or "(command file not found)"
        if final:
            shutil.copy2(final, os.path.join(dst, f"{tag}.log"))
        cells.append(f"{0 if final else 'NA'}|{tag}|{cmd}")
        for i, fp in enumerate(fails, 1):
            shutil.copy2(fp, os.path.join(dst, f"{tag}.failed{i}.log"))
            failed.append(f"{last_rc(fp)}|{tag}|{os.path.basename(fp)}")
        evals.append(f"{ds} {arm} {seed} {rd}")
        summary.append({"tag": tag, "dataset": ds, "arm": arm, "seed": int(seed), "run_dir": rd,
                        "eval": eval_row(rd), "a3_S": a3_of(tag, spec.get("scores", []))})
    # Merge, never overwrite: stageL/run_queue_main.sh appends its own cells (e.g. *_N19_*) to the
    # same files while it runs; keep every existing line whose tag this builder does not produce.
    def _tag_of(name, line):
        if name == "eval_runs.txt":
            parts = line.split()
            m = RUN_RE.match(os.path.basename(parts[3])) if len(parts) >= 4 else None
            return m.group("tag") if m else None
        parts = line.split("|")
        return parts[1] if len(parts) >= 2 else None
    for name, lines in (("cells.txt", cells), ("cells_failed.txt", failed), ("eval_runs.txt", evals)):
        path = os.path.join(dst, name)
        keep = []
        if os.path.exists(path):
            for line in open(path).read().splitlines():
                if line.strip() and _tag_of(name, line) not in tags:
                    keep.append(line)
        merged = keep + lines
        if merged or name != "cells_failed.txt":
            tmp = path + ".tmp"
            open(tmp, "w").write("\n".join(merged) + ("\n" if merged else ""))
            os.replace(tmp, path)
    # scoring JSONs
    for sd in spec.get("scores", []):
        src = os.path.join(TD, sd)
        if not os.path.isdir(src):
            continue
        out = os.path.join(dst, sd.split("/")[-1])
        os.makedirs(out, exist_ok=True)
        for f in glob.glob(os.path.join(src, "*")):
            b = os.path.basename(f)
            if any(t in b for t in tags) or sd.endswith("pmem"):
                shutil.copy2(f, out) if os.path.isfile(f) else None
    for e in spec.get("extra", []):
        p = os.path.join(TD, e)
        if os.path.exists(p):
            shutil.copy2(p, dst)
    # preregistration
    if "prereg" in spec:
        shutil.copy2(spec["prereg"][0], os.path.join(dst, "PREREGISTRATION.md"))
    else:
        secs = prereg_sections(spec["prereg_sections"])
        open(os.path.join(dst, "PREREGISTRATION.md"), "w").write(
            "<!-- extracted from the worktree's result/analysis/textdiag_2026-10/stage3/PREREGISTRATION.md -->\n\n" + "\n".join(secs))
    ents = modi_entries(spec["modi"])
    open(os.path.join(dst, "RESULT.md"), "w").write(
        "<!-- copied from docs/MODI_PROJECT_LOG.md (branch text-diag-2026-09) -->\n\n" + ("\n".join(ents) if ents else "(no completed MODI entry yet)\n"))
    json.dump(summary, open(os.path.join(dst, "summary.json"), "w"), indent=1)
    with open(os.path.join(dst, "summary.txt"), "w") as fh:
        fh.write(f"{spec['folder']} — last-epoch validation (log.csv) and deployed A3 v2 S (codon / codeword)\n")
        fh.write("tag | epochs evaluated: mAP@R / unique / dead | A3 S per scoring set\n")
        for s in summary:
            ev = "; ".join(f"e{r['epoch']}: {r['mAP_at_R']}/{r['unique']}/{r['dead']}" for r in s["eval"])
            a3 = "; ".join(f"{k.split('/')[-1]} codon {v['codon']:+.3f} codeword {v['codeword']:+.3f}" for k, v in s["a3_S"].items())
            fh.write(f"{s['tag']} | {ev} | {a3}\n")
    open(os.path.join(dst, "analysis_batch.sh"), "w").write(
        "#!/usr/bin/env bash\n# Scoring used for this stage (worktree /home/yschoi/gdna_textdiag, branch text-diag-2026-09):\n"
        "#   training cells: result/analysis/textdiag_2026-10/stage1/run_cell.sh <gpu> <cmd file> (or stageL/run_queue_main.sh)\n"
        "#   A3 v2 (2,000 Flickr rows) + d4d6: result/analysis/textdiag_2026-10/stage3/score_list.sh <runs.txt>\n"
        "#   (= d0/run_batch.sh … a3_v2.py --caption_file cache_eval/flickr25k_v4_train_plus_evaldb.jsonl\n"
        "#      --reference_text_cache cache_eval/ref_text_flickr_v4_plus_evaldb --extra_rows_from_caption_file\n"
        "#      --oracle_mix 0.0 0.3 1.0 ; d4d6/run_batch.sh … d4d6_train_vs_deploy.py --boot 1000)\n"
        f"cd {WT} && bash result/analysis/textdiag_2026-10/stage3/score_list.sh <(awk '{{print $4}}' {os.path.join(dst, 'eval_runs.txt')})\n")
    print(f"{stage}: {len(tags)} runs -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    runs = all_runs()
    for st, spec in STAGES.items():
        if a.only and st not in a.only:
            continue
        build(st, spec, runs, a.move)


if __name__ == "__main__":
    main()
