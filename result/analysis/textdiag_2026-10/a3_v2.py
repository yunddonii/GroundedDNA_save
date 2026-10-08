"""A3 v2 -- cross-image slot consistency with a fixed reference, a lexical pair rule, bootstrap
intervals, codeword- and codon-level readings and an oracle-mixing calibration. CPU, deployment
forward (no caption reaches the model). The 2026-09-29 script is left untouched; this file is the
Stage-0 (D0) instrument of the text-path plan.

Differences from the 09-29 script
  * --caption_file / --reference_text_cache: every arm in a comparison is scored on the SAME pairs,
    taken from one caption file and one text cache (keyed by image_id), not from the arm's own cache.
  * all held-out validation rows by default (--n 0).
  * primary pair rule = LEXICAL "shared element": two rows share an axis-m element when their axis-m
    captions share at least one element word of that axis's kind (object axes: noun-like content
    words; colour axis: colour / material words; relation axis: verb-like words), the word being used
    by at most --max_df of the rows. This is a different relation from the caption-embedding
    neighbourhood a training target may use, so it does not evaluate the model on its own target.
    The caption-cosine top-q rule is kept as the secondary, "same relation as training" rule.
  * statistic per axis R(m) = log(lift_own / mean lift_other); S = mean over axes. Scale-free.
  * bootstrap over images (--boot) for R(m), S and their codon-level versions.
  * "axis-exclusive" pairs: share the axis-m element and share NO element on the other axes.
  * calibration: the slot-m code of a fraction f of rows is replaced by an oracle (the k-means id of
    the row's axis-m reference caption embedding); S(f) for f in --oracle_mix tells which role
    fraction the instrument can see. f = 0 must reproduce the plain reading; f = 1 is the positive
    control.
"""
import argparse, hashlib, json, math, os, re, sys
import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

AXES = ["C_global", "C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture"]
KIND = {"C_primary_object": "noun", "C_secondary_object": "noun", "C_activity_or_relation": "verb", "C_color_texture": "colour"}
STOP = set("""a an the and or of to in on at with by for from as is are was were be been being this that these those it its
their his her they them he she we you i our your into onto over under above below near beside between behind front
against along across through around while during before after up down out off no not none very more most some
any each other another such than then there here where when which who whom what how all both few many several
also just only own same so too can could may might will would shall should do does did done has have had having
one two three four five six seven eight nine ten first second third left right top bottom center centre middle
side sides background foreground image scene photo picture shot view visible shown seen appears appear appearing
positioned located situated set placed standing sitting lying stands sits lies large small big little long short
wide narrow tall high low upper lower main central nearby distant partially partly fully slightly mostly
suggesting indicating creating showing featuring including surrounded surrounding""".split())
COLOUR = set("""red orange yellow green blue purple pink brown black white gray grey beige tan cream ivory gold golden
silver bronze navy teal turquoise maroon olive violet magenta cyan amber crimson scarlet lavender khaki rust copper
dark light pale bright vivid muted pastel neon dull warm cool monochrome sepia colorful colourful multicolored
wooden wood metal metallic steel iron glass plastic fabric cloth cotton leather fur furry feathered wool woolen stone
brick concrete sand sandy rocky grassy leafy smooth rough glossy matte shiny textured striped spotted checkered
patterned plain soft hard wet dry rusty worn weathered polished translucent transparent reflective fluffy""".split())
VERB_HINT = set("""hold holds holding carry carries carrying ride rides riding walk walks walking run runs running sit sits
sitting stand stands standing lie lies lying jump jumps jumping play plays playing eat eats eating drink drinks drinking
talk talks talking look looks looking watch watches watching read reads reading write writes writing cook cooks cooking
fly flies flying swim swims swimming climb climbs climbing pull pulls pulling push pushes pushing throw throws throwing
catch catches catching kick kicks kicking hit hits hitting cut cuts cutting pose poses posing smile smiles smiling
laugh laughs laughing wear wears wearing lean leans leaning rest rests resting sleep sleeps sleeping wait waits waiting
work works working perform performs performing dance dances dancing sing sings singing drive drives driving park parks
parked grasp grasps grasping touch touches touching point points pointing reach reaches reaching face faces facing
interact interacting gather gathered gathering""".split())
WORD = re.compile(r"[a-z][a-z\-']+")


def norm(w):
    """Rule-based lemma: plurals and simple verb endings (no lemmatizer in the environment)."""
    if w in COLOUR or w in VERB_HINT:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content(s):
    return [norm(w) for w in WORD.findall(str(s).lower()) if w not in STOP and len(w) > 2]


def elements(s, kind):
    ws = content(s)
    if kind == "colour":
        return set(w for w in ws if w in COLOUR)
    if kind == "verb":
        return set(w for w in ws if w in VERB_HINT or (w.endswith("ing") and len(w) > 5))
    # noun-like: drop colour/material words, verb-like words and adverbs
    return set(w for w in ws if w not in COLOUR and w not in VERB_HINT and not w.endswith(("ing", "ly", "ed")) and len(w) > 3)


def quad(W, A):
    """W [B,N] resample multiplicities, A [N,N] upper-triangular 0/1: weighted number of pairs i<j."""
    return ((W @ A) * W).sum(1)


class Reader:
    """Lift / R / S readings with a vectorised image bootstrap. same_list: per slot [N,N] bool."""

    def __init__(self, N, boot, seed):
        self.iu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        g = torch.Generator().manual_seed(seed)
        W = torch.zeros(boot, N)
        for b in range(boot):
            W[b] += torch.bincount(torch.randint(0, N, (N,), generator=g), minlength=N).float()
        self.W = torch.cat([torch.ones(1, N), W])            # row 0 = point estimate
        self.B_all = quad(self.W, self.iu.float())

    def lifts(self, same_list, mask):
        """returns [B+1, M] lifts under `mask` (vs all pairs), or None if too few pairs."""
        m = (mask & self.iu).float()
        n_pairs = quad(self.W, m)
        if float(n_pairs[0]) < 5:
            return None, 0
        out = []
        for s in same_list:
            p = quad(self.W, (s & mask & self.iu).float()) / n_pairs.clamp_min(1)
            b = quad(self.W, (s & self.iu).float()) / self.B_all.clamp_min(1)
            out.append(p / b.clamp_min(1e-9))
        return torch.stack(out, 1), int(n_pairs[0])

    def reading(self, same_list, rules, rule_name):
        M = len(same_list)
        out = {"per_axis": {}, "S": None}
        Rs = []
        for m in range(M):
            ax = AXES[m + 1]
            L, n = self.lifts(same_list, rules[ax][rule_name])
            if L is None:
                out["per_axis"][ax] = None; continue
            others = torch.stack([L[:, k] for k in range(M) if k != m], 1).mean(1)
            R = torch.log(L[:, m].clamp_min(1e-9) / others.clamp_min(1e-9))
            out["per_axis"][ax] = {"pairs": n, "lift_own": round(float(L[0, m]), 3), "lift_other_mean": round(float(others[0]), 3),
                                   "R": round(float(R[0]), 4),
                                   "R_ci95": [round(float(torch.quantile(R[1:], .025)), 4), round(float(torch.quantile(R[1:], .975)), 4)] if R.shape[0] > 1 else None}
            Rs.append(R)
        if Rs:
            S = torch.stack(Rs, 1).mean(1)
            out["S"] = {"value": round(float(S[0]), 4), "axes": len(Rs), "positive_axes": int(sum(float(r[0]) > 0 for r in Rs)),
                        "ci95": [round(float(torch.quantile(S[1:], .025)), 4), round(float(torch.quantile(S[1:], .975)), 4)] if S.shape[0] > 1 else None}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--caption_file", default=None, help="reference caption jsonl (default: the run's own)")
    ap.add_argument("--reference_text_cache", default=None, help="dir with text_part.f16.npy + image_ids.json (default: the run's own cache)")
    ap.add_argument("--n", type=int, default=0, help="0 = all held-out validation rows")
    ap.add_argument("--extra_rows_from_caption_file", action="store_true",
                    help="also score every image of --caption_file that is NOT a training row (e.g. evaluation-only database captions); "
                         "their visual features are read from the run's feature cache by image_id")
    ap.add_argument("--cos_top_frac", type=float, default=0.02)
    ap.add_argument("--max_df", type=float, default=0.20, help="an element word used by more than this share of rows does not define a pair")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min_shared_colour", type=int, default=2, help="colour axis: shared colour/material words needed for a pair")
    ap.add_argument("--oracle_mix", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 1.0])
    ap.add_argument("--oracle_k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an args.txt field after parsing (e.g. anchor_source=memory, anchor_memory_tau=0.02); repeatable")
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    import model_siglip2 as MS
    import dataloaders as DL
    import dna_utils.runtime_state as RS
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from slot_role_probe import parse_args_txt
    provenance = {mod.__name__: {"file": mod.__file__, "sha256": hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()}
                  for mod in (MS, DL, RS)}
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    import ast as _ast
    for _kv in a.set:                      # (2026-10-08, Stage 3-C) deploy-time overrides, e.g. anchor_source=memory
        _k, _v = _kv.split("=", 1)
        try:
            _v = _ast.literal_eval(_v)
        except Exception:
            pass
        setattr(args, _k, _v)
    print("overrides:", a.set, flush=True)
    args.lambda_mec = 0.0
    args.device = "cpu"
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = MS.SigLIP2SemanticOTModel(args).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
                            setting=getattr(args, "setting", "setting1"), train_transform=None,
                            test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)]
    if a.n > 0:
        idx = idx[:a.n]
    cache_rows = [int(rows[i]) for i in idx]
    image_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    ids = [str(image_ids[cr]) for cr in cache_rows]

    # ---- reference captions and reference text embeddings, keyed by image_id
    cap_file = a.caption_file or args.qwen_text_cache_path
    captions = {}
    for line in open(cap_file):
        r = json.loads(line)
        captions[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    caps = [captions.get(i) for i in ids]
    keep = [k for k, c in enumerate(caps) if c is not None]
    if len(keep) < len(caps):
        idx = [idx[k] for k in keep]; cache_rows = [cache_rows[k] for k in keep]; ids = [ids[k] for k in keep]; caps = [caps[k] for k in keep]
    n_val = len(idx)
    extra_rows = []
    if a.extra_rows_from_caption_file:
        pos = {str(v): k for k, v in enumerate(image_ids)}
        for iid, c in captions.items():
            r = pos.get(iid)
            if r is None or r in allr or r in set(cache_rows):
                continue
            extra_rows.append(r); ids.append(iid); caps.append(c); cache_rows.append(r)
        print(f"rows: {n_val} validation + {len(extra_rows)} extra (non-training) from the caption file")
    ref_dir = a.reference_text_cache or cache
    ref_ids = json.load(open(os.path.join(ref_dir, "image_ids.json")))
    ref_pos = {str(v): k for k, v in enumerate(ref_ids)}
    tp = np.load(os.path.join(ref_dir, "text_part.f16.npy"), mmap_mode="r")
    traw = torch.tensor(np.stack([np.asarray(tp[ref_pos[i]][1:5], dtype=np.float32) for i in ids]))   # [N,4,512]

    # ---- deployment forward
    idx_cb, bases = [], []
    vt_np = np.load(os.path.join(cache, "visual_tokens.f16.npy"), mmap_mode="r")
    vg_np = np.load(os.path.join(cache, "visual_global.f16.npy"), mmap_mode="r")
    def _batch(k0, k1):
        vts, vgs = [], []
        for k in range(k0, k1):
            if k < n_val:
                s = tr[idx[k]]; vts.append(s["cached_visual_tokens_raw"]); vgs.append(s["cached_visual_global"])
            else:   # same fp16 -> fp32 conversion as _SigLIP2FeatureCache.get
                r = cache_rows[k]
                vts.append(torch.from_numpy(np.asarray(vt_np[r], dtype=np.float32))); vgs.append(torch.from_numpy(np.asarray(vg_np[r], dtype=np.float32)))
        return torch.stack(vts), torch.stack(vgs)
    with torch.no_grad():
        for s0 in range(0, len(ids), 50):
            vt, vg = _batch(s0, min(s0 + 50, len(ids)))
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None, return_routing=True,
                      cached_visual_tokens_raw=vt, cached_visual_global=vg, cached_text_part_raw=None, cached_has_text=None)
            idx_cb.append(o["codebook_indices"][:, 1:].cpu()); bases.append(o["base_indices_per_codebook"][:, 1:].cpu())
    idx_cb, bases = torch.cat(idx_cb).long(), torch.cat(bases).long()
    N, M = idx_cb.shape
    codon = bases[:, :, 0] * 16 + bases[:, :, 1] * 4 + bases[:, :, 2]                                   # [N,4]
    same_cw = [(idx_cb[:, m][:, None] == idx_cb[:, m][None, :]) for m in range(M)]
    same_cd = [(codon[:, m][:, None] == codon[:, m][None, :]) for m in range(M)]
    iu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    # ---- pair rules
    E = {}
    df = {}
    for m in range(M):
        ax = AXES[m + 1]
        E[ax] = [elements((c or {}).get(ax, ""), KIND[ax]) for c in caps]
        cnt = {}
        for s in E[ax]:
            for w in s:
                cnt[w] = cnt.get(w, 0) + 1
        df[ax] = cnt
    share = {}
    for m in range(M):
        ax = AXES[m + 1]
        ok = set(w for w, c in df[ax].items() if c / N <= a.max_df)
        El = [s & ok for s in E[ax]]
        need = a.min_shared_colour if KIND[ax] == "colour" else 1
        S = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            if len(El[i]) < need:
                continue
            for j in range(i + 1, N):
                if len(El[i] & El[j]) >= need:
                    S[i, j] = True
        share[ax] = S
    trn = F.normalize(traw, dim=-1)
    rules = {}
    for m in range(M):
        ax = AXES[m + 1]
        excl = share[ax].clone()
        for k in range(M):
            if k != m:
                excl &= ~share[AXES[k + 1]]
        cos = trn[:, m] @ trn[:, m].t()
        thr = torch.quantile(cos[iu], 1 - a.cos_top_frac)
        rules[ax] = {"lexical": share[ax] & iu, "lexical_axis_exclusive": excl & iu,
                     f"caption_cos_top{int(a.cos_top_frac * 100)}pct": (cos >= thr) & iu}

    # ---- readings with bootstrap
    reader = Reader(N, a.boot, a.seed)
    reading = lambda same_list, rule: reader.reading(same_list, rules, rule)

    res = {"result_dir": a.result_dir, "dataset": args.dataset, "axis_center": getattr(args, "axis_center", None),
           "text_supervision_disabled": bool(getattr(args, "disable_text_supervision", False)),
           "use_gumbel_softmax": bool(getattr(args, "use_gumbel_softmax", False)),
           "rows": N, "rows_validation": n_val, "rows_extra": len(extra_rows), "caption_file": cap_file, "caption_sha256": hashlib.sha256(open(cap_file, "rb").read()).hexdigest(),
           "reference_text_cache": ref_dir, "module_provenance": provenance,
           "pair_rule": {"max_df": a.max_df, "cos_top_frac": a.cos_top_frac, "min_shared_colour": a.min_shared_colour, "boot": a.boot,
                         "elements_per_row_mean": {ax: round(float(np.mean([len(s) for s in E[ax]])), 2) for ax in AXES[1:]},
                         "rows_with_element": {ax: int(sum(bool(s) for s in E[ax])) for ax in AXES[1:]}},
           "codeword": {r: reading(same_cw, r) for r in ("lexical", "lexical_axis_exclusive", f"caption_cos_top{int(a.cos_top_frac * 100)}pct")},
           "codon": {r: reading(same_cd, r) for r in ("lexical", "lexical_axis_exclusive")}}

    # ---- calibration: mix an oracle code into a fraction of rows (codeword level, lexical rule)
    from sklearn.cluster import KMeans
    oracle = torch.zeros(N, M, dtype=torch.long)
    ctr = traw - traw.mean(0, keepdim=True)
    for m in range(M):
        oracle[:, m] = torch.tensor(KMeans(n_clusters=min(a.oracle_k, N // 2), n_init=4, random_state=0).fit_predict(ctr[:, m].numpy()))
    cal = {}
    gmix = torch.Generator().manual_seed(a.seed + 1)
    for f in a.oracle_mix:
        mixed = idx_cb.clone()
        if f > 0:
            sel = torch.rand(N, generator=gmix) < f
            # oracle ids live in their own space: offset so they never collide with model codewords
            mixed[sel] = oracle[sel] + 10_000
        same_mix = [(mixed[:, m][:, None] == mixed[:, m][None, :]) for m in range(M)]
        rd = reading(same_mix, "lexical")
        cal[str(f)] = rd["S"]
    res["calibration_lexical_codeword"] = cal
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "axis_center", "text_supervision_disabled", "use_gumbel_softmax", "rows")}))
    for lvl in ("codeword", "codon"):
        for r, v in res[lvl].items():
            print(lvl, r, "S=", v["S"], {ax: (x["pairs"], x["lift_own"], x["lift_other_mean"], x["R"]) if x else None for ax, x in v["per_axis"].items()})
    print("calibration", {f: (v["value"], v["ci95"]) if v else None for f, v in cal.items()})


if __name__ == "__main__":
    main()
