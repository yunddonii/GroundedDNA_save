"""D4 control (recorded version of the 2026-10-06 fork experiment): does the S of the caption-routed
codes come from THIS image's caption being present as an input?

For each run: (A) caption-routed codes with the image's OWN captions, (A-shuf) caption-routed codes
with ANOTHER image's captions (a fixed permutation of the validation rows), (B) deployed codes. For a
text-OFF run, --force_text routes it with captions through its untrained text adapter.
If S(A) > 0 but S(A-shuf) ~ 0, the signal is carried by the caption input; if a text-OFF model routed
with captions also gives S ~ 0, the carrier is the learned text adapter, not a mechanical leak.
Usage: d4_caption_swap_control.py --out out.json [--force_text] run_dir [run_dir ...]
"""
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = HERE
for _ in range(3):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
from a3_v2 import AXES, KIND, elements, Reader  # noqa: E402
from d4d6_train_vs_deploy import forward  # noqa: E402


class Shuffled:
    """dataset view whose caption fields come from a permuted row."""
    def __init__(self, tr, perm_map):
        self.tr, self.pm = tr, perm_map

    def __getitem__(self, i):
        s = dict(self.tr[i]); o = self.tr[self.pm[i]]
        for k in ("cached_text_part_raw", "cached_text_tokens", "cached_text_token_mask", "has_text"):
            if k in o:
                s[k] = o[k]
        return s


def run(rd, force_text, boot, seed):
    import model_siglip2 as MS
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from slot_role_probe import parse_args_txt
    args = parse_args_txt(os.path.join(rd, "args.txt")); args.lambda_mec = 0.0; args.device = "cpu"
    ckpt = os.path.join(rd, "model_state_dict.pth")
    model = MS.SigLIP2SemanticOTModel(args).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    text_off = bool(getattr(args, "disable_text_supervision", False))
    if force_text:
        model.disable_text_supervision = False
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1", train_transform=None,
                            test_transform=None, load_train=True, load_database=False, load_test=False, return_index=True,
                            qwen_text_cache_path=args.qwen_text_cache_path, siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)]
    ids = json.load(open(os.path.join(cache, "image_ids.json")))
    caps = {}
    for line in open(args.qwen_text_cache_path):
        r = json.loads(line); caps[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    C = [caps[str(ids[int(rows[i])])] for i in idx]
    rng = np.random.default_rng(seed); p = rng.permutation(len(idx))
    shuf = Shuffled(tr, {idx[k]: idx[p[k]] for k in range(len(idx))})
    prune = bool(model.bidirectional_token_prune)
    A = forward(model, tr, idx, with_text=True, training=True, prune=prune)
    AS = forward(model, shuf, idx, with_text=True, training=True, prune=prune)
    B = forward(model, tr, idx, with_text=False, training=False, prune=prune)
    N, M = len(idx), 4
    share = {}
    for m in range(M):
        ax = AXES[m + 1]; E = [elements(c.get(ax, ""), KIND[ax]) for c in C]; cnt = {}
        for s in E:
            for w in s:
                cnt[w] = cnt.get(w, 0) + 1
        ok = set(w for w, c in cnt.items() if c / N <= 0.2); El = [s & ok for s in E]; need = 2 if KIND[ax] == "colour" else 1
        S = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            if len(El[i]) < need:
                continue
            for j in range(i + 1, N):
                if len(El[i] & El[j]) >= need:
                    S[i, j] = True
        share[ax] = S
    iu = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
    rules = {AXES[m + 1]: {"lexical": share[AXES[m + 1]] & iu} for m in range(M)}
    R = Reader(N, boot, seed)
    out = {"run": rd, "dataset": args.dataset, "rows": N, "text_off_run": text_off, "forced_text": bool(force_text),
           "use_gumbel_softmax": bool(getattr(args, "use_gumbel_softmax", False)),
           "mode_A": A["mode"], "mode_A_shuffled": AS["mode"], "mode_B": B["mode"]}
    for name, X in (("caption_routed_own", A), ("caption_routed_shuffled", AS), ("deployed", B)):
        same = [(X["cb"][:, m][:, None] == X["cb"][:, m][None, :]) for m in range(M)]
        out[name] = R.reading(same, rules, "lexical")["S"]
    out["p_same_codeword_own_vs_deployed"] = [round(float((A["cb"][:, m] == B["cb"][:, m]).float().mean()), 3) for m in range(M)]
    out["p_same_codeword_own_vs_shuffled"] = [round(float((A["cb"][:, m] == AS["cb"][:, m]).float().mean()), 3) for m in range(M)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--force_text", action="store_true")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    res = {}
    for rd in a.runs:
        r = run(rd, a.force_text, a.boot, a.seed); res[os.path.basename(rd.rstrip("/"))] = r
        print(json.dumps({k: r[k] for k in ("run", "text_off_run", "forced_text", "caption_routed_own", "caption_routed_shuffled", "deployed", "p_same_codeword_own_vs_shuffled")}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
