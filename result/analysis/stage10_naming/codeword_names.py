"""Stage 10 (lever 1, no training): name every local codeword by the captions nearest to it.

Each local codeword lives in the text-adapter space (the space the text path maps captions into), so
it can be named without any dictionary fitted on image codes: for slot m, codeword k, take the opt
captions of axis m, map them through the model's own text path (`_adapt_pooled_text_for_loss`) and
rank them by cosine to the codeword. The name of k is the words shared by its top-5 captions.

Quantities (held-out val rows, deployment forward, no text):
  - axis specificity: among the top-5 captions when ALL four axes' captions compete, the share
    that belongs to slot m's own axis (chance .25);
  - name precision: for a val image deployed to codeword k in slot m, the AP of k's top-5 caption
    word bag against the image's own axis-m distinctive words, with the stage-2 posterior;
  - a readable table of names for the 12 most-used codewords per slot.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "result", "analysis", "stage5_bigidea"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    from slot_role_probe import parse_args_txt, distinctive_vocab, multi_hot, AXES, WORD
    tokenize = lambda s: [w.lower() for w in WORD.findall(s)]  # noqa: E731  the probe's own word rule
    from heldout_codon_decoding import label_ranking_ap
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from zscr_like import read_scores

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.lambda_mec = 0.0
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True, load_database=False,
                            load_test=False, return_index=True, qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    io = [i for i in range(len(tr)) if int(rows[i]) in opt]
    iv = [i for i in range(len(tr)) if int(rows[i]) not in opt]
    fc_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    name = lambda i: ("images/" + os.path.basename(str(tr[i]["image_path"]))      # noqa: E731
                      if "image_path" in tr[i] else str(fc_ids[int(rows[i])]))
    caps = {}
    for line in open(args.qwen_text_cache_path):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}
    caps.update({"images/" + os.path.basename(k): v for k, v in list(caps.items())
                 if "images/" + os.path.basename(k) not in caps})
    id_o, id_v = [name(i) for i in io], [name(i) for i in iv]

    # opt captions through the model's own text path -> [n_opt, 5, D]
    raw_o = torch.stack([tr[i]["cached_text_part_raw"] for i in io]).float()
    with torch.no_grad():
        adapted = torch.cat([model._adapt_pooled_text_for_loss(raw_o[s:s + 256].to(a.device)).float().cpu()
                             for s in range(0, len(io), 256)])                    # [n_opt, 5, D]
    cb = model.quantizer.get_effective_codebooks().detach().float().cpu()          # [5, K, D]
    K = cb.shape[1]
    # deployed codewords of val rows
    cw_v = []
    with torch.no_grad():
        for s0 in range(0, len(iv), 128):
            smp = [tr[i] for i in iv[s0:s0 + 128]]
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None, return_routing=True,
                      cached_visual_tokens_raw=torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(a.device),
                      cached_visual_global=torch.stack([s["cached_visual_global"] for s in smp]).to(a.device),
                      cached_text_part_raw=None, cached_has_text=None)
            cw_v.append(o["codebook_indices"].cpu())
    cw_v = torch.cat(cw_v).numpy()                                                  # [n_val, 5]

    vocab = distinctive_vocab(caps, id_o)
    rep = {"result_dir": a.result_dir, "n_opt": len(io), "n_val": len(iv), "top": a.top, "slots": {},
           "centring": "captions minus their axis mean (opt); codewords minus the mean deployed codeword "
                       "of their slot (val) -- the quant_gap probe's centring, on the pool"}
    # Centred as the code->axis probe centres: every axis's captions lose their own mean, every
    # slot's codewords lose the mean deployed codeword of that slot. Raw cosines rank the same
    # captions first for every codeword (a shared direction), which is what this removes.
    cent = adapted[:, 1:, :] - adapted[:, 1:, :].mean(0, keepdim=True)                 # [n_opt, 4, D]
    pooled = F.normalize(cent.reshape(-1, adapted.shape[-1]), dim=-1)                    # all axes' captions
    pooled_axis = np.repeat(np.arange(4), len(io))
    for m in range(1, 5):
        ax = AXES[m - 1]
        q_mean = cb[m][torch.as_tensor(cw_v[:, m])].mean(0, keepdim=True)              # mean deployed codeword
        c = F.normalize(cb[m] - q_mean, dim=-1)                                         # [K, D]
        t_own = F.normalize(cent[:, m - 1, :], dim=-1)                                  # [n_opt, D]
        top_own = (c @ t_own.T).topk(a.top, dim=1).indices.numpy()                  # [K, top]
        top_all = (c @ pooled.T).topk(a.top, dim=1).indices.numpy()                # [K, top]
        specificity = float((pooled_axis[top_all] == (m - 1)).mean())
        # name = word bag of the top-5 own-axis captions; precision on val images deployed to k
        Tv = multi_hot(caps, id_v, ax, vocab[ax])
        words = vocab[ax]
        widx = {w: i for i, w in enumerate(words)}
        bag = np.zeros((K, len(words)))
        for k in range(K):
            for j in top_own[k]:
                for w in set(tokenize(caps[id_o[j]][ax])):
                    if w in widx:
                        bag[k, widx[w]] += 1
        To = multi_hot(caps, id_o, ax, vocab[ax]); prior, n_prior = To.sum(0).astype(float), float(len(To))
        S = read_scores(bag, np.full(K, float(a.top)), prior, n_prior, min_support=a.top)
        used = np.bincount(cw_v[:, m], minlength=K)
        precision = float(np.nanmean(label_ranking_ap(S[cw_v[:, m]], Tv)))
        prior_ap = float(np.nanmean(label_ranking_ap(np.tile((prior + 1) / (n_prior + 2), (len(Tv), 1)), Tv)))
        table = []
        for k in np.argsort(-used)[:12]:
            top_words = [words[i] for i in np.argsort(-bag[k])[:6] if bag[k, i] >= 2]
            table.append({"codeword": int(k), "val_images": int(used[k]), "name": " / ".join(top_words),
                          "example_caption": caps[id_o[top_own[k][0]]][ax][:90]})
        rep["slots"][ax] = {"axis_specificity_top5": specificity, "chance": 0.25,
                            "name_precision_ap": precision, "prior_ap": prior_ap,
                            "codewords_used_val": int((used > 0).sum()), "names": table}
        print(f"{ax:24s} specificity {specificity:.3f} (chance .25)  name-precision AP {precision:.4f} vs prior {prior_ap:.4f}")
        for row in table[:4]:
            print(f"    cw {row['codeword']:3d} ({row['val_images']:3d} imgs): {row['name']}  | e.g. {row['example_caption']}")
    rep["mean"] = {k: float(np.mean([rep["slots"][ax][k] for ax in AXES])) for k in ("axis_specificity_top5", "name_precision_ap", "prior_ap")}
    json.dump(rep, open(a.out, "w"), indent=1)
    print("MEAN", rep["mean"])


if __name__ == "__main__":
    main()
