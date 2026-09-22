"""Stage 2 pilot: zero-shot codon reading (ZSCR) -- can a codon be read in words without a decoder
fitted on image-code co-occurrence, and does that need captions?

For one finished Flickr25K stage-1 run (opt rows trained, val rows held out):

TEXT-PATH reading (needs a trained caption path).
  Every OPT caption of axis m is sent through the model's own text path
  (`_adapt_pooled_text_for_loss` -> `_encode_text_tokens_to_dna`) to a slot-m codeword k. The image
  path does not read a codeword alone -- its local codon head sees C_m[k] + g_m * C_0[k0] -- so each
  k is turned into an IMAGE-STYLE codon distribution by marginalising over the global codewords k0
  that opt images actually use. A caption's words are then credited to the codons it lands on.
  No image is paired with its own caption anywhere in this table.

CLIP-ONLY reading (C2b; available to ANY code, including a caption-free one).
  The prototype of (slot m, codon c) is the mean CLIP image embedding of the opt images whose
  deployed codon at slot m is c; it is matched to the pool of opt axis-m caption embeddings by CLIP
  similarity, and the words of the nearest captions name the codon.

EVALUATION. For each held-out val image and local slot m, the deployed codon ranks the axis-m
distinctive vocabulary by lift over the prior; AP against that image's own axis-m caption words.
Floors: shuffled codons, prior only. Ceiling: the supervised dictionary (opt image codes paired with
their own captions, `decode_slot`) -- the reading s4.7-style probes do.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def codon_of(bases):                                    # [..., 3] -> [...]
    return bases[..., 0] * 16 + bases[..., 1] * 4 + bases[..., 2]


def ranking_ap(scores, targets):
    from heldout_codon_decoding import label_ranking_ap
    return label_ranking_ap(scores, targets)


def read_scores(counts, support, prior, n_prior, alpha=1.0, min_support=10):
    """Posterior P(word | codon) with the same Beta(alpha, alpha) smoothing and support fallback
    as the supervised dictionary (`build_dictionary`), so the readers differ only in WHERE the
    word counts come from. counts [64, V] (fractional allowed), support [64], prior [V]."""
    p = (counts + alpha) / (support[:, None] + 2.0 * alpha)
    base = (prior + alpha) / (n_prior + 2.0 * alpha)
    return np.where((support >= min_support)[:, None], p, base[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clip_temperature", type=float, default=0.01)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    from slot_role_probe import parse_args_txt, distinctive_vocab, multi_hot, AXES
    from heldout_codon_decoding import decode_slot, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.lambda_mec = 0.0
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
                            setting=getattr(args, "setting", "setting1"), train_transform=None,
                            test_transform=None, load_train=True, load_database=False,
                            load_test=False, return_index=True,
                            qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx_opt = [i for i in range(len(tr)) if int(rows[i]) in opt]
    idx_val = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)]

    def forward(indices):
        codons, cw, vg_all, txt = [], [], [], []
        with torch.no_grad():
            for s0 in range(0, len(indices), 128):
                smp = [tr[i] for i in indices[s0:s0 + 128]]
                vt = torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(a.device)
                vg = torch.stack([s["cached_visual_global"] for s in smp]).to(a.device)
                o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                          return_routing=True, cached_visual_tokens_raw=vt,
                          cached_visual_global=vg, cached_text_part_raw=None, cached_has_text=None)
                codons.append(codon_of(o["base_indices_per_codebook"]).cpu())       # [b, 5]
                cw.append(o["codebook_indices"].cpu())                               # [b, 5]
                vg_all.append(vg.float().cpu())
                txt.append(torch.stack([s["cached_text_part_raw"] for s in smp]).float())
        return (torch.cat(codons).numpy(), torch.cat(cw).numpy(), torch.cat(vg_all),
                torch.cat(txt))

    c_opt, k_opt, vg_opt, raw_opt = forward(idx_opt)
    c_val, k_val, _, raw_val = forward(idx_val)

    # ---- image-style codon distribution for every (local slot m, codeword k) -----------------
    cb = model.quantizer.get_effective_codebooks().detach()                 # [5, K, D]
    gate = torch.sigmoid(model.global_gate_logits).detach()                # [4]
    k0, k0_count = np.unique(k_opt[:, 0], return_counts=True)
    w0 = torch.tensor(k0_count / k0_count.sum(), dtype=torch.float32)
    K = cb.shape[1]
    p_img = np.zeros((5, K, 64))
    with torch.no_grad():
        for m in range(1, 5):
            x = cb[m][:, None, :] + gate[m - 1] * cb[0][torch.as_tensor(k0)][None, :, :]  # [K, n0, D]
            out = model.codon_heads[m](x.reshape(-1, x.shape[-1]), residual=None, gamma=0.0,
                                       text_chunks=None)
            cod = codon_of(out["base_indices"]).reshape(K, len(k0)).cpu()           # [K, n0]
            for kk in range(K):
                np.add.at(p_img[m, kk], cod[kk].numpy(), w0.numpy())

    # ---- captions and vocabulary --------------------------------------------------------
    caps = {}
    for line in open(args.qwen_text_cache_path):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}
    # NUS-WIDE / MS-COCO key images under subfolders; the ids below are "images/" + basename.
    caps.update({"images/" + os.path.basename(k): v for k, v in list(caps.items())
                 if "images/" + os.path.basename(k) not in caps})
    fc_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    # CIFAR-10 samples carry no image_path; their captions are keyed by the cache row id.
    name = lambda i: ("images/" + os.path.basename(str(tr[i]["image_path"]))      # noqa: E731
                      if "image_path" in tr[i] else str(fc_ids[int(rows[i])]))
    id_opt, id_val = [name(i) for i in idx_opt], [name(i) for i in idx_val]
    vocab = distinctive_vocab(caps, id_opt)
    T_opt = {ax: multi_hot(caps, id_opt, ax, vocab[ax]) for ax in AXES}
    T_val = {ax: multi_hot(caps, id_val, ax, vocab[ax]) for ax in AXES}

    # ---- text path: captions -> slot codewords -------------------------------------------
    def text_codewords(raw):
        with torch.no_grad():
            out = []
            for s0 in range(0, len(raw), 256):
                t = model._adapt_pooled_text_for_loss(raw[s0:s0 + 256].to(a.device))
                enc = model._encode_text_tokens_to_dna(t, allow_mm_ema=False, deterministic_codon=True)
                out.append(enc["codebook_indices"].cpu())
            return torch.cat(out).numpy()                                           # [n, 5]
    k_txt = text_codewords(raw_opt)
    # Held-out rows: how often a caption lands on the codeword its own image deploys (local slots).
    agree_val = (text_codewords(raw_val)[:, 1:] == k_val[:, 1:]).mean(0)            # [4]

    # ---- CLIP-only prototypes (C2b) -------------------------------------------------------
    img = F.normalize(vg_opt, dim=-1)
    rng = np.random.default_rng(a.shuffle_seed)
    perm = rng.permutation(len(idx_val))
    report = {"result_dir": a.result_dir, "n_opt": len(idx_opt), "n_val": len(idx_val),
              "gate": gate.cpu().numpy().round(4).tolist(), "slots": {}}
    for m in range(1, 5):
        ax = AXES[m - 1]
        Topt, Tval = T_opt[ax], T_val[ax]
        prior, n_prior = Topt.sum(0).astype(float), float(len(Topt))
        # text-path counts: each opt caption's words credited to the image-style codons it lands on
        mass = p_img[m][k_txt[:, m]]                                                 # [n_opt, 64]
        text_counts, text_support = mass.T @ Topt, mass.sum(0)
        # CLIP-only counts: codon c takes as many nearest opt captions (CLIP image-text cosine to the
        # prototype of the opt images carrying c) as it has member images, so support matches.
        txt = F.normalize(raw_opt[:, m, :], dim=-1)
        clip_counts, clip_support = np.zeros((64, Topt.shape[1])), np.zeros(64)
        for c in range(64):
            members = np.where(c_opt[:, m] == c)[0]
            if len(members) == 0:
                continue
            proto = F.normalize(img[members].mean(0), dim=-1)
            nearest = torch.topk(txt @ proto, len(members)).indices.numpy()
            clip_counts[c], clip_support[c] = Topt[nearest].sum(0), len(members)
        res = {}
        for label, counts, support in (("text_path", text_counts, text_support),
                                       ("clip_only", clip_counts, clip_support)):
            S = read_scores(counts, support, prior, n_prior)
            res[label] = float(np.nanmean(ranking_ap(S[c_val[:, m]], Tval)))
            res[label + "_shuffled"] = float(np.nanmean(ranking_ap(S[c_val[perm, m]], Tval)))
        res["prior_only"] = float(np.nanmean(ranking_ap(
            np.tile((prior + 1.0) / (n_prior + 2.0), (len(idx_val), 1)), Tval)))
        res["supervised_ceiling"] = decode_slot(c_opt[:, m], Topt, c_val[:, m], Tval, 64,
                                                DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT)["concept_mAP"]
        res["text_codeword_agrees_with_image_codeword"] = float(agree_val[m - 1])
        report["slots"][ax] = res
        print(f"slot {m} {ax:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in res.items() if v is not None))
    means = {k: float(np.mean([report["slots"][ax][k] for ax in AXES]))
             for k in ("text_path", "text_path_shuffled", "clip_only", "clip_only_shuffled",
                       "prior_only", "supervised_ceiling", "text_codeword_agrees_with_image_codeword")}
    report["mean_over_local_slots"] = means
    print("MEAN " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
