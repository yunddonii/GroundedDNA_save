"""arch-exp-3: where is the axis-specific signal lost -- before or at quantisation?

On held-out rows, matches each local slot's representation to the caption
embedding of each axis OF THE SAME IMAGE (4-way, batch-centred, exactly as the
P4 term scores it) and reports the accuracy at two points of the same forward:
the slot token the quantiser receives, and the codeword it returns.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")

def match_acc(q, t):                      # [B, M, D] each
    q = q.float(); t = t.float()
    q = F.normalize(q - q.mean(0, keepdim=True), dim=-1)
    t = F.normalize(t - t.mean(0, keepdim=True), dim=-1)
    S = torch.einsum("bmd,bad->bma", q, t)                    # [B, code, axis]
    M = S.shape[1]
    tgt = torch.arange(M, device=S.device).expand(S.shape[0], M)
    code_picks_axis = (S.argmax(dim=2) == tgt).float().mean().item()
    axis_picks_code = (S.argmax(dim=1) == tgt).float().mean().item()
    diag = S.diagonal(dim1=1, dim2=2).mean().item()
    offd = (S.sum(dim=(1, 2)) - S.diagonal(dim1=1, dim2=2).sum(dim=1)).mean().item() / (M * M - M)
    return dict(code_picks_own_axis=round(code_picks_axis, 4),
                axis_picks_own_code=round(axis_picks_code, 4),
                chance=round(1.0 / M, 4),
                mean_diag_cos=round(diag, 4), mean_offdiag_cos=round(offd, 4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    from slot_role_probe import parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt")); args.lambda_mec = 0.0
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"), train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None), siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)][:a.n]
    pre, quant, txt = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(idx), 64):
            smp = [tr[i] for i in idx[s0:s0+64]]
            vt = torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(a.device)
            vg = torch.stack([s["cached_visual_global"] for s in smp]).to(a.device)
            raw = torch.stack([s["cached_text_part_raw"] for s in smp]).to(a.device)
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                      return_routing=True, cached_visual_tokens_raw=vt, cached_visual_global=vg,
                      cached_text_part_raw=None, cached_has_text=None)     # DEPLOYMENT forward
            z = o.get("semantic_visual_tokens")
            if z is None: z = o.get("quantizer_input")
            pre.append(z[:, 1:, :].cpu()); quant.append(o["quantized_tokens"][:, 1:, :].cpu())
            txt.append(model._adapt_pooled_text_for_loss(raw)[:, 1:, :].cpu())
    pre, quant, txt = torch.cat(pre), torch.cat(quant), torch.cat(txt)
    res = {"result_dir": a.result_dir, "n_images": int(pre.shape[0]),
           "note": "deployment forward (no text reaches the model); captions used only as the target",
           "pre_quantisation_slot_token": match_acc(pre, txt),
           "quantised_codeword": match_acc(quant, txt)}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
main()
