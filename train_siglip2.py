"""Training loop for the SigLIP2 + semantic-OT + DNA-codon model.

Mirrors the structure of legacy `train.py` (now archived under `backup/`):
    - argparse-driven Config (re-uses repo's `config.py`)
    - explicit module instantiation (no Lightning)
    - per-group lr in optimizer (`backbone_lr`, `proj_lr`)
    - `one_epoch(train, loader, epoch)` inner function
    - per-epoch CSV logger + tqdm + SummaryWriter
    - state_dict save with the same `args.save_model_state_path` pattern

Required batch keys provided by `dataloaders.ImgRtvCIFAR10` /
`dataloaders.ImgRtvDataset`:
    img                   FloatTensor [B, 3, H, W]                (raw pixel mode)
    part_input_ids        LongTensor  [B, M=6, L]                 (Qwen text tokens)
    part_attention_mask   Tensor      [B, M=6, L]
    label                 LongTensor  [B, n_class]                (one-hot or multi-hot)
  --or, when `siglip2_feature_cache_dir` is set, the cached-features fast path:
    cached_visual_tokens_raw   FloatTensor [B, 196, H_v]
    cached_visual_global       FloatTensor [B, D_proj]
    cached_text_part_raw       FloatTensor [B, 6, D_proj]
    has_text                   BoolTensor  [B]                    True = Qwen text valid
"""

import os
import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataloaders import load_dataset
from config import set_random_seed, Config

from model_siglip2 import SigLIP2SemanticOTModel
from loss_siglip2  import DNACodonHashLoss
# All training-side utilities live under `dna_utils` so that this script does
# NOT depend on legacy `utils.py` (which transitively pulls in DWKM /
# sinkhornbarycenters that are not used by the SigLIP2 framework).
from dna_utils import (
    EpochCSVLogger,
    get_transform,
    print_one_epoch_info,
    get_summarywriter,
)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# ----------------------------- flat save-path helper

def _resolve_save_path(args: Config) -> str:
    """Build a flat result directory with a consistent naming schema.

    Schema:
        result/<date>+<dataset>_<setting>_<user_tag>+bs+<bs>+e+<epoch>+proj_lr+<lr>/

    The `<dataset>_<setting>_` prefix is auto-prepended (lowercased) so all
    run directories on disk are uniformly tagged with the dataset and
    split they correspond to. If `--tag` already starts with that prefix
    (e.g. legacy callers passed `--tag flickr25k_setting1_v18_xxx`), it is
    used verbatim to avoid double-prefixing.

    Schema artifacts inside the directory:
        model_state_dict.pth        (final-epoch checkpoint)
        criterion_state_dict.pth    (final; preserves EMA buffer)
        log.csv                     (per-epoch metrics, append mode)
        args.txt                    (Config.print_info dump)
        train/  val/                (tensorboard event subdirs)
        config.pt                   (Config.save_arg() output)
        extract_{db,query}.npz / evaluation_*.json
                                    (extraction & evaluation outputs)

    Trailing separator preserved so the legacy
    ``Config.print_info()`` line ``open(self.save_log_path + 'args.txt', ...)``
    (no separator between path and filename) resolves correctly.
    """
    auto_prefix = f"{str(args.dataset).lower()}_{str(args.setting)}"
    user_tag_parts = args.tag if args.tag is not None else []
    user_tag = "_".join(user_tag_parts) if user_tag_parts else ""
    if user_tag and user_tag.startswith(auto_prefix):
        full_tag = user_tag                              # already prefixed
    elif user_tag:
        full_tag = f"{auto_prefix}_{user_tag}"           # auto-prepend
    else:
        full_tag = auto_prefix                           # no user tag

    tag = f"{args.date}+{full_tag}"
    for k, v in [
        ("bs",      args.batch_size),
        ("e",       args.epoch),
        ("proj_lr", args.proj_lr),
    ]:
        tag = f"{tag}+{k}+{v}"

    base = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "result",
        tag,
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "")


# ----------------------------- mid-training retrieval eval helper

@torch.no_grad()
def _collect_split_data(model, loader, device, M: int, K_max: int):
    """v78a: single-pass sweep over `loader` (train loader expected) to
    collect, for each (m, k):
      - List of z_m vectors that landed in codeword k (for 2-means at split)
      - Collision pressure: average duplicate-group size of the full DNA
        code among samples that landed in (m, k).

    Returns:
        z_per_cw: nested list [M][K_max], each entry None or tensor [N_mk, D]
        collision_pressure: [M, K_max] tensor
    """
    from collections import Counter
    model.eval()
    z_chunks = [[[] for _ in range(K_max)] for _ in range(M)]
    full_codes = []           # list[tuple[int, int, int, int, int, int]] (cb indices per sample)
    cb_idx_per_sample = []    # list of np arrays [M] per sample
    for batch in loader:
        # Pick the right cached-feature key, mirroring the training loop's
        # paired-aug convention (cached_visual_tokens_aug0 / _aug1) when
        # available, else the un-augmented cache.
        if "cached_visual_tokens_aug0" in batch and "cached_visual_global_aug0" in batch:
            cached_vt = batch["cached_visual_tokens_aug0"].to(device)
            cached_vg = batch["cached_visual_global_aug0"].to(device)
        else:
            cached_vt = batch.get("cached_visual_tokens_raw", None)
            cached_vg = batch.get("cached_visual_global", None)
            if cached_vt is not None: cached_vt = cached_vt.to(device)
            if cached_vg is not None: cached_vg = cached_vg.to(device)
        cached_tp = batch.get("cached_text_part_raw", None)
        cached_ht = batch.get("cached_has_text", None)
        if cached_tp is not None: cached_tp = cached_tp.to(device)
        if cached_ht is not None: cached_ht = cached_ht.to(device)
        out = model(
            cached_visual_tokens_raw=cached_vt,
            cached_visual_global   =cached_vg,
            cached_text_part_raw   =cached_tp,
            cached_has_text        =cached_ht,
            return_routing=True,
        )
        z   = out["semantic_visual_tokens"].detach()   # [B, M, D]
        cbi = out["codebook_indices"].detach()         # [B, M]
        B = z.shape[0]
        for b in range(B):
            for m in range(M):
                k = int(cbi[b, m].item())
                z_chunks[m][k].append(z[b, m].cpu())
            full_codes.append(tuple(int(x) for x in cbi[b].tolist()))
            cb_idx_per_sample.append(cbi[b].cpu().numpy())
    # collation: stack per (m, k)
    z_per_cw: list = [[None] * K_max for _ in range(M)]
    for m in range(M):
        for k in range(K_max):
            if z_chunks[m][k]:
                z_per_cw[m][k] = torch.stack(z_chunks[m][k], dim=0)
    # collision pressure: for each sample, group-size minus 1 (excluding self)
    code_counts = Counter(full_codes)
    cp = torch.zeros(M, K_max)
    cp_cnt = torch.zeros(M, K_max)
    for sample_cb, code in zip(cb_idx_per_sample, full_codes):
        gs = code_counts[code]               # group size including self
        for m in range(M):
            k = int(sample_cb[m])
            cp[m, k]    += float(gs)
            cp_cnt[m, k] += 1.0
    cp = cp / cp_cnt.clamp_min(1.0)          # average group size per (m, k)
    return z_per_cw, cp


def _build_active_loss_types(args) -> list:
    """Return ONLY the loss-dict keys that correspond to active losses for
    this run. "Active" = the gating lambda is > 0 OR the structural flag
    that turns the loss on is set.

    Goal: stop polluting log.csv / tensorboard with always-zero columns
    for losses that are turned off, and stop SILENTLY dropping newly-added
    losses (v119 CIBHash, v121 SwAV-assign, v123 codeword text proto, ...)
    that the legacy hardcoded list did not know about.

    Conventions:
      - 'loss' (total) and the two routing diagnostics are always logged.
      - Sub-component scalars (e.g. loss_entropy / loss_base_balance for
        loss_dna; loss_cb_balance / loss_cb_uncorr for loss_bu;
        eff_lambda_codeword_codon_sinkhorn for the bij; the text-cluster
        and text-codon-rel diagnostics) are emitted together with their
        parent loss.
    """
    def _on(name: str) -> bool:
        return float(getattr(args, name, 0.0)) > 0.0

    def _flag(name: str) -> bool:
        return bool(getattr(args, name, False))

    keys: list = ['loss']

    # ---- core VQ / quantization commitments
    if _on('lambda_vq'):    keys.append('loss_vq')
    if _on('lambda_quant'): keys.append('loss_quant')

    # ---- DNA + base regularizers
    if _on('lambda_anchor'): keys.append('loss_anchor')
    if _on('lambda_dna'):
        keys.extend(['loss_dna', 'loss_entropy', 'loss_base_balance'])
    if _on('lambda_bu'):
        keys.extend(['loss_bu', 'loss_cb_balance', 'loss_cb_uncorr'])

    # ---- hash / text-hash supervision
    if _on('lambda_hash'):              keys.append('loss_hash')
    if _on('lambda_hash_hard'):         keys.append('loss_hash_hard')
    # loss_text_hash (MSE / swap form) and loss_text_hash_ntxent_add (v100
    # additive InfoNCE form) are SEPARATE dict keys in the loss output.
    # Emit each only when its own lambda is on so the CSV column reflects
    # the actual value being optimized.
    if _on('lambda_text_hash'):
        keys.append('loss_text_hash')
    if _on('lambda_text_hash_ntxent'):
        keys.append('loss_text_hash_ntxent_add')
    if _on('lambda_cw_xmodal'):         keys.append('loss_cw_xmodal')

    # ---- routing-side alignment
    if _on('lambda_wasserstein'): keys.append('loss_wasserstein')
    if _on('lambda_ortho_text'):  keys.append('loss_ortho_text')

    # ---- paired-aug NtXent on DNA codes (v29)
    if _flag('use_paired_aug_ntxent') and _on('lambda_ntxent'):
        keys.append('loss_ntxent')

    # ---- codeword-codon bijection family (v106 / v111 / v112)
    if _on('lambda_codeword_codon_sinkhorn'):
        keys.extend(['loss_codeword_codon_sinkhorn',
                     'eff_lambda_codeword_codon_sinkhorn'])
    if _on('lambda_text_cluster_codon_ot'):
        keys.extend(['loss_text_cluster_codon_ot',
                     'text_cluster_conf_mean',
                     'text_cluster_usage_entropy'])
    if _on('lambda_hierarchical_cluster_codon'):
        keys.append('loss_hierarchical_cluster_codon')
    if _on('lambda_proto_cluster_cos'):
        keys.append('loss_proto_cluster_cos')

    # ---- text-codon relational (v113)
    if _on('lambda_text_codon_rel'):
        keys.extend(['loss_text_codon_rel',
                     'text_codon_rel_pos_sim',
                     'text_codon_rel_neg_sim'])

    # ---- codeword text prototype CE (v123)
    if _on('lambda_codeword_text_proto'):
        keys.append('loss_codeword_text_proto')

    # ---- CIBHash family (v119)
    if _on('lambda_cibhash_ntxent'):
        keys.append('loss_cibhash_ntxent')
    if _on('lambda_cibhash_kl'):
        keys.append('loss_cibhash_kl')

    # ---- SwAV-style swapped balanced assignment (v121)
    if _on('lambda_swav_assign'):
        keys.append('loss_swav_assign')

    # ---- v66 per-codon text-anchored prototype CE
    if _flag('codon_text_anchor') and _on('lambda_codon_text_anchor'):
        keys.append('loss_text_anchor')

    # ---- reconstruction (v28)
    if _on('lambda_recon'):
        keys.append('loss_recon')

    # ---- hash reconstruction decoder (v70a)
    if _flag('use_hash_recon') and _on('lambda_hash_recon'):
        keys.append('loss_hash_recon')

    # ---- dual hash projection heads (v72a)
    if _flag('use_dual_hash_proj'):
        if _on('lambda_dual_semantic'):
            keys.append('loss_dual_semantic')
        if _on('lambda_dual_instance'):
            keys.append('loss_dual_instance')

    # ---- routing diagnostics: always informative
    keys.extend(['routing_mean_effective_k', 'routing_fraction_top1'])

    # de-dupe while preserving order (defensive)
    seen = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def _mid_train_eval(model, loader, device, distance_mode: str, codebook_size: int):
    """Quick retrieval + collapse eval on a single (test) split.

    Treats the split as both query and db with self-match removed. Uses
    `extraction_siglip2.encode_split` + `evaluation_siglip2.evaluate_*` so
    that mid-training metrics are computed by the SAME functions used at
    final evaluation -- no metric drift between checkpoints.
    """
    from extraction_siglip2 import encode_split
    from evaluation_siglip2 import evaluate_retrieval, evaluate_code_collapse

    was_training = model.training
    model.eval()
    try:
        ext = encode_split(model, loader, device, split_name="mid-eval")
    finally:
        if was_training:
            model.train()
    retrieval = evaluate_retrieval(
        ext, ext,
        distance_mode=distance_mode,
        precision_at_k_list=(1, 5, 10, 50, 100),
        remove_self_match=True,
    )
    collapse = evaluate_code_collapse(ext, codebook_size=codebook_size)
    return retrieval, collapse


def main(args: Config):

    # ---------- model -------------------------------------------------------
    model = SigLIP2SemanticOTModel(args).to(args.device)
    model.assert_no_shared_trainable_params(verbose=True)
    model.print_parameter_summary()

    # ---------- v78a: warm-start codebook from a K=K_init checkpoint --------
    warm_path = getattr(args, "warm_start_codebook_from", None)
    if warm_path and os.path.exists(warm_path):
        print(f"[v78a] warm-starting codebook from {warm_path}")
        sd_full = torch.load(warm_path, map_location=args.device, weights_only=False)
        # quantizer codebook/EMA buffers (K=K_init in source)
        cb_src = sd_full["quantizer.codebooks"]
        cs_src = sd_full["quantizer.cluster_size"]
        ea_src = sd_full["quantizer.embed_avg"]
        K_init = cb_src.shape[1]
        model.quantizer.warm_start_from_state(cb_src, cs_src, ea_src, K_init=K_init)
        # Load the rest of the model weights (skip quantizer state we just
        # warm-started — different tensor shape on the target).
        skip = {"quantizer.codebooks", "quantizer.cluster_size",
                "quantizer.embed_avg", "quantizer.embed_sqavg",
                "quantizer.active_mask"}
        filtered_sd = {k: v for k, v in sd_full.items() if k not in skip}
        miss, unexp = model.load_state_dict(filtered_sd, strict=False)
        print(f"[v78a] warm-start state_dict load: "
              f"missing={len(miss)}, unexpected={len(unexp)}")

    # ---------- dataset (untouched) -----------------------------------------
    transform      = get_transform('train')
    test_transform = get_transform('test')

    qwen_text_cache_path = getattr(args, "qwen_text_cache_path", None)
    feature_cache_dir    = getattr(args, "siglip2_feature_cache_dir", None)
    # v28a (PixelDecoder) needs the raw pixel image as reconstruction
    # target. v29 (paired-aug NtXent) needs paired augmented views: if
    # the SigLIP2 cache contains `visual_{tokens,global}_aug{0,1}.f16.npy`
    # the trainer uses them directly and we can skip the live PIL decode;
    # otherwise we fall back to live-decode + live backbone.
    _aug_cache_present = (
        feature_cache_dir is not None
        and os.path.exists(os.path.join(feature_cache_dir, "visual_tokens_aug0.f16.npy"))
        and os.path.exists(os.path.join(feature_cache_dir, "visual_tokens_aug1.f16.npy"))
    )
    force_pixel_decode = bool(
        (getattr(args, "use_decoder", False)
         and getattr(args, "decoder_target", "siglip_feat") == "pixel")
        or (getattr(args, "use_paired_aug_ntxent", False) and not _aug_cache_present)
    )
    if getattr(args, "use_paired_aug_ntxent", False):
        if _aug_cache_present:
            print(f"[train_siglip2] paired-aug NtXent: using CACHED aug views "
                  f"from {feature_cache_dir} (live backbone skipped).")
        else:
            print(f"[train_siglip2] paired-aug NtXent: aug cache NOT found in "
                  f"{feature_cache_dir}; falling back to live PIL decode + "
                  f"live backbone (slow).")
    # v29 needs strong CIBHash-style augmentations on the paired views.
    # The default `get_transform('train')` is too weak (color jitter 0,
    # scale (1.0, 1.0)), which would make img_tr1 ≈ img_tr2 and turn the
    # NtXent task into a trivial identity match. Override with a SimCLR
    # transform when --use_paired_aug_ntxent is on.
    if getattr(args, "use_paired_aug_ntxent", False):
        from torchvision import transforms as T
        cj = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([cj], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    trainset, testset, _ = load_dataset(
        args.dataset_dir, args.dataset, setting='setting1',
        train_transform=transform, test_transform=test_transform,
        load_train=True, load_database=False, load_test=True,
        return_index=True, return_paired_aug_img=True,
        qwen_text_cache_path=qwen_text_cache_path,
        siglip2_feature_cache_dir=feature_cache_dir,
        force_pixel_decode=force_pixel_decode,
    )
    if feature_cache_dir is not None:
        print(f"[train_siglip2] using cached SigLIP2 features from {feature_cache_dir}; "
              f"encoder pass will be skipped.")
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
    )

    # ---------- v41 (5-G): text-supervised codebook init --------------------
    # Replace the random Gaussian codebook init with K vectors derived from
    # train-set `cached_text_part_raw`. Text routing path is preserved
    # (still runs every forward) -- this is purely additive supervision for
    # the codebook starting point. See docs/ANALYSIS_2026-05-19.md sec 5-G.
    _text_init_mode = str(getattr(args, "text_init_codebook", "none"))
    if _text_init_mode != "none":
        _N_target = int(getattr(args, "text_init_subset", 4096))
        _seed     = int(getattr(args, "text_init_seed",   42))
        text_buffer = []
        n_collected = 0
        print(f"[text_init_codebook] gathering up to {_N_target} text vectors "
              f"from train_loader for mode={_text_init_mode!r} ...")
        for _batch in train_loader:
            _tp = _batch.get("cached_text_part_raw", None)
            if _tp is None:
                raise RuntimeError(
                    "[text_init_codebook] train batch has no "
                    "'cached_text_part_raw'; rebuild the SigLIP2 feature "
                    "cache or pass --qwen_text_cache_path."
                )
            text_buffer.append(_tp.detach().cpu())
            n_collected += _tp.shape[0]
            if n_collected >= _N_target:
                break
        text_anchors = torch.cat(text_buffer, dim=0)[:_N_target]  # [N, M, D_text]
        # v41 (CLIP backbone fix 2026-06-03): cached_text_part_raw is the raw
        # text-encoder output (512-dim for CLIP-B/16) but the codebook is
        # d_model-dim (768 for CLIP). Project the raw text vectors through
        # the model's text_adapter(s) so they live in the codebook space.
        # For SigLIP2 backbone (text/visual both 768-dim) this is a no-op
        # at the dim level but still applies the trained MLP/Linear.
        if int(text_anchors.shape[-1]) != int(model.d_model):
            print(f"[text_init_codebook] projecting text_anchors "
                  f"{text_anchors.shape[-1]} -> {model.d_model} via text_adapter ...")
            text_anchors = text_anchors.to(args.device)
            with torch.no_grad():
                N_a, M_a, D_a = text_anchors.shape
                if isinstance(model.text_adapter, torch.nn.ModuleList):
                    # per-slot text adapter: apply slot m to column m
                    out_cols = [
                        model.text_adapter[m](text_anchors[:, m, :])
                        for m in range(M_a)
                    ]
                    text_anchors = torch.stack(out_cols, dim=1)  # [N, M, d_model]
                else:
                    text_anchors = model.text_adapter(
                        text_anchors.reshape(N_a * M_a, D_a)
                    ).reshape(N_a, M_a, -1)
            text_anchors = text_anchors.detach().cpu()
            print(f"[text_init_codebook] projected text_anchors shape="
                  f"{tuple(text_anchors.shape)}")
        print(f"[text_init_codebook] collected text_anchors shape="
              f"{tuple(text_anchors.shape)}; calling init...")
        _diag = model.quantizer.initialize_from_text_anchors(
            text_anchors, mode=_text_init_mode, seed=_seed,
        )
        print(f"[text_init_codebook] done: {_diag}")
        del text_buffer, text_anchors

    # ---------- optimizer ---------------------------------------------------
    backbone_params = [p for p in model.backbone.parameters()             if p.requires_grad]
    # v45: allow text_adapter to have its own LR (typically higher than proj_lr).
    # When --text_adapter_lr is None this collapses to the legacy 2-group setup.
    text_adapter_lr = getattr(args, "text_adapter_lr", None)
    if text_adapter_lr is not None:
        text_adapter_params = [p for n, p in model.named_parameters()
                               if p.requires_grad and n.startswith("text_adapter.")]
        other_params        = [p for n, p in model.named_parameters()
                               if p.requires_grad
                               and not n.startswith("backbone.")
                               and not n.startswith("text_adapter.")]
        n_ta = sum(p.numel() for p in text_adapter_params)
        n_ot = sum(p.numel() for p in other_params)
        print(f"[optimizer] text_adapter group: {len(text_adapter_params)} tensors, "
              f"{n_ta:,} params @ lr={text_adapter_lr}")
        print(f"[optimizer] other group:        {len(other_params)} tensors, "
              f"{n_ot:,} params @ lr={args.proj_lr}")
    else:
        text_adapter_params = []
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith("backbone.")]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr,
                             'weight_decay': args.weight_decay})
    if text_adapter_params:
        param_groups.append({'params': text_adapter_params, 'lr': float(text_adapter_lr)})
    if other_params:
        param_groups.append({'params': other_params, 'lr': args.proj_lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters in the model.")
    optimizer = torch.optim.Adam(param_groups)
    # ---- LR scheduler ------------------------------------------------------
    # The legacy default was StepLR(step_size=10, gamma=1e-4) which decayed
    # lr to ~0 after epoch 10 and froze training. Default is now 'cosine'
    # (annealed across the full run); 'step' / 'none' are still selectable
    # via the new --lr_scheduler config.
    sched_kind = str(getattr(args, 'lr_scheduler', 'cosine'))
    if sched_kind == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epoch),
            eta_min=float(getattr(args, 'lr_eta_min', 1e-5)),
        )
    elif sched_kind == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, 'lr_step_size', 20)),
            gamma=float(getattr(args, 'lr_gamma', 0.5)),
        )
    elif sched_kind == 'none':
        scheduler = None
    else:
        raise ValueError(f"[train_siglip2] unknown --lr_scheduler={sched_kind!r}")
    print(f"[train_siglip2] lr scheduler = {sched_kind}")

    # ---------- criterion ---------------------------------------------------
    # Reads its weights / hyperparams via getattr fallbacks on `args`, so no
    # config.py edits are required. See `loss_siglip2.DNACodonHashLoss`.
    criterion = DNACodonHashLoss(args).to(args.device)

    # Loss components recorded per epoch in log.csv. After the 2026-05-13
    # cleanup of inactive R-series / v15 / `loss_global` terms, this list
    # matches the active set defined in `loss_siglip2.DNACodonHashLoss`.
    # v122: only ACTIVE losses (gating lambda > 0 OR enabling flag set) are
    # logged. Keeps log.csv / tensorboard relevant to *this* run's recipe
    # and ensures newly-added losses (cibhash v119, swav_assign v121,
    # codeword_text_proto v123, etc.) are automatically picked up when
    # their lambdas turn on. See `_build_active_loss_types`.
    loss_types = _build_active_loss_types(args)
    print(f"[csv-logger] active loss keys ({len(loss_types)}): {loss_types}")

    # ---------- per-epoch CSV logger ---------------------------------------
    csv_fields = ["epoch"]
    for phase in ("train", "val"):
        for k in loss_types:
            csv_fields.append(f"{phase}_{k}")
    csv_fields += [
        "eval_mAP",
        "eval_mean_positive_distance",
        "eval_mean_negative_distance",
        "eval_dead_code_ratio_mean",
        "eval_unique_code_ratio",
        "eval_duplicate_rate",
        "eval_mean_base_normalized_entropy",
        "eval_mean_per_codebook_unique_ratio",
    ]
    csv_path = os.path.join(args.save_log_path, "log.csv")
    csv_logger = EpochCSVLogger(csv_path, csv_fields)
    print(f"[csv-logger] writing per-epoch metrics to {csv_path}")

    # ---------- loop --------------------------------------------------------
    def one_epoch(train, loader, epoch):
        result = {k: 0.0 for k in loss_types}

        for i, batch in tqdm(enumerate(loader)):
            # ---- cached SigLIP2 features (when extract_siglip2_features.py
            # has been run) take priority. They short-circuit the encoder
            # pass entirely.
            cached_vt  = batch.get("cached_visual_tokens_raw", None)
            cached_vg  = batch.get("cached_visual_global",     None)
            cached_tp  = batch.get("cached_text_part_raw",     None)
            cached_ht  = batch.get("has_text",                 None)
            cached_tt  = batch.get("cached_text_tokens",       None)
            cached_ttm = batch.get("cached_text_token_mask",   None)
            # v29 paired-aug NtXent (train-only): prefer the cached
            # `visual_{tokens,global}_aug{0,1}` views when present
            # (built by extract_siglip2_features.py --save_aug_views 2).
            # If they are not in the batch, fall back to running the
            # live backbone on img_tr1 / img_tr2.
            v29_aug_cached = bool(
                train and getattr(args, 'use_paired_aug_ntxent', False)
                and ('cached_visual_tokens_aug0' in batch)
                and ('cached_visual_tokens_aug1' in batch)
            )
            v29_aug_live = bool(
                train and getattr(args, 'use_paired_aug_ntxent', False)
                and (not v29_aug_cached)
                and ('img_tr1' in batch) and ('img_tr2' in batch)
            )
            v29_train = v29_aug_cached or v29_aug_live
            if v29_train and not v29_aug_cached:
                # live-backbone path: drop the deterministic *visual* cache
                # so the encoder runs on img_tr1 (and img_tr2 for view 2).
                # KEEP cached text inputs -- text-supervised compositional
                # code is a core contribution; routing centroids must come
                # from the Qwen captions when available. Caption is
                # image-content-agnostic, so the same text is correct for
                # both augmented views of the same image.
                cached_vt = cached_vg = None
                # `cached_tp`, `cached_ht`, `cached_tt`, `cached_ttm`
                # deliberately preserved (v162 grounded text routing needs
                # the per-image token-level text cache here).
            if v29_aug_cached:
                # cached-aug path: swap visual cache for aug-0 tensors.
                # Text path preserved as above.
                cached_vt = batch['cached_visual_tokens_aug0']
                cached_vg = batch['cached_visual_global_aug0']
                # `cached_tp`, `cached_ht`, `cached_tt`, `cached_ttm` preserved.
            using_cache = cached_vt is not None
            if using_cache:
                cached_vt = cached_vt.to(args.device)
                cached_vg = cached_vg.to(args.device) if cached_vg is not None else None
                cached_tp = cached_tp.to(args.device) if cached_tp is not None else None
                cached_ht = cached_ht.to(args.device) if cached_ht is not None else None
                cached_tt  = cached_tt .to(args.device) if cached_tt  is not None else None
                cached_ttm = cached_ttm.to(args.device) if cached_ttm is not None else None
                pixel_values   = None
                part_input_ids = None
                part_attn      = None
            else:
                # v29 path: use img_tr1 as view-1 input. Otherwise default
                # to the deterministic pixel_values.
                if v29_train:
                    pixel_values = batch['img_tr1'].to(args.device)
                else:
                    pixel_values = batch.get("pixel_values", batch.get("img", None))
                    if pixel_values is None:
                        raise RuntimeError(
                            "[train_siglip2] batch must contain 'pixel_values' / 'img' "
                            "or cached SigLIP2 features."
                        )
                    pixel_values = pixel_values.to(args.device)
                if 'part_input_ids' in batch and 'part_attention_mask' in batch:
                    part_input_ids = batch['part_input_ids'].to(args.device)
                    part_attn      = batch['part_attention_mask'].to(args.device)
                else:
                    part_input_ids = None
                    part_attn      = None

            # ---- forward --------------------------------------------------
            out = model(
                pixel_values=pixel_values,
                part_input_ids=part_input_ids,
                part_attention_mask=part_attn,
                return_routing=True,
                cached_visual_tokens_raw=cached_vt,
                cached_visual_global=cached_vg,
                cached_text_part_raw=cached_tp,
                cached_has_text=cached_ht,
                cached_text_tokens=cached_tt,
                cached_text_token_mask=cached_ttm,
            )

            # ---- v29 paired-aug NtXent: forward a SECOND augmented view --
            # Train-only path. View 2 also uses the SAME text path inputs
            # as view 1 (cached_tp / cached_ht / live part_input_ids) --
            # captions describe the original image which is shared
            # between the two augmented views, so text routing belongs in
            # both forwards. Without this the text-supervised
            # compositional-code claim has no training signal at all
            # under paired-aug NtXent.
            out_view2 = None
            if v29_aug_cached:
                out_view2 = model(
                    pixel_values=None,
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                    cached_visual_tokens_raw=batch['cached_visual_tokens_aug1'].to(args.device),
                    cached_visual_global=batch['cached_visual_global_aug1'].to(args.device),
                    cached_text_part_raw=cached_tp,
                    cached_has_text=cached_ht,
                )
            elif v29_aug_live and ('img_tr2' in batch):
                pix_v2 = batch['img_tr2'].to(args.device)
                out_view2 = model(
                    pixel_values=pix_v2,
                    part_input_ids=part_input_ids,
                    part_attention_mask=part_attn,
                    return_routing=True,
                )

            # ---- labels ----------------------------------------------------
            mh_labels = batch.get('label',
                          batch.get('labels',
                          batch.get('multi_hot_labels',
                          batch.get('multi_label', None))))
            if mh_labels is not None:
                mh_labels = mh_labels.to(args.device)
            single_labels = batch.get('target', batch.get('class_id', None))
            if single_labels is not None:
                single_labels = single_labels.to(args.device)

            # ---- loss -----------------------------------------------------
            # pixel target for v28a: ImageNet-normalized image tensors. The
            # dataloader keeps them under 'pixel_values' (or 'img') even when
            # cached SigLIP2 features are loaded -- needed by PixelDecoder's
            # MSE target. Use the cached visual_global path (out["visual_global_feat"])
            # for v28b -- handled inside the criterion.
            pixel_target = batch.get('pixel_values', batch.get('img', None))
            if pixel_target is not None:
                pixel_target = pixel_target.to(args.device)
            loss_dict = criterion(
                outputs=out,
                labels=single_labels,
                multi_hot_labels=mh_labels,
                epoch=epoch,
                pixel_target=pixel_target,
                outputs_view2=out_view2,
            )
            loss = loss_dict['loss']

            # ---- backward (train only) ------------------------------------
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ---- accumulate -----------------------------------------------
            B = (cached_vt.shape[0] if using_cache else pixel_values.shape[0])
            for k in loss_types:
                if k.startswith("routing_"):
                    v = out.get(k, None)
                else:
                    v = loss_dict.get(k, None)
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.detach().item()
                result[k] += float(v) * B

        # mean over the dataset
        n = len(loader.dataset)
        for k in loss_types:
            result[k] = result[k] / max(n, 1)
        return result

    # ---------- writers + epoch loop ---------------------------------------
    train_writer, val_writer = get_summarywriter(loss_types, args.save_log_path)

    eval_every       = int(getattr(args, "eval_every",         5))
    distance_mode    = str(getattr(args, "dna_distance_mode",  "base"))
    codebook_size_cf = int(getattr(args, "codebook_size",      32))

    # ---------- Gumbel-tau cosine annealing config -------------------------
    # When `gumbel_tau_anneal` is on we override the static `gumbel_tau` set
    # at model build time with a cosine schedule from `gumbel_tau_init` down
    # to `gumbel_tau_final` across the full run. Effective `tau` is propagated
    # to every codon head via `model.set_gumbel_tau(...)` at the start of
    # each epoch. Higher tau early -> more exploration -> less code collapse.
    tau_anneal = bool(getattr(args, "gumbel_tau_anneal", True))
    tau_init   = float(getattr(args, "gumbel_tau_init",  2.0))
    tau_final  = float(getattr(args, "gumbel_tau_final", 0.3))
    def _gumbel_tau_for_epoch(e: int) -> float:
        if not tau_anneal:
            return float(getattr(args, "gumbel_tau", 1.0))
        denom = max(int(args.epoch) - 1, 1)
        cos_w = 0.5 * (1.0 + math.cos(math.pi * (e / denom)))
        return tau_final + (tau_init - tau_final) * cos_w

    # v78a: parse split epochs once
    _split_epochs_str = str(getattr(args, "split_epochs", ""))
    _split_epochs = set()
    if _split_epochs_str:
        try:
            _split_epochs = {int(x) for x in _split_epochs_str.split(",") if x.strip()}
        except ValueError:
            _split_epochs = set()
    _split_max = int(getattr(args, "split_max_per_epoch", 12))

    for e in range(args.epoch):

        # propagate annealed gumbel-softmax temperature to codon heads
        cur_tau = _gumbel_tau_for_epoch(e)
        model.set_gumbel_tau(cur_tau)

        # v33a: propagate current epoch to model so Sinkhorn router can
        # compute annealed epsilon (no-op when annealing is off).
        if hasattr(model, "set_current_epoch"):
            model.set_current_epoch(e)

        # v112: hierarchical codon decomposition -- refresh text-similarity
        # clusters every N epochs after a warmup, by running k-means on the
        # codebook codewords. Cluster labels are stored as a buffer in the
        # criterion and consumed by _loss_hierarchical_cluster_codon.
        if float(getattr(criterion, "lambda_hierarchical_cluster_codon", 0.0)) > 0.0:
            _warmup = int(getattr(criterion, "hierarchical_cluster_warmup_epochs", 5))
            _refresh_every = max(1, int(getattr(criterion, "hierarchical_cluster_refresh_every", 5)))
            _not_init = not bool(criterion.hierarchical_cluster_initialized.item())
            _refresh_now = (
                e >= _warmup
                and (_not_init or ((e - _warmup) % _refresh_every == 0))
            )
            if _refresh_now:
                cb_buf = model.quantizer.codebooks                 # [M, K_max, D]
                cb_mask = model.quantizer.active_mask              # [M, K_max] bool
                _method = str(getattr(args, "hierarchical_cluster_method", "kmeans"))
                _diag = criterion.refresh_clusters(
                    cb_buf, cb_mask, method=_method, seed=42,
                )
                print(f"[v112-hierarchical] epoch {e}: refreshed clusters -- "
                      f"C={_diag['C']}, active_per_cb={_diag['active_per_cb']}, "
                      f"elapsed={_diag['elapsed_sec']:.2f}s")

        # v78a: codeword split at scheduled epochs (BEFORE this epoch's training)
        if e in _split_epochs and hasattr(model.quantizer, "do_split"):
            print(f"[v78a] split hook at epoch {e}: sweeping train loader to "
                  f"collect z_per_cw + full-code collision pressure")
            with torch.no_grad():
                model.eval()
                z_per_cw, cp = _collect_split_data(model, train_loader, args.device,
                                                    M=model.num_codebooks,
                                                    K_max=model.quantizer.K_max)
            info = model.quantizer.do_split(
                max_splits=_split_max, z_per_cw=z_per_cw,
                collision_pressure=cp,
            )
            print(f"[v78a] split @ ep{e}: n_split={info['n_split_total']}, "
                  f"per_codebook={info['split_count_per_codebook']}, "
                  f"active_K={info['active_K_per_codebook']}")

        model.train()
        train_result = one_epoch(train=True, loader=train_loader, epoch=e)
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_result = one_epoch(train=False, loader=test_loader, epoch=e)

        # tensorboard
        for loss in loss_types:
            train_writer.add_scalar(f'train/loss/{loss}', train_result[loss], e)
            val_writer.add_scalar  (f'val/loss/{loss}',   val_result[loss],   e)

        # ---------- mid-training retrieval eval (every eval_every epochs)
        eval_row: dict = {}
        run_mid_eval = (
            eval_every > 0
            and ((e + 1) % eval_every == 0 or (e + 1) == args.epoch)
        )
        if run_mid_eval:
            try:
                model.eval()
                print(f"[mid-eval] running retrieval+collapse eval at epoch {e}")
                retrieval, collapse = _mid_train_eval(
                    model, test_loader, args.device,
                    distance_mode=distance_mode,
                    codebook_size=codebook_size_cf,
                )
                eval_row = {
                    "eval_mAP":                          retrieval["mAP"],
                    "eval_mean_positive_distance":       retrieval["mean_positive_distance"],
                    "eval_mean_negative_distance":       retrieval["mean_negative_distance"],
                    "eval_dead_code_ratio_mean":         float(np.mean(collapse["dead_code_ratio"])),
                    "eval_unique_code_ratio":            collapse["unique_code_ratio"],
                    "eval_duplicate_rate":               collapse["duplicate_rate"],
                    "eval_mean_base_normalized_entropy": collapse["mean_base_normalized_entropy"],
                    "eval_mean_per_codebook_unique_ratio": collapse.get(
                        "mean_per_codebook_unique_ratio", 0.0,
                    ),
                }
                for k, v in eval_row.items():
                    val_writer.add_scalar(f"eval/{k}", float(v), e)
                print(
                    f"[mid-eval] epoch {e}: mAP={retrieval['mAP']:.4f}, "
                    f"unique={collapse['unique_code_ratio']:.4f}, "
                    f"per-cb-unique={collapse.get('mean_per_codebook_unique_ratio', 0.0):.4f}, "
                    f"dead={float(np.mean(collapse['dead_code_ratio'])):.4f}"
                )
            except Exception as ex:
                print(f"[mid-eval] WARNING: skipped due to error: {ex}")

        # ---------- per-epoch CSV append
        csv_row = {"epoch": e}
        for k in loss_types:
            csv_row[f"train_{k}"] = train_result.get(k, "")
            csv_row[f"val_{k}"]   = val_result  .get(k, "")
        csv_row.update(eval_row)
        csv_logger.log(csv_row)

        if ((e + 1) % args.print_epoch) == 0:
            print_one_epoch_info(e, [train_result, val_result])

        # Track best mid-eval mAP checkpoint. SigLIP2 backbone is frozen so
        # only the trainable adapter / codebook params get saved — small
        # disk footprint (<300 MB), safe to keep alongside final.
        _eval_mAP = float(eval_row.get("eval_mAP", -1.0)) if eval_row else -1.0
        if _eval_mAP > getattr(args, "_best_mid_mAP", -1.0):
            args._best_mid_mAP = _eval_mAP
            args._best_mid_epoch = int(e)
            best_model_path = os.path.join(args.save_model_state_path, "model_state_dict_best.pth")
            best_crit_path  = os.path.join(args.save_model_state_path, "criterion_state_dict_best.pth")
            torch.save(model.state_dict(),     best_model_path)
            torch.save(criterion.state_dict(), best_crit_path)
            print(f"[best-ckpt] new best mid-eval mAP={_eval_mAP:.4f} at epoch {e} — saved to model_state_dict_best.pth")

        # Save ONLY the final-epoch checkpoint. Per-epoch intermediates were
        # ~1.5 GB each (SigLIP2 backbone serialized), filling /home quickly.
        # `args.best_save` is now a no-op for intermediate epochs.
        if (e + 1) == args.epoch:
            model_path = os.path.join(args.save_model_state_path, "model_state_dict.pth")
            crit_path  = os.path.join(args.save_model_state_path, "criterion_state_dict.pth")
            torch.save(model.state_dict(),     model_path)
            # Persist the criterion too so its EMA buffer survives a resume.
            torch.save(criterion.state_dict(), crit_path)
            print(f"Final checkpoint saved to `{args.save_model_state_path}`")
            # If best-checkpoint differs from final, replace final with best
            # for the downstream evaluation / extraction. Best is preserved
            # as model_state_dict_best.pth.
            if hasattr(args, "_best_mid_epoch") and args._best_mid_epoch != e:
                best_model_path = os.path.join(args.save_model_state_path, "model_state_dict_best.pth")
                if os.path.exists(best_model_path):
                    import shutil
                    print(f"[best-ckpt] swapping final checkpoint with best (epoch {args._best_mid_epoch}, mAP={args._best_mid_mAP:.4f}) for final eval")
                    shutil.copy2(best_model_path, model_path)
                    best_crit_path = os.path.join(args.save_model_state_path, "criterion_state_dict_best.pth")
                    if os.path.exists(best_crit_path):
                        shutil.copy2(best_crit_path, crit_path)
                    # Reload model from best checkpoint for in-memory use too
                    model.load_state_dict(torch.load(best_model_path, map_location=args.device, weights_only=False))

    args.save_arg()

    train_writer.flush(); train_writer.close()
    val_writer.flush();   val_writer.close()

    # ---------- end-of-training extraction + full evaluation -------------
    if getattr(args, "evaluation", False):
        from extraction_siglip2 import extract_code as _extract_code
        from evaluation_siglip2 import evaluation as _evaluation
        # Optionally swap cache for final evaluation (e.g., train on FAIRrank
        # multi-view cache, evaluate on whole-image cache for paper-claim
        # inference mode). Restores after eval so any post-hoc analysis still
        # has training cache pointer if needed.
        _eval_cache = getattr(args, "eval_cache_dir", None)
        _saved_cache = args.siglip2_feature_cache_dir
        _saved_whiten = getattr(args, "text_whiten_npz", None)
        if _eval_cache and os.path.exists(_eval_cache):
            print(f"[final-eval] OVERRIDE cache: {_saved_cache} -> {_eval_cache}")
            args.siglip2_feature_cache_dir = _eval_cache
            _maybe_whiten = os.path.join(_eval_cache, "text_whiten.npz")
            if os.path.exists(_maybe_whiten):
                args.text_whiten_npz = _maybe_whiten
                print(f"[final-eval] OVERRIDE text_whiten: {_maybe_whiten}")
        try:
            print("[final-eval] running extraction ...")
            _extract_code(args)
        except Exception as ex:
            print(f"[final-eval] extraction failed: {ex} -- continuing to viz.")
        try:
            print(f"[final-eval] running evaluation (distance_mode={distance_mode}) ...")
            _evaluation(
                args.save_result_path,
                distance_mode=distance_mode,
                codebook_size=codebook_size_cf,
            )
        except Exception as ex:
            print(f"[final-eval] evaluation failed: {ex} -- continuing to viz. "
                  f"Re-run evaluation_siglip2.py externally to recover metrics.")
        # Restore (post-eval compositional uses _cache below; keep it on eval cache too)
        # so compositional_eval reads correctly on the eval-cache features.

        # ---------- post-eval compositional analysis ---------------------
        # NMI + drop ablation + B0/B1/B2 lift. Each wrapped in try/except so
        # a single failure does not block the others or viz below. Disable
        # with --no-post_eval_compositional.
        if bool(getattr(args, "post_eval_compositional", True)):
            _rd = args.save_result_path
            _cache = getattr(args, "siglip2_feature_cache_dir", None)
            _droot = os.path.join(args.dataset_dir, args.dataset)
            try:
                print("[post-eval] pairwise NMI ...")
                import subprocess, sys
                subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "scripts/pairwise_nmi.py"),
                    "--results", _rd], check=False, timeout=600)
            except Exception as ex:
                print(f"[post-eval] pairwise_nmi failed: {ex}")
            try:
                import numpy as _np
                _ndb = int(_np.load(os.path.join(_rd, "extract_db.npz"))["base_indices"].shape[0])
                _subset = 1000 if _ndb > 25000 else 2000
                print(f"[post-eval] drop ablation (subset={_subset}) ...")
                import subprocess, sys
                subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__),
                                 "scripts/codebook_drop_ablation_fast.py"),
                    "--result_dir", _rd,
                    "--subset_queries", str(_subset)],
                    check=False, timeout=1800)
            except Exception as ex:
                print(f"[post-eval] drop_ablation failed: {ex}")
            try:
                if _cache and os.path.exists(_cache):
                    print("[post-eval] compositional B0/B1/B2 ...")
                    import subprocess, sys
                    subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__),
                                     "compositional_eval.py"),
                        "--result_dir", _rd,
                        "--cache_dir", _cache,
                        "--dataset_root", _droot,
                        "--skip_grids"],
                        check=False, timeout=1200)
                else:
                    print(f"[post-eval] skipped compositional (cache_dir missing: {_cache})")
            except Exception as ex:
                print(f"[post-eval] compositional_eval failed: {ex}")

    # ---------- end-of-training diagnostic plots --------------------------
    # Routing heatmap + per-codebook t-SNE go into the run directory next to
    # log.csv / evaluation_*.json. Disable with --no_visualize.
    if bool(getattr(args, "visualize", True)):
        try:
            from dna_utils import visualize_routing, visualize_codebook_tsne
        except ImportError as ie:
            print(f"[visualize] skipped (missing dep: {ie})")
        else:
            qwen_jsonl = getattr(args, "qwen_text_cache_path", None)
            n_routing  = int(getattr(args, "viz_routing_samples", 12))
            n_tsne     = int(getattr(args, "viz_tsne_samples",    2000))
            # When --eval_cache_dir is set, rebuild a viz_trainset on the
            # whole-image cache so the routing heatmap reflects the
            # paper-claim inference mode (1 image, 196 tokens) rather than
            # the training-cache K-crop layout.
            _viz_trainset = trainset
            _eval_cache_for_viz = getattr(args, "eval_cache_dir", None)
            if _eval_cache_for_viz and os.path.exists(_eval_cache_for_viz):
                try:
                    print(f"[visualize] rebuilding viz_trainset on eval cache: {_eval_cache_for_viz}")
                    _viz_trainset, _, _ = load_dataset(
                        args.dataset_dir, args.dataset, setting='setting1',
                        train_transform=transform, test_transform=test_transform,
                        load_train=True, load_database=False, load_test=False,
                        return_index=True, return_paired_aug_img=False,
                        qwen_text_cache_path=qwen_jsonl,
                        siglip2_feature_cache_dir=_eval_cache_for_viz,
                        force_pixel_decode=False,
                    )
                except Exception as e:
                    print(f"[visualize] viz_trainset rebuild failed: {e} — falling back to trainset")
                    _viz_trainset = trainset
            try:
                # Use viz_trainset (whole-image cache when override is set).
                visualize_routing(
                    model, _viz_trainset,
                    save_path=os.path.join(args.save_result_path, "viz_routing_heatmap.png"),
                    qwen_jsonl_path=qwen_jsonl,
                    num_samples=n_routing,
                    device=args.device,
                )
            except Exception as e:
                print(f"[visualize_routing] failed: {e}")
            try:
                visualize_codebook_tsne(
                    model, testset,
                    save_path=os.path.join(args.save_result_path, "viz_codebook_tsne.png"),
                    num_samples=n_tsne,
                    device=args.device,
                )
            except Exception as e:
                print(f"[visualize_codebook_tsne] failed: {e}")


if __name__ == '__main__':

    set_random_seed(42)

    args = Config()
    args.set_args()
    # Override the legacy nested layout (`result/<date>/<tag>/log/...` +
    # `result/<date>/<tag>/model_state/...`) with a flat per-run directory.
    # `set_args()` eagerly mkdir's those legacy paths even though we never
    # use them under SigLIP2. Capture and clean them up so `result/` is not
    # littered with empty `<date>/<tag>/log,model_state/` skeletons.
    legacy_log         = args.save_log_path
    legacy_model_state = args.save_model_state_path
    legacy_result_path = args.save_result_path

    flat_dir = _resolve_save_path(args)
    args.save_result_path      = flat_dir.rstrip(os.sep)
    args.save_log_path         = flat_dir
    args.save_model_state_path = flat_dir

    for _p in (legacy_log, legacy_model_state, legacy_result_path):
        try:
            if os.path.isdir(_p) and not os.listdir(_p):
                os.rmdir(_p)
        except OSError:
            pass
    _date_parent = os.path.dirname(legacy_result_path) if legacy_result_path else None
    try:
        if _date_parent and os.path.isdir(_date_parent) and not os.listdir(_date_parent):
            os.rmdir(_date_parent)
    except OSError:
        pass

    args.print_info()
    main(args)
