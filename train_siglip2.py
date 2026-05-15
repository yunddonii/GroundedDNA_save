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
    """Build a flat result directory tailored for SigLIP2 runs.

    Schema:
        result/<date>+<tag>+bs+<batch_size>+e+<epoch>+proj_lr+<proj_lr>/
            ├── model_state_dict.pth          (final-epoch checkpoint)
            ├── criterion_state_dict.pth      (final; preserves EMA buffer)
            ├── log.csv                       (per-epoch metrics, append mode)
            ├── args.txt                      (Config.print_info dump)
            ├── train/  val/                  (tensorboard event subdirs)
            ├── config.pt                     (Config.save_arg() output)
            └── extract_db.npz / extract_query.npz / evaluation_*.json
                                              (extraction & evaluation outputs)

    Trailing separator preserved so the legacy
    ``Config.print_info()`` line ``open(self.save_log_path + 'args.txt', ...)``
    (no separator between path and filename) resolves correctly.
    """
    tag = str(args.date)
    if args.tag is not None:
        for t in args.tag:
            tag = "+".join([tag, t])
    for k, v in [
        ("bs",      args.batch_size),
        ("e",       args.epoch),
        ("proj_lr", args.proj_lr),
    ]:
        tag = "+".join([tag, k, str(v)])

    base = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "result",
        tag,
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "")


# ----------------------------- mid-training retrieval eval helper

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

    # ---------- dataset (untouched) -----------------------------------------
    transform      = get_transform('train')
    test_transform = get_transform('test')

    qwen_text_cache_path = getattr(args, "qwen_text_cache_path", None)
    feature_cache_dir    = getattr(args, "siglip2_feature_cache_dir", None)
    # v28a (PixelDecoder) and v29 (paired-aug NtXent) both need the raw
    # image at training time -- v28a as a reconstruction target, v29 as
    # paired SimCLR-style augmented views. We keep the PIL decode active
    # whenever either is on. The flag is a no-op for v27b/v28b.
    force_pixel_decode = bool(
        (getattr(args, "use_decoder", False)
         and getattr(args, "decoder_target", "siglip_feat") == "pixel")
        or getattr(args, "use_paired_aug_ntxent", False)
    )
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

    # ---------- optimizer ---------------------------------------------------
    backbone_params = [p for p in model.backbone.parameters()             if p.requires_grad]
    other_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("backbone.")]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr,
                             'weight_decay': args.weight_decay})
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
    loss_types = [
        'loss',
        'loss_hash', 'loss_hash_hard',
        'loss_vq', 'loss_quant',
        'loss_anchor',
        'loss_dna', 'loss_entropy', 'loss_base_balance',
        'loss_bu', 'loss_cb_balance', 'loss_cb_uncorr',
        'loss_wasserstein',
        'loss_recon',
        'loss_ntxent',
    ]

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
            # v29 paired-aug NtXent (train-only): bypass cache entirely
            # and feed two augmented views (img_tr1, img_tr2) live through
            # the (frozen) backbone. The cache only holds the deterministic
            # view, which would defeat the purpose of contrastive training.
            v29_train = bool(
                train and getattr(args, 'use_paired_aug_ntxent', False)
                and ('img_tr1' in batch) and ('img_tr2' in batch)
            )
            if v29_train:
                cached_vt = cached_vg = cached_tp = cached_ht = None
                cached_tt = cached_ttm = None
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
            # Train-only path. Forces live backbone (cached_*=None) on
            # `img_tr2`, no text routing (text path is unchanged across
            # augmentations so it would be redundant). The resulting DNA
            # code is contrasted against `out['dna_hash_code_st']` inside
            # the criterion.
            out_view2 = None
            if (
                train and getattr(args, 'use_paired_aug_ntxent', False)
                and ('img_tr2' in batch)
            ):
                pix_v2 = batch['img_tr2'].to(args.device)
                out_view2 = model(
                    pixel_values=pix_v2,
                    part_input_ids=None,
                    part_attention_mask=None,
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

    for e in range(args.epoch):

        # propagate annealed gumbel-softmax temperature to codon heads
        cur_tau = _gumbel_tau_for_epoch(e)
        model.set_gumbel_tau(cur_tau)

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
                }
                for k, v in eval_row.items():
                    val_writer.add_scalar(f"eval/{k}", float(v), e)
                print(
                    f"[mid-eval] epoch {e}: mAP={retrieval['mAP']:.4f}, "
                    f"unique={collapse['unique_code_ratio']:.4f}, "
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

    args.save_arg()

    train_writer.flush(); train_writer.close()
    val_writer.flush();   val_writer.close()

    # ---------- end-of-training extraction + full evaluation -------------
    if getattr(args, "evaluation", False):
        from extraction_siglip2 import extract_code as _extract_code
        from evaluation_siglip2 import evaluation as _evaluation
        print("[final-eval] running extraction ...")
        _extract_code(args)
        print(f"[final-eval] running evaluation (distance_mode={distance_mode}) ...")
        _evaluation(
            args.save_result_path,
            distance_mode=distance_mode,
            codebook_size=codebook_size_cf,
        )

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
            try:
                visualize_routing(
                    model, testset,
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
