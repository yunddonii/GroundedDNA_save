import torch
import numpy as np

import os
import time
import random

import argparse
import torch.distributed
import yaml

from dataloaders import NUM_CLASSES
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-gpu
    torch.backends.cudnn.deterministic = True # reduce operation speed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

class Config():
    
    def __init__(self):
        
        self.date = time.strftime("%y%m%d", time.localtime(time.time()))
        
    @staticmethod
    def get_config():
    
        # parser = argparse.ArgumentParser()
        
        parser = argparse.ArgumentParser(description='the hyperparameters for training')
        
        parser.add_argument('--config_path', dest='config_path', default=None, help='If you have config file, input the file path')
        
        save_arg = parser.add_argument_group("save information")
        save_arg.add_argument('--tag', dest='tag', nargs='+')
        save_arg.add_argument('-s', '--save', dest='best_save', action='store_true', help='Save best model state')
        save_arg.add_argument('--print_epoch', nargs='?', type=int, default=1, help='Epoch print infomation period')
        save_arg.add_argument('--log_dir', nargs='?', default='result', help='The directory name to save log file \n\t default: %(default)s')
        save_arg.add_argument('--test', dest='test', action='store_true', help='If true, model is tested after training.')
        save_arg.add_argument('--tsne', dest='tsne', action='store_true', help='If true, the emb of model is visualized.')
        
        train_arg = parser.add_argument_group("training parameters")
        train_arg.add_argument('-nd', '--num_devices', dest='num_devices', nargs='?', type=int, default=1, help='The number of devices for distributed training (default: %(default)s)')
        train_arg.add_argument('-mg', '--multi_gpu', dest='multi_gpu', action='store_true')
        train_arg.add_argument('-e', '--epoch', dest='epoch', nargs='?', type=int, default=20, help='# of total epoch')
        train_arg.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, default=64, help='Batch size')
        # Active learning rates in the SigLIP2 framework:
        #   -lr1 / --backbone_lr : SigLIP2 backbone (used only when --no_freeze_backbone)
        #   -lr3 / --proj_lr     : adapters + quantizer.codebooks + global_gate +
        #                          codon_heads (everything trainable from scratch)
        # The other lr flags below were used by the legacy pipeline only and are
        # kept commented out so they can be revived by uncommenting if needed.
        # ─── unused in SigLIP2 framework ───────────────────────────────────
        # train_arg.add_argument('-lr', '--lr', dest='lr', nargs='?', type=float, default=1e-3, help='Learning rate')                       # never referenced
        train_arg.add_argument('-lr1', '--backbone_lr', dest='backbone_lr', nargs='?', type=float, default=1e-6, help='Learning rate')
        # train_arg.add_argument('-lr2', '--code_emb_lr', dest='code_emb_lr', nargs='?', type=float, default=1e-3, help='Learning rate')    # legacy train.py / pl_train.py only (code_embedding)
        train_arg.add_argument('-lr3', '--proj_lr', dest='proj_lr', nargs='?', type=float, default=1e-3, help='Learning rate')
        # v45 (BERT-text): text_adapter often needs a stronger update than
        # visual_adapter / quantizer / codon_heads, especially when the
        # text encoder is swapped to BERT (no co-training with images).
        # When this flag is None (default) the text_adapter shares proj_lr;
        # when set, it gets its own group with this LR (typically 3-10x).
        train_arg.add_argument('--text_adapter_lr', dest='text_adapter_lr',
            type=float, default=None,
            help='Dedicated learning rate for text_adapter parameters. '
                 'None (default) -> shares proj_lr. Set higher (e.g. 5e-3) '
                 'when text encoder is swapped to BERT or when text path '
                 'needs to catch up.')
        # train_arg.add_argument('-lr4', '--proxy_lr', dest='proxy_lr', nargs='?', type=float, default=1e-6, help='Learning rate')           # never referenced
        train_arg.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.3)
        # train_arg.add_argument('--max_lr', dest='max_lr', nargs='?', type=float, default=0.001, help='maximum learning rate in cosine lr scheduler')  # legacy pl_train.py cosine scheduler only
        train_arg.add_argument('--scheduler', dest='scheduler', nargs='?', type=str, choices=['step', 'lambda', 'exponential', 'cosine', 'reduce'], default='reduce', help='scheduler (default: %(default)s)')
        train_arg.add_argument('--weight_decay', dest='weight_decay', nargs='?', type=int, default=6e-2)
        train_arg.add_argument('--num_workers', dest='num_workers', nargs='?', type=int, default=16)
        
        model_arg = parser.add_argument_group("Transformer parameters")
        model_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', nargs='?', type=int, default=64, help='d_model in transformer')
        model_arg.add_argument('--n_ref', dest='n_ref', nargs='?', type=int, default=50)

        data_arg = parser.add_argument_group("data arguments")
        data_arg.add_argument('--dataset', dest='dataset', default='CIFAR10')
        data_arg.add_argument('--dataset_dir', dest='dataset_dir', default='/home/yschoi/DNAemb_quat_hashing/dataset')
        data_arg.add_argument('--hash_path', dest='hash_path', default='/home/yschoi/DNAemb_quat_hashing/converted_data.csv')
        data_arg.add_argument('--setting', type=int, choices=[1, 2], default=1)
        
        test_arg = parser.add_argument_group("test parameters")
        test_arg.add_argument('-ev', '--eval', dest='evaluation', action='store_true')

        # ---------- siglip2 / dna hashing parameters --------------------
        # All flags below are additive — every model/loss module also has
        # `getattr(args, ..., default)` fallback, so omitting them keeps the
        # legacy behavior unchanged. Specifying them here lets us drive runs
        # via CLI without editing source.
        siglip2_arg = parser.add_argument_group("siglip2 / dna hashing parameters")
        siglip2_arg.add_argument('--siglip2_backbone', dest='siglip2_backbone',
            default='google/siglip2-base-patch16-224',
            help='Hugging Face SigLIP2 backbone name (default: %(default)s)')
        siglip2_arg.add_argument('--qwen_text_cache_path', dest='qwen_text_cache_path',
            default=None,
            help='JSONL cache produced by preprocess_qwen_codebook_texts.py.')
        siglip2_arg.add_argument('--siglip2_feature_cache_dir', dest='siglip2_feature_cache_dir',
            default=None,
            help='Directory written by extract_siglip2_features.py (memmaped '
                 'fp16 visual+text feature tensors). When set, training and '
                 'extraction skip the SigLIP2 encoder pass and read these '
                 'features directly. Big speedup for frozen-backbone runs.')
        siglip2_arg.add_argument('--d_model', dest='d_model', type=int, default=None,
            help='Adapter output dim; None = use SigLIP2 projection_dim.')
        # v30 ablation: control adapter capacity. 'mlp' (default) keeps
        # the LayerNorm + Linear(D, 2D) + GELU + Dropout + Linear(2D, D)
        # + residual block. 'linear' swaps it for LayerNorm + Linear(D, D)
        # (no hidden, no GELU) -- ~5x fewer trainable params, closer in
        # spirit to CIBHash's 30K-param flat projection.
        siglip2_arg.add_argument('--adapter_type', dest='adapter_type',
            type=str, default='mlp', choices=['mlp', 'linear'],
            help="visual/text adapter architecture (default mlp).")
        # v59: text-only bottleneck. When set, overrides adapter_hidden_dim
        # ONLY for the text adapter (visual adapter keeps its own setting).
        # Use to compress text features through a low-rank bottleneck
        # (e.g. 768 -> 64 -> 768) while leaving visual capacity intact.
        siglip2_arg.add_argument('--text_adapter_hidden_dim',
            dest='text_adapter_hidden_dim',
            type=int, default=None,
            help='Override hidden_dim for text_adapter only. None defaults '
                 'to --adapter_hidden_dim (shared). Set to a small value '
                 '(e.g. 64, 128) to bottleneck text features asymmetrically.')
        siglip2_arg.add_argument('--adapter_hidden_dim', dest='adapter_hidden_dim',
            type=int, default=None,
            help="MLP hidden dim (only used when adapter_type=mlp). "
                 "Default = 2 * d_model.")
        siglip2_arg.add_argument('--adapter_dropout', dest='adapter_dropout',
            type=float, default=0.0,
            help="Dropout inside the MLP adapter (only used when adapter_type=mlp).")
        siglip2_arg.add_argument('--use_gumbel_softmax', dest='use_gumbel_softmax',
            action='store_true', default=True,
            help='Use Gumbel-Softmax for the codon hard path (default: True).')
        siglip2_arg.add_argument('--no_gumbel_softmax', dest='use_gumbel_softmax',
            action='store_false',
            help='Disable Gumbel-Softmax (fall back to deterministic STE).')
        siglip2_arg.add_argument('--gumbel_tau', dest='gumbel_tau',
            type=float, default=1.0)
        siglip2_arg.add_argument('--freeze_backbone', dest='freeze_backbone',
            action='store_true', default=True,
            help='Freeze SigLIP2 backbone (default: True).')
        siglip2_arg.add_argument('--no_freeze_backbone', dest='freeze_backbone',
            action='store_false')
        siglip2_arg.add_argument('--num_codebooks', dest='num_codebooks',
            type=int, default=6)
        # K=32 default chosen empirically on CIFAR10 setting1 (5K train).
        # K=64 collapsed in v1 (gradient mode); K=48 with EMA + revival also
        # regressed (-0.04 mAP) because 5K samples cannot populate 6×48 codes
        # densely enough. Larger K may pay off for bigger datasets (e.g.
        # ImageNet100 train=13K, MSCOCO=10K) -- override per dataset if needed.
        siglip2_arg.add_argument('--codebook_size', dest='codebook_size',
            type=int, default=32)
        # ---------- C_global slot source -------------------------------
        # `mean_pool`     -- (default) mean of visual_tokens (post-adapter).
        # `siglip2_global` -- SigLIP2's MAP-pooled + projection-head output
        #                     (`get_image_features(...)`), routed through a
        #                     dedicated `global_adapter` linear into d_model.
        # Empirical: on CIFAR10 setting1, `siglip2_global` regressed mAP by
        # 0.09 (v6 ablation) -- visual_global is too class-compressed to
        # preserve intra-class variation needed for fine-grained retrieval.
        # Worth re-trying on natural-resolution datasets where MAP-pooling
        # is less noisy.
        siglip2_arg.add_argument('--c_global_source', dest='c_global_source',
            type=str, default='mean_pool',
            choices=['mean_pool', 'siglip2_global'],
            help='How the C_global codebook slot is fed (default mean_pool).')
        # Per-slot text adapter (Option A from the text-collapse diagnostic).
        # Replaces the shared TextAdapter MLP with an nn.ModuleList of 6
        # independent MLPs (one per slot) so the near-identical SigLIP2
        # text-encoder outputs (measured pairwise cos sim ~0.88) can be
        # pushed into distinct slot-specific subspaces.
        siglip2_arg.add_argument('--per_slot_text_adapter', dest='per_slot_text_adapter',
            action='store_true', default=False,
            help='Use 6 independent text adapters (one per codebook slot) '
                 'instead of a single shared MLP. Targets the diagnosed '
                 'SigLIP2 text-encoder cross-slot collapse.')
        siglip2_arg.add_argument('--use_text_token_attention', dest='use_text_token_attention',
            action='store_true', default=False,
            help='Replace the static pooled text embedding per slot with a '
                 'visual-attention-pooled representation over slot-specific '
                 'text TOKENS. Requires the cache to have been built with '
                 '--save_text_tokens. Option B for fixing the SigLIP2 text '
                 'cross-slot collapse (~cos 0.88 -> cos ?).')
        siglip2_arg.add_argument('--disable_global_gate', dest='disable_global_gate',
            action='store_true', default=False,
            help='Skip the C_0 -> C_1..5 gated addition before the codon '
                 'heads. Each local codon head sees its own codeword in '
                 'isolation (v23b ablation).')
        # v32 text-injection ablation: at train time only, add per-part
        # text token to the routed visual token before codebook lookup.
        # The codeword embeddings absorb text-semantic structure during
        # training, but at inference we keep the forward pass text-free.
        siglip2_arg.add_argument('--text_inject_train_only', dest='text_inject_train_only',
            type=str, default='none', choices=['none', 'add'],
            help="Train-only text injection mode. 'add' = "
                 "z' = z_v + alpha * t (combined before quantizer; "
                 "skipped at eval). 'none' = legacy v29 behaviour.")
        siglip2_arg.add_argument('--text_inject_alpha', dest='text_inject_alpha',
            type=float, default=0.2,
            help="Mixing weight for the train-only text injection. "
                 "Small values (0.1-0.3) keep the train-test shift small.")
        siglip2_arg.add_argument('--text_inject_detach', dest='text_inject_detach',
            action='store_true', default=True,
            help="Detach the text features so the gradient only updates "
                 "the codebook / encoder, not the text adapter (v32 "
                 "variant b). Pass --no_text_inject_detach to override.")
        siglip2_arg.add_argument('--no_text_inject_detach', dest='text_inject_detach',
            action='store_false')
        siglip2_arg.add_argument('--text_attn_num_heads', type=int, default=4,
            help='Number of heads in the visual-cross-attention text pooling '
                 'block (use_text_token_attention).')
        # ---------- router type ----------------------------------------
        # 'sinkhorn'  : balanced OT (legacy). Forces every part to receive
        #               ~equal patch mass even on images that lack a part.
        # 'attention' : cross-attention (text part = query, visual = K/V).
        #               Each part softmax-attends over patches. No marginal
        #               balance -> per-part heatmap focuses on relevant region.
        siglip2_arg.add_argument('--router_type', dest='router_type',
            type=str, default='sinkhorn',
            choices=['sinkhorn', 'attention'],
            help='Router for the 5 local parts (default sinkhorn).')
        siglip2_arg.add_argument('--attention_router_temperature',
            dest='attention_router_temperature', type=float, default=0.1,
            help='Softmax temperature for attention router (smaller -> sharper).')
        # v33a: Sinkhorn epsilon annealing.
        siglip2_arg.add_argument('--sinkhorn_epsilon_init',
            dest='sinkhorn_epsilon_init', type=float, default=None,
            help='If set together with --sinkhorn_epsilon_final, the '
                 'Sinkhorn router uses a cosine-annealed epsilon from '
                 '*_init (epoch 0) to *_final (last epoch). Smaller '
                 'epsilon -> sharper (harder) routing. v33a ablation.')
        siglip2_arg.add_argument('--sinkhorn_epsilon_final',
            dest='sinkhorn_epsilon_final', type=float, default=None,
            help='Companion to --sinkhorn_epsilon_init.')
        # v33b: top-k routing mask per patch.
        siglip2_arg.add_argument('--routing_topk',
            dest='routing_topk', type=int, default=None,
            help='If set, keep only the top-k largest routing weights per '
                 'patch (along the M=5 part axis) and renormalize so each '
                 'patch row keeps its original marginal. k=1 -> hard '
                 'argmax (Sinkhorn balance broken); k>=M -> no-op. '
                 'v33b ablation.')
        # v46: adaptive-k routing via cumulative-mass threshold (top-p /
        # nucleus). For each patch, keep the smallest set of parts whose
        # sorted probabilities cumulatively reach `topp`. Top-1 always
        # preserved (every patch must route to at least one part). Lets
        # clearly-localized patches concentrate on 1-2 parts (forcing
        # codebook specialization) while ambiguous patches keep more spread.
        # Mutually compatible with --routing_topk (top-k applied first).
        siglip2_arg.add_argument('--routing_topp',
            dest='routing_topp', type=float, default=None,
            help='Top-p (cumulative-mass) routing threshold per patch. '
                 'None disables. Try 0.5-0.9. Works best when Sinkhorn '
                 'output is sharp (low epsilon or epsilon-annealed); '
                 'flat softmax over near-uniform text centroids makes '
                 'top-p effectively keep all M parts.')
        # v55: Unbalanced OT (Chizat et al. NeurIPS 2018). KL-relaxed
        # marginals let some patches have row sum < 1/N (i.e. patches that
        # are uninformative — background, blur — can be partially "rejected"
        # from routing rather than forced into a part). lambda → ∞ recovers
        # balanced Sinkhorn; lambda → 0 leaves marginal completely free.
        siglip2_arg.add_argument('--sinkhorn_lambda_a',
            dest='sinkhorn_lambda_a', type=float, default=None,
            help='UOT KL penalty on visual marginal. None = balanced (hard '
                 'constraint). Finite value (e.g. 0.5-5.0) allows some patches '
                 'to have row sum < a[n] = 1/N — effectively rejecting '
                 'uninformative tokens from semantic routing.')
        siglip2_arg.add_argument('--sinkhorn_lambda_b',
            dest='sinkhorn_lambda_b', type=float, default=None,
            help='UOT KL penalty on part marginal. None = balanced. Finite '
                 'allows part loads to be unbalanced (e.g. images with no '
                 'C_color_texture content can route less mass to that part).')
        # v56: Null/background centroid. Adds one extra learnable part
        # (M+1=6 locals) whose codebook column is discarded after routing.
        # Patches that match none of the 5 semantic part centroids well
        # route to this null centroid; their mass is then masked out before
        # codebook usage. Practical alternative to UOT — no Sinkhorn change.
        siglip2_arg.add_argument('--use_null_centroid',
            dest='use_null_centroid', action='store_true', default=False,
            help='Add a learnable null/background centroid to the Sinkhorn '
                 'router (alongside the 5 text part centroids). After '
                 'routing, the null column is sliced out, so patches '
                 'preferring the null get effectively rejected from codebook '
                 'updates. v56 ablation.')
        # ---------- VQ codebook update mode -----------------------------
        # `gradient` (default, legacy) -- codebook is an nn.Parameter,
        #   updated by the VQ loss MSE term. Prone to dead-code collapse.
        # `ema` -- codebook is a buffer, updated in-place via VQ-VAE EMA
        #   (DALL-E / VQ-VAE-2 style). Tends to keep more codewords alive.
        siglip2_arg.add_argument('--codebook_update', dest='codebook_update',
            type=str, default='ema', choices=['gradient', 'ema'],
            help="how the per-part codebook tensors get updated each step")
        siglip2_arg.add_argument('--codebook_ema_decay', dest='codebook_ema_decay',
            type=float, default=0.99,
            help='EMA decay for codebook update (only used when codebook_update=ema).')
        siglip2_arg.add_argument('--codebook_ema_eps', dest='codebook_ema_eps',
            type=float, default=1e-5,
            help='Laplace-smoothed denominator for EMA codebook normalization.')
        siglip2_arg.add_argument('--codebook_revive', dest='codebook_revive',
            action='store_true', default=True,
            help='Replace EMA-dead codewords with random batch z samples '
                 'every `--codebook_revive_every` steps. Counters code collapse.')
        siglip2_arg.add_argument('--no_codebook_revive', dest='codebook_revive',
            action='store_false')
        siglip2_arg.add_argument('--codebook_revive_threshold', dest='codebook_revive_threshold',
            type=float, default=0.01,
            help='A codeword is "dead" when its EMA cluster_size falls below '
                 '`threshold * max_cluster_size_in_codebook`.')
        siglip2_arg.add_argument('--codebook_revive_every', dest='codebook_revive_every',
            type=int, default=50,
            help='Run dead-code rejuvenation every N training forwards (steps).')
        # Option α: Codeword Repulsion. After each EMA update, push each
        # codeword in a codebook away from its closest neighbours
        # (Gaussian-weighted, auto-sigma per codebook). Disabled by default.
        siglip2_arg.add_argument('--codebook_repel_strength', dest='codebook_repel_strength',
            type=float, default=0.0,
            help='Option α: per-step repulsion step size as a multiplier on '
                 'the unit repulsion direction. 0 (default) disables.')
        siglip2_arg.add_argument('--codebook_repel_sigma_factor', dest='codebook_repel_sigma_factor',
            type=float, default=0.5,
            help='Option α: Gaussian width sigma = sigma_factor * median '
                 'pairwise distance per codebook. Smaller = sharper '
                 '(only very close pairs repel).')
        siglip2_arg.add_argument('--codebook_repel_every', dest='codebook_repel_every',
            type=int, default=1,
            help='Option α: apply repulsion every N EMA update steps.')
        # ---------- v41 (5-G): text-supervised codebook initialization ---
        # Before training starts, replace the random Gaussian codebook init
        # with K vectors derived from train-set `cached_text_part_raw`
        # (SigLIP2 text-encoder pooled embeddings per part). The text path
        # is KEPT active during training as well (still routes Sinkhorn
        # centroids each forward) -- this is purely additive supervision
        # for the codebook's starting point. See docs/ANALYSIS_2026-05-19.md
        # section 5-G.
        siglip2_arg.add_argument('--text_init_codebook',
            dest='text_init_codebook',
            type=str, default='none', choices=['none', 'mean', 'kmeans'],
            help="How to initialize codebooks from text part embeddings. "
                 "'none' (default) = random Gaussian, legacy behaviour. "
                 "'mean' = K random samples per codebook from train-set "
                 "text_part_raw (fast, ~1s). 'kmeans' = K-means cluster "
                 "centers per codebook (slower, ~30s for K=128 N=4096).")
        siglip2_arg.add_argument('--text_init_subset',
            dest='text_init_subset', type=int, default=4096,
            help='Number of train-set text vectors to gather for '
                 'codebook init (per codebook). Capped at train set size. '
                 'Default 4096 is enough for stable K-means with K<=128.')
        siglip2_arg.add_argument('--text_init_seed',
            dest='text_init_seed', type=int, default=42,
            help='Random seed for codebook init sampling / K-means n_init.')
        # ---------- Gumbel tau annealing --------------------------------
        # When `gumbel_tau_init` is set we override the static `gumbel_tau`
        # with a cosine schedule from init -> final across `args.epoch`.
        # v62 (Option A): residual-conditioned codon head. When >0, each codon
        # head receives both the quantized codeword AND a residual signal
        # (z - q) scaled by gamma. Lets two images sharing the same codeword
        # index produce different codons -> higher unique-code ratio without
        # breaking compositional structure (codebook indices unchanged).
        siglip2_arg.add_argument('--codon_residual_gamma',
            dest='codon_residual_gamma', type=float, default=0.0,
            help='Weight gamma in [0, 1] for residual-conditioned codon head. '
                 '0 disables (CodonHead works as before). Try 0.1-0.5. Larger '
                 'gamma -> more image-specific codon variation, but risks '
                 'breaking compositional interpretation if too large.')
        # v65: expand the per-position 4-class fc inside CodonHead into a
        # small MLP (Linear(chunk, H) -> GELU -> Linear(H, 4)). chunk = D/3 =
        # 256 in the default setting; head_hidden_dim=0 (default) keeps the
        # legacy single Linear(256, 4).
        siglip2_arg.add_argument('--codon_head_hidden_dim',
            dest='codon_head_hidden_dim', type=int, default=0,
            help='Hidden dim for the per-position MLP in CodonHead. 0 (default) '
                 'preserves the legacy single Linear(chunk, 4) path.')
        # v66: Per-Codon Text-Anchored Classifier. Replaces fc with cosine
        # similarity to a learnable [3, 4, chunk] prototype tensor and adds an
        # auxiliary CE loss between the visual codon logits and a text-derived
        # target class (target = argmax(cos(text_chunk, prototype))). Provides
        # ongoing text supervision directly on the codon decoding stage.
        # Mutually exclusive with --codon_head_hidden_dim (when this flag is
        # set, the MLP path is bypassed in favour of the prototype layer).
        siglip2_arg.add_argument('--codon_text_anchor',
            dest='codon_text_anchor', action='store_true', default=False,
            help='Enable per-codon text-anchored prototype classifier (v66).')
        siglip2_arg.add_argument('--codon_anchor_temperature',
            dest='codon_anchor_temperature', type=float, default=0.1,
            help='Temperature for cos-sim prototype logits in CodonHead.')
        siglip2_arg.add_argument('--lambda_codon_text_anchor',
            dest='lambda_codon_text_anchor', type=float, default=0.1,
            help='Loss weight for the v66 text-anchored CE aux loss '
                 '(applied only when --codon_text_anchor is set).')
        siglip2_arg.add_argument('--gumbel_tau_init',  dest='gumbel_tau_init',
            type=float, default=2.0)
        siglip2_arg.add_argument('--gumbel_tau_final', dest='gumbel_tau_final',
            type=float, default=0.3)
        siglip2_arg.add_argument('--gumbel_tau_anneal', dest='gumbel_tau_anneal',
            action='store_true', default=True,
            help='Cosine-anneal codon-head tau from init->final across epochs.')
        siglip2_arg.add_argument('--no_gumbel_tau_anneal',
            dest='gumbel_tau_anneal', action='store_false')

        # ---------- LR scheduler ----------------------------------------
        # The legacy default `StepLR(step_size=10, gamma=1e-4)` killed lr
        # to ~0 after epoch 10, freezing training. Cosine is the safe new
        # default; 'step' / 'none' kept for explicit selection.
        sched_arg = parser.add_argument_group("learning rate scheduler")
        sched_arg.add_argument('--lr_scheduler', dest='lr_scheduler',
            type=str, default='cosine', choices=['cosine', 'step', 'none'])
        sched_arg.add_argument('--lr_step_size', dest='lr_step_size',
            type=int, default=20)
        sched_arg.add_argument('--lr_gamma',     dest='lr_gamma',
            type=float, default=0.5)
        sched_arg.add_argument('--lr_eta_min',   dest='lr_eta_min',
            type=float, default=1e-5)

        # ---------- loss weights (DNACodonHashLoss) ---------------------
        loss_arg = parser.add_argument_group("dna hashing loss weights")
        # Active loss weights after the 2026-05-13 cleanup. R1–R4 alignment
        # family, v15 codebook orthogonality, and the deprecated `loss_global`
        # were removed -- see `docs/PROJECT_LOG.md`.
        loss_arg.add_argument('--lambda_hash',         type=float, default=1.0)
        loss_arg.add_argument('--lambda_hash_hard',    type=float, default=0.5)
        loss_arg.add_argument('--lambda_hash_type',    type=str, default='mse',
            choices=['mse', 'hashnet'],
            help="Form of loss_hash. 'mse' = legacy Jaccard-MSE on continuous "
                 "code (default). 'hashnet' = HashNet-style class-weighted "
                 "logistic likelihood on a centred-scaled DNA-sim score "
                 "(less aggressive collapse, follows HashNet ICCV'17).")
        loss_arg.add_argument('--hashnet_alpha',       type=float, default=1.0,
            help="Continuation parameter for hashnet loss_hash. score = "
                 "alpha * (2*sim_dna - 1). Larger -> sharper sign saturation. "
                 "Try 1.0 (default), 2.0, 5.0.")
        loss_arg.add_argument('--hashnet_use_jaccard', action='store_true', default=False,
            help="Use FRACTIONAL Jaccard S in the HashNet logistic instead of "
                 "binary any-shared S. With binary S most multi-label pairs "
                 "are positive -> single positive cluster collapse (90% dup "
                 "rate). Jaccard S preserves per-label-combo granularity in "
                 "the target probability, hopefully reducing collision.")
        loss_arg.add_argument('--hashnet_S_cap',       type=float, default=1.0,
            help="Cap on the hashnet logistic target similarity S_target. "
                 "Default 1.0 = no cap (full-overlap pairs aim for sigmoid=1, "
                 "v18 behaviour). Setting <1.0 leaves a residual degree of "
                 "freedom inside each same-powerset cluster -> distinct codes "
                 "per cluster -> higher unique_code_ratio. Try 0.95 or 0.90.")
        # ---------- Unsupervised hash target (v27a) ---------------------
        # Replaces the label-derived Jaccard pairwise similarity S used by
        # loss_hash / loss_hash_hard with a self-supervised similarity
        # computed from the frozen SigLIP2 visual_global embedding. Makes
        # the retrieval objective truly unsupervised (cf. CIBHash, CIMON,
        # SPQ, MLS3RDUH). 'jaccard' (default) preserves legacy supervised
        # behaviour. 'siglip_cos' rescales cosine sim ∈ [-1, 1] → [0, 1].
        loss_arg.add_argument('--hash_target_mode',    type=str, default='jaccard',
            choices=['jaccard', 'siglip_cos', 'siglip_cos_topk'],
            help="How to build the pairwise similarity target S for "
                 "loss_hash/loss_hash_hard. 'jaccard' uses multi_hot labels "
                 "(supervised). 'siglip_cos' uses frozen SigLIP2 visual_global "
                 "cosine similarity rescaled to [0,1] (unsupervised, but "
                 "v27a showed it collapses because the cos distribution is "
                 "narrow around 0.7-0.9). 'siglip_cos_topk' binarizes the "
                 "cosine sim at the (1 - siglip_cos_pos_rate) quantile "
                 "within the batch -> top fraction get S=1, rest S=0 "
                 "(CIBHash/CIMON-style pseudo-positives, v27b).")
        loss_arg.add_argument('--siglip_cos_pos_rate',  type=float, default=0.2,
            help="When hash_target_mode='siglip_cos_topk', fraction of "
                 "off-diagonal pairs marked positive (S=1). Default 0.2 "
                 "(top-20% feature-similar pairs are pseudo-positives).")
        # ---------- v28: reconstruction decoder -------------------------
        # Adds a decoder head that maps the 6 selected codewords back to
        # either the original RGB image (v28a) or the cached SigLIP2
        # visual_global (v28b). Provides a strong dense per-sample
        # unsupervised signal that fixes v27b's batch-cosine pseudo-positive
        # weakness. Off by default.
        loss_arg.add_argument('--use_decoder',          action='store_true', default=False,
            help="Attach a decoder head (PixelDecoder or FeatureDecoder) "
                 "that reconstructs from the 6 selected codewords. Adds a "
                 "loss_recon term to the aggregator (weight --lambda_recon).")
        loss_arg.add_argument('--decoder_target',       type=str, default='siglip_feat',
            choices=['pixel', 'siglip_feat'],
            help="What the decoder reconstructs. 'siglip_feat' targets the "
                 "frozen SigLIP2 visual_global (v28b, fast MLP). 'pixel' "
                 "targets the 224x224 RGB image (v28a, ConvTranspose stack).")
        loss_arg.add_argument('--lambda_recon',         type=float, default=1.0,
            help="Weight on loss_recon. Default 1.0 puts it on par with "
                 "loss_hash. Set 0 to disable even when --use_decoder is on.")
        # ---------- v29: paired-aug NtXent on DNA code ------------------
        # CIBHash-style instance-discrimination contrastive loss applied
        # directly on the DNA code (forward = STE one-hot per position).
        # Requires the dataloader to return paired augmented views per
        # image (img_tr1, img_tr2) and the trainer to forward the model
        # twice. Goal: replace v27b's noisy batch top-k pseudo-positive
        # with the sharper "same image, different view" positive signal
        # that drove CIBHash to mAP 0.6543 vs our 0.5639 on Flickr25k.
        loss_arg.add_argument('--use_paired_aug_ntxent', action='store_true', default=False,
            help="Enable paired-augmentation NtXent loss on DNA codes. "
                 "Requires force_pixel_decode + paired_aug in dataloader; "
                 "the trainer runs the model twice (once per view) and "
                 "the criterion contrasts the two DNA codes per image.")
        loss_arg.add_argument('--lambda_ntxent',         type=float, default=1.0,
            help="Weight on loss_ntxent_dna. With --use_paired_aug_ntxent "
                 "set to 1.0 you typically also want --lambda_hash 0 and "
                 "--lambda_hash_hard 0 (NtXent replaces the pairwise "
                 "similarity-target retrieval signal entirely).")
        loss_arg.add_argument('--ntxent_temperature',    type=float, default=0.3,
            help="Temperature for the NtXent softmax. CIBHash uses 0.3.")
        loss_arg.add_argument('--ntxent_mode',           type=str, default='global',
            choices=['global', 'per_codebook'],
            help="NtXent target granularity (v31 ablation). "
                 "'global' (default, v29) contrasts whole-image DNA "
                 "codes [B, 18, 4]. 'per_codebook' (v31b) computes 6 "
                 "separate NtXents on each codebook's 3-codon group "
                 "[B, 3, 4], summed -> each codebook is forced to "
                 "independently discriminate samples (compositional "
                 "independence test).")
        # ---------- v42 (Q2): dynamic temperature from text similarity ---
        # Inspired by Wang et al. CVPR 2021 "Understanding the Behaviour of
        # Contrastive Loss": small tau is hardness-aware (sharp push on
        # nearest negative) but breaks semantic neighborhood; large tau is
        # tolerant but loses uniformity. v42 modulates tau per (anchor i,
        # negative j) pair using their text-caption cosine similarity:
        #     tau_ij = base_tau * (1 + alpha * cos(text_i, text_j))
        # so semantically-similar samples get a higher tau (soft push) and
        # semantically-distant ones get a lower tau (hard push). Only
        # active for ntxent_mode='per_codebook' (each codebook m uses its
        # own text slot text_part_raw[:, m, :]).
        loss_arg.add_argument('--ntxent_dynamic_tau',
            action='store_true', default=False,
            help="Enable per-pair dynamic NtXent temperature modulated by "
                 "text-caption cosine similarity (v42 / Q2 proposal). "
                 "Requires --ntxent_mode per_codebook and a non-None text "
                 "path. Off by default (legacy single-tau behaviour).")
        # v60: skip dynamic-tau for the C_0 (global) codebook only. Uses
        # static base_tau for m=0 even when dynamic_tau is enabled.
        # Motivation: C_0 receives whole-image mean-pooled visual; the
        # C_global Qwen caption is a topic-level summary that may not
        # be the right signal for per-pair tau modulation.
        loss_arg.add_argument('--ntxent_dynamic_tau_skip_global',
            action='store_true', default=False,
            help='If set, the m=0 (C_global) codebook uses static base_tau '
                 'in per-codebook NtXent (v60 ablation), even when '
                 '--ntxent_dynamic_tau is on. C_1..5 still get dynamic '
                 'modulation as usual.')
        # v61: use mean of the 5 local text_part_raw vectors as the
        # similarity source for m=0 (C_global) dynamic-tau, instead of
        # the C_global caption embedding. Motivation: aggregated local-
        # part similarity may capture a richer image-level signal than
        # a single topic-summary embedding.
        loss_arg.add_argument('--ntxent_global_use_local_mean',
            action='store_true', default=False,
            help='If set, when computing dynamic-tau for m=0 use the mean '
                 'of text_part_raw[:, 1:, :] (5 local parts) instead of '
                 'text_part_raw[:, 0, :] (C_global caption). v61 ablation.')
        loss_arg.add_argument('--ntxent_dynamic_tau_alpha',
            type=float, default=0.5,
            help="Modulation strength alpha in "
                 "tau_ij = base_tau * (1 + alpha * cos(text_i, text_j)). "
                 "alpha in [0, 1). alpha=0 -> static base_tau (off). "
                 "alpha=0.5 -> tau in [0.5*base, 1.5*base]. Larger alpha "
                 "is more aggressive but caps at alpha<1 to avoid tau<=0.")
        # ---------- v44 (B1): cross-slot text orthogonality reg ---------
        # Pushes the per-slot text_part_tokens (post adapter) apart so the
        # 6 codebook-routing centroids are not collapsed onto the same
        # direction. Direct counterpoint to the diagnosed SigLIP2 cross-
        # slot cos sim ~0.97 (V1) / 0.97 (V3) -- prompt engineering alone
        # only chips ~0.01 off; this regularizer pushes the *adapted*
        # text features (which we control) much further apart.
        # Loss form (per sample b):
        #     T_n     = normalize(text_part_tokens, dim=-1)  # [M, D]
        #     G_b     = T_n @ T_n.T                          # [M, M]
        #     L_ortho = ((G_b - I)**2).sum() / (M * (M - 1))
        # then averaged over batch. Active only when text_part_tokens
        # is not None (text path on).
        loss_arg.add_argument('--lambda_ortho_text',
            type=float, default=0.0,
            help="Weight on cross-slot text orthogonality loss "
                 "(v44 / B1 ablation). 0 disables. Try 0.02-0.1 -- "
                 "larger values risk dominating loss_ntxent (lambda=1.0).")
        loss_arg.add_argument('--lambda_wasserstein',  type=float, default=0.0,
            help="Weight for the entropic-OT Wasserstein loss (per-sample "
                 "<pi, cost> from the Sinkhorn router, restored for v24a). "
                 "v11 found 0.05 to be the Flickr sweet spot.")
        loss_arg.add_argument('--eta_base_balance',    type=float, default=1.0,
            help="Coefficient on the per-position base-balance term inside "
                 "loss_dna (= loss_entropy + eta * loss_base_balance). v18 "
                 "shows mean per-position entropy 0.745 (severe G/T bias). "
                 "Larger eta enforces uniform A/C/G/T usage more strongly. "
                 "Try 1.0 (default), 5.0, 10.0.")
        loss_arg.add_argument('--lambda_vq',           type=float, default=0.25)
        loss_arg.add_argument('--lambda_quant',        type=float, default=0.05)
        loss_arg.add_argument('--lambda_anchor',       type=float, default=0.05)
        # Bumped 0.01 -> 0.05: the DNA entropy + base-balance signal was
        # essentially zero at lambda=0.01 (loss * weight ~ 0.007); the
        # codon head needs more pressure to spread base usage.
        loss_arg.add_argument('--lambda_dna',          type=float, default=0.05)
        loss_arg.add_argument('--lambda_bu',           type=float, default=0.02)
        loss_arg.add_argument('--bu_warmup_epochs',    type=int,   default=0)
        loss_arg.add_argument('--anchor_ema_momentum', type=float, default=0.99)

        # ---------- retrieval / extraction parameters -------------------
        retr_arg = parser.add_argument_group("retrieval evaluation parameters")
        retr_arg.add_argument('--eval_every', type=int, default=5,
            help='Run mid-training retrieval eval every N epochs (0=disable).')
        retr_arg.add_argument('--extract_batch_size', type=int, default=256)
        retr_arg.add_argument('--dna_distance_mode', type=str, default='base',
            choices=['base', 'bit2'])

        # ---------- visualization ---------------------------------------
        viz_arg = parser.add_argument_group("training-end visualization")
        viz_arg.add_argument('--visualize', dest='visualize',
            action='store_true', default=True,
            help='Save routing heatmaps + codebook t-SNE PNGs at end of run.')
        viz_arg.add_argument('--no_visualize', dest='visualize',
            action='store_false')
        viz_arg.add_argument('--viz_routing_samples', dest='viz_routing_samples',
            type=int, default=12)
        viz_arg.add_argument('--viz_tsne_samples',    dest='viz_tsne_samples',
            type=int, default=2000)

        config = parser.parse_args()

        return config
    
    def load_args(self):
        
        config = self.get_config()
        
        torch.cuda.set_device(config.num_devices)
        
        args_info_path = os.path.join(config.config_path, "model_state", "config.pt")
        args_info_yaml_path = os.path.join(config.config_path, 'lightning_logs', 'csv', 'hparams.yaml')
        
        if os.path.exists(args_info_path):
            args = torch.load(args_info_path, map_location='cpu')
            
            for kargs, vargs in args.items():
                setattr(self, f"{kargs}", vargs)
                
        elif os.path.exists(args_info_yaml_path):
            with open(args_info_yaml_path, "r") as f:
                args = yaml.safe_load(f)
                
            for k, v in args.items():
                setattr(self, k, v)
                            
        self.save_result_path = config.config_path
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        self.device = torch.device(f'cuda:{config.num_devices}' if torch.cuda.is_available() else 'cpu')
        self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
        
    def set_args(self, args=None):
        
        if args is None: args = self.get_config()
        
        # tag = '+'.join([self.date, args.batch_size, args.epoch])
        tag = str(self.date)
        if args.tag is not None:
            for t in args.tag:
                tag = "+".join([tag, t])
                
        tag_args = ["dataset", "batch_size", "n_ref", ]
        
        for ktag in tag_args:
            vtag = getattr(args, ktag)
            tag = "+".join([tag, str(ktag), str(vtag)])
        
        self.save_result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.log_dir, self.date, tag)

        # if os.path.exists(self.save_result_path +'/model_state'):
        #     i=0
        #     while True:
        #         i+=1
        #         self.save_result_path = self.save_result_path.split('+') + f"+v{i}"
        #         if not os.path.exists(self.save_result_path):
        #             break
            
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        os.makedirs(self.save_log_path, exist_ok=True)
        self.device = torch.device(f'cuda:{args.num_devices}' if torch.cuda.is_available() else 'cpu')
        
        for karg, varg in args._get_kwargs():
            setattr(self, f"{karg}", varg)
            
        self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
        os.makedirs(self.save_model_state_path, exist_ok=True) 
            
        # self.dataset_dir = '/home/yschoi/ConstrainedDeepHashing/default_dataset'
        self.setting = 'setting' + str(self.setting)
        self.num_classes = NUM_CLASSES.get(self.dataset)[self.setting]
        

        # delattr(self, "num_devices")
        delattr(self, "log_dir")
        
    def save_arg(self):
        
        """call only in training phase
        """
        
        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        torch.save(args, self.save_model_state_path + "/config.pt")

    def print_info(self):

        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        print(f"{' PARAMETERS INFO ':=^100s}")

        with open(self.save_log_path + 'args.txt', 'w', encoding='utf-8') as args_txt:
            for k, v in args.items():
                arg = f"{k:-<30s}{str(v):->70s}"
                print(arg)
                args_txt.write(str(arg) + '\n')  
            print('\n')