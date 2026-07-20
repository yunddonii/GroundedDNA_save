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
        test_arg.add_argument('--post_eval_compositional',
            dest='post_eval_compositional', action=argparse.BooleanOptionalAction,
            default=True,
            help='After --eval, automatically run scripts/pairwise_nmi.py, '
                 'scripts/codebook_drop_ablation_fast.py, and '
                 'compositional_eval.py (B0/B1/B2). Pass '
                 '--no-post_eval_compositional to disable.')

        # ---------- siglip2 / dna hashing parameters --------------------
        # All flags below are additive — every model/loss module also has
        # `getattr(args, ..., default)` fallback, so omitting them keeps the
        # legacy behavior unchanged. Specifying them here lets us drive runs
        # via CLI without editing source.
        siglip2_arg = parser.add_argument_group("siglip2 / dna hashing parameters")
        # ---------- backbone family selector ---------------------------------
        # 'siglip2' (default, legacy) -- google/siglip2-base-patch16-224 dual encoder.
        # 'clip'                      -- openai/clip-vit-base-patch16. Both image
        #                                 and text encoder come from CLIP and are
        #                                 frozen (image encoder = CLIP vision tower,
        #                                 text encoder = CLIP text tower).
        # Cache schema is identical across backbones (image_ids.json,
        # visual_tokens / visual_global / text_part / has_text / meta), so
        # downstream `_SigLIP2FeatureCache` works for either family. Choose the
        # cache dir consistent with this flag via --siglip2_feature_cache_dir.
        siglip2_arg.add_argument('--backbone_type', dest='backbone_type',
            type=str, default='siglip2', choices=['siglip2', 'clip'],
            help="Pretrained backbone family (default: %(default)s). 'siglip2' "
                 "uses --siglip2_backbone; 'clip' uses --clip_backbone. Both "
                 "are kept frozen by --freeze_backbone (default True).")
        siglip2_arg.add_argument('--siglip2_backbone', dest='siglip2_backbone',
            default='google/siglip2-base-patch16-224',
            help='Hugging Face SigLIP2 backbone name (default: %(default)s). '
                 'Used only when --backbone_type=siglip2.')
        siglip2_arg.add_argument('--clip_backbone', dest='clip_backbone',
            default='openai/clip-vit-base-patch16',
            help='OpenAI CLIP backbone name (default: %(default)s). Used only '
                 'when --backbone_type=clip. Image and text encoders both come '
                 'from this single CLIPModel checkpoint.')
        siglip2_arg.add_argument('--qwen_text_cache_path', dest='qwen_text_cache_path',
            default=None,
            help='JSONL cache produced by preprocess_qwen_codebook_texts.py.')
        siglip2_arg.add_argument('--siglip2_feature_cache_dir', dest='siglip2_feature_cache_dir',
            default=None,
            help='Directory written by extract_siglip2_features.py (memmaped '
                 'fp16 visual+text feature tensors). When set, training and '
                 'extraction skip the SigLIP2 encoder pass and read these '
                 'features directly. Big speedup for frozen-backbone runs.')
        siglip2_arg.add_argument('--eval_cache_dir', dest='eval_cache_dir',
            default=None,
            help='Optional separate cache for FINAL extraction + evaluation. '
                 'When set, training uses --siglip2_feature_cache_dir but the '
                 'end-of-training extract_db + extract_query + evaluation use '
                 'this cache instead. Use case: train on FAIRrank multi-view '
                 'cache but report final results on whole-image cache.')
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
        # v122: codon length per codebook (L). The DNA hash has shape
        # [N, M*L] base indices in [0..3] = A/C/G/T = 2 bits each, so the
        # total hash length is M*L*2 bits (= 36 at default L=3, M=6).
        # The Sinkhorn codeword-codon bijection (--lambda_codeword_codon_sinkhorn)
        # requires K == 4^L per codebook; default L=3 forces K=64, while L=4
        # frees K up to 256 and removes the K=128 pigeonhole forced collision
        # diagnosed in the v118a → v120 family. d_model must be divisible by L.
        siglip2_arg.add_argument('--num_codons_per_codebook',
            dest='num_codons_per_codebook', type=int, default=3,
            choices=[3, 4],
            help='v122: number of codon positions per codebook (default 3 '
                 '→ 36-bit DNA, K up to 64 codons; 4 → 48-bit DNA, K up to '
                 '256 codons). Requires d_model %% L == 0.')
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
        siglip2_arg.add_argument('--global_gate_init_logit', dest='global_gate_init_logit',
            type=float, default=-3.0,
            help='Initial logit for the per-local-codebook C_0->C_m gate '
                 '(sigmoid). Default -3.0 -> initial gate ~0.047. Use '
                 '-4.595 for ~0.01 (v88c-style weak addition).')
        siglip2_arg.add_argument('--use_stop_grad_global', dest='use_stop_grad_global',
            action=argparse.BooleanOptionalAction, default=True,
            help='Detach C_0 codeword before the gated addition so the C_0 '
                 'pathway does not receive gradient from C_1..5 codon heads '
                 '(v88c default). Pass --no-use_stop_grad_global to allow '
                 'gradients through.')
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
            choices=['sinkhorn', 'attention', 'slot', 'cluster_attn', 'cross_attn'],
            help='Router for the 5 local parts (default sinkhorn). '
                 '"slot" = Slot Attention (Locatello et al. NeurIPS 2020). '
                 '"cluster_attn" = v141 DiVT-inspired soft Sinkhorn cluster '
                 '+ masked cross-attention. Operates on RAW patches; pair '
                 'with --visual_adapter_after_router to relocate the '
                 'visual_adapter to AFTER the router output (M=6 tokens).')
        # v141: cluster-attention router hyperparameters.
        siglip2_arg.add_argument('--cluster_attn_heads',
            dest='cluster_attn_heads', type=int, default=4,
            help='v141: number of attention heads in ClusterAttentionRouter.')
        siglip2_arg.add_argument('--cluster_attn_mlp_ratio',
            dest='cluster_attn_mlp_ratio', type=float, default=4.0,
            help='v141: MLP hidden ratio in ClusterAttentionRouter.')
        siglip2_arg.add_argument('--cluster_attn_sinkhorn_eps',
            dest='cluster_attn_sinkhorn_eps', type=float, default=0.1,
            help='v141: Sinkhorn eps for soft cluster assignment.')
        siglip2_arg.add_argument('--cluster_attn_sinkhorn_iters',
            dest='cluster_attn_sinkhorn_iters', type=int, default=3,
            help='v141: Sinkhorn iteration count for soft cluster.')
        siglip2_arg.add_argument('--cluster_attn_pool_temperature',
            dest='cluster_attn_pool_temperature', type=float, default=0.3,
            help='v141: temperature for attention-pooled centroid.')
        siglip2_arg.add_argument('--visual_adapter_after_router',
            dest='visual_adapter_after_router', action='store_true',
            default=False,
            help='v141: place visual_adapter AFTER router (operates on M=6 '
                 'tokens). Required for --router_type cluster_attn so the '
                 'adapter gradient stays connected through the centroid + '
                 'attention path back to the encoder.')
        # v146: text-as-query cross-attention router hyperparameters.
        siglip2_arg.add_argument('--cross_attn_heads',
            dest='cross_attn_heads', type=int, default=4,
            help='v146: number of attention heads in TextCrossAttentionRouter.')
        siglip2_arg.add_argument('--cross_attn_dropout',
            dest='cross_attn_dropout', type=float, default=0.1,
            help='v146: attention dropout probability.')
        siglip2_arg.add_argument('--cross_attn_temp_init',
            dest='cross_attn_temp_init', type=float, default=0.2,
            help='v146: initial softmax temperature (annealed to temp_final).')
        siglip2_arg.add_argument('--cross_attn_temp_final',
            dest='cross_attn_temp_final', type=float, default=0.07,
            help='v146: final softmax temperature after annealing.')
        siglip2_arg.add_argument('--cross_attn_warmup_epochs',
            dest='cross_attn_warmup_epochs', type=int, default=20,
            help='v146: warmup epochs for alpha blend AND temperature anneal.')
        siglip2_arg.add_argument('--cross_attn_near_identity_scale',
            dest='cross_attn_near_identity_scale', type=float, default=0.1,
            help='v146: std for near-identity init noise on W_Q/W_K/W_V.')
        siglip2_arg.add_argument('--cross_attn_alpha_final',
            dest='cross_attn_alpha_final', type=float, default=1.0,
            help='v146: final alpha (cross-attn weight) after warmup. 1.0 = '
                 'pure cross-attn; 0.7 = 70% cross-attn + 30% Sinkhorn baseline.')
        siglip2_arg.add_argument('--attention_router_temperature',
            dest='attention_router_temperature', type=float, default=0.1,
            help='Softmax temperature for attention router (smaller -> sharper).')
        # v107a-slot: slot attention iteration count (paper default 3).
        siglip2_arg.add_argument('--slot_attention_iters',
            dest='slot_attention_iters', type=int, default=3,
            help='v107a-slot: number of slot-attention iterations '
                 '(Locatello et al. paper uses 3). Only used when '
                 '--router_type slot.')
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
        # v80/v81: confidence-adaptive sparse routing. These are default-off
        # extensions of the existing top-p/top-k masks. They keep confident
        # patches sparse while preserving multi-part assignments for ambiguous
        # patches.
        siglip2_arg.add_argument('--routing_ambiguity_topk',
            dest='routing_ambiguity_topk', action='store_true', default=False,
            help='If set, confident patches keep top-1 routing while '
                 'ambiguous patches keep --routing_ambiguity_k parts.')
        siglip2_arg.add_argument('--routing_ambiguity_threshold',
            dest='routing_ambiguity_threshold', type=float, default=0.6,
            help='Confidence threshold max_m P[b,n,m] for ambiguity-aware '
                 'top-k routing. Patches above this keep top-1.')
        siglip2_arg.add_argument('--routing_ambiguity_k',
            dest='routing_ambiguity_k', type=int, default=2,
            help='Number of parts retained for ambiguous patches in '
                 '--routing_ambiguity_topk.')
        siglip2_arg.add_argument('--routing_adaptive_topp',
            dest='routing_adaptive_topp', action='store_true', default=False,
            help='If set, use patch-specific top-p threshold based on '
                 'routing confidence instead of a fixed scalar threshold.')
        siglip2_arg.add_argument('--routing_adaptive_topp_min',
            dest='routing_adaptive_topp_min', type=float, default=0.5,
            help='Minimum patch-specific top-p threshold for confident '
                 'patches when --routing_adaptive_topp is enabled.')
        siglip2_arg.add_argument('--routing_adaptive_topp_max',
            dest='routing_adaptive_topp_max', type=float, default=0.9,
            help='Maximum patch-specific top-p threshold for ambiguous '
                 'patches when --routing_adaptive_topp is enabled.')
        siglip2_arg.add_argument('--routing_adaptive_topp_entropy',
            dest='routing_adaptive_topp_entropy', action='store_true', default=False,
            help='If set with --routing_adaptive_topp, compute patch-specific '
                 'top-p threshold from normalized routing entropy instead '
                 'of max-probability confidence.')
        siglip2_arg.add_argument('--routing_specificity_marginal',
            dest='routing_specificity_marginal', action='store_true',
            default=False,
            help='Replace the uniform Sinkhorn visual marginal with detached '
                 'slot-specificity weights 1-H(softmax_slot(cos))/log(M). '
                 'Slot-common patches receive little transport mass without '
                 'a visual top-k threshold; default off.')
        siglip2_arg.add_argument('--routing_centered_consensus_mask',
            dest='routing_centered_consensus_mask', action='store_true',
            default=False,
            help='Mask a visual patch from local Sinkhorn routing only when '
                 'its cosine similarity exceeds each valid slot\'s per-image '
                 'patch mean. Remaining patches use the legacy uniform visual '
                 'marginal; default off.')
        siglip2_arg.add_argument('--routing_cls_verified_consensus_mask',
            dest='routing_cls_verified_consensus_mask', action='store_true',
            default=False,
            help='In frozen CLIP shared space, mask a visual patch from local '
                 'OT only when it is above the per-image mean similarity for '
                 'every local text slot but below the per-image mean cosine '
                 'to the CLIP global/CLS embedding. Default off.')
        # v184: inference routing centroid source. codebook_mean (default) uses
        # learned visual prototypes. text_prototype uses EMA of trainset
        # text_part_tokens as routing centroids. Zero cost increase, addresses
        # the diagnostic finding that codebook_mean routing has flat cost
        # matrix (0.01 max) vs text-anchored alignment sharp peaks (0.6).
        siglip2_arg.add_argument('--eval_routing_mode',
            dest='eval_routing_mode',
            choices=['codebook_mean', 'text_prototype'],
            default='codebook_mean',
            help='v184: inference routing centroid source. text_prototype '
                 'uses EMA text_part_tokens average as centroids.')
        siglip2_arg.add_argument('--text_prototype_ema_decay',
            dest='text_prototype_ema_decay', type=float, default=0.999,
            help='v184: EMA decay for text_prototype tracker during training.')
        siglip2_arg.add_argument('--routing_perplexity_topk',
            dest='routing_perplexity_topk', action='store_true', default=False,
            help='v84a: per-patch top-k routing where k = ceil(M^H_norm). '
                 'k is the perplexity of the patch routing distribution: '
                 'confident patches (H≈0) get k=1 (hard routing), uniform '
                 'patches (H=log M) get k=M (dense). Zero hyperparameters. '
                 'Mutually exclusive with --routing_adaptive_topp.')
        siglip2_arg.add_argument('--routing_codebook_choice',
            dest='routing_codebook_choice', action='store_true', default=False,
            help='v85: after patch-wise routing masks, let each codebook '
                 'retain only its highest-mass visual tokens under a '
                 'capacity budget, then restore per-patch row mass. This '
                 'is inspired by expert-choice routing and is default-off.')
        siglip2_arg.add_argument('--routing_codebook_choice_capacity',
            dest='routing_codebook_choice_capacity', type=float, default=1.5,
            help='Capacity factor for --routing_codebook_choice. Each '
                 'codebook keeps ceil(N / M * capacity) tokens per sample, '
                 'with top-1 row fallback if all codebooks for a patch are '
                 'filtered out.')
        siglip2_arg.add_argument('--routing_codebook_choice_beta',
            dest='routing_codebook_choice_beta', type=float, default=1.0,
            help='v124: blend strength for --routing_codebook_choice. '
                 '1.0 reproduces the hard v85 expert-choice filter; values '
                 'in (0,1) softly interpolate from the pre-choice routing '
                 'matrix to the filtered expert-choice matrix.')
        siglip2_arg.add_argument('--routing_codebook_choice_warmup_epochs',
            dest='routing_codebook_choice_warmup_epochs', type=int, default=0,
            help='v124: linearly warm up routing_codebook_choice_beta over '
                 'this many epochs. 0 disables warm-up.')
        siglip2_arg.add_argument('--routing_text_evidence_beta',
            dest='routing_text_evidence_beta', type=float, default=0.0,
            help='v176a: soft text-evidence routing prior. Positive values '
                 'lower Sinkhorn transport cost for visual patches whose '
                 'embedding matches the routed text/codebook centroid. '
                 '0.0 disables and preserves the legacy router.')
        siglip2_arg.add_argument('--routing_text_evidence_warmup_epochs',
            dest='routing_text_evidence_warmup_epochs', type=int, default=0,
            help='v176a: linearly warm up routing_text_evidence_beta over '
                 'this many epochs. 0 disables warm-up.')
        siglip2_arg.add_argument('--routing_text_evidence_keep_ratio',
            dest='routing_text_evidence_keep_ratio', type=float, default=1.0,
            help='v177: per-local-slot evidence-aware keep ratio. Values in '
                 '(0,1) identify the top visual tokens per routed local slot; '
                 'tokens outside this set receive routing_text_evidence_penalty. '
                 '1.0 disables the candidate penalty and recovers v176a.')
        siglip2_arg.add_argument('--routing_text_evidence_penalty',
            dest='routing_text_evidence_penalty', type=float, default=0.0,
            help='v177: soft cost penalty applied to visual tokens outside the '
                 'per-local-slot evidence keep set. 0.0 disables and preserves '
                 'the v176a behavior.')
        siglip2_arg.add_argument('--routing_token_ot_evidence',
            dest='routing_token_ot_evidence', action='store_true',
            help='v179a: add a token-level text-to-visual evidence prior to '
                 'Sinkhorn routing. Each local semantic part uses its cached '
                 'text-token embeddings to softly reward visual patches that '
                 'support the part description. Default-off.')
        siglip2_arg.add_argument('--routing_token_ot_beta',
            dest='routing_token_ot_beta', type=float, default=0.0,
            help='v179a: strength for --routing_token_ot_evidence. Positive '
                 'values lower Sinkhorn cost on patches supported by local '
                 'text tokens.')
        siglip2_arg.add_argument('--routing_token_ot_eps',
            dest='routing_token_ot_eps', type=float, default=0.05,
            help='v179a: temperature for token-to-visual evidence transport. '
                 'Smaller values make each text token focus on fewer patches.')
        siglip2_arg.add_argument('--routing_token_ot_topk_text',
            dest='routing_token_ot_topk_text', type=int, default=8,
            help='v179a: number of text tokens kept per local semantic part '
                 'before computing token-level evidence.')
        siglip2_arg.add_argument('--routing_token_ot_warmup_epochs',
            dest='routing_token_ot_warmup_epochs', type=int, default=0,
            help='v179a: linearly warm up routing_token_ot_beta over this many '
                 'epochs. 0 disables warm-up.')
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
        siglip2_arg.add_argument('--foreground_text_mask_topk_ratio',
            dest='foreground_text_mask_topk_ratio', type=float, default=None,
            help='If set, keep only top-K%% of patches by cosine similarity to '
                 'C_global text embedding before Sinkhorn routing. K = N*ratio. '
                 'Rejected (background) patches get visual_attention_mask=0 so '
                 'they are excluded from all 6 codebook updates. Designed for '
                 'single-object fine-grained datasets (CUB-200) where '
                 'background tokens dominate the router by mass conservation. '
                 'Disabled (None) for multi-object scenes.')
        # v175: text source for the foreground mask. "global" (default) uses
        # the cb0 C_global text embedding; "local_pooled" pools the local
        # text slots cb1..cb5 (per-slot adapter output) by GAP. The local
        # pooled variant focuses pruning on per-anatomy-slot textual content
        # rather than the global scene caption.
        siglip2_arg.add_argument('--foreground_text_mask_source',
            dest='foreground_text_mask_source',
            choices=['global', 'local_pooled', 'per_slot_union',
                     'per_slot_token_attention', 'per_slot_token_attention_low'],
            default='global',
            help='v175: anchor text for foreground mask. global = cb0 text '
                 '(default, legacy). local_pooled = GAP over local cb1..cb5 '
                 'text slots (anatomy-focused). per_slot_union = independent '
                 'top-K per local slot, take UNION (patches relevant to ANY '
                 'slot survive) — preserves per-slot localization. Only used '
                 'when --foreground_text_mask_topk_ratio is set.')
        # v185 BIDIRECTIONAL token pruning (2026-07-11):
        # Extension of v182 (visual-only pruning) with an EXPLICIT text-side
        # pruning step. When enabled, model computes per-slot visual→text
        # attention on the [B, N, 512] shared-space visual patches and the
        # [B, M, T, 512] per-token text embeddings, then:
        #   Direction A (visual): keep top-K% patches per slot, UNION across
        #     slots → visual_attention_mask.
        #   Direction B (text): keep top-K% tokens per slot (softmax over
        #     patches, sum), then REBUILD text_part_raw as mean over the
        #     KEPT text tokens ONLY. All downstream text-embedding paths
        #     (per_slot_text_adapter, whiten, text_token_attention, losses,
        #     compositional analysis) then observe ONLY pruned text tokens.
        # Requires: --backbone_type clip, cached_text_tokens available,
        # cached_text_token_mask available.
        siglip2_arg.add_argument('--final_epoch_eval',
            dest='final_epoch_eval', action='store_true', default=False,
            help='Leakage-free protocol: do NOT swap in the best-mid-eval '
                 'checkpoint; evaluate the FINAL-epoch weights. Removes '
                 'test-based checkpoint selection (mid-eval still logged).')
        # ---- P0: held-out validation protocol (leakage-free epoch selection) --
        # Carves a stratified subset out of the TRAIN split. Training uses only
        # the remaining optimization-train rows; mid-eval retrieval runs
        # val_query (held-out) vs val_db (= optimization-train) so the official
        # test split is never touched before the single final evaluation.
        # Checkpoint selection then uses val mAP@R instead of test mAP.
        siglip2_arg.add_argument('--val_split_ratio', type=float, default=0.0,
            help='P0 protocol: fraction of the TRAIN split held out as the '
                 'validation query set for epoch selection (e.g. 0.1). '
                 '0.0 = off (legacy: select on test = leaky).')
        siglip2_arg.add_argument('--val_split_seed', type=int, default=42,
            help='Seed for the deterministic train/val carve-out.')
        # ---- P0 stage 2: refit on the full train split, stop at E* -----------
        # The val split picks E* (the best epoch) from a 90% optimization-train
        # run; stage 2 then refits on 100% of train and stops there, so the
        # evaluated model saw all the training data while the epoch count was
        # still chosen without touching test.
        # `--epoch` MUST stay at the original budget (e.g. 60): the LR scheduler
        # is built with T_max=args.epoch, so re-running with `-e 5` would
        # complete a whole cosine cycle in 5 epochs and produce a completely
        # different model than epoch 4 of a 60-epoch schedule. This flag stops
        # the loop while leaving the schedule intact -- which also matches how
        # the baselines' epoch_XXX.pth checkpoints were produced.
        siglip2_arg.add_argument('--stop_after_epoch', type=int, default=None,
            help='Stop training after this 0-indexed epoch, keeping the LR '
                 'schedule defined by --epoch. Use with --final_epoch_eval to '
                 'evaluate exactly this epoch (P0 stage-2 refit).')
        siglip2_arg.add_argument('--val_select_metric', type=str,
            default='mAP_at_R', choices=['mAP_at_R', 'mAP'],
            help='Which validation metric selects the checkpoint under the P0 '
                 'protocol. Default mAP@R = the reported paper metric.')
        siglip2_arg.add_argument('--share_codebook',
            dest='share_codebook', action='store_true', default=False,
            help='A4 ablation: tie all M slots to one shared codebook (slot 0). '
                 'Use with matched-capacity --codebook_size (e.g. 768=6x128) to '
                 'test whether SEPARATE per-slot codebooks are needed.')
        siglip2_arg.add_argument('--disable_text_supervision',
            dest='disable_text_supervision', action='store_true', default=False,
            help='A2 ablation: disable all text supervision (visual-only '
                 'codebook_mean routing during training; text-derived losses '
                 'become inactive because the text path is never taken).')
        siglip2_arg.add_argument('--bidirectional_token_prune',
            dest='bidirectional_token_prune', action='store_true',
            default=False,
            help='v185: enable bidirectional visual + text token pruning. '
                 'Downstream text embeddings are recomputed by mean-pool '
                 'over KEPT text tokens only.')
        siglip2_arg.add_argument('--bidirectional_token_prune_visual_ratio',
            dest='bidirectional_token_prune_visual_ratio',
            type=float, default=0.5,
            help='v185: fraction of visual patches to KEEP per slot before '
                 'UNION (default 0.5).')
        siglip2_arg.add_argument('--bidirectional_token_prune_text_ratio',
            dest='bidirectional_token_prune_text_ratio',
            type=float, default=0.5,
            help='v185: fraction of text tokens to KEEP per slot when '
                 'rebuilding text_part_raw (default 0.5).')
        siglip2_arg.add_argument('--bidirectional_token_prune_mode',
            dest='bidirectional_token_prune_mode',
            choices=[
                'legacy', 'mutual_dual_softmax',
                'mutual_consensus_residual',
            ], default='legacy',
            help='Bidirectional pruning score. legacy preserves the v185 '
                 'softmax-sum implementation for reproducibility. '
                 'mutual_dual_softmax uses the geometric mean of visual-to-text '
                 'and text-to-visual attention and applies a distinct visual '
                 'candidate mask to every local routing slot. '
                 'mutual_consensus_residual additionally removes visual '
                 'evidence shared uniformly across local slots.')
        siglip2_arg.add_argument('--bidirectional_prune_only',
            dest='bidirectional_prune_only', action='store_true',
            default=False,
            help='Bypass the learned/OT router and uniformly pool the visual '
                 'tokens selected by each slot-specific pruning mask. At '
                 'image-only evaluation, masks are rebuilt from the learned '
                 'slot anchors with the same consensus-residual selection.')
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
        # v76 (Exp): cosine VQ lookup. Use 1 - cos(z, codeword) instead of
        # squared L2 for the nearest-neighbor argmin selection. EMA codebook
        # update logic unchanged (still tracks z mean), but the geometry
        # used for *lookup* becomes scale-invariant and consistent with the
        # rest of the model (router, NtXent, anchor, prototype heads).
        siglip2_arg.add_argument('--vq_distance_mode',
            dest='vq_distance_mode', type=str, default='euclidean',
            choices=['euclidean', 'cosine'],
            help='Distance metric for codebook nearest-neighbour lookup '
                 'in SemanticCodebookQuantizer. "euclidean" (default) = '
                 'legacy squared-L2. "cosine" = 1 - cosine_similarity '
                 '(scale-invariant, consistent with rest of model).')
        # v76b: also use cosine geometry in loss_vq (the VQ + commitment loss).
        # Only meaningful when --vq_distance_mode=cosine.
        siglip2_arg.add_argument('--vq_loss_cosine',
            dest='vq_loss_cosine', action='store_true', default=False,
            help='If set, loss_vq becomes (1 - cos) instead of MSE for both '
                 'codebook and commitment terms. Pair with --vq_distance_mode '
                 'cosine for the full cosine VQ ablation (v76b).')
        # v78a: adaptive K / codeword split. K_max > codebook_size allocates
        # codebook tensor of size K_max with only `codebook_size` codewords
        # active initially. The training loop can call quantizer.do_split()
        # to convert inactive slots into active ones. Default 0 = use
        # codebook_size (no extra capacity, legacy behaviour bit-exact).
        siglip2_arg.add_argument('--codebook_K_max',
            dest='codebook_K_max', type=int, default=0,
            help='v78a: K_max for adaptive codeword split. 0 (default) = '
                 'use codebook_size (legacy). Set >= codebook_size to '
                 'reserve inactive slots for split.')
        siglip2_arg.add_argument('--warm_start_codebook_from',
            dest='warm_start_codebook_from', type=str, default=None,
            help='v78a: path to a model_state_dict.pth whose '
                 'quantizer.codebooks / .cluster_size / .embed_avg are '
                 'copied into the first K_init slots of this run\'s '
                 'codebook (with active_mask set accordingly). Rest of the '
                 'state_dict is loaded as well (skip quantizer EMA shape '
                 'mismatches).')
        siglip2_arg.add_argument('--split_epochs',
            dest='split_epochs', type=str, default='',
            help='v78a: comma-separated epochs at which to call '
                 'quantizer.do_split. Example: "10,20,30". Empty = no splits.')
        siglip2_arg.add_argument('--split_max_per_epoch',
            dest='split_max_per_epoch', type=int, default=12,
            help='v78a: maximum number of (m, k) splits per split epoch.')
        # v79a (#1.2): cross-codebook orthogonality loss on z (per-codebook
        # batch means). Has gradient via visual_adapter. 0 disables.
        siglip2_arg.add_argument('--lambda_codebook_ortho',
            type=float, default=0.0,
            help='v79a: weight on cross-codebook orthogonality loss '
                 '(on z batch means; gradient via visual_adapter). 0 disables.')
        # v79b (#2.3): learnable per-codebook text prompts.
        # Adds nn.Parameter [M, D] biased onto text_part_raw before the
        # text_adapter, giving each slot a learnable text bias.
        siglip2_arg.add_argument('--use_codebook_text_prompts',
            dest='use_codebook_text_prompts', action='store_true', default=False,
            help='v79b: enable learnable per-codebook text prompt bias.')
        siglip2_arg.add_argument('--codebook_text_prompt_init_scale',
            dest='codebook_text_prompt_init_scale', type=float, default=0.02,
            help='v79b: init std of the learnable text prompt parameter.')
        # v79c (#4.1): hard (Gumbel-Softmax) routing on Sinkhorn output.
        # After Sinkhorn produces soft routing matrix, apply Gumbel-Softmax
        # hard=True on the per-patch part probabilities so each patch routes
        # to exactly one part.
        siglip2_arg.add_argument('--routing_hard',
            dest='routing_hard', action='store_true', default=False,
            help='v79c: apply Gumbel-Softmax hard=True on routing matrix '
                 '(per-patch one-hot part assignment).')
        siglip2_arg.add_argument('--routing_hard_tau',
            dest='routing_hard_tau', type=float, default=1.0,
            help='v79c: Gumbel-Softmax temperature for hard routing.')
        # v79d (#1.3-lite): per-codebook learnable attention queries pooling
        # visual_tokens instead of using Sinkhorn-routed semantic tokens.
        # Each local codebook m (m∈1..5) gets its own query q_m ∈ R^D that
        # attention-pools the cached visual_tokens; replaces the per-cb
        # input to the quantizer. cb0 (global) still uses visual_global.
        siglip2_arg.add_argument('--use_per_cb_attn_pool',
            dest='use_per_cb_attn_pool', action='store_true', default=False,
            help='v79d (#1.3-lite): per-codebook learnable attention pooling '
                 'over visual_tokens for cb1..5. cb0 unchanged.')
        siglip2_arg.add_argument('--per_cb_attn_pool_temp',
            dest='per_cb_attn_pool_temp', type=float, default=1.0,
            help='v79d: softmax temperature for attention pooling.')
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
        # v69a (Exp 1): position-specific CodonHead. Replaces the shared
        # Linear(chunk, 4) with 3 independent Linear(chunk, 4) — one per
        # codon position. Disabled by default; mutually exclusive with
        # --codon_text_anchor and --codon_head_hidden_dim (legacy MLP path).
        siglip2_arg.add_argument('--codon_position_specific_head',
            dest='codon_position_specific_head', action='store_true', default=False,
            help='v69a: per-codon-position separate Linear(chunk, 4). 3 fc '
                 'layers replace the shared one. Allows codon positions 0/1/2 '
                 'to specialize.')
        # v87a: shared CodonHead + zero-initialized position residual adapters.
        # Keeps the legacy shared Linear(chunk, 4) path as the main classifier,
        # then adds a small per-position correction to logits. This preserves
        # v81a at initialization while allowing codon positions to specialize.
        siglip2_arg.add_argument('--codon_position_residual_adapter',
            dest='codon_position_residual_adapter', action='store_true', default=False,
            help='v87a: keep the shared CodonHead classifier and add zero-init '
                 'per-position Linear(chunk, 4) residual adapters to logits.')
        # v69b (Exp 2): residual-split CodonHead. Position 0,1 use the
        # codeword chunk; position 2 uses gamma * (z - q) chunk. Requires
        # --codon_residual_gamma > 0. Bypasses the existing concat -> input_proj
        # path; uses 3 separate Linears (2 semantic + 1 residual).
        siglip2_arg.add_argument('--codon_residual_split',
            dest='codon_residual_split', action='store_true', default=False,
            help='v69b: split codon position 0,1 -> codeword path, '
                 'position 2 -> gamma * residual path. 3 separate Linears. '
                 'Falls back to legacy when residual_gamma=0.')
        # v71a (Exp 5): sample-adaptive residual gate. Multiplies gamma by
        # sigmoid(a_m * ||residual||_2 + b_m) per codebook before injecting
        # into the codon head. a_m, b_m are learnable scalars per CodonHead.
        siglip2_arg.add_argument('--codon_residual_gate',
            dest='codon_residual_gate', action='store_true', default=False,
            help='v71a: dataset-/sample-adaptive residual gate '
                 'sigmoid(a*||z-q|| + b) applied to gamma*residual. '
                 'Only used when --codon_residual_gamma > 0.')
        # v105 (codeword->DNA collision mitigation via decoder expressivity):
        # Replace Linear(chunk=d_model/3, 4) shared across 3 codon positions
        # with a single Linear(d_model, 12) -> view [B, 3, 4]. Each (position, base)
        # output uses ALL d_model dims instead of just its chunk. This is
        # equivalent to "v69a position-specific head" + "no chunk partition",
        # i.e., 3 independent codon-position decoders each seeing the full
        # codeword. Per-head params 1028 -> 9228 (still negligible vs backbone).
        # Default OFF -> bit-exact identical to current v103a behavior.
        # Mutually exclusive with --codon_text_anchor, --codon_residual_split,
        # --codon_position_specific_head, --codon_head_hidden_dim>0.
        siglip2_arg.add_argument('--codon_full_linear',
            dest='codon_full_linear', action='store_true', default=False,
            help='v105: CodonHead decoder uses Linear(d_model, 12)+view[B,3,4] '
                 'on the full codeword embedding, instead of '
                 'Linear(chunk, 4)+share-across-3-positions on chunked input. '
                 'Tests whether removing the chunk-partition information '
                 'bottleneck reduces codeword->DNA codon collisions. Default OFF.')
        # v106 (codeword<->DNA codon bijection losses): operate on the codon
        # decoder applied to codebook codewords directly (no residual, no
        # sample dependence). Three variants:
        #   - Sinkhorn-OT (recommended): hard marginal constraints enforce
        #     K codewords -> K distinct codons mapping when K=64. Strongest.
        #   - Aggregated entropy: KL(uniform || P_bar) + per-codeword
        #     sharpness. Cheaper but K' < K local-min vulnerable.
        #   - Pairwise distinctness: off-diagonal inner product sum.
        #     Cheapest, weakest.
        # Combine with the standard v103a recipe to test the codeword->codon
        # collision hypothesis (DNA unique 0.24 -> targeting 0.55+).
        siglip2_arg.add_argument('--lambda_codeword_codon_sinkhorn',
            type=float, default=0.0,
            help='v106a: Sinkhorn-OT bijection loss weight. 0=disabled. '
                 'Recommended 0.05-0.2 (loss scale ~ log K).')
        siglip2_arg.add_argument('--codeword_codon_sinkhorn_warmup_epochs',
            type=int, default=0,
            help='v111b: linearly ramp lambda_codeword_codon_sinkhorn from 0 '
                 'to its configured value over this many epochs. 0 keeps the '
                 'legacy static lambda behavior.')
        siglip2_arg.add_argument('--codeword_codon_sinkhorn_eps',
            type=float, default=0.1,
            help='v106: entropy regularization for Sinkhorn iterations. '
                 'Smaller -> sharper assignment (closer to Hungarian). '
                 'Default 0.1 (stable + sharp).')
        siglip2_arg.add_argument('--codeword_codon_sinkhorn_iters',
            type=int, default=30,
            help='v106: number of Sinkhorn iterations. Default 30.')
        siglip2_arg.add_argument('--lambda_codeword_codon_agg_ent',
            type=float, default=0.0,
            help='v106b: aggregated-entropy bijection loss weight. 0=disabled. '
                 'Recommended 0.1-0.3.')
        siglip2_arg.add_argument('--codeword_codon_agg_ent_alpha',
            type=float, default=0.5,
            help='v106b: weight of per-codeword sharpness term relative to '
                 'aggregate uniform term. Default 0.5.')
        siglip2_arg.add_argument('--lambda_codeword_codon_pairwise',
            type=float, default=0.0,
            help='v106c: pairwise distinctness bijection loss weight. 0=disabled. '
                 'Recommended 0.5-2.0.')
        siglip2_arg.add_argument('--lambda_text_cluster_codon_ot',
            type=float, default=0.0,
            help='v112b: confidence-weighted text-cluster to DNA-codon OT '
                 'loss. 0=disabled. Uses EMA text prototypes per codebook.')
        siglip2_arg.add_argument('--text_cluster_count',
            type=int, default=8,
            help='v112b: number of textual semantic clusters per codebook.')
        siglip2_arg.add_argument('--text_cluster_temperature',
            type=float, default=0.1,
            help='v112b: soft assignment temperature for text clusters.')
        siglip2_arg.add_argument('--text_cluster_ema_momentum',
            type=float, default=0.95,
            help='v112b: EMA momentum for text cluster prototypes and '
                 'cluster-codeword usage statistics.')
        siglip2_arg.add_argument('--text_cluster_conf_gamma',
            type=float, default=1.0,
            help='v112b: exponent applied to max cluster probability when '
                 'weighting cluster-codeword assignments.')
        siglip2_arg.add_argument('--text_cluster_codon_ot_eps',
            type=float, default=0.1,
            help='v112b: entropy regularization for cluster-to-codon OT.')
        siglip2_arg.add_argument('--text_cluster_codon_ot_iters',
            type=int, default=30,
            help='v112b: Sinkhorn iterations for cluster-to-codon OT.')
        siglip2_arg.add_argument('--lambda_text_codon_rel',
            type=float, default=0.0,
            help='v113: weak rank-based text-codon relational loss weight. '
                 '0=disabled. Encourages text-similar samples to use codon '
                 'distributions closer than text-dissimilar samples without '
                 'forcing cluster-code bijection.')
        siglip2_arg.add_argument('--text_codon_rel_top_frac',
            type=float, default=0.10,
            help='v113: top fraction of centered text-similarity pairs used '
                 'as positive semantic-neighbor pairs.')
        siglip2_arg.add_argument('--text_codon_rel_bottom_frac',
            type=float, default=0.30,
            help='v113: bottom fraction of centered text-similarity pairs '
                 'used as negative semantic-neighbor pairs.')
        siglip2_arg.add_argument('--text_codon_rel_margin',
            type=float, default=0.05,
            help='v113: margin for mean positive codon similarity over mean '
                 'negative codon similarity.')
        siglip2_arg.add_argument('--text_codon_rel_min_pairs',
            type=int, default=8,
            help='v113: minimum number of pairwise examples per codebook for '
                 'the rank-based relational loss.')
        # v107 (prototype cosine clustering): InfoNCE between visual semantic
        # tokens z [B, M, D] and the codebook codewords (prototypes) [M, K, D]
        # in cosine similarity space. Each z is pulled toward its assigned
        # codeword and pushed away from other codewords. Replaces the
        # Sinkhorn-OT bijection enforcement with a "prototype-clustering"
        # interpretation: codewords act as cluster centers, z's cluster around
        # them. The visual-visual paired-aug NtXent and the visual-text DNA
        # NtXent are KEPT (orthogonal contrastive signals).
        siglip2_arg.add_argument('--lambda_proto_cluster_cos',
            type=float, default=0.0,
            help='v107: prototype cosine clustering InfoNCE weight. 0=disabled. '
                 'Recommended 0.3-1.0.')
        siglip2_arg.add_argument('--proto_cluster_cos_tau',
            type=float, default=0.1,
            help='v107: InfoNCE temperature for prototype cosine clustering. '
                 'Smaller = sharper assignment. Default 0.1.')
        # v112 (Hierarchical Codon Decomposition):
        # Decompose the 3-base codon into [cluster_bit, intra_bit_1, intra_bit_2]:
        # - First base = cluster identifier (4 clusters per codebook)
        # - Remaining 2 bases = within-cluster variation (16 sub-states each = 64)
        # K=64 codewords organized into 4 text-similarity clusters of 16 each.
        # First base's logit supervised by cluster label.
        siglip2_arg.add_argument('--lambda_hierarchical_cluster_codon',
            type=float, default=0.0,
            help='v112: Hierarchical codon decomposition loss weight. 0=disabled. '
                 'Recommended 0.2-0.5.')
        siglip2_arg.add_argument('--hierarchical_cluster_n_clusters',
            type=int, default=4,
            help='v112: Number of text-similarity clusters per codebook. '
                 'Default 4 (matches 4 base options for first codon position).')
        siglip2_arg.add_argument('--hierarchical_cluster_refresh_every',
            type=int, default=5,
            help='v112: Re-cluster codewords every N epochs. Default 5.')
        siglip2_arg.add_argument('--hierarchical_cluster_warmup_epochs',
            type=int, default=5,
            help='v112: Delay first clustering by N epochs (let codebook '
                 'stabilize from EMA updates first). Default 5.')
        # v70a (Exp 3): final DNA hash reconstruction. Small decoder maps
        # the flattened hash code [B, 72] back to the SigLIP2 visual_global
        # or text_global embedding via cosine loss.
        siglip2_arg.add_argument('--use_hash_recon',
            dest='use_hash_recon', action='store_true', default=False,
            help='v70a: add a small MLP decoder mapping the 18*4=72-dim '
                 'flattened DNA hash to a SigLIP2 embedding target. '
                 'Inference unchanged (decoder unused at retrieval).')
        siglip2_arg.add_argument('--hash_recon_target',
            dest='hash_recon_target', type=str, default='siglip_visual',
            choices=['siglip_visual', 'text_global', 'both'],
            help='v70a target embedding for the hash-recon decoder.')
        siglip2_arg.add_argument('--hash_recon_hidden',
            dest='hash_recon_hidden', type=int, default=256,
            help='v70a hidden dim for the hash-recon decoder MLP.')
        siglip2_arg.add_argument('--lambda_hash_recon',
            dest='lambda_hash_recon', type=float, default=0.0,
            help='v70a weight on the 1-cos(decoder(hash), target.detach()) '
                 'reconstruction loss. 0 = off.')
        # v72a (Exp 6): dual projection auxiliary heads. Two MLPs from the
        # flattened hash code [B, 72] -> D_proj for separate semantic
        # alignment and instance discrimination losses. Inference unchanged.
        siglip2_arg.add_argument('--use_dual_hash_proj',
            dest='use_dual_hash_proj', action='store_true', default=False,
            help='v72a: add semantic_proj + instance_proj heads from the '
                 'flattened DNA hash. semantic aligns to text/visual global; '
                 'instance does NtXent across paired-aug views.')
        siglip2_arg.add_argument('--dual_hash_proj_hidden',
            dest='dual_hash_proj_hidden', type=int, default=256,
            help='v72a hidden dim for both projection heads.')
        siglip2_arg.add_argument('--dual_hash_proj_target',
            dest='dual_hash_proj_target', type=str, default='siglip_visual',
            choices=['siglip_visual', 'text_global'],
            help='v72a semantic-side target embedding.')
        siglip2_arg.add_argument('--lambda_dual_semantic',
            dest='lambda_dual_semantic', type=float, default=0.0,
            help='v72a weight on semantic 1-cos loss.')
        siglip2_arg.add_argument('--lambda_dual_instance',
            dest='lambda_dual_instance', type=float, default=0.0,
            help='v72a weight on instance NtXent loss between two views.')
        siglip2_arg.add_argument('--dual_hash_proj_ntxent_tau',
            dest='dual_hash_proj_ntxent_tau', type=float, default=0.5,
            help='v72a temperature for the instance NtXent on projection.')
        siglip2_arg.add_argument('--gumbel_tau_init',  dest='gumbel_tau_init',
            type=float, default=2.0)
        siglip2_arg.add_argument('--gumbel_tau_final', dest='gumbel_tau_final',
            type=float, default=0.3)
        siglip2_arg.add_argument('--gumbel_tau_anneal', dest='gumbel_tau_anneal',
            action='store_true', default=True,
            help='Cosine-anneal codon-head tau from init->final across epochs.')
        siglip2_arg.add_argument('--no_gumbel_tau_anneal',
            dest='gumbel_tau_anneal', action='store_false')

        # ---------- v113: text-embedding pre-transform (anisotropy mitigation) ---
        # Applied to `raw = feats["text_part_raw"]` BEFORE the text_adapter.
        #   none           : pass-through (legacy behavior).
        #   per_image_mean : T_local <- T_local - T_local.mean(dim=1); normalize.
        #                    Removes the per-image cross-slot common direction.
        #   global_residual: T_local <- T_local - T_global (slot 0); normalize.
        #                    Uses caption-derived global as the reference.
        #                    Pair with --residualize_visual_for_routing so the
        #                    Sinkhorn cost is computed in matched subspaces.
        #   partial_whiten : T <- ((T - mu) @ W_gamma); requires
        #                    --text_whiten_npz pointing at a precomputed
        #                    {mu, U, S, D, gamma} bundle (see
        #                    scripts/build_text_whiten_matrix.py).
        #   phrase_concept : NO runtime transform; instead point the cache
        #                    directory at a phrase-aggregated text_part.f16.npy
        #                    built by extract_clip_text_phrase_features.py.
        siglip2_arg.add_argument('--text_embed_transform',
            dest='text_embed_transform', type=str, default='none',
            choices=['none', 'per_image_mean', 'global_residual',
                     'partial_whiten', 'global_residual_whiten',
                     'phrase_concept'],
            help='v113/v117: pre-adapter transform applied to cached '
                 'text_part_raw to mitigate CLIP/SigLIP text anisotropy. '
                 'global_residual_whiten = keep slot 0 raw + subtract slot 0 '
                 'from local slots 1..5 + apply partial whitening to the '
                 'residualized local slots only. Pairs with '
                 '--text_whiten_npz pointing at a "_localres.npz" bundle '
                 'built by scripts/build_text_whiten_matrix.py '
                 '--residualize_first.')
        siglip2_arg.add_argument('--text_whiten_npz',
            dest='text_whiten_npz', type=str, default=None,
            help='Path to .npz with keys {mu [D], U [D,D], S [D], gamma} for '
                 '--text_embed_transform partial_whiten.')
        siglip2_arg.add_argument('--text_whiten_gamma',
            dest='text_whiten_gamma', type=float, default=0.25,
            help='Override the stored gamma in --text_whiten_npz (0 = identity, '
                 '1 = full whitening). Default 0.25 (partial).')
        siglip2_arg.add_argument('--text_whiten_eps',
            dest='text_whiten_eps', type=float, default=1e-5)
        siglip2_arg.add_argument('--residualize_visual_for_routing',
            dest='residualize_visual_for_routing', action='store_true',
            default=False,
            help='Companion to --text_embed_transform global_residual: '
                 'subtract patch-mean from visual_tokens and L2-normalize '
                 'BEFORE feeding the Sinkhorn router; losses still see the '
                 'unmodified visual_tokens.')
        # v116: split text path so the transform feeds ONLY routing centroids.
        # Losses (anchor, paired-aug NtXent dynamic tau, text-DNA path) then
        # see the *original* untransformed text_part_tokens. The text_adapter
        # is run twice (once on transformed raw for routing, once on original
        # raw for loss) — small extra cost.
        siglip2_arg.add_argument('--text_transform_routing_only',
            dest='text_transform_routing_only', action='store_true',
            default=False,
            help='When --text_embed_transform != none, apply the transform '
                 'ONLY to the routing-centroid path. Loss-side text embeddings '
                 'use the original (untransformed) text_part_raw -> adapter '
                 'output. Useful for isolating the OT-routing effect of the '
                 'transform from its impact on downstream losses.')
        siglip2_arg.add_argument('--route_global_text',
            dest='route_global_text', action='store_true', default=False,
            help='v119: include slot-0 global caption embedding in the '
                 'Sinkhorn router so C_0 is pooled from visual tokens rather '
                 'than always using the visual CLS/global token.')
        siglip2_arg.add_argument('--routed_cls_add_gamma',
            dest='routed_cls_add_gamma', type=float, default=0.0,
            help='v119: after text-routed visual pooling and before '
                 'quantization, add gamma * stopgrad(visual CLS/global) to '
                 'the routed token(s). 0 disables this path.')
        siglip2_arg.add_argument('--routed_cls_add_scope',
            dest='routed_cls_add_scope', type=str, default='local',
            choices=['local', 'all'],
            help='v119 companion to --routed_cls_add_gamma. local adds CLS '
                 'only to C_1..C_5; all also adds it to C_0.')
        siglip2_arg.add_argument('--slot_sequential_residual',
            dest='slot_sequential_residual', action='store_true', default=False,
            help='2026-07-21: pool each local slot from the TOKEN residual left '
                 'by the previous slots, so slot m only sees what slots 1..m-1 '
                 'did not explain. Targets the measured redundancy between '
                 'local slots (each pools ~62%% of the same patches; pairwise '
                 'NMI 0.74-0.82). Constructive, not a redundancy penalty. '
                 'Distinct from --local_residual_quant, which removes only the '
                 'global C_0 projection from the slot vectors. Default off.')
        siglip2_arg.add_argument('--slot_seq_residual_gamma',
            dest='slot_seq_residual_gamma', type=float, default=1.0,
            help='Strength of the --slot_sequential_residual token projection '
                 'removal. 1.0 = full removal, 0.0 = disabled.')
        siglip2_arg.add_argument('--local_residual_quant',
            dest='local_residual_quant', action='store_true', default=False,
            help='v122: before codeword assignment, remove each local slot '
                 "token's projection onto C_0/global. C_0 itself is kept "
                 'unchanged. Default off.')
        siglip2_arg.add_argument('--local_residual_gamma',
            dest='local_residual_gamma', type=float, default=1.0,
            help='v122: strength for --local_residual_quant projection '
                 'removal. 1.0 = full orthogonal residual.')
        siglip2_arg.add_argument('--local_residual_detach_global',
            dest='local_residual_detach_global',
            action=argparse.BooleanOptionalAction, default=True,
            help='v122: detach C_0/global when computing the projection '
                 'direction. Default True keeps C_0 from being optimized '
                 'only to explain away local residuals.')
        siglip2_arg.add_argument('--local_residual_text',
            dest='local_residual_text', action='store_true', default=False,
            help='v122/v123: apply the same C_0-residualization to '
                 'text_part_tokens before the text-only DNA/codeword path '
                 'and codeword text-prototype loss.')

        # ---------- v162: grounded text routing (Stage-2 top-k token pruning) -----
        # Visually-routed semantic_visual_tokens (Stage-1 OT output) act as
        # queries against per-codebook local text tokens; top-k_t tokens are
        # softmax-weighted and pooled to form a refined per-codebook text
        # embedding. Overwrites text_part_tokens flowing into the loss layer.
        # Requires cached_text_tokens (extract_clip_text_token_features.py).
        siglip2_arg.add_argument('--grounded_text_routing',
            dest='grounded_text_routing', action='store_true', default=False,
            help='v162: enable Stage-2 OT-based token pruning to refine '
                 'per-codebook text embeds. Requires text_tokens.f16.npy '
                 'cache (extract_clip_text_token_features.py).')
        siglip2_arg.add_argument('--grounded_text_k_t',
            dest='grounded_text_k_t', type=int, default=5,
            help='v162: top-k_t local text tokens to keep per codebook in '
                 'Stage-2 routing.')
        siglip2_arg.add_argument('--grounded_text_eps',
            dest='grounded_text_eps', type=float, default=0.05,
            help='v162: temperature (entropic-OT epsilon) for Stage-2 '
                 'softmax-attention scoring. Smaller = sharper top-k.')
        siglip2_arg.add_argument('--grounded_text_stage1_sg',
            dest='grounded_text_stage1_sg',
            action=argparse.BooleanOptionalAction, default=True,
            help='v162: stop-gradient on Stage-1 semantic_visual_tokens when '
                 'used as Stage-2 query. Default True (decoupled cascade).')
        siglip2_arg.add_argument('--grounded_text_skip_global',
            dest='grounded_text_skip_global',
            action=argparse.BooleanOptionalAction, default=True,
            help='v162: keep C_0 (global slot) text embed UNCHANGED; only '
                 'refine local 5 codebooks. Default True (global caption is '
                 'a whole-image summary; token-pruning is meaningful only '
                 'for local semantic parts).')
        siglip2_arg.add_argument('--soft_visual_grounded_text_pool',
            dest='soft_visual_grounded_text_pool', action='store_true',
            default=False,
            help='Use detached local OT mass to aggregate visual-query to '
                 'text-token cross-attention, then residual-pool all valid '
                 'text tokens into the local semantic text embeddings. No '
                 'content top-k or text hard mask is used; C_0 is unchanged.')
        siglip2_arg.add_argument('--cosine_visual_grounded_text_pool',
            dest='cosine_visual_grounded_text_pool', action='store_true',
            default=False,
            help='Refine each local text embedding with parameter-free cosine '
                 'attention over all valid caption tokens, aggregated by '
                 'detached local OT mass. A zero-initialized learned slot gate '
                 'makes the initial forward exactly match the base text path.')

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
        # v67: dynamic-tau variant selector. "text_cos" (default) reproduces
        # the legacy v42 per-pair tau. "neg_only_norm_model" adopts the
        # MACL/PromptHash-informed redesign: positive pair uses static
        # base_tau, negative pair uses base * model_scale_m * semantic_scale_ij.
        loss_arg.add_argument('--ntxent_dynamic_tau_variant',
            type=str, default='text_cos',
            choices=['text_cos', 'neg_only_norm_model'],
            help="Dynamic-tau variant. 'text_cos' = legacy v42 per-pair tau. "
                 "'neg_only_norm_model' = positive pair uses static base_tau, "
                 "negative pair uses base * model_scale * semantic_scale "
                 "where semantic is batch-normalized + tanh-clipped text "
                 "affinity and model is a MACL-style per-codebook factor.")
        loss_arg.add_argument('--ntxent_dynamic_tau_model_beta',
            type=float, default=0.5,
            help="v67: beta in model_scale_m = clamp(1 + beta * (A_m - A0), "
                 "r_min, r_max). Larger beta = stronger model-state coupling.")
        loss_arg.add_argument('--ntxent_dynamic_tau_model_a0',
            type=float, default=0.6,
            help="v67: alignment threshold A0 (positive-pair agreement at "
                 "which model_scale = 1). Below A0 -> tau shrinks (sharper); "
                 "above -> tau grows (softer).")
        loss_arg.add_argument('--ntxent_dynamic_tau_model_scale_min',
            type=float, default=0.75,
            help="v67: lower clamp on model_scale_m.")
        loss_arg.add_argument('--ntxent_dynamic_tau_model_scale_max',
            type=float, default=1.25,
            help="v67: upper clamp on model_scale_m.")
        loss_arg.add_argument('--ntxent_dynamic_tau_semantic_scale_min',
            type=float, default=0.7,
            help="v67: lower clamp on semantic_scale_ij (negative-pair "
                 "text-affinity modulation).")
        loss_arg.add_argument('--ntxent_dynamic_tau_semantic_scale_max',
            type=float, default=1.3,
            help="v67: upper clamp on semantic_scale_ij.")
        # v87 (Wang & Liu, CVPR 2021): explicit hard-negative sampling in
        # per-codebook NtXent. For each anchor, keep only the top-alpha
        # fraction of *informative* negatives (highest similarity) in the
        # softmax denominator; below-threshold negatives are masked out.
        # alpha=1.0 = legacy (keep all negatives, bit-exact). alpha<1.0 =
        # Wang Eq 9 hard-contrastive loss adapted per-codebook.
        loss_arg.add_argument('--ntxent_hard_neg_alpha',
            type=float, default=1.0,
            help="v87: keep top-alpha fraction of hardest negatives per "
                 "anchor in per-codebook NtXent (Wang & Liu CVPR 2021 Eq 9). "
                 "1.0 disables (legacy, all negatives kept). Recommended "
                 "0.25-0.5 for compositional retrieval; the positive pair "
                 "is always preserved.")
        # v88 (Huang et al., ICML 2023; MACL Algorithm 1 adapted per-codebook):
        # model-aware temperature that adapts to paired-augmentation positive
        # alignment magnitude. For each codebook m, compute the batch-mean
        # cosine A_m = E_i[cos(semantic_v_m^view1[i], semantic_v_m^view2[i])]
        # (detached), then scale tau as:
        #     tau_m = tau_0 * (1 + macl_alpha * (A_m - macl_a0))
        # Combined MULTIPLICATIVELY with the legacy v42 text-cosine
        # dynamic-tau when both are active; either disables to scalar 1.
        # macl_alpha=0 disables (legacy, bit-exact). Recommended 0.3-0.7.
        loss_arg.add_argument('--ntxent_macl_alpha',
            type=float, default=0.0,
            help="v88: MACL-paired model-aware tau alpha (Huang et al. ICML "
                 "2023, Eq 7). 0=disabled (legacy). Per-codebook tau "
                 "tau_m = tau_0 * (1 + alpha * (A_m - A_0)) where A_m is "
                 "the batch-mean cosine of paired-aug semantic_v[m]. "
                 "Composes multiplicatively with --ntxent_dynamic_tau when "
                 "both are enabled.")
        loss_arg.add_argument('--ntxent_macl_a0',
            type=float, default=0.0,
            help="v88: MACL initial alignment baseline A_0. 0.0 means tau "
                 "starts at tau_0 (early training, low alignment) and grows "
                 "as paired-aug alignment improves. Set higher (e.g. 0.5) "
                 "to start with tau below tau_0 and have it converge to "
                 "tau_0 only at full alignment.")
        # v91 (text-to-DNA-hash matching): force the same 36-bit DNA code
        # to be retrievable from either the image visual_tokens path OR a
        # text-only path that reuses the same quantizer + codon heads.
        # Implements Option F from the 2026-05-28 dynamic-tau discussion:
        # text_part_tokens [B, 6, D] -> quantizer (no EMA) -> codon_heads
        # -> text_continuous_code [B, 18, 4]. MSE between this and the
        # image-side continuous_code is added to the total loss.
        # 0 (default) disables (legacy bit-exact). Recommended 0.05-0.10.
        loss_arg.add_argument('--lambda_text_hash',
            type=float, default=0.0,
            help='v91: weight for text-to-image DNA-hash matching loss. '
                 'Computes MSE between text-derived continuous_code (via '
                 'a parallel text-only pass through the shared quantizer '
                 'and codon_heads) and image-derived continuous_code. '
                 '0 disables (legacy). Recommended 0.05-0.10.')
        # v93: per-codebook cross-modal codeword InfoNCE between the visual
        # quantized codeword (image path) and the text quantized codeword
        # (v91 text path through the shared quantizer, EMA-disabled).
        # Positive: (visual_cw_m[i], text_cw_m[i]) per cb_m, per sample i.
        # Negatives: other samples' text_cw_m. Bidirectional (CLIP-style).
        # Reuses the same `text_part_tokens -> quantizer.eval()` path the v91
        # text-hash matching uses, so a single text-path forward serves both
        # lambda_text_hash (MSE on continuous_code) and lambda_cw_xmodal
        # (InfoNCE on codeword). 0 disables.
        loss_arg.add_argument('--lambda_cw_xmodal',
            type=float, default=0.0,
            help='v93: per-codebook cross-modal codeword InfoNCE between '
                 'visual quantized codeword and text quantized codeword. '
                 '0 disables (legacy bit-exact). Recommended 0.05-0.20.')
        loss_arg.add_argument('--cw_xmodal_temperature',
            type=float, default=0.07,
            help='Temperature for v93 cross-modal codeword InfoNCE. '
                 'Default 0.07 (CLIP-style).')
        # v160: cross-modal commitment loss (Uni-Code Eq.(8), NeurIPS 2023).
        # Adds (beta/2)*||phi^a(x) - sg[e^b]||^2 to the encoder commitment,
        # i.e., visual encoder commits to text-derived quantized codeword
        # AND text encoder symmetrically commits to visual quantized codeword.
        # Standard self-modality commitment (lambda_quant) stays ON.
        loss_arg.add_argument('--lambda_xmodal_commit',
            dest='lambda_xmodal_commit', type=float, default=0.0,
            help='v160 (Uni-Code Eq.8): cross-modal commitment loss weight = '
                 'beta/2 in the paper notation. Recommended 0.025 when '
                 'lambda_quant = 0.05 (so weight ratio matches beta/(beta/2) '
                 '= 2:1). Symmetric across visual and text. Requires the '
                 'text-quantization path to be active (i.e., one of '
                 'lambda_text_hash / lambda_text_hash_ntxent / lambda_cw_xmodal '
                 '/ lambda_xmodal_commit > 0). Default 0.0 disables.')
        # v176: skip cb0 (C_global) from xmodal_commit. C_global uses pooled
        # visual_global (siglip2_global source) NOT text-anchored routing, so
        # text supervision on cb0 is an architectural mismatch. Local slots
        # (cb1..cb5) remain xmodal-supervised.
        loss_arg.add_argument('--xmodal_commit_skip_global',
            dest='xmodal_commit_skip_global',
            action='store_true', default=False,
            help='v176: skip cb0 (C_global) from xmodal_commit. Only local '
                 'codebooks (cb1..cb5) contribute. C_global codebook then '
                 'learns purely from visual + CIBHash NtXent.')
        # v161 (Uni-Code Section 4.3 simplified): bi-modal EMA codebook update.
        loss_arg.add_argument('--mm_ema',
            dest='mm_ema', action='store_true', default=False,
            help='v161 (Uni-Code MM-EMA, simplified). Currently the text path '
                 'runs the quantizer in eval mode (EMA-disabled), so the '
                 'codebook is updated only from visual signals. With --mm_ema, '
                 'the text path KEEPS the quantizer in train mode, so both '
                 'modalities contribute to the EMA codebook update. This is a '
                 'minimum-viable Uni-Code MM-EMA without the cross-attention '
                 'intermediary (r^a, r^b) which is a separate v161b/c step. '
                 'Default OFF preserves legacy behavior.')
        # ---------- v144: text -> code KL distillation -----------------
        # Per-codebook distribution matching between visual and text views.
        # For each local codebook m (cb0/global excluded by default), build
        # K-way categorical distributions
        #   p_v[m] = softmax(z_v_m @ C_m.T / tau_v)
        #   p_t[m] = softmax(t_m   @ C_m.T / tau_t)
        # over the K codewords, then minimize KL(p_t.detach() || p_v) with
        # per-sample confidence weighting conf = 1 - H(p_t)/log(K). Only
        # samples with conf > --text_code_kl_conf_threshold contribute.
        # Provides a *soft* codebook-level text supervision: text decides
        # which codeword each visual feature should go to. Asymmetric --
        # text gets no gradient (handled by other text-side losses).
        loss_arg.add_argument('--lambda_text_code_kl',
            type=float, default=0.0,
            help='v144: weight for text->code KL distillation. '
                 '0 disables (default). Recommended 0.01-0.05.')
        loss_arg.add_argument('--text_code_kl_tau_v',
            type=float, default=0.1,
            help='v144: visual softmax temperature over codebook. '
                 'Larger -> visual exploration ↑ (less commit).')
        loss_arg.add_argument('--text_code_kl_tau_t',
            type=float, default=0.07,
            help='v144: text softmax temperature over codebook. '
                 'Smaller -> text gets sharper / more confident.')
        loss_arg.add_argument('--text_code_kl_conf_threshold',
            type=float, default=0.0,
            help='v144: skip samples with text confidence below this '
                 'threshold. 0.0 = no filtering. Recommended 0.2.')
        loss_arg.add_argument('--text_code_kl_skip_global',
            action='store_true', default=False,
            help='v144: skip codebook 0 (C_global) when computing the '
                 'text_code_kl loss. Only local codebooks (cb1..cb5) '
                 'contribute. Recommended.')
        # v172 (routing-text supervision): directly supervise per-patch
        # routing weights using cosine sim to per-slot text embeddings.
        # Closes the gap that text_code_kl only touches codeword-INDEX
        # distribution but not which patches feed each codebook.
        # Per (image, codebook m, patch p):
        #   target[m, p] = softmax_p(cos(text_part[m], visual_token[p]) / tau)
        # Loss: KL(target || routing_matrix[m]) averaged over m, B.
        loss_arg.add_argument('--lambda_routing_text',
            type=float, default=0.0,
            help='v172: weight for per-patch routing supervision via text. '
                 '0 disables (default). Recommended 0.05-0.20. '
                 'Replaces lambda_text_code_kl when both used in tandem.')
        loss_arg.add_argument('--routing_text_tau',
            type=float, default=0.1,
            help='v172: softmax temperature on per-patch text-similarity target. '
                 'Smaller -> sharper target routing.')
        loss_arg.add_argument('--routing_text_skip_global',
            action='store_true', default=False,
            help='v172: skip cb0 (C_global) from routing supervision. '
                 'C_global is supposed to aggregate full-image content; per-patch '
                 'text supervision does not apply.')
        # v173 (text-codeword contrastive): per-codebook InfoNCE between RAW
        # text_part_tokens (NOT quantized) and quantized_tokens (post-VQ visual,
        # = codebook[m, k_visual*]).  Positive = same image, same slot; negative
        # = other images, same slot.  Bypasses text-quantization noise.
        loss_arg.add_argument('--lambda_text_codeword_contrastive',
            type=float, default=0.0,
            help='v173: weight for text-codeword contrastive (RAW text vs '
                 'quantized visual codeword, per-codebook InfoNCE). '
                 '0 disables (default). Recommended 0.05-0.20.')
        loss_arg.add_argument('--text_codeword_contrastive_tau',
            type=float, default=0.07,
            help='v173: NtXent temperature for text-codeword contrastive. '
                 'CLIP default 0.07; smaller -> sharper supervision.')
        loss_arg.add_argument('--text_codeword_contrastive_skip_global',
            action='store_true', default=False,
            help='v173: skip cb0 (C_global) from text-codeword contrastive.')
        # v174 Option alpha: text vs PRE-QUANT semantic_visual_tokens InfoNCE.
        # Bypasses quantization-induced collapse attractor of v173.
        loss_arg.add_argument('--lambda_text_preq_contrastive',
            type=float, default=0.0,
            help='v174alpha: weight for text vs PRE-QUANT semantic_visual_tokens '
                 'per-codebook InfoNCE. 0 disables (default). Continuous space, no '
                 'collapse attractor unlike v173 quantized variant.')
        loss_arg.add_argument('--text_preq_contrastive_tau',
            type=float, default=0.07,
            help='v174alpha: temperature.')
        loss_arg.add_argument('--text_preq_contrastive_skip_global',
            action='store_true', default=False,
            help='v174alpha: skip cb0.')
        # v174 Option gamma: hash-level text<->visual InfoNCE.
        # Uses 18x4 codon-base continuous codes; positive=same image, negative=other.
        loss_arg.add_argument('--lambda_text_visual_hash_contrastive',
            type=float, default=0.0,
            help='v174gamma: weight for hash-code level text<->visual InfoNCE '
                 '(continuous_code [B, 18, 4] flat vs text_continuous_code [B, 18, 4] '
                 'flat). 0 disables (default).')
        loss_arg.add_argument('--text_visual_hash_contrastive_tau',
            type=float, default=0.07,
            help='v174gamma: temperature.')
        loss_arg.add_argument('--lambda_codeword_text_proto',
            type=float, default=0.0,
            help='v123: weight for codeword-level text prototype alignment. '
                 'Maintains an EMA text prototype per (codebook, codeword) '
                 'from assigned samples and classifies each visual '
                 'quantizer input against those prototypes. 0 disables.')
        loss_arg.add_argument('--codeword_text_proto_tau',
            type=float, default=0.1,
            help='v123: temperature for codeword text-prototype CE.')
        loss_arg.add_argument('--codeword_text_proto_momentum',
            type=float, default=0.95,
            help='v123: EMA momentum for per-codeword text prototypes.')
        loss_arg.add_argument('--codeword_text_proto_min_count',
            type=int, default=4,
            help='v123: minimum EMA assignment count before a text prototype '
                 'can be used as a negative/target.')
        loss_arg.add_argument('--codeword_text_proto_include_global',
            dest='codeword_text_proto_include_global',
            action='store_true', default=False,
            help='v123: include slot 0/C_global in text-prototype alignment. '
                 'Default false focuses the loss on local codebooks.')
        # v97: replace text_hash MSE with symmetric NtXent treating the
        # text-derived continuous_code [B, 18, 4] as an augmented view of
        # the image-derived continuous_code. Positive pair = same sample's
        # (image_dna, text_dna); negatives = other samples in batch.
        # Operates on the FLATTENED 72-dim DNA representation.
        # Only effective when --lambda_text_hash > 0; swaps the loss form
        # for the same weight.
        loss_arg.add_argument('--text_hash_use_ntxent',
            dest='text_hash_use_ntxent',
            action=argparse.BooleanOptionalAction, default=False,
            help='v97: swap text_hash MSE → symmetric NtXent. Treats text '
                 'DNA as augmented view of image DNA. Use with '
                 '--lambda_text_hash > 0. Pass --no-text_hash_use_ntxent '
                 'to keep legacy MSE.')
        loss_arg.add_argument('--text_hash_ntxent_temperature',
            type=float, default=0.07,
            help='Temperature for v97 text-DNA NtXent. Default 0.07.')
        # v100: ADDITIVE text-DNA NtXent (extra term, NOT a swap of MSE).
        # Computes symmetric InfoNCE on flattened DNA continuous_code [B, 72]
        # exactly like v97's swap form, but adds it to total loss with its
        # own weight INSTEAD OF replacing the MSE term. Use when you want
        # both (a) image-text DNA alignment via MSE AND (b) cross-batch
        # discrimination via NtXent.
        loss_arg.add_argument('--lambda_text_hash_ntxent',
            type=float, default=0.0,
            help='v100: weight for ADDITIVE text-DNA NtXent. 0=disabled. '
                 'Composes on top of lambda_text_hash (MSE). Uses '
                 '--text_hash_ntxent_temperature for tau.')
        loss_arg.add_argument('--text_hash_ntxent_mode',
            dest='text_hash_ntxent_mode',
            type=str, default='global',
            choices=['global', 'per_codebook'],
            help='Granularity of the ADDITIVE text-DNA NtXent.\n'
                 '  global       (default; v100-v125d) one symmetric InfoNCE '
                                'on the full flattened DNA code [B, R*4]. '
                                'Aligns the WHOLE hash sequence as one vector.\n'
                 '  per_codebook (v128) M=num_codebooks independent symmetric '
                                'InfoNCEs, one per codebook, each over its L '
                                'codons flattened to a (L*4)-dim DNA segment. '
                                'Matches the "each codebook = one semantic '
                                'part" compositional contribution claim: text '
                                'directly supervises each codebook\'s DNA '
                                'segment instead of only the whole hash.')
        # v176: skip cb0 (C_global) from text_hash_ntxent per_codebook.
        # Only valid in per_codebook mode. Global mode ignores this flag.
        loss_arg.add_argument('--text_hash_ntxent_skip_global',
            dest='text_hash_ntxent_skip_global',
            action='store_true', default=False,
            help='v176: skip cb0 (C_global) from text_hash_ntxent (per_codebook '
                 'mode). Only local codebooks cb1..cb5 contribute. Match '
                 'architectural role of C_global (visual-pooled input, not '
                 'text-anchored routing).')
        # v73 (Exp 7): global DNA NtXent auxiliary loss alongside per-codebook.
        # When `ntxent_mode=per_codebook`, also compute the global NtXent
        # (whole 18-codon DNA code) using static base temperature and add
        #     loss_ntxent = loss_local + lambda_global * loss_global
        # Helps the final retrieval-time DNA code regain cross-codebook
        # coherence that per-codebook NtXent alone can fragment. Default 0
        # keeps the legacy per-codebook-only behaviour.
        loss_arg.add_argument('--lambda_global_dna_ntxent',
            type=float, default=0.0,
            help='v73: weight for the global-NtXent auxiliary loss on full '
                 '[B, 18, 4] DNA code (uses static base ntxent_temperature, '
                 'NO dynamic-tau). Only active when ntxent_mode=per_codebook. '
                 'Try 0.05 / 0.10. 0 (default) keeps legacy behaviour.')
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
        # ---------- v119: CIBHash-style per-codebook hash supervision ---------
        # CIBHash (Hu et al., 2021) loss: paired-augmented binary hash codes
        # supervised via (1) NtXent on signed-thresholded sigmoid probs and
        # (2) symmetric Bernoulli KL between the two views' bit-probabilities.
        # Per-codebook variant: split the 36-bit DNA hash into 6 codebook
        # groups of 6 bits each (3 codons * 2 bits / A=00 C=01 G=10 T=11) and
        # average the 6 per-codebook losses. Operates on continuous_code
        # [B, 18, 4] (softmax probs over A/C/G/T) by deriving the 2-bit
        # marginal probs per codon:
        #     bit_0_prob = P(G) + P(T)   (== marginal of the upper-half bases)
        #     bit_1_prob = P(C) + P(T)   (== marginal of bases with second bit 1)
        # then z = STE-sign(prob - 0.5), and the per-codebook 12-dim z [B, 6]
        # feeds the NtXent. The KL operates on the prob vectors directly.
        loss_arg.add_argument('--lambda_cibhash_ntxent', type=float, default=0.0,
            help='v119: weight for CIBHash-style NtXent on per-codebook '
                 'binary hash codes (paired-aug). 0 disables.')
        loss_arg.add_argument('--lambda_cibhash_kl', type=float, default=0.0,
            help='v119: weight for CIBHash-style symmetric Bernoulli KL on '
                 'per-codebook bit-probabilities (paired-aug). CIBHash paper '
                 'uses 0.001 (relative to ntxent 1.0). 0 disables.')
        loss_arg.add_argument('--cibhash_temperature', type=float, default=0.3,
            help='v119: temperature for CIBHash NtXent (paper default 0.3).')
        # v120: CIBHash extension knobs.
        loss_arg.add_argument('--cibhash_mode',
            dest='cibhash_mode', type=str, default='per_codebook',
            choices=['per_codebook', 'global'],
            help='v120: per_codebook (v119a default) splits the 36-bit hash '
                 'into 6 codebook-groups and averages 6 NtXent losses; '
                 'global flattens to one 36-bit hash and computes one '
                 'NtXent on the full code (CIBHash original design).')
        loss_arg.add_argument('--cibhash_dynamic_tau',
            dest='cibhash_dynamic_tau', action='store_true', default=False,
            help='v120e: enable text-cos dynamic temperature on the CIBHash '
                 'per-codebook NtXent (mirrors v42 ntxent_dynamic_tau). '
                 'Requires cibhash_mode=per_codebook.')
        loss_arg.add_argument('--cibhash_dynamic_tau_alpha',
            dest='cibhash_dynamic_tau_alpha', type=float, default=0.3,
            help='v120e: alpha for cibhash dynamic tau. tau_ij = T * '
                 '(1 + alpha * cos(text_i, text_j)). 0 disables.')
        # v149: continuous NtXent (no STE-sign quantization) to restore
        # cosine granularity for uniformity gradient -- aligns with the
        # original CIBHash paper which operates on continuous Bernoulli
        # probabilities, not STE-binarized signs.
        loss_arg.add_argument('--cibhash_ntxent_continuous',
            dest='cibhash_ntxent_continuous', action='store_true', default=False,
            help='v149: replace STE-sign(bit_probs) with linearly-shifted '
                 'continuous bit_probs (2*p - 1) for the cibhash NtXent. '
                 'Removes the 7-level cosine granularity ceiling imposed '
                 'by 6-bit signed slices; full continuous uniformity '
                 'gradient. DNA code path is unchanged (computed at '
                 'inference via argmax). Default OFF preserves legacy '
                 'v119-v148 behavior.')
        # v150: contrastive on pre-VQ routed visual tokens instead of
        # post-VQ continuous_code -- D-dim continuous uniformity gradient
        # at the routing layer.
        loss_arg.add_argument('--cibhash_ntxent_source',
            dest='cibhash_ntxent_source', type=str, default='continuous_code',
            choices=['continuous_code', 'visual_token'],
            help='v150: input source for cibhash NtXent. '
                 '"continuous_code" (default) = legacy v119-v149 behavior, '
                 'NtXent on bit_probs derived from post-VQ codon outputs. '
                 '"visual_token" = NtXent on pre-VQ semantic_visual_tokens '
                 '[B, M, D] from the router (D-dim continuous, full cosine '
                 'granularity). KL term is implicitly disabled in visual_token '
                 'mode (Bernoulli KL undefined on continuous vectors).')
        loss_arg.add_argument('--cibhash_visual_projection_head',
            dest='cibhash_visual_projection_head', action='store_true',
            default=False,
            help='Apply the visual-token CIBHash NtXent to six independent '
                 'two-layer projection heads instead of directly to the '
                 'pre-VQ semantic tokens. Projection outputs are train-only '
                 'and do not alter the VQ/DNA inference path.')
        # ---------- v138: prototype passthrough + paired-view InfoNCE -----
        # Two flags to *remove the VQ codebook bottleneck* for codon_head
        # input while keeping prototype-based clustering as a separate
        # supervision signal.
        loss_arg.add_argument('--codon_input_source',
            dest='codon_input_source', type=str, default='quantized',
            choices=['quantized', 'routed'],
            help='v138: source of codon_head input. quantized=legacy VQ '
                 'output (post-gate quantized_tokens); routed=raw router '
                 'weighted-sum vectors (semantic_visual_tokens). When '
                 '"routed" is set, codon_head sees continuous tokens; VQ '
                 'still runs to provide codebook_distances for the '
                 'prototype InfoNCE loss but does NOT bottleneck codon '
                 'output. Use with --lambda_vq 0 --lambda_quant 0 '
                 '--lambda_anchor 0 --lambda_bu 0.')
        loss_arg.add_argument('--lambda_proto_cluster',
            dest='lambda_proto_cluster', type=float, default=0.0,
            help='v138: weight for paired-view prototype-cluster InfoNCE. '
                 'Operates on softmax(-codebook_distances/tau) per '
                 'codebook. Forces paired augmented views to map to the '
                 'same prototype assignment distribution (SwAV-lite '
                 'without Sinkhorn balancing).')
        loss_arg.add_argument('--proto_cluster_temperature',
            dest='proto_cluster_temperature', type=float, default=0.3,
            help='v138: NtXent temperature for the prototype-cluster '
                 'paired-view InfoNCE loss.')
        # ---------- v121: SwAV-style swapped balanced assignment loss --------
        # Operates at the codeword-assignment level (BEFORE codon decoding).
        # Each paired-aug view computes a soft codeword-assignment distribution
        # P(k | z, m) = softmax(-codebook_distances[m] / tau_pred), and a
        # Sinkhorn-balanced *target* Q is built that is row-uniform (each
        # sample has a distribution) and column-uniform (each codeword used
        # ~B/K times in the batch). The swapped CE pulls view1's prediction
        # toward view2's balanced target and vice versa, encouraging
        # (a) two-view assignment consistency and (b) batch-level codeword
        # usage balance — directly counteracting the dead-codeword collapse
        # the CIBHash regime induces. Applied to local codebooks (slot 1..5)
        # by default; slot 0 (C_global) excluded unless --swav_assign_include_global.
        loss_arg.add_argument('--lambda_swav_assign', type=float, default=0.0,
            help='v121: weight for SwAV-style swapped balanced codeword-'
                 'assignment loss (additive). 0 disables.')
        loss_arg.add_argument('--swav_assign_tau', type=float, default=0.1,
            help='v121: prediction temperature; logits = -distances/tau.')
        loss_arg.add_argument('--swav_sinkhorn_eps', type=float, default=0.05,
            help='v121: Sinkhorn entropy regularizer; smaller = sharper '
                 'balanced target.')
        loss_arg.add_argument('--swav_sinkhorn_iters', type=int, default=3,
            help='v121: number of Sinkhorn-Knopp iterations.')
        loss_arg.add_argument('--swav_assign_include_global',
            dest='swav_assign_include_global', action='store_true', default=False,
            help='v121: include slot 0 (C_global) in the swav-assign loss. '
                 'Default OFF (loss only applies to local slots 1..5).')
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
