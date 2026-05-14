# GroundedDNA

VLM-grounded compositional DNA hashing for image retrieval.

## Overview

`GroundedDNA` encodes each image into an 18-codon DNA sequence (36-bit hash)
that supports fast Hamming-distance retrieval while staying interpretable
at the codeword level. The full pipeline is:

```
  image
    │
    ▼
  SigLIP2 vision tower (frozen)          ←── visual patches (196 tokens)
    │                                        + global pooled feature
    ▼
  visual adapter
    │
  Sinkhorn OT router ◄────── Qwen2.5-VL              per-image, per-codebook
    │                          part captions  ──►   text embedding
    │                          (6 semantic slots)
    ▼
  6 codebooks (K = 32 by default, EMA-updated)
    │
    ▼
  per-codebook codeword  →  3 DNA codons  →  6 base ∈ {A, C, G, T}
                                         ×6 = 18 codons = 36 bits
```

Key design choices:

- **Frozen SigLIP2 backbone** — only the adapters, the per-codebook codebooks
  and the codon head are trained. Features are pre-extracted to disk for
  speed (`extract_siglip2_features.py`).
- **Sinkhorn OT routing** with marginal balance — every codebook slot receives
  approximately equal patch mass per image, which we found is what keeps the
  codebook diverse during training.
- **VLM text supervision** — Qwen2.5-VL produces 6 short captions per image,
  one per codebook semantic role (global / primary object / ... / scene type).
  The SigLIP2 text tower encodes these and the result drives the routing
  centroids.
- **Codon head with Gumbel-Softmax / STE** — soft 4-base probabilities during
  training, hard one-hot codes at retrieval time.
- **DNA-style hash** — the 18-base sequence is the unit of similarity at
  retrieval (Hamming over bases, equivalent to the natural 36-bit packing).

The full version history, ablation log, and design-decision rationale live
in [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md). This README only covers the
setup needed to reproduce or extend the current best variant.

## Software requirements

Tested on Ubuntu 22.04 with CUDA 12.x.

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA support (project uses `cuda:0` by default)
- See `requirements.txt` for the pinned Python dependencies.
- Recommended setup: create the conda environment from `hashing.yml`:
  ```bash
  conda env create -f hashing.yml
  conda activate hashing
  ```
  This pulls the exact deps used during development.

## Repository layout

```
GroundedDNA/
├── README.md
├── requirements.txt
├── hashing.yml                 # conda env (exact deps)
├── .gitignore
├── docs/
│   └── PROJECT_LOG.md          # full design / ablation history
├── models/                     # backbone wrappers, adapters, routers
├── dna_utils/                  # DNA / codebook / VLM utilities
├── baseline/                   # HashNet / DPSH / CSQ / OrthoHash baselines
├── scripts/                    # shell entry points
├── config.py                   # CLI + Config class
├── dataloaders.py              # ImgRtv dataset + cache loader
├── model_siglip2.py            # SigLIP2SemanticOTModel
├── loss_siglip2.py             # DNACodonHashLoss aggregate
├── evaluation_siglip2.py       # retrieval / collapse metrics
├── extraction_siglip2.py       # in-training feature extraction
├── extract_siglip2_features.py # offline cache builder
├── preprocess_qwen_codebook_texts.py
├── interpret_compositional_code.py
└── train_siglip2.py            # main training entry point
```

Result directories, parameter caches, slide assets, and the local dataset
folder are excluded from version control — see `.gitignore`.

## Quick start

1. Pre-extract SigLIP2 visual + text features (one-time per dataset):
   ```bash
   python extract_siglip2_features.py \
       --mode pathlist \
       --pathlist_root ./dataset/Flickr25k \
       --pathlist_setting setting1 \
       --qwen_cache_path ./cache/flickr25k_qwen.jsonl \
       --cache_dir ./cache/flickr25k_siglip2
   ```
2. Train the current best variant (HashNet-style loss, K=64):
   ```bash
   python train_siglip2.py \
       --tag flickr25k_setting1 \
       --dataset Flickr25k --setting 1 -bs 64 -e 60 --proj_lr 1e-3 \
       --qwen_text_cache_path ./cache/flickr25k_qwen.jsonl \
       --siglip2_feature_cache_dir ./cache/flickr25k_siglip2 \
       --codebook_size 64 --c_global_source siglip2_global \
       --router_type sinkhorn --lambda_hash_type hashnet \
       --eval_every 10 --dna_distance_mode base -ev
   ```

Results land under `result/<run-tag>/`.
