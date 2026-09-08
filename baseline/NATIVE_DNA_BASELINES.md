# Native-DNA direct-predecessor baselines

This directory now contains a clean-room PyTorch implementation of four
computational baselines:

| CLI method | Supervision regime | Paper lineage | Original protocol | Matched comparison |
|---|---|---|---|---|
| `bee2018` | **`U0-FD` — unsupervised; target-label-free encoder objective** | [Stewart et al., DNA24 2018](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/08/dna24.pdf) | Caltech-256, VGG-FC2→PCA-10, 30 nt, analytic Hamming→yield, NUPACK/wet-lab diagnostics | separate matched 18-/24-base runs with the same cached features/splits and projected base-Hamming |
| `bee2021` | **`U0-FD` — unsupervised; target-label-free encoder objective** | [Bee et al., Nature Communications 2021](https://www.nature.com/articles/s41467-021-24991-z) | OpenImages, VGG-FC2, 80 nt, alternating NUPACK-trained yield predictor, molecular candidate-set recall | separate 18-/24-base runs; requires an explicitly supplied official predictor and is labelled a frozen-predictor length-transfer adaptation |
| `koike2024` | **`S` — supervised** | [DATE 2024](https://past.date-conference.com/proceedings-archive/2024/DATA/478_pdf_upload.pdf) / [DAC 2024](https://doi.org/10.1145/3649329.3657320) | CIFAR, 80 nt, supervised semi-hard triplet, class-consensus DDH classification | same cached features/splits, 18 or 24 nt, per-image query, projected base-Hamming mAP@R |
| `koike2026` | **`S` — supervised** | [TCBB 2026](https://pubmed.ncbi.nlm.nih.gov/41824343/) | Koike triplet plus probability/HP/GC loss and an HP correction heuristic | same as above, followed by the common exact DP projection |

## Supervision taxonomy

The tags below are repository-local information-condition labels rather than
names claimed by the source papers.

| Tag | Explicit meaning | Methods |
|---|---|---|
| **`U0-FD`** | **Unsupervised with respect to the target benchmark; target-label-free encoder objective.** Target ground-truth labels, taxonomy, and captions do not enter the encoder objective; pseudo-pair targets come from frozen optimization-train visual-feature distances. This is feature-distance pseudo-supervision, not from-scratch self-supervision. | DNA24, PRIMO |
| **`S`** | **Supervised.** Ground-truth train labels directly define positive/negative relations in the encoder objective. | Koike DATE/DAC 2024, Koike TCBB 2026 |

As for the other common-P0 baselines, held-out labels are used only to score
validation retrieval and select \(E^*\). Therefore, `U0-FD` describes the
encoder objective's information condition, not a label-blind end-to-end
evaluation protocol.

PRIMO additionally consumes a frozen external thermodynamic-yield predictor.
That changes its information condition and reproduction boundary, but it does
not turn PRIMO into a target-label-supervised (`S`) method.

## 논문별 문제 설정과 핵심 방법

### Stewart et al., DNA24 2018

이 연구는 파일명을 미리 아는 key-based 접근을 넘어, DNA 데이터베이스 안에서 이미지 내용에 따른 근사 유사도 검색을 분자 수준으로 수행하는 문제를 다룬다. VGG16-FC2 특징을 PCA 10차원으로 축소한 뒤, 두 개의 shared-weight pointwise convolution과 fully connected layer로 각 이미지를 30개 위치의 4채널 softmax DNA 표현으로 변환한다. 이미지 쌍의 PCA 거리를 임계값으로 이진화하고, soft cosine Hamming distance를 NUPACK 기반 sigmoid hybridization-yield 근사에 연결해 cross-entropy로 학습하며, feature 영역과 ID 영역의 cross-talk을 줄이는 strand/query 설계까지 wet-lab으로 검증한다.

### Bee et al., Nature Communications 2021 (PRIMO)

PRIMO는 고차원 이미지 특징을 DNA로 직접 인코딩하면서도 비선형 hybridization 반응을 효율적으로 근사해, molecular similarity search를 160만 이미지 규모로 확장하는 문제를 다룬다. 고정 VGG16-FC2의 4096차원 특징을 2048차원 ReLU hidden layer와 80×4 softmax 출력으로 변환하고, 별도의 convolutional yield predictor가 두 서열 사이의 비정렬 3-mer local match를 이용해 hybridization yield를 예측한다. Encoder는 유사도 label과 예측 yield의 cross-entropy 및 position-wise entropy regularization으로, predictor는 이산화한 서열의 NUPACK yield로 번갈아 학습된다.

### Koike et al., DATE 2024

이 연구는 기존 DNA 이미지 검색의 feature-distance 대 yield 비교만으로는 시각적 검색 정확도를 직접 측정하기 어렵다는 점을 지적하고, class-query를 이용한 이미지 분류를 정량 평가 과제로 도입한다. CNN 특징을 dense layer와 80×4 softmax DNA 출력으로 보내고, 같은 class의 anchor-positive는 가깝게, 다른 class의 semi-hard negative는 margin 밖으로 밀어내도록 normalized DNA Hamming distance 기반 supervised triplet loss로 학습한다. 각 class의 학습 서열에서 위치별 최빈 염기를 모아 대표 query를 만들고, 가장 높은 DNA-DNA hybridization yield를 보이는 query의 class로 예측한다.

### Koike et al., DAC 2024

DAC 논문은 DATE 논문과 같은 문제 설정 및 triplet-network DNA encoder를 다루는 동일 계보의 확장 연구로, PRIMO의 반복적인 hybridization-simulator 학습을 supervised metric learning으로 대체해 검색 정확도와 학습 효율을 함께 높이는 데 초점을 둔다. Dense softmax encoder, semi-hard triplet mining, normalized DNA distance, class-consensus query와 DDH 기반 분류가 핵심 구조이며, 저자 [공식 코드](https://github.com/tkoike-kuee/dna-triplet-network)도 이 공통 방법을 구현한다. 따라서 이 문서에서는 DATE와 DAC를 별개의 성능 baseline으로 중복 집계하지 않고 하나의 `koike2024` 구현 계보로 취급한다.

### Koike et al., IEEE TCBB 2026

이 연구는 triplet-network DNA encoder의 검색 정확도와 학습 속도를 유지하면서, 생성 서열이 합성·hybridization에 불리한 긴 homopolymer와 치우친 GC 함량을 피하도록 만드는 문제를 다룬다. 기존 dense softmax encoder와 normalized-distance semi-hard triplet objective에 base probability를 명확하게 만드는 entropy/probability penalty, homopolymer loss, GC-balance loss를 결합해 end-to-end로 학습한다. 추론 시에는 과도한 homopolymer run 안의 낮은-confidence 위치에서 현재 염기를 억제하고 차선 염기로 바꾸는 저자 heuristic을 반복 적용하며, [공식 코드](https://github.com/tkoike-kuee/DNA-Encoder-under-Bioconstraints)는 이 출력을 별도로 생성한다.

DATE and DAC describe the same core method and must not be counted as two
independent baseline rows.  Koike methods use ground-truth train labels; the
paper table must mark them as supervised direct-prior baselines.  On
multi-label datasets, positive pairs are defined by any label overlap.  That is
an explicit adaptation, not part of the original CIFAR protocol.

## What is implemented

- `baseline/native_dna.py`
  - exact predecessor alphabet handling (`A,T,C,G` internally,
    `A,C,G,T` in repository artifacts);
  - DNA24 PCA/pointwise-convolution encoder, soft cosine Hamming, and the
    reported sigmoid yield approximation;
  - PRIMO dense encoder and local 3-mer interaction predictor;
  - PRIMO source-level balanced-pair pooling and random truncation (therefore
    each realized batch is not forcibly 50/50);
  - Koike masked-softmax DNA distance and semi-hard mining;
  - Keras activity-regularizer batch scaling and Keras-compatible Glorot
    initialization for the PRIMO/Koike dense encoders; DNA24's paper does not
    specify an initializer, so its convolution/linear layers retain the
    explicitly declared PyTorch defaults;
  - TCBB source-level entropy, probability, homopolymer, and GC losses;
  - TCBB homopolymer inference heuristic;
  - versioned common 18-/24-base extraction, base-Hamming evaluation, and
    minimum-edit DP projection.
- `scripts/train_native_dna_baseline.py`
  - held-out P0 selection (`val_query` against optimization-train DB);
  - full-train refit/evaluation path;
  - separate neural-raw, author-HP/deployment, and common-DP artifacts; the
    standard `extract_{query,db}.npz` is the deployment code immediately before
    common DP.
- `scripts/convert_primo_predictor.py`
  - converts the official Keras predictor weights to PyTorch layout and records
    source/output SHA-256; no NUPACK code is included or invoked.

## Matched 18-base sealed P0 run

Stage 1 selects a zero-based epoch `E*` using neural-raw 18-base validation
mAP@R (the shared P0 rule) and never instantiates the test/database splits.
The driver then performs a weight-free scratch refit on the full designated
train split for exactly `E*+1` epochs.  Its refit subprocess is the only code
allowed to extract the official query/database, and raw, author-postprocessed,
and common-DP metrics all come from that one extraction.

```bash
python3 scripts/run_native_dna_p0.py \
  --method koike2024 \
  --dataset Flickr25k \
  --seed 42 \
  --result-root /data/USER/groundeddna_native_p0/runs \
  --device cuda:0 \
  --allow-main-ineligible-diagnostic
```

The paper matrix fixes seeds `{42,43,44}`, all four datasets, length 18, raw
base-Hamming selection every five epochs, and each method's source horizon:

| Method | Stage-1 horizon | Batch/steps | Optimizer |
|---|---:|---|---|
| DNA24 | 65 epochs | 500 × 1000 steps/epoch | Adam, LR `1e-3` (declared reimplementation choice) |
| PRIMO | 100 epochs | 100 × 1000 steps/epoch | Keras-compatible Adagrad, LR `1e-3` |
| Koike DATE/DAC | 150 full epochs | 128 | Keras-compatible Adagrad, LR `.01` |
| Koike TCBB | 1000 full epochs | 128 | Keras-compatible Adagrad, LR `.01` |

The official PRIMO predictor used by the matched run is pinned to
`uwmisl/primo-similarity-search`, tag `pub`, commit
`5cf3656b163e1ae49f01b39e30a7fba985552cbf`:

- `yield-model.h5`: 53,600 bytes, SHA-256
  `fb81610f09a901e22e979c7b828d6226fa6166c5eb129289fc5de9040f8dc580`;
- converted `primo_yield_predictor.npz`: SHA-256
  `a17d51f2d288472f5a4ddc7aefd81c4d4737f8ae2dd67ceb8bc064092a98133f`.

```bash
python3 scripts/convert_primo_predictor.py \
  --keras_h5 /path/to/official/yield-model.h5 \
  --out /data/USER/groundeddna_native_p0/artifacts/primo_yield_predictor.npz

python3 scripts/run_native_dna_p0_matrix.py \
  --data-root /data/USER/groundeddna_native_p0 \
  --gpus 0 1 2 4 5 \
  --primo-predictor-npz \
    /data/USER/groundeddna_native_p0/artifacts/primo_yield_predictor.npz \
  --allow-main-ineligible-diagnostic

python3 scripts/aggregate_native_dna_p0.py \
  --root /data/USER/groundeddna_native_p0/runs
```

The official PRIMO predictor was calibrated for 80 nt at 21 °C and 1 nM.  Its
use at 18 nt is therefore named `PRIMO-18 frozen-predictor length-transfer`; a
physically calibrated `PRIMO-18` requires new train-only 18-mer thermodynamic
yields and alternating predictor refits.

DNA24's original similarity label is PCA-space Euclidean distance `<=0.2`, and
PRIMO's is VGG-FC2 distance `<=75`.  Those numbers are only valid in their
original feature spaces.  A matched cached-feature run therefore estimates a
threshold from optimization-train pairs only (controlled by
`--feature_positive_quantile`), records it in `config.json`, and never calibrates
on validation/test rows.  Pass `--feature_distance_threshold` only when the
feature space has a predeclared threshold.

## Matched 24-base sealed P0 run

The 24-base capacity uses a versioned driver, trainer, matrix launcher, and
aggregator so that the sealed 18-base implementation and artifacts remain
unchanged:

- `scripts/run_native_dna_p0_24.py`
- `scripts/train_native_dna_baseline_24.py`
- `scripts/run_native_dna_p0_matrix_24.py`
- `scripts/aggregate_native_dna_p0_24.py`

It retains the same datasets, seeds `{42,43,44}`, source horizons, losses,
optimizers, raw base-Hamming E* selection, scratch refit, and exact-once
official-test extraction. The capacity-specific biological contract is 24
bases, inclusive GC count `[10,14]`, and homopolymer run≤3. Its outputs live
under a separate root; the launcher fails closed if that root already contains
18-base cells.

```bash
python3 scripts/run_native_dna_p0_matrix_24.py \
  --data-root /data/USER/groundeddna_native_p0_24base \
  --gpus 0 1 2 3 4 \
  --primo-predictor-npz \
    /data/USER/groundeddna_native_p0/artifacts/primo_yield_predictor.npz \
  --allow-main-ineligible-diagnostic

python3 scripts/aggregate_native_dna_p0_24.py \
  --root /data/USER/groundeddna_native_p0_24base/runs \
  --out-json docs/native_dna_p0_24base_aggregate.json \
  --out-md docs/native_dna_p0_24base_aggregate.md
```

The unchanged official PRIMO predictor is an 80-nt model. Its use here is
therefore named `PRIMO-24 frozen-predictor length-transfer` and carries the
explicit blocker `primo_frozen_predictor_length_transfer_80_to_24nt`; it is
not presented as a physically recalibrated 24-mer PRIMO reproduction.

Both standalone aggregators enforce versioned, per-method protocol locks in
addition to the outer manifest digest. The lock binds the declared source
profile/audit, candidate grid, stage and biological contracts, run-era
implementation SHA mapping, stable execution sizes, and PRIMO predictor/
length-transfer metadata. Historical 18-base implementation SHAs and the
24-base v1 SHAs are pinned separately; old results are not invalidated by
rehashing an unrelated newer worktree, while a modified declaration cannot be
admitted merely by recomputing its outer protocol digest.

## 18-base sealed P0 results

The authoritative run root is
`/data/yschoi/groundeddna_native_p0_sealed`. All `48/48` cells completed for
four methods, four datasets, and seeds `{42,43,44}` with no launcher failure,
invalid manifest, duplicate, or missing cell. The generated
[Markdown aggregate](../docs/native_dna_p0_aggregate.md) and
[machine-readable JSON](../docs/native_dna_p0_aggregate.json) contain
checkpoint/evaluation SHA-256 values, seed-specific E*, raw and post-DP mAP,
database uniqueness, protocol-family hashes, and every strict-main blocker.

The historical CLIP caches do not contain complete immutable transform and
model-weight provenance. Consequently all 48 cells are diagnostic-only and
the strict paper table remains:

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| DNA24-18 analytic-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs | - | - | - | - |
| PRIMO-18 frozen-predictor length-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs + frozen external predictor | - | - | - | - |
| Koike DATE/DAC 2024 | `S`, ground-truth train labels | - | - | - | - |
| Koike TCBB 2026 | `S`, labels + bio-aware losses/HP heuristic | - | - | - | - |

The post-DP diagnostic results below are three-seed mean ± sample standard
deviation. `†` means diagnostic-only and is not a supervision marker.

### Unsupervised (`U0-FD`; target-label-free encoder objective) feature-distance-pair direct predecessors

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| DNA24-18 analytic-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs | 0.7808 ± 0.0071† | 0.6330 ± 0.0127† | 0.7427 ± 0.0055† | 0.7786 ± 0.0101† |
| PRIMO-18 frozen-predictor length-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs + frozen external predictor | 0.7882 ± 0.0209† | 0.6251 ± 0.0129† | 0.7320 ± 0.0109† | 0.7344 ± 0.0123† |

### Supervised (`S`) direct-prior baselines

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| Koike DATE/DAC 2024 | `S`, ground-truth train labels | 0.8353 ± 0.0047† | 0.5795 ± 0.0151† | 0.7453 ± 0.0419† | 0.8850 ± 0.0090† |
| Koike TCBB 2026 | `S`, labels + bio-aware losses/HP heuristic | 0.8916 ± 0.0037† | 0.6189 ± 0.0045† | 0.8156 ± 0.0009† | 0.9304 ± 0.0054† |

GroundedDNA's current `.8723/.8063/.8274/.9009` values are historical
single-run context, not a paired three-seed comparison. In particular, the
Koike rows are supervised direct priors and must not be ranked as U0 or
label-free baselines. PRIMO has the additional 80-mer-to-18-mer predictor
transfer boundary described above.

## 24-base sealed P0 results

The authoritative 24-base run root is
`/data/yschoi/groundeddna_native_p0_24base_sealed`. All `48/48` run records
completed for four methods, four datasets, and seeds `{42,43,44}` with no
launcher failure, invalid manifest, duplicate, unkeyed, blocked, or missing
record. All `16/16` method×dataset three-seed diagnostic aggregates are
complete. The generated
[Markdown aggregate](../docs/native_dna_p0_24base_aggregate.md) and
[machine-readable JSON](../docs/native_dna_p0_24base_aggregate.json) are the
numeric source of truth.

The historical CLIP caches do not contain complete immutable transform and
model-weight provenance. Consequently all 48 records are diagnostic-only and
the strict paper table remains:

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| DNA24-24 analytic-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs | - | - | - | - |
| PRIMO-24 frozen-predictor length-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs + frozen external predictor | - | - | - | - |
| Koike DATE/DAC 2024 | `S`, ground-truth train labels | - | - | - | - |
| Koike TCBB 2026 | `S`, labels + bio-aware losses/HP heuristic | - | - | - | - |

The post-DP diagnostic results below are three-seed mean ± sample standard
deviation. `†` means diagnostic-only and is not a supervision marker.

### Unsupervised (`U0-FD`; target-label-free encoder objective) feature-distance-pair direct predecessors

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| DNA24-24 analytic-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs | 0.7927 ± 0.0068† | 0.6412 ± 0.0110† | 0.7542 ± 0.0046† | 0.7814 ± 0.0090† |
| PRIMO-24 frozen-predictor length-transfer | `U0-FD`, target-label-free encoder objective; train-feature distance pairs + frozen external predictor | 0.7962 ± 0.0125† | 0.6326 ± 0.0049† | 0.7522 ± 0.0033† | 0.7569 ± 0.0085† |

### Supervised (`S`) direct-prior baselines

| Method | Information condition | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---|---:|---:|---:|---:|
| Koike DATE/DAC 2024 | `S`, ground-truth train labels | 0.8167 ± 0.0071† | 0.5788 ± 0.0203† | 0.7598 ± 0.0113† | 0.8812 ± 0.0049† |
| Koike TCBB 2026 | `S`, labels + bio-aware losses/HP heuristic | 0.8939 ± 0.0043† | 0.6359 ± 0.0128† | 0.8285 ± 0.0016† | 0.9295 ± 0.0014† |

The 24-base contract is inclusive GC count `[10,14]`, homopolymer run≤3,
and the same minimum-Hamming DP projection for query and database. DNA24
transfers its original 30-mer analytic yield mapping to 24 bases. PRIMO uses
the unchanged official 80-mer predictor and is therefore an 80-to-24-nt
length-transfer adaptation, not a physically recalibrated PRIMO-24
reproduction.

## 15-base (30-bit) panel — computed, but ineligible

**The paper's main panel is 15 bases / 30 bits, and these direct-predecessor
baselines have no admissible result at that length.** Only 18-base and 24-base
matrices are sealed above.

This is not an unfinished run. Measured 2026-09-08 at
`/data/yschoi/groundeddna_native_p0_15base_attempt_20260813T181511/runs`:

```
manifests            48
base_length          15   (48/48)
methods              bee2018 12 · bee2021 12 · koike2024 12 · koike2026 12
cells carrying metrics   48 / 48        <- the computation finished
main_protocol_eligible   False 48/48
blockers   legacy_cache_missing_strict_provenance      48
           primo_frozen_predictor_length_transfer      12
```

The matrix ran to the same 4×4×3 scale as the sealed 18-/24-base matrices and
produced metrics for every cell. What disqualifies it is **cache lineage**: all
48 cells ran on a legacy cache rather than the provenance-strict
`groundeddna_cache_v6prov`, so every one carries
`legacy_cache_missing_strict_provenance`. Three further partial roots exist
(`..._15base`, `..._15base_attempt_20260812T172527`, `..._15base_SMOKE`, with 3,
18 and 1 manifests); none is sealed and no 15-base aggregate document exists.

The remedy is therefore a **full 48-cell re-run on the strict cache**, not a
continuation. That is a deferred item, not a new blocker:
`docs/TODO_reexperiments.md` P0 #4 already records it as *"native-DNA 15-base 48
cells — full re-run after protocol-label fix · 48 cells · deferred
(diagnostic-only, below P0)"*, matching the cell count exactly.

Until that re-run, the 15-base predecessor rows stay `-`. This is the third
disclosed gap of its kind, alongside exact DUH-EG (no ordered noun bank) and
Bi-half/NUS-WIDE (no authored NUS-WIDE trainer). Each is a refusal to publish a
number the protocol cannot support, and each is stated rather than left blank.
The gap is independent of the U0 modern-baseline panel, whose completion is not
affected by it.

## Reproduction boundaries

- DNA24 did not release an official implementation and omits
  optimizer/seed/initializer details; this is an independent paper-based
  reimplementation. Adam at `1e-3` and PyTorch-default initialization for its
  convolution/linear layers are declared implementation choices.
- The PRIMO and Koike GitHub repositories expose source but no explicit code
  license.  Their source was inspected to audit equations; it was not copied
  into this repository.
- NUPACK/CuPyCK has separate restrictive terms.  It is an optional external
  thermodynamic oracle only and must not be vendored.
- Original DDH classification accuracy or molecular recall is not inserted in
  the matched image-query mAP table.  Original and adapted protocols are
  reported separately.
- A 30/80-nt run on the repository cache is recorded as
  `original_length_only_adaptation`; sequence length alone does not make it an
  original-protocol reproduction.
- TCBB artifacts distinguish neural argmax codes from the author's HP heuristic
  output.  Common DP projection consumes the latter; `soft_probs_atcg` aligns
  with `neural_raw_base_indices`.
- PRIMO/Koike use Keras-compatible Adagrad defaults
  (`initial_accumulator_value=0.1`, `epsilon=1e-7`); PyTorch defaults are not
  numerically equivalent.
- None of the audited predecessor protocols specifies gradient clipping, so
  the sealed driver explicitly passes and verifies `--grad_clip 0.0`.
- `koike2024` implements the fully symmetric normalized L1 distance stated in
  DATE Eq. (2) and used by the TCBB successor. The released DATE/DAC helper
  fills only its lower triangle and then adds an empty upper triangle; this
  apparent source-code symmetrization defect is documented in the manifest and
  intentionally not reproduced.
- The TCBB source monitors its test-fed validation stream with
  `EarlyStopping(patience=10, min_delta=1e-4)`. The matched P0 experiment
  instead uses the common train-only validation grid and scratch refit, so test
  labels cannot select a stopping point.
- The current historical CLIP caches omit immutable transform/model-weight
  provenance. Both sealed capacity matrices completed all `96/96` run records
  (`48/48` at 18 bases and `48/48` at 24 bases), producing `32/32` complete
  method×dataset×capacity three-seed diagnostic aggregates. All records carry
  `legacy_cache_missing_strict_provenance`, so strict-main eligibility is
  `0/96`.
- All `12/12` PRIMO-18 records additionally carry
  `primo_frozen_predictor_length_transfer`. All `12/12` PRIMO-24 records
  additionally carry
  `primo_frozen_predictor_length_transfer_80_to_24nt`.
- The saved `codebook_indices` are merely base-4 codon IDs for evaluator
  compatibility; these baselines do not learn GroundedDNA-style codebooks.
