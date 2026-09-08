# Audited modern image-hashing baselines

This directory contains clean-room, 36-/48-bit matched-protocol implementations
for recent uni-modal image-hashing work.  They are comparison runners, not
claimed reproductions of published tables: GroundedDNA fixes one frozen CLIP
ViT-B/16 cache, its own train/query/database manifests, 36 or 48 output bits,
raw 18- or 24-base Hamming selection, and the common biological
post-projection.

No result is entered into the paper until a full P0 refit is run.  Missing
results remain `-`; published 16/32/64-bit numbers are not interpolated.

`PROJECT_LOG.md`와 draft에 붙는 `-cache`, `-cache2v`, `-cache3v`, `-probe`
같은 접미사는 논문명의 일부가 아니라 이 저장소의 matched-protocol
runner 태그다. 예를 들어 `CroVCA-cache2v-probe`는 CroVCA 논문을 두
개의 고정 cache view와 frozen-backbone probing으로 적응한
실험을 뜻한다.

## Information conditions

The following are repository-local audit tags, not supervision names claimed
verbatim by every source paper.

| Tier | Training-time information | Implemented runners |
|---|---|---|
| VLM-T | target ground-truth labels and benchmark taxonomy are absent from the representation objective, but VLM-generated instance captions and a frozen text encoder provide text supervision | GroundedDNA |
| U0 | target labels, benchmark class names, captions, and external text banks are all absent | CIBHash, CIMON, MLS³RDUH, UGH (GreedyHash unsupervised setting), Bi-half, SDC, HHCH, CroVCA, OH |
| U0-FD | native-DNA subtag of U0: target labels, taxonomy, and captions are absent from the encoder objective; pseudo-pair targets come from frozen optimization-train visual-feature distances | DNA24, PRIMO |
| U1 | no target labels/taxonomy, but generic external knowledge is used | DUH-EG only after the authors' ordered WordNet bank provenance is established |
| U2 | no instance labels, but benchmark class-name taxonomy is used | UMRCH; any DUH-EG bank matching a target taxonomy |
| U? | an external term source or its selection provenance is not auditable | current user-supplied DUH-EG adapter |
| S | target labels are consumed directly by the training objective | CRH |

`VLM-T`, the U tiers/subtags, and the supervised `S` panel must be explicit
columns or separate tables. Calling every target-label-free method simply
“unsupervised” would hide materially different information access:
GroundedDNA is `VLM-T`, not visual-only `U0`; DNA24/PRIMO are `U0-FD`; and CRH
is **not** an unsupervised baseline. For
`VLM-T`/U0/U0-FD/U1/U2/U?, the common P0 protocol uses held-out training
labels only to score validation retrieval and fix E* and never feeds those
labels into the representation objective. CRH necessarily feeds labels into
its supervised class-center objective.

## Implemented and blocked variants

| Runner | Source-matched core | Matched-protocol boundary |
|---|---|---|
| `cibhash` | paired-view contrastive learning on a stochastic binary representation plus the symmetric Bernoulli information-bottleneck regularizer | two fixed augmented cached globals replace online image augmentation and the original image backbone |
| `cimon` | globally refined/statistically weighted pseudo-similarity, parallel/cross semantic consistency, and contrastive consistency | fixed cached canonical/augmented globals replace the paper's pretrained VGG-F feature and online image paths |
| `mls3rduh` | paper-defined manifold local-similarity reconstruction, paper `o=0.06N` neighborhood policy, Xavier hash projection, momentum SGD, and log-cosh hashing loss | the canonical frozen cache replaces the paper's pretrained image feature path; the runner is explicitly `paper-cache` because the public implementation differs in neighborhood count, initialization, and optimizer momentum |
| `greedyhash` | official UGH one-linear-layer head, hard `sign` forward/identity backward, shuffled half-pair cosine-similarity MSE, and cubic quantization penalty | deterministic frozen cache replaces frozen VGG16-fc7 whose classifier remained in train-mode dropout; the official unsupervised experiment reports CIFAR-10 and 16/32/64 bits, not the repository's four-dataset 36-bit setting |
| `bihalf` | released per-bit descending rank/scatter assignment, exact proxy gradient, half-pair cosine-similarity MSE, and inference-time `sign` | frozen matched cache replaces frozen VGG16; paper and release disagree on the proxy coefficient, and the release has no NUS-WIDE training script |
| `sdc-release` | public default D→D→36+BN, SDC reconstruction and cosine quantization; no contrastive term | canonical cached global feature |
| `sdc-release-simclr` | public executable D→D head; four-block within-view SDC pairs on both views, clipped target, NT-Xent | two fixed cached augmented globals |
| `sdc-paper` | paper Algorithm 1: 4096-hidden head; SDC + quantization on the first view, NT-Xent across both, unclipped target | two fixed cached augmented globals |
| `hhch` | paper Poincaré projection, K-Means++, Einstein midpoint, HIC/HPC, LogCosh quantization; official release LR schedule supplies the paper-unreported schedule | cached canonical + two augmented globals replace frozen VGG/image transforms |
| `crovca` | small HashCoder, symmetric stop-gradient BCE, executable Algorithm-1 coding-rate loss, AdamW/cosine, five epochs | frozen-cache probing; not the paper table's LoRA/asymmetric-Hamming setting |
| `oh` | release-executable 0/1 sigmoid-STE hash head, normalized continuous head, EMA key encoder, P/B/C FIFO queues, overview attention, two contrastive CE terms, and TensorFlow-matched Glorot/Adam-epsilon defaults | two cached augmented globals replace the online ResNet-50/pixel views; extractor sees the equivalent signed code `2b-1` |
| `duheg` | released external-guidance and three-way multi-positive objective | user-supplied noun-bank adapter only; arbitrary nouns are never accepted as an exact paper result |
| `umrch` | released CLIP global/local semantic reconstruction objective | benchmark taxonomy is U2; cached pre-LN patch tokens receive the exact CLIP post-LN/projection adapter |
| `crh-supervised` | paper-equation codebook-centric learning with `M=2C` fixed and local uniformly sampled candidate subcodes; label-guided class-to-candidate reassignment composes dynamic class centers while margin classification learns the hash function, with tanh-L2 quantization weighted `.1` for single-label and `0` for multi-label datasets | **supervised S panel only**; the release-executable smallest-divisor head rule is used at the repository's novel 36/48-bit budgets, a frozen CLIP cached global replaces the paper's pixel/ResNet-34 input, and a linear hash head precedes common DNA conversion/projection |
| FSCH | regional-common/global-local method is relevant U0 prior work | **blocked**: public repository lacks trainer, loss, optimizer, and entry point and contains model defects |

`baseline/MBE.py` is a retained, unregistered historical prototype.  It lacks
the released Bi-half proxy gradient and inference separation and must not be
used for comparison runs; `--method bihalf` resolves only to `baseline/BiHalf.py`.

The source boundary matters.  SDC's paper and release, and HHCH's paper and
release, make different choices.  The runner names prevent a silent hybrid.
Bi-half likewise uses the executable image-release proxy
`dL/dU = dL/dB + 6(U-B)/(NK)`; the paper states coefficient 3.  The released
Flickr25K, MSCOCO, and CIFAR-10 profiles are respectively
`100 epochs/1e-4/step 60`, `150/1e-3/60`, and `300/1e-4/120`, all with batch
size 32. NUS-WIDE uses the Flickr/paper profile as an explicitly marked
common-cache adaptation because the public repository supplies no NUS-WIDE
trainer; its admitted diagnostic cells carry `‡`.
OH's release is CIFAR-10-only and consumes 50,000 repeated rows per epoch;
the fair P0 adapter instead makes one finite pass over the repository's
designated optimization split. Its FIFO dictionary still has the released
length 4096, but is an executable approximation to the paper's “with all”
language, not a literal full-dataset matrix.

## Audited primary sources

- [CIBHash paper (IJCAI 2021)](https://www.ijcai.org/proceedings/2021/133).
  - **논문 요약:** CIBHash는 입력 복원 중심의 비지도 해싱이 배경처럼
    검색에 불필요한 정보까지 보존하는 문제를 지적한다. 두 증강 view의
    이진 표현을 contrastive objective로 정렬하고, 확률적 binary layer와
    information-bottleneck regularization을 결합하여 판별적인 의미는
    보존하되 불필요한 입력 정보는 억제한다.
- [CIMON paper (IJCAI 2021)](https://www.ijcai.org/proceedings/2021/125).
  - **논문 요약:** CIMON은 pretrained feature에서 만든 국소
    pseudo-similarity에 false positive/negative와 불확실한 pair가 섞이고,
    증강 교란에 hash code가 불안정해지는 문제를 함께 다룬다. 전역
    refinement와 통계적 confidence weighting으로 similarity guidance를
    다듬고, parallel/cross semantic consistency와 contrastive consistency를
    학습해 견고하고 판별적인 코드를 만든다.
- [MLS³RDUH paper (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/479).
  - **논문 요약:** MLS³RDUH는 단순 k-nearest-neighbor graph에 의미가
    다른 noisy neighbor가 포함되어 hash supervision을 훼손하는 문제를
    다룬다. feature manifold의 상호 이웃 구조와 cosine similarity로 국소
    의미 관계를 재구성하고, 그 similarity를 log-cosh hashing loss의
    guidance로 사용해 compact binary code를 학습한다.
- [GreedyHash paper](https://papers.neurips.cc/paper_files/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html)
  and [repository](https://github.com/ssppp/GreedyHash), commit
  `e5089b2354b8f283d662bb0584f1b006faf2c2db`.
  - **논문 요약:** GreedyHash는 `sign` 활성화의 이산 제약과 소실
    기울기 때문에 심층 해싱을 직접 최적화하기 어렵고, 연속 완화에는
    양자화 오차가 남는 문제를 다룬다. 전방향에서는 엄격한 이진 부호를
    생성하되 역방향에서는 기울기를 그대로 전달하는 hash coding layer를
    사용하여, 반복마다 유력한 이산 해를 향해 네트워크를 탐욕적으로
    갱신한다.
- [Bi-half paper](https://ojs.aaai.org/index.php/AAAI/article/view/16296)
  and [repository](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing),
  commit `1cd671a33fa010e24fdb36b625837dea9858301f`.
  - **논문 요약:** Bi-half는 라벨 없는 해싱에서 각 bit가 한 값으로
    치우쳐 이진 코드의 정보량이 줄어드는 문제를 bit entropy 최대화로
    다룬다. 별도의 entropy loss 대신 각 bit의 두 값이 절반씩 나타나는
    분포를 강제하는 parameter-free Bi-Half layer를 두며, 이를 연속 특징과
    이상적인 half-half 분포 사이의 penalized Wasserstein distance
    최소화로 해석한다.
- [SDC paper](https://papers.bmvc2023.org/0053.pdf) and
  [repository](https://github.com/kamwoh/sdc), commit
  `9981beb206f6be8199185bd6fcd92eb1b597a00d`.
  - **논문 요약:** SDC는 연속 특징의 편향된 pairwise similarity를
    제한된 Hamming similarity 값으로 옮길 때 양·음성 관계가 충분히
    분리되지 않는 similarity collapse를 다룬다. 개별 유사도를 일대일로
    복원하는 대신 hash similarity의 전체 분포를 충분히 퍼진 대칭 Beta
    calibration distribution에 Wasserstein loss로 맞추고, contrastive 및
    quantization objective와 함께 학습한다.
- [HHCH paper](https://pubmed.ncbi.nlm.nih.gov/38442063/) and
  [repository](https://github.com/HUST-IDSM-AI/HHCH), commit
  `47eac18aedf3a2f8623002a702da28aec8aafd0b`.
  - **논문 요약:** HHCH는 실제 이미지 데이터에 내재한 계층 구조를
    기존 self-supervised hashing이 활용하지 못해 데이터 관계를 정확히
    보존하기 어려운 문제를 다룬다. 연속 hash code를 Poincaré ball에
    투영하고 hyperbolic K-Means로 계층을 적응적으로 구성한 뒤,
    hierarchical instance-wise 및 prototype-wise contrastive learning으로
    그 구조를 코드에 반영한다.
- [CroVCA paper](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Moummad_Image_Hashing_via_Cross-View_Code_Alignment_in_the_Age_of_CVPRW_2026_paper.html)
  and [repository](https://github.com/ilyassmoummad/cross-view-code-alignment),
  commit `c59aa7cb981777a27e0bc311fc511761bd6e783c`.
  - **논문 요약:** CroVCA는 foundation-model embedding의 높은 검색
    비용을 줄이면서 기존 해싱의 복잡한 다중 목적과 긴 학습 절차를
    피하는 문제를 다룬다. 의미적으로 정렬된 두 view의 이진 code를
    단일 BCE objective로 맞추고 coding-rate maximization으로 collapse를
    억제하며, 마지막 batch normalization을 둔 경량 MLP HashCoder를
    사용한다.
- [OH paper](https://doi.org/10.1145/3581783.3611977) and
  [repository](https://github.com/RosieYuu/OH), commit
  `181691b2adc6fa0767a27a589e034de871764e7b`.
  - **논문 요약:** OH는 mini-batch 안의 관계만으로 학습할 때 한
    이미지와 전체 데이터 사이의 전역 구조를 충분히 이용하기 어려운
    문제를 다룬다. EMA key encoder와 binary·continuous·overview
    memory queue를 유지하고, hash-space affinity attention으로 continuous
    memory를 집계한 overview representation에 queue-level 및 in-batch
    contrastive objective를 적용한다.
- [DUH-EG paper](https://proceedings.mlr.press/v267/song25h.html) and
  [repository](https://github.com/XLearning-SCU/2025-ICML-DUHEG), commit
  `30757367c06d24c351bcc610f637dd1ff9d3f867`.
  - **논문 요약:** DUH-EG는 이미지 내부의 시각 정보만 활용하는
    비지도 해싱이 갖는 의미 판별력의 상한을 외부 텍스트 지식으로
    넘으려 한다. 외부 말뭉치에서 중복이 적은 대표 명사를 선택해
    이미지와 대응되는 external feature를 만들고, 내부·외부 공간의 hash
    code 사이 agreement를 양방향 contrastive learning으로 최대화한다.
- [UMRCH paper](https://www.sciencedirect.com/science/article/pii/S0957417425040291)
  and [repository](https://github.com/Lab201A/UMRCH), commit
  `5e60e8d3d35e435506f86dd135e90763c07b5592`.
  - **논문 요약:** UMRCH는 여러 객체가 공존하는 multi-label
    이미지에서 단일 global feature나 단순 pseudo-cluster가 복합적인
    의미 관계를 표현하지 못하는 문제를 다룬다. vision-language model의
    global·local patch feature와 text semantics를 집계해 multi-semantic
    pseudo-label을 만들고, 이를 이용한 similarity reconstruction과
    contrastive learning으로 hash code를 학습한다.
- [CRH paper (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/38190)
  and [repository](https://github.com/iFamilyi/CRH), audited commit
  `bc3efd3757501f11d4f308be81167be7e2bd5342`.
  - **논문 요약:** CRH는 무작위로 고정한 class-to-center 대응이
    클래스 사이의 의미 관계를 반영하지 못하고, 별도의 semantic-center
    사전 최적화는 계산 비용과 two-stage mismatch를 만든다는 문제를
    다룬다. 클래스 수 `C`에 대해 각 head에 `M=2C`개의 서로 다른 이진
    candidate subcode를 두고, label-guided adaptive assignment가
    클래스별 subcode 조합을 반복 갱신해 하나의 full class center를
    구성한다. Margin classification이 neural hash function을 학습하며,
    tanh-L2 quantization은 single-label에서만 가중치 `.1`로 더해지고
    multi-label에서는 가중치 `0`으로 기록만 된다. 클래스 라벨이 center
    reassignment와 classification loss에 직접 들어가므로 이 방법은
    **supervised S baseline**이다.
- [FSCH record](https://dblp.org/rec/journals/tcsv/CaoHNW24) and
  [incomplete repository](https://github.com/huanglab-research/FSCH), commit
  `66f5f3697038ed1c181905ab7a44aa54992b6085`.
  - **논문 요약:** FSCH는 global-only image representation에 의존한
    similarity mining이 배경이나 비관심 객체의 간섭을 받아 세밀한
    의미 관계를 놓치는 문제를 다룬다. patch matching으로 얻은 local
    pairwise structure와 global hash similarity의 일관성을 보존하고,
    증강 view 사이의 common regional feature를 대조하여 local
    fine-grained similarity를 학습한다.

No `LICENSE`/`COPYING` file was found in these audited snapshots.  Therefore
the repository does not vendor or copy their programs; the modules are written
from published equations and independently tested against observable behavior.
For CRH specifically, no upstream license file was detected at the audited
commit. Its runner is therefore a clean-room, paper-equation matched-cache
adapter, **not** a vendored copy or a claim of exact released-code
reproduction. The release does not ship the authors' sampled codebook
realization, and its persistent unseeded/reused codebook state plus an
undocumented best-of-1000 center-candidate heuristic prevent byte-for-byte
reproduction of their exact centers.

## Fair P0 execution

Use the repository environment explicitly:

```bash
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python

$PY scripts/run_modern_baseline_p0.py \
  --variant greedyhash --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant bihalf --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant sdc-release --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant sdc-paper --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant sdc-release-simclr --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant hhch --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant crovca --dataset Flickr25k --device cuda:0

$PY scripts/run_modern_baseline_p0.py \
  --variant oh --dataset Flickr25k --device cuda:0
```

CRH is launched and reported separately because it is supervised:

```bash
$PY scripts/run_modern_baseline_p0.py \
  --variant crh-supervised --dataset Flickr25k --device cuda:0

$PY scripts/run_baseline_p0_matrix.py \
  --panel supervised \
  --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
  --bits 36 48 --seeds 42 43 44 --gpus 0
```

This command is the repository's fair matched-cache adapter: it retains the
paper-equation CRH codebook/assignment/loss design while replacing the original
pixel/ResNet-34 feature path with the same frozen CLIP global cache used by the
other comparison runners. Its 36-/48-bit outputs then follow the identical
bit-pair DNA encoding and common biological post-processing. It must be
reported under S, never U0/U1/U2.

The paper/release evaluates Stanford Cars, NABirds, and MS COCO. Thus only the
matched MSCOCO row overlaps the original dataset scope; Flickr25K, NUS-WIDE,
and CIFAR-10 are explicit cross-dataset adaptations. For the four matched
datasets, the release-executable “smallest divisor `d` with `2^d >= M`” rule
gives `(subcode dimension d, number of heads H)` as follows:

| Dataset | 36 bits | 48 bits |
|---|---:|---:|
| Flickr25K | `(6,6)` | `(6,8)` |
| MSCOCO | `(9,4)` | `(8,6)` |
| NUS-WIDE | `(6,6)` | `(6,8)` |
| CIFAR-10 | `(6,6)` | `(6,8)` |

This follows the public helper rather than silently forcing the paper prose's
“power-of-two `d`” wording at code lengths not studied in the original paper.
Flickr25K contains zero-label training rows, for which CRH's label-normalized
objective and reassignment cost are undefined. The adapter therefore filters
training-loader rows only: stage 1 uses `4,424/4,500` rows and scratch refit
uses `4,919/5,000`; validation, query, and database rows are not filtered.

The driver performs the following sequence by construction:

1. Stage 1 holds out 10% with `val_split.carve_val_indices(..., seed=42)`.
2. Test and official database are not loaded in stage 1.
3. E* is selected by raw base-Hamming mAP@R in the applicable 18- or 24-base
   space; bit-Hamming is diagnostic only. Candidate cadence is fixed before
   test evaluation and recorded in the run manifest.
4. Stage 2 starts from scratch and trains on the full designated train split
   for exactly `E*+1` epochs, while retaining the method's nominal stage-1
   schedule horizon. Early stopping therefore does not compress a cosine
   schedule into the shorter refit.
5. One final refit checkpoint is fixed without test feedback. Raw and
   post-projection test metrics are then computed from that same checkpoint;
   this legacy diagnostic runner may reopen the fixed test split during
   independent selector verification and code extraction, but cannot change
   any model or epoch choice. A future strict-P0 run must instead reuse one
   sealed extraction without reopening the test loader.
6. Query/database codes are packed `00,01,10,11 → A,C,G,T`, projected by the
   same minimum-Hamming biological DP, and evaluated with stable tie ordering.

The driver hashes its variant, protocol, cache metadata/image IDs, dataset
splits, relevant implementation files, and semantic manifests into every trial
name. It refuses to reuse any existing model/result/extraction directory, and
the selector rechecks every candidate checkpoint plus the final checkpoint
before recomputing metrics. This prevents stale epochs or JSON files from being
silently mixed into a run.

Repeat with the declared seed set for the final mean±standard-deviation table.
The current paper draft requires three seeds; a one-run smoke test is not a
paper result.

Current CLIP cache choices are encoded in the driver and can be overridden by
`--cache-dir`.  The runner fails immediately if a method-defining augmented
global or local-token file is missing; it never substitutes the canonical view.
The shared cache uses CLIP ViT-B/16 and fixed precomputed views. These are
controlled adapters, not published-table reproductions: for example UMRCH's
release uses ViT-B/32/49 patches and its own online transforms, whereas the
matched cache supplies ViT-B/16/196 patches and the repository's fixed views.

The currently checked-in default cache metadata predates the strict contract:
it omits at least the exact canonical/augmentation specification, augmentation
seed, or immutable Hugging Face revision/weight provenance. It remains usable
for explicitly labelled main-ineligible diagnostic matrices and integration
smokes only with `--allow-main-ineligible-smoke`; otherwise the runner stops
before training. Its manifest remains main-ineligible. Main-table runs require
regenerating the cache with the current extractor; consumed NPY files are then
content-bound into the protocol and every checkpoint. The archived 78-cell
U0/U2 matrix predates the sealed exact-once guard. The newer CRH supervised
matrix uses the same legacy diagnostic test-access allowance: after all
choices are fixed, selector verification and independent code extraction may
reopen the fixed test split. Neither matrix therefore establishes sealed
exact-once compliance, irrespective of the additional legacy-cache provenance
blockers.

## Current matched-protocol comparison

All values are common post-processed DNA mAP@R after raw-code validation
selection. U0/U2 numbers come from
`docs/baseline_p0_matrix_seeds42_legacy_cache.json` and are **single-seed
(`42`) diagnostics**. The separately reported CRH S panel comes from
`docs/baseline_p0_matrix_seeds42-43-44_supervised_legacy_cache.json` and is
the mean ± sample standard deviation over seeds `{42,43,44}`. Both aggregates
use legacy caches and have `main_protocol_eligible=false`; `†` prevents
accidental promotion into the paper main table. A final U0 paper table still
requires strict-provenance three-seed reruns.

### U0 visual-only — 15 bases (30 bits), D6 `author_fixed_final` — PAPER MAIN PANEL

This is the only panel that matches the paper's 5-slot / 15-base / 30-bit
geometry. Unlike every table below it, these cells are **not** `†` diagnostics:
all 105 completed cells are `main_protocol_eligible` with zero blockers and zero
protocol deviations, on the provenance-strict `groundeddna_cache_v6prov` cache.
Three seeds `{42,43,44}`, mean<sub>±sample sd</sub> of post-projection
base-Hamming mAP@R. Source of truth:
[JSON](../docs/baseline_p0_matrix_seeds42-43-44_author_fixed_30b.json) ·
[Markdown](../docs/baseline_p0_matrix_seeds42-43-44_author_fixed_30b.md).

| Method | Flickr25K @5000 | MS-COCO @5000 | NUS-WIDE @5000 | CIFAR-10 @1000 |
|---|---:|---:|---:|---:|
| CIBHash | 0.7242<sub>±0.0039</sub> | 0.7692<sub>±0.0076</sub> | 0.7449<sub>±0.0085</sub> | 0.8164<sub>±0.0056</sub> |
| CIMON | 0.8141<sub>±0.0034</sub> | 0.6845<sub>±0.0012</sub> | 0.7980<sub>±0.0046</sub> | 0.8746<sub>±0.0055</sub> |
| MLS³RDUH (paper-cache) | 0.7507<sub>±0.0081</sub> | 0.6301<sub>±0.0061</sub> | 0.7590<sub>±0.0026</sub> | 0.6226<sub>±0.0396</sub> |
| GreedyHash-UGH | 0.6484<sub>±0.0062</sub> | 0.5621<sub>±0.0060</sub> | 0.6504<sub>±0.0146</sub> | 0.1059<sub>±0.0000</sub> ✗ |
| Bi-half | 0.8158<sub>±0.0119</sub> | 0.7191<sub>±0.0037</sub> | **-** | 0.7598<sub>±0.0023</sub> |
| SDC-paper | 0.7267<sub>±0.0034</sub> | 0.8114<sub>±0.0031</sub> | 0.7692<sub>±0.0031</sub> | 0.7856<sub>±0.0095</sub> |
| OH | 0.8366<sub>±0.0057</sub> | 0.7653<sub>±0.0103</sub> | 0.8053<sub>±0.0019</sub> | 0.8665<sub>±0.0096</sub> |
| HHCH | 0.6144<sub>±0.0218</sub> | 0.4102<sub>±0.0126</sub> | 0.3844<sub>±0.0217</sub> | 0.2794<sub>±0.0833</sub> |
| CroVCA-cache2v-probe | 0.7715<sub>±0.0017</sub> | 0.8146<sub>±0.0159</sub> | 0.8002<sub>±0.0041</sub> | 0.8916<sub>±0.0065</sub> |

**`-` (Bi-half / NUS-WIDE) is a refusal, not a missing run.** The public release
ships no NUS-WIDE training script, so the cell would require running the Flickr
profile as an adapter. `run_modern_baseline_p0.py:1165` blocks that from the main
comparison (`bihalf_public_release_has_no_nuswide_training_script;
paper_flickr_profile_adapter`), and all three seeds refuse before training. The
`‡` adaptation reported in the 36-/48-bit tables below is exactly what this
protocol declines to do; those `‡` values are not comparable to this panel.

**`✗` (GreedyHash / CIFAR-10) marks a degenerate run, not a weak one.** Its
database code diversity is `dna_unique = 0.000017`, i.e. **one distinct code for
all 59,000 database images**, identically across all three seeds; 0.1059 is
simply the CIFAR-10 class prior. It must not be read as a functioning baseline
at this budget. The 36-/48-bit tables below do not show this collapse
(`0.1546†` / `0.2379†`), so it is specific to the 30-bit budget.

Code diversity is worth reading beside mAP@R throughout this panel. Mean
post-projection `dna_unique` on the database split:

| | Flickr25K | MS-COCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.933 | 0.588 | 0.665 | 0.642 |
| CIMON | 0.699 | 0.335 | 0.370 | 0.149 |
| MLS³RDUH | 0.618 | 0.449 | 0.498 | 0.012 |
| GreedyHash-UGH | 0.225 | 0.120 | 0.098 | **0.000017** |
| Bi-half | 0.489 | 0.174 | - | 0.081 |
| SDC-paper | 0.918 | 0.476 | 0.611 | 0.848 |
| OH | 0.512 | 0.217 | 0.236 | 0.082 |
| HHCH | 0.021 | 0.004 | 0.003 | 0.001 |
| CroVCA-cache2v-probe | 0.866 | 0.372 | 0.551 | 0.268 |

HHCH's `0.001`–`0.021` explains its mAP@R directly: it emits very few distinct
codes on every dataset. These ratios are computed on the **database** split
(107,218 MS-COCO / 59,000 CIFAR-10 rows), never on the query split.

The matrix is 36 cells; 35 are `complete_paper_table_eligible` with all three
seeds, and the single `missing` cell is exactly Bi-half/NUS-WIDE. Filling it
would itself be the protocol violation. `--require-paper-eligible` therefore
cannot pass: it expects a 108-cell rectangle and cannot express a structurally
excluded pair. Its nonzero exit reports the matrix's non-rectangularity, not an
incomplete table.

### U0 visual-only — 18 bases (36 bits)

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.7824† | 0.7739† | 0.7829† | 0.8968† |
| CIMON | 0.8165† | 0.6751† | 0.7928† | 0.8478† |
| MLS³RDUH (paper-cache) | 0.7577† | 0.6311† | 0.7584† | 0.6409† |
| GreedyHash-cache | 0.6077† | 0.5639† | 0.6511† | 0.1851† |
| Bi-half | 0.8161† | 0.7062† | 0.7489†‡ | 0.7581† |
| SDC-paper | 0.7230† | 0.8185† | 0.7520† | 0.8442† |
| OH | 0.8362† | 0.7587† | 0.8023† | 0.8737† |
| HHCH | 0.5867† | 0.4709† | 0.4329† | 0.2992† |
| CroVCA-cache2v-probe | 0.7682† | 0.8257† | 0.7944† | 0.8819† |

### U0 visual-only — 24 bases (48 bits)

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.8077† | 0.7894† | 0.8100† | 0.9054† |
| CIMON | 0.8261† | 0.6877† | 0.8096† | 0.8576† |
| MLS³RDUH (paper-cache) | 0.7576† | 0.6414† | 0.7766† | 0.5780† |
| GreedyHash-cache | 0.6234† | 0.5693† | 0.6731† | 0.2379† |
| Bi-half | 0.8207† | 0.7164† | 0.7571†‡ | 0.7582† |
| SDC-paper | 0.7427† | 0.8410† | 0.7854† | 0.8700† |
| OH | 0.8467† | 0.7740† | 0.8139† | 0.8797† |
| HHCH | 0.6209† | 0.5084† | 0.4735† | 0.3189† |
| CroVCA-cache2v-probe | 0.7634† | 0.8344† | 0.7967† | 0.8739† |

### U2 taxonomy-assisted — 18 bases (36 bits)

UMRCH consumes the exact target benchmark taxonomy and is excluded from U0.

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| UMRCH | 0.7994† | 0.8009† | 0.8224† | n/a (release scope) |

### U2 taxonomy-assisted — 24 bases (48 bits)

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| UMRCH | 0.7993† | 0.8237† | 0.8336† | n/a (release scope) |

### Unresolved or implementation-blocked — 18 bases (36 bits)

These rows are shown explicitly so that a missing paper-faithful result cannot
be mistaken for an omitted experiment.

| Method | Information condition | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---|---:|---:|---:|---:|
| DUH-EG | U?; authoritative ordered WordNet selection artifact required | - | - | - | - |
| FSCH | U0; upstream executable implementation required | - | - | - | - |

### Unresolved or implementation-blocked — 24 bases (48 bits)

| Method | Information condition | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---|---:|---:|---:|---:|
| DUH-EG | U?; authoritative ordered WordNet selection artifact required | - | - | - | - |
| FSCH | U0; upstream executable implementation required | - | - | - | - |

### S supervised — 18 bases (36 bits)

CRH consumes target labels during training and is never ranked as an
unsupervised result.

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CRH | 0.8628 ± 0.0068† | 0.8435 ± 0.0013† | 0.8532 ± 0.0025† | 0.9346 ± 0.0019† |

### S supervised — 24 bases (48 bits)

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CRH | 0.8816 ± 0.0068† | 0.8668 ± 0.0011† | 0.8634 ± 0.0037† | 0.9383 ± 0.0019† |

Legend: `†` = completed but main-table-ineligible diagnostic; `-` = no completed
and admitted result yet. DUH-EG remains blocked from exact aggregation because
the ordered selected-WordNet bank or an unambiguous selection specification is
not available. FSCH remains implementation-blocked because its public
repository lacks the trainer, loss, optimizer, and entry point and contains
material model defects. `‡` marks the Bi-half/NUS-WIDE common-cache adaptation
outside the datasets covered by the released training scripts.

The CRH [JSON aggregate](../docs/baseline_p0_matrix_seeds42-43-44_supervised_legacy_cache.json)
and [Markdown audit](../docs/baseline_p0_matrix_seeds42-43-44_supervised_legacy_cache.md)
are authoritative for its S-panel values: `24/24`
`complete_diagnostic_only`, strict `0/24`, and no missing, invalid, duplicate,
malformed, extraneous, source-excluded, or implementation-blocked cell. Every
CRH cell carries `cache_meta_missing_canonical_transform` and
`cache_meta_missing_immutable_hf_provenance`; the non-sealed diagnostic
test-reopen boundary described above is an additional reporting limitation.
CRH is not included in U0/U2 ranking or unsupervised SOTA margins.

## External semantic assets

The intended DUH-EG condition is U1 only when the authors' ordered selected
WordNet bank and its provenance are available. `duheg-selected` merely embeds
a supplied list and does not reproduce the missing selection artifact, so the
current arbitrary/user-supplied adapter is recorded as U?. If its semantic term
set matches a target benchmark taxonomy, it is conservatively reclassified U2:

```bash
$PY scripts/prepare_modern_hashing_text.py duheg-selected \
  --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
  --terms_file /path/to/audited_selected_wordnet_nouns.txt \
  --out_prefix artifacts/duheg_selected_nouns

$PY scripts/run_modern_baseline_p0.py \
  --variant duheg --dataset Flickr25k --device cuda:0 \
  --duheg-noun-embeddings artifacts/duheg_selected_nouns_embeddings.npy \
  --duheg-asset-manifest artifacts/duheg_selected_nouns_manifest.json \
  --duheg-allow-unverified-selection \
  --allow-main-ineligible-smoke
```

The preparation command records `duheg_selection_included=false`: the public
repository does not supply an unambiguous selected-noun artifact and its
selection code conflicts with the paper at a material branch. Consequently,
the explicit opt-in above is required by the common driver before it launches
training and is only a released-objective adapter smoke/run and
must not be labelled an exact DUH-EG paper baseline. A publishable exact row
remains blocked until the authors provide the ordered selected noun bank or an
unambiguous selection specification.

UMRCH is U2 and requires the exact ordered benchmark taxonomy.  The preparation
script verifies the official dataset-specific ordered taxonomy hash and records
its prompt, immutable Hugging Face model commit, model/tokenizer hashes, cache
metadata hash, output hashes, and dimensions:

```bash
$PY scripts/prepare_modern_hashing_text.py umrch \
  --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
  --terms_file /path/to/flickr25k_24_class_names.txt \
  --out_prefix artifacts/umrch_flickr25k

$PY scripts/run_modern_baseline_p0.py \
  --variant umrch --dataset Flickr25k --device cuda:0 \
  --umrch-concept-embeddings artifacts/umrch_flickr25k_embeddings.npy \
  --umrch-vision-adapter artifacts/umrch_flickr25k_vision_adapter.npz \
  --umrch-asset-manifest artifacts/umrch_flickr25k_manifest.json
```

Do not place DUH-EG or UMRCH in the strict U0 headline table.

## Tests and audit gates

```bash
$PY -m unittest tests.test_modern_unsupervised_baselines
$PY -m unittest tests.test_greedyhash_baseline
$PY -m unittest tests.test_bihalf_baseline
$PY -m unittest tests.test_hhch_baseline
$PY -m unittest tests.test_crovca_baseline
$PY -m unittest tests.test_oh_baseline
$PY -m unittest tests.test_crh_baseline
$PY -m unittest tests.test_crh_supervised_matrix
$PY -m unittest tests.test_baseline_checkpoint_protocol
$PY -m unittest tests.test_asset_provenance
$PY -m unittest tests.test_cache_provenance
$PY -m unittest tests.test_modern_driver_protocol
$PY -m unittest tests.test_clip_cache_extractor_safety
$PY -m unittest tests.test_bio_projection_pipeline
$PY -m unittest tests.test_native_dna_baselines
```

The tests cover paper equations, architecture/shape invariants, gradients,
hyperbolic clustering, complete method-specific checkpoint restoration,
base-Hamming versus bit-Hamming selection, and the stage-1 no-test boundary.
The pre-existing seven runnable variants have passed a one-epoch Flickr25K real-cache
train/held-out-validation/checkpoint smoke test. GreedyHash and Bi-half additionally
passed source-equation and gradient parity tests plus a one-epoch Flickr25K CPU
end-to-end smoke through held-out E* selection, scratch refit, official
query/database extraction, biological projection, and artifact SHA manifests.
Those runs used a one-epoch/nonstandard cadence and a legacy-provenance cache, so
they explicitly passed `--allow-main-ineligible-smoke` and their manifests
correctly mark them main-ineligible. This verifies integration,
not performance, and does not replace full-horizon, three-seed execution. OH
also passed a deliberately nonstandard one-epoch Flickr25K integration smoke
(reduced hidden/continuous/queue widths), independent held-out rescoring,
query/database extraction, and explicit-cell biological projection. Its source
defaults remain unit-checked, but no smoke number is a paper result. Custom-head
checkpoint reconstruction has been tested for all registered modern methods.
Earlier DUH-EG/UMRCH manifests passed real-cache integration, independent
raw-base validation rescoring, and (for UMRCH) extraction plus biological
projection, but those caches/manifests predate the strict v2 provenance
contract. They are diagnostic smoke evidence only; v2 assets and regenerated
caches must pass the complete gate before any main-table run.
