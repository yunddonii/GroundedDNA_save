# Compositional code vs flat code — 정성적 interpretability 증명 방법 제안

작성일 2026-07-11.
목적: 우리 compositional 6-slot code (K=64 codewords/slot, 4^3=64 codon 매핑) 가 CIBHash 등 flat 36-bit sign hash 보다 **interpretability가 높다는 것을 정성적으로 보이는 방법** 3가지 제안. 최종 paper Figure 후보.

---

## 배경 — 왜 정성적 증명이 필요한가

정량 metric (NMI, B0/B1/B2 lift) 은 이미 우리 champion 이 baseline 대비 orders of magnitude 우세임을 보임 (Flickr NMI 0.55 vs baseline 0.19–0.39). 하지만 paper reviewer 는 "그래서 실제로 slot 이 무슨 의미를 학습하는가?" 를 시각적으로 확인하고 싶어함.

**본질적 차이:**
- **Ours**: code = `(cb0=42, cb1=17, cb2=8, cb3=53, cb4=21, cb5=6)` — 각 slot m 은 학습된 semantic axis (예: cb1 ≈ scene type, cb2 ≈ primary object, ...). 각 codeword k ∈ [0, K−1] 는 그 axis 위 discrete value.
- **CIBHash**: code = `0b101101...110001` — 36 bit 각각은 continuous projection 이 sign 을 취한 것으로, 어떤 bit 도 개별적 semantic 없음. Bit 7 = 1 인 이미지 집합은 아무 semantic theme 을 공유하지 않음.

이 구조적 차이를 3가지 방식으로 시각화 제안.

---

## Method A — Per-slot Codeword Atlas (가장 강력한 방법)

### 개념
"주어진 slot m 과 codeword k 에서, 그 codeword 로 라우팅되는 이미지들이 공통된 semantic property 를 공유하는가?"

### 프로토콜
1. Ours champion 에서 DB extraction 후 `codebook_indices` (shape `[N, 6]`) 획득.
2. Slot m ∈ {1, 2, 3, 4, 5} 각각에 대해:
   - Top-8 usage codeword k 를 frequency 로 선택.
   - 각 codeword k 마다 `codebook_indices[:, m] == k` 인 이미지 중 8-16 장 무작위 sample.
   - Grid layout: rows = codewords (8개), cols = sample images (8-16장).
3. **Baseline 비교**: CIBHash 36-bit hash 를 6개 6-bit chunk (chunk_m = bits[6m:6m+6]) 로 임의 분할.
   - Chunk m ∈ {0, 1, ..., 5} 마다 top-8 chunk value 를 선택.
   - 각 chunk value 별 8-16장 sample 표시.
   - 예상: sample 이 공통 semantic 없이 무작위로 보임.
4. Grid 를 side-by-side (Ours vs CIBHash) 로 정렬.

### 예상 결과 (paper Figure 후보)
- Ours: `(slot=2, codeword=17)` → 모두 새 이미지 (predator birds). `(slot=2, codeword=42)` → 모두 wading birds. `(slot=4, codeword=8)` → 모두 blue-dominant color.
- CIBHash: `(chunk=2, value=0x1A)` → 새/개/자동차/음식/사람 무작위 혼합.

### 구현 난이도
- **낮음 (2-3시간).** 필요한 것은 이미 있음:
  - Ours codebook_indices: `result/.../extract_db.npz`
  - CIBHash 36-bit: `result_baseline/.../extract_db.npz`
  - Image paths from cache metadata.
- 스크립트 하나 (`scripts/qualitative_codeword_atlas.py`) 로 Ours + CIBHash 각각 그리드 PNG 저장.

### 기존 자산 활용
- `codeword_concept_atlas/` 폴더가 이미 CIBHash Flickr 결과 dir 안에 존재 (line above `ls result_baseline/260527/cibhash_flickr25k_clip_unsup60/`). Ours 도 유사 구현 있는지 확인 후 재활용.

---

## Method B — Slot-swap Retrieval (compositional controllability 증명)

### 개념
Compositional code 의 대표적 특성 = **한 slot 만 바꿔 검색하면 그 slot 이 담당하는 semantic axis 만 변한다**. Flat hash 는 이런 controllability 가 없음.

### 프로토콜
1. Query 이미지 x → code `(c_0, c_1, c_2, c_3, c_4, c_5)`.
2. DB retrieval: `(c_0, c_1, c_2, c_3, c_4, c_5)` 튜플과 정확히 같은 이미지 top-3 표시 (그림 왼쪽 = anchor).
3. **Slot m swap**: `c_m → c_m'` (예: cb2 = "예/새 종" 을 다른 codeword 로) 로 하나만 바꾸고, 나머지 5개 slot 은 고정한 채로 근접 이미지 top-3 표시.
4. Slot m ∈ {1, ..., 5} 각각에 대해 반복. 총 5개 세로 스트립.
5. **Baseline 비교**: CIBHash 36-bit code 를 6개 6-bit chunk 로 나누고, chunk m 만 다른 값으로 flip 후 최근접 이미지 top-3.

### 예상 결과
- Ours: cb2 swap → "새 → 개" (다른 primary object) but 배경/색상 유지. cb4 swap → 같은 새 but 다른 색 배경.
- CIBHash: chunk swap → semantic 관련성 없이 아무 이미지.

### 구현 난이도
- **중간 (반나절-1일).**
- Distance-based retrieval 필요: exact tuple match 는 rare 하므로 "5개 slot 은 exact match, m-th slot 은 임의" 로 완화된 filter 후 hash distance 로 rank.
- Baseline 은 XOR distance 로 chunk-swap 후 top-3.

### 이 method 의 paper-value
- Compositional code 의 **generative/editable** property 를 보임 — 검색 뿐 아니라 attribute 제어 도 가능함을 시사.

---

## Method C — Text-conditional Slot Activation Heatmap (routing 시각화)

### 개념
우리 slot 은 text-supervised. "text sub-caption 이 slot m 을 어떻게 활성화시키는지" 이미지 위에 heatmap 으로 표시. Flat hash 는 text path 자체가 없음.

### 프로토콜
1. Query 이미지 x + Qwen 6-sub-caption (`C_global`, `C_primary_object`, ..., `C_scene_type`).
2. Model forward → `routing_matrix` (shape `[B, N_patch, M+1]`) 획득.
3. Slot m ∈ {1, ..., 5} 각각에 대해:
   - `routing_matrix[b, :, m]` 을 14×14 grid 로 reshape → 이미지 위에 heatmap overlay.
   - Caption text 를 heatmap 아래에 표시: e.g., "C_primary_object: sparrow"
4. Baseline 비교 는 **불가능** — CIBHash 는 per-patch routing 개념이 없음. 대신 baseline 은 `raw feature attention` (patch-level cosine to global feature) 로 대체.

### 예상 결과
- Ours: slot 2 (primary_object) → 새 몸통 활성화. slot 3 (secondary_object) → 새의 발/부리. slot 5 (scene_type) → 배경 (하늘/나뭇가지).
- Baseline: attention 이 patch 전체에 uniform 하거나 CLIP global feature 방향만 향함 (slot 별 구분 불가).

### 구현 난이도
- **낮음 (2-3시간).** 이미 `scripts/diagnostic_text_alignment_viz.py` 에 유사 로직 존재.
- Baseline 대체 시각화 는 CLIP `visual_projection` 결과와 caption text embed 의 patch-level cos-sim 으로 계산.

### 이 method 의 paper-value
- Ours 만 가능한 시각화 = **text-visual grounding** 의 학습된 spatial 패턴 을 직접 보임.
- ICML 계열 리뷰어가 좋아하는 "interpretable AI" 각도.

---

## 종합 - Paper Figure 후보 우선순위

| Priority | Method | Panels/Figure | 구현 시간 | Paper-value |
|:---:|---|---|---:|---|
| 🟢 1위 | **Method A** — codeword atlas | 2 (Ours vs CIBHash 8×8 grid) | 2-3시간 | 매우 높음 (직관적) |
| 🟢 2위 | **Method C** — routing heatmap | 1 (Ours 5-slot × N-samples) | 2-3시간 | 매우 높음 (paper contribution) |
| 🟡 3위 | **Method B** — slot swap retrieval | 1 (5-slot swap 스트립) | 반나절-1일 | 높음 (editable code) |

**추천 조합**: Method A + Method C 는 상보적.
- A: DB 내 semantic clustering 이 어떻게 이루어지는가 (post-hoc analysis).
- C: 학습 신호 (text) 가 어떻게 spatial routing 을 유도하는가 (mechanism).
Paper Section "Qualitative Analysis" 에 A 를 main figure, C 를 supplementary/appendix.

**Method B 는 강력하지만 3-dataset 에 적용하기 위한 코드 부담이 더 큼** — main paper 에는 A/C 만 넣고, B 는 supplementary 에 CUB 하나만 예시로 시연 하는 것도 고려.

---

## Baseline 정성적 열세를 명시적으로 보이는 요약 그림 후보

Paper 의 최종 "interpretability" 주장 정리 표:

| 관점 | Ours | Flat baseline |
|---|:---:|:---:|
| 학습된 semantic slot 존재 | ✅ | ❌ |
| Per-slot codeword atlas coherent | ✅ (A) | ❌ (A) |
| Text 로 특정 slot 지목 가능 | ✅ (C) | ❌ (개념 부재) |
| Single-slot swap → semantic axis 변화 | ✅ (B) | ❌ (B) |
| Per-slot routing spatial map | ✅ (C) | ❌ (개념 부재) |
| NMI (off-diag mean, 정량) | 0.55–0.68 | 0.19–0.41 |

---

## 실행 계획

1. **Method A 구현 (즉시 시작 가능)**: `scripts/qualitative_codeword_atlas.py` 작성. Flickr champion + Flickr CIBHash 각각 8×8 codeword-sample grid 생성. 결과 → `docs/figs/codeword_atlas_flickr_2026-07-11.png`.
2. **Method C 구현**: 기존 `scripts/diagnostic_text_alignment_viz.py` 기반 확장. Flickr 5-slot routing heatmap 3 sample × 5 slot grid. 결과 → `docs/figs/routing_heatmap_flickr_2026-07-11.png`.
3. Method B 는 위 2개 완료 후 시간 여유 있을 시 진행.
