# 손실함수 정리 — 실제로 학습에 기여하는 항만

2026-08-12 확정. 5 슬롯 / 15 염기 / 30 bit 아키텍처 기준.

`DNACodonHashLoss`가 반환하는 항은 56개지만, 그중 **총합에 자기 λ로 들어가면서
gradient가 0이 아닌 항은 10개뿐**이다. 나머지는 λ=0이거나, 값은 찍히지만
gradient가 정확히 0이거나, 이 레시피가 타지 않는 경로에서만 생성된다.

판정은 읽어서 한 것이 아니라 측정했다 — `scripts/audit_loss_gradients.py`가
실제 배치 1개로 forward한 뒤 항마다

```
‖∇_θ L_term‖   over all trainable parameters      (torch.autograd.grad, allow_unused=True)
```

를 구한다. λ만 보면 판별할 수 없기 때문이다. 실제로 `lambda_recon=1.0`,
`lambda_codon_text_anchor=0.1`, `lambda_anchor=0.05`는 모두 양수지만 기여가 0이다.

> **이름 규칙.** 아래 "논문 이름"은 산문·수식·표에서 쓰는 이름이다. 괄호 안 코드
> 식별자는 **논문에 절대 쓰지 않는다.** 코드·플래그·기록된 `args.txt`는 그대로
> 두는데, 바꾸면 모든 런처와 이미 기록된 실행 디렉토리가 깨지기 때문이다.
> 부록에 이 대응표를 싣는다.

---

## 기여도 순위 (Flickr25k, 배치 8, epoch 5 실측)

`λ·‖∇‖`는 그 항이 한 스텝에서 파라미터를 실제로 얼마나 미는지에 비례한다.

| 순위 | 논문 이름 | λ | value | ‖∇‖ | **λ·‖∇‖** |
|---:|---|---:|---:|---:|---:|
| 1 | \(\mathcal L_{\rm contrastive}\) | 1.00 | .787 | .726 | **.7263** |
| 2 | \(\mathcal L_{\rm text\text{-}code\text{-}contrastive}\) | 0.05 | .274 | 4.884 | **.2442** |
| 3 | \(\mathcal L_{\rm xmodal}\) | 0.05 | 2.520 | 2.023 | **.1011** |
| 4 | \(\mathcal L_{\rm codon\text{-}joint}\) | 0.02 | .945 | 3.876 | **.0775** |
| 5 | \(\mathcal L_{\rm transport}\) | 0.15 | 1.766 | .485 | **.0727** |
| 6 | \(\mathcal L_{\rm text\text{-}code\text{-}KL}\) | 0.05 | .936 | 1.063 | **.0531** |
| 7 | \(\mathcal L_{\rm base\text{-}prior}\) | 0.05 | .712 | .732 | **.0366** |
| 8 | \(\mathcal L_{\rm VQ}\) | 0.25 | .256 | .093 | **.0233** |
| 9 | \(\mathcal L_{\rm commit}\) | 0.05 | .044 | .153 | **.0077** |
| 10 | \(\mathcal L_{\rm codebook\text{-}balance}\) | 0.02 | .001 | .078 | **.0016** |

**정규화된 gradient 크기의 순위는 λ 순위와 다르다.**
\(\mathcal L_{\rm text\text{-}code\text{-}contrastive}\)는 λ가 20분의 1인데 ‖∇‖가
가장 크고(4.88), \(\mathcal L_{\rm VQ}\)는 λ가 5배인데 ‖∇‖가 0.09로 가장 작다.
λ 표만 싣고 "가중치가 큰 항이 중요하다"고 읽히게 두면 안 된다.

---

## 1. \(\mathcal L_{\rm contrastive}\)  *(코드: `loss_cibhash_ntxent`)*

**역할.** 두 증강 뷰 사이의 **슬롯별 InfoNCE**. 라벨 없이 이미지 정체성을
학습하는 유일한 항이며, 검색 성능의 주된 원천이다. 기여 1위.

**개명 이유.** 코드 이름은 CIBHash(Qiu et al., NeurIPS'21)에서 왔지만 (i) 그
논문의 핵심인 variational IB 항은 우리 레시피에서 `λ=0.001`이면서 gradient가 0이라
쓰지 않고, (ii) CIBHash는 해시 **비트**에 거는 반면 우리는 **VQ 이전 슬롯
임베딩**에 걸며, (iii) CIBHash는 우리 비교 baseline이기도 해서 방법과 baseline이
섞여 읽힌다. 이 항이 전달하는 것은 결국 대조 신호 그 자체이므로
`L_contrastive`가 정확하다.

**수식.** 슬롯 \(m\), 두 뷰 \(s^m_{(1)},s^m_{(2)}\), 배치 \(B\):

\[
\mathcal L_{\rm contrastive}=\frac1M\sum_{m=1}^{M}
\mathrm{NTXent}\big(\{s^m_{(1)},s^m_{(2)}\};\,T_m\big),\qquad
T_m=T\big(1-\alpha\,\overline{\cos}(a^m,a^{\cdot})\big)
\]

\(T_m\)은 그 슬롯의 텍스트 앵커가 다른 앵커와 얼마나 가까운지에 따라 조절되는
per-slot 온도다(축이 서로 겹칠수록 온도를 낮춰 더 세게 민다).

**구현** — `loss_siglip2.py:_loss_cibhash_visual_per_codebook`

```python
B, M, D = visual_tokens_view1.shape
T = max(float(temperature), 1e-6)
ntxent_per_cb = []
for m in range(M):
    vm1 = visual_tokens_view1[:, m, :]          # [B, D]  pre-VQ
    vm2 = visual_tokens_view2[:, m, :]          # [B, D]
    v   = torch.cat([vm1, vm2], dim=0)          # [2B, D]
    v_n = F.normalize(v, dim=-1, eps=1e-8)
    sim_raw = v_n @ v_n.T                       # [2B, 2B] cosine
    ...                                         # per-slot temperature, then NTXent
```

---

## 2. \(\mathcal L_{\rm text\text{-}code\text{-}contrastive}\)  *(코드: `loss_text_hash_ntxent`)*

**역할.** 텍스트가 **코드 자체**를 배치 안에서 판별적으로 만든다. `per_codebook`
모드에서는 각 코드북의 염기 분절이 독립적으로 감독된다. 정규화 기준 ‖∇‖가 4.88로
가장 크다 — λ가 작아도 방향을 강하게 잡는 항이다.

**개명 이유.** 코드 이름의 `text_hash`가 **λ=0으로 죽어 있는 별도 항**
`loss_text_hash`와 혼동된다. 작용 대상이 hash가 아니라 코드이므로
`text-code-contrastive`가 정확하다.

**수식.** 이미지 코드 \(u\), 텍스트에서 유도한 코드 \(u^{t}\):

\[
\mathcal L=\tfrac12\Big[\mathrm{CE}\big(\tfrac{\hat u\hat u^{t\top}}{\tau},\,I\big)
+\mathrm{CE}\big(\tfrac{\hat u^{t}\hat u^{\top}}{\tau},\,I\big)\Big]
\]

**구현** — `loss_siglip2.py:2636`

```python
t_n = F.normalize(t_flat, dim=-1)
i_n = F.normalize(i_flat, dim=-1)
logits_it = (i_n @ t_n.T) / tau                 # [B, B]
labels = torch.arange(B, device=logits_it.device)
loss_text_hash_ntxent_add = 0.5 * (
    F.cross_entropy(logits_it,   labels) +
    F.cross_entropy(logits_it.T, labels))
```

---

## 3. \(\mathcal L_{\rm xmodal}\)  *(코드: `loss_xmodal_commit`)*

**역할.** 슬롯 표현과 축 앵커를 **양방향**으로 맞춘다. 각 방향의 표적은 상대
모달리티의 **양자화된** 토큰이며 stop-gradient가 걸린다.

**주의 — 논문에 반드시 적어야 할 성질.** 두 번째 항은 앵커를 시각 쪽으로
이동시킨다. 따라서 앵커는 학습 중 고정 상수가 아니며, 수송 계획이 평탄할 때
(\(\varepsilon\) 큼) 모든 \(q_v^m\)이 비슷해지면 모든 앵커가 같은 표적으로 끌려가
서로 붕괴한다. §4.2c의 \(\varepsilon\) 정렬이 이 되먹임을 푼다.

\[
\mathcal L_{\rm xmodal}=\tfrac12\Big(\big\lVert s^m-\mathrm{sg}[q^m_t]\big\rVert^2
+\big\lVert a^m-\mathrm{sg}[q^m_v]\big\rVert^2\Big)
\]

**구현** — `loss_siglip2.py:2882`

```python
loss_xmodal_visual = F.mse_loss(z_v[:, start_m:, :], q_t[:, start_m:, :].detach())
loss_xmodal_text   = F.mse_loss(t_v[:, start_m:, :], q_v[:, start_m:, :].detach())
loss_xmodal_commit = 0.5 * (loss_xmodal_visual + loss_xmodal_text)
```

---

## 4. \(\mathcal L_{\rm codon\text{-}joint}\)  *(코드: `loss_codon_joint`, 이름 유지)*

**역할.** 슬롯별 코돈의 **결합 분포**를 균등 쪽으로 민다. 위치별 주변 분포가
균등해도 결합 분포는 소수 조합에 몰릴 수 있으므로 규제를 \(4^{\ell}\)개 조합
위에서 정의한다 — 이것이 본 논문이 도입한 지점이다.

\[
\mathcal L_{\rm codon\text{-}joint}=\frac1{|S|}\sum_{m\in S}
\mathrm{KL}\big(\mathcal U_{4^{\ell}}\,\Vert\,\bar J^{m}\big),\qquad
\bar J^{m}=\mathbb E_{x}\Big[\textstyle\bigotimes_{j=1}^{\ell}u^{m}_{j}\Big]
\]

**구현** — `loss_siglip2.py:3553`

```python
_p = u.float().view(B, _M, _L, 4)
_j = _p[:, :, 0, :]                                   # [B, M, 4]
for _l in range(1, _L):                               # outer product over positions
    _j = (_j.unsqueeze(-1) * _p[:, :, _l, :].unsqueeze(-2)).flatten(-2)
_Q = _j.mean(dim=0)                                   # [M, 4**L]
_Q = _Q / _Q.sum(-1, keepdim=True).clamp_min(1e-12)
_logQ = _Q.clamp_min(self.codon_joint_floor).log()
_unif = torch.full_like(_Q, 1.0 / _Q.shape[-1])
loss_codon_joint = F.kl_div(_logQ, _unif, reduction="batchmean")
```

---

## 5. \(\mathcal L_{\rm transport}\)  *(코드: `loss_wasserstein`)*

**역할.** 수송 비용 자체를 목적에 넣어 라우팅이 앵커에 가까운 패치를 모으게 한다.
라우터에 직접 걸리는 **유일한** 항이다.

**개명 이유.** 엔트로피 정규화된 불균형 OT의 계획 비용 \(\langle P,C\rangle\)이지
Wasserstein **거리**가 아니다. 거리로 읽히면 잘못된 성질(삼각부등식 등)을
가정하게 된다.

\[
\mathcal L_{\rm transport}=\mathbb E_{x}\big[\langle P, C\rangle\big],\qquad
C_{nm}=1-\cos(v_n,a^m)
\]

**구현** — `loss_siglip2.py:2565`

```python
ot = outputs.get("ot_cost")          # per-sample <pi, cost> from the router
loss_wasserstein = ot.mean()
```

---

## 6. \(\mathcal L_{\rm text\text{-}code\text{-}KL}\)  *(코드: `loss_text_code_kl`, 이름 유지)*

**역할.** 슬롯별로 코드북 위 \(K\)-way 분포를 시각·텍스트 양쪽에서 만들고 텍스트
쪽을 교사로 증류한다. **캡션이 모호한 표본은 확신도로 가중해 걸러낸다** — 그렇지
않으면 애매한 캡션이 코드를 끌고 간다.

\[
p_v[k]\propto\exp\tfrac{\cos(s^m,e^m_k)}{\tau_v},\quad
p_t[k]\propto\exp\tfrac{\cos(a^m,e^m_k)}{\tau_t},\quad
\mathcal L=\mathbb E\big[w\cdot\mathrm{KL}(\mathrm{sg}[p_t]\,\Vert\,p_v)\big]
\]
\[
w=1-\frac{H(p_t)}{\log K},\qquad w<w_{\min}\ \text{인 표본은 제외}
\]

---

## 7. \(\mathcal L_{\rm base\text{-}prior}\)  *(코드: `loss_dna`)*

**상태 확인 (2026-08-12).** 이 항은 **켜져 있다** — 4개 데이터셋 모두
`lambda_dna=0.05`, `eta_base_balance=0.3`이며 gradient가 0이 아니다.
`L_codon_joint` 도입 시 껐다는 기록은 PROJECT_LOG에 없고, 오히려 "core losses
kept"로 유지 기록이 있다.

**역할.** 염기 분포에 두 방향의 사전(prior)을 건다: **위치별로는 뾰족하게**
(하나의 염기로 확정), **데이터셋 전체로는 균등하게**.

**개명 이유.** 코드 이름 `loss_dna`는 전체 DNA 손실 또는 GC/homopolymer 생물학적
제약으로 오해된다. 실제 내용은 염기 분포에 대한 사전이다.

\[
\mathcal L_{\rm base\text{-}prior}
=\underbrace{\mathbb E_{x}\Big[-\sum_{\ell,\alpha}u_{\ell\alpha}\log u_{\ell\alpha}\Big]}_{\text{위치별 첨예화}}
+\;\eta\underbrace{\mathrm{KL}\big(\mathcal U_4\,\Vert\,\bar p_{\ell}\big)}_{\text{전역 균등화}},
\qquad \bar p_{\ell}=\mathbb E_x[u_\ell],\ \ \eta=0.3
\]

**구현** — `loss_siglip2.py:1839`

```python
entropy = -(u * (u + self.eps).log()).sum(dim=-1)        # [B, L]
loss_entropy = entropy.mean()
base_usage = u.mean(dim=0)                                # [L, 4]
log_base_usage = (base_usage + self.eps).log()
uniform = torch.full_like(base_usage, 0.25)
loss_base_balance = F.kl_div(log_base_usage, uniform, reduction="batchmean")
loss = loss_entropy + self.eta_base_balance * loss_base_balance
```

균등화는 **forward KL**(mode-covering)이라 쓰이지 않는 염기에 발산 페널티가
걸린다. \(\eta\)는 내부 가중이므로 균등화 항의 실효 가중은
\(\lambda\cdot\eta=0.05\times0.3=0.015\)다.

---

## 8. \(\mathcal L_{\rm VQ}\)  *(코드: `loss_vq`, 이름 유지)*

**역할.** 표준 VQ-VAE 코드북 + 커밋먼트. **이름을 유지하는 이유는 VQ-VAE가 조상임을
그대로 드러내기 때문이다.**

\[
\mathcal L_{\rm VQ}=\big\lVert q-\mathrm{sg}[s]\big\rVert^2+\big\lVert s-\mathrm{sg}[q]\big\rVert^2
\]

```python
codebook_loss   = F.mse_loss(q, z.detach())
commitment_loss = F.mse_loss(z, q.detach())
```

`--vq_loss_cosine`이 켜지면 두 항 모두 \(1-\cos\)이 되어, 코사인 VQ 조회와 같은
기하에서 당긴다.

---

## 9. \(\mathcal L_{\rm commit}\)  *(코드: `loss_quant`)*

**역할.** \(\mathcal L_{\rm VQ}\)의 **염기 확률 공간 판본**. 연속 코돈 분포를 그것이
현재 고르는 hard one-hot 쪽으로 당긴다.

**개명 이유.** `quant`와 `vq`가 둘 다 "양자화"로 읽혀 어느 공간인지 구분되지
않는다. 이 항은 커밋먼트 한 방향뿐이다.

\[
\mathcal L_{\rm commit}=\big\lVert u-\mathrm{sg}[\tilde u]\big\rVert^2,
\qquad \tilde u=\mathrm{onehot}(\arg\max u)
\]

```python
return F.mse_loss(continuous_code, dna_hash_code_hard.detach())
```

---

## 10. \(\mathcal L_{\rm codebook\text{-}balance}\)  *(코드: `loss_bu`)*

**역할.** 코드북 붕괴 방지. 사용률 균등화 + 배정 상관 억제.

**개명 이유.** `bu`가 무엇의 약어인지 본문만으로 알 수 없다.

\[
\mathcal L=\mathrm{MSE}\Big(\bar P,\tfrac1K\Big)
+\rho\cdot\overline{\big(G\odot(1-I)\big)^{2}},\qquad
G_{m}= \tfrac{K}{B}\,P_m^{\top}P_m
\]

**구현** — `loss_siglip2.py:2452`

```python
P = F.softmax(-distances / tau, dim=-1)                  # [B, M, K]
usage = P.mean(dim=0)                                    # [M, K]
loss_cb_balance = F.mse_loss(usage, torch.full_like(usage, 1.0 / K))
G = (K / float(B)) * torch.einsum("bmk,bml->mkl", P, P)  # [M, K, K]
offdiag = G * (1.0 - torch.eye(K, ...).unsqueeze(0))
loss_cb_uncorr = (offdiag ** 2).mean()
loss = loss_cb_balance + self.rho_cb_uncorr * loss_cb_uncorr
```

---

# λ > 0 인데 기여가 0인 항 — 논문에 목적함수로 쓰지 않는다

| 코드 항 | λ | 원인 |
|---|---:|---|
| `loss_recon` | **1.00** | 값 0, 그래프 없음. 픽셀 디코더 경로 미사용 |
| `loss_codon_text_anchor` | **0.10** | `use_text_anchor=False`(기본값) → 구조적으로 항상 0 |
| `loss_anchor` | 0.05 | 값 1.018이 로그에 찍히지만 **‖∇‖ = 정확히 0** |
| `loss_cibhash_kl` | 0.001 | 연속 임베딩에는 KL이 정의되지 않아 0 반환 |

`loss_anchor`가 특히 위험하다 — 로그에 값이 찍히므로 눈으로는 살아 있어 보이고,
gradient를 재지 않으면 잡히지 않는다.

`lambda_hash`, `lambda_hash_hard`, `lambda_entropy`, `lambda_cb_uncorr`은 **λ=0**이라
gradient는 있어도 기여가 없다.

---

# 총합 조립 시 주의 — 내부 항을 따로 세지 말 것

총합에는 **10개 항만** 자기 λ로 들어간다 (`loss_siglip2.py:3270` 부근).
`loss_base_balance`·`loss_entropy`는 \(\mathcal L_{\rm base\text{-}prior}\) **안**에,
`loss_cb_balance`·`loss_cb_uncorr`는 \(\mathcal L_{\rm codebook\text{-}balance}\)
**안**에 있으며 총합에 별도 항으로 더해지지 않는다. `loss_dict`에는 진단용으로
따로 보고되므로, 이를 독립 항으로 세면 이중 계산이 된다.

---

# 논문 이름 ↔ 코드 식별자 대응표 (부록용)

| 논문 이름 | 코드 식별자 | λ 플래그 |
|---|---|---|
| \(\mathcal L_{\rm contrastive}\) | `loss_cibhash_ntxent` | `--lambda_cibhash_ntxent` |
| \(\mathcal L_{\rm text\text{-}code\text{-}contrastive}\) | `loss_text_hash_ntxent` | `--lambda_text_hash_ntxent` |
| \(\mathcal L_{\rm xmodal}\) | `loss_xmodal_commit` | `--lambda_xmodal_commit` |
| \(\mathcal L_{\rm codon\text{-}joint}\) | `loss_codon_joint` | `--lambda_codon_joint` |
| \(\mathcal L_{\rm transport}\) | `loss_wasserstein` | `--lambda_wasserstein` |
| \(\mathcal L_{\rm text\text{-}code\text{-}KL}\) | `loss_text_code_kl` | `--lambda_text_code_kl` |
| \(\mathcal L_{\rm base\text{-}prior}\) | `loss_dna` | `--lambda_dna`, `--eta_base_balance` |
| \(\mathcal L_{\rm VQ}\) | `loss_vq` | `--lambda_vq` |
| \(\mathcal L_{\rm commit}\) | `loss_quant` | `--lambda_quant` |
| \(\mathcal L_{\rm codebook\text{-}balance}\) | `loss_bu` | `--lambda_bu` |

재현 자료: `docs/newmodel_analysis/loss_gradient_audit_flickr_s5.json`,
측정 스크립트 `scripts/audit_loss_gradients.py`.
