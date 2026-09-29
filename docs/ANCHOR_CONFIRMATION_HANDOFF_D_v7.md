# Anchor Confirmation — the frozen N record and the stage-D request (response to audit §726)

**Status: the §726-approved CPU N reduction ran and published the frozen N record. The stage-D
plan and request below are non-executable until a new audit approval names them.**
- No D training, lease, probe or other GPU work ran.
- The reduction hashed the 16 named `config.pt` files and deserialised nothing. The D plan replayed
  the same selection, the same bounded read.

## 1. The N reduction (§726.3)

- **Source list.** The audit's `ancS7_select_sources.json` was copied byte-for-byte (`cp
  --no-clobber`; no earlier file existed) to `artifacts/anchor_confirmation/ancS7_select_sources.json`.
  SHA256 `0cfab95106bfadb03e5e9750d8e5a17d7ff16523ff14b66bba1cab5519067656`, as required: 16
  receipt-sourced coordinates, receipt `5915768767e3…`.
- **Pins checked before the run:**
  - reducer `79fbe1318a35e15ed501faec09ff9fafc9bd1fe4652938c850972e7351da3bf5`;
  - snapshot file `6f2c03527b26…`;
  - manifest `c0612963…`;
  - tree clean at `b916fb7`.
- **Command:** exactly §726.3's, saved as `artifacts/anchor_confirmation/ancS7_reduction/command.txt`
  (`aed0e57123cf…`).
- **Outcome:** 2026-09-29 12:41:31 → 12:41:37 UTC, **rc 0**, stderr empty.
  - stdout: `[anchor-reducer] wrote artifacts/anchor_confirmation/ancS7_selected_n.json sha256 5cda7adb…`.
  - rc, stdout, stderr, start and end are kept in `ancS7_reduction/`.

**The frozen N record** `artifacts/anchor_confirmation/ancS7_selected_n.json`, SHA256
**`5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff`**:
- reducer `anchor-confirm-reducer/3`, arms `['anchors']`, the fixed-architecture declaration;
- generation: manifest `c0612963…`, commit `59477d8`, contract `e6c4978e…`;
- rule: argmax raw base-Hamming mAP@R at the own terminal epoch, ties to the smallest N, seed 42.

| dataset | N=4 | N=9 | N=19 | N=39 | frozen N |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | **.782328** | .761542 | .710839 | .717689 | **4** |
| Flickr25K | **.764194** | .735947 | .719277 | .706278 | **4** |
| NUS-WIDE | **.722516** | .719036 | .694194 | .684803 | **4** |
| MS-COCO | .621615 | .626046 | .630374 | **.634066** | **39** |

It equals §726.2's cross-check. These are seed-42 train-only validation values, not a result or an
anchor-versus-control claim.

## 2. The stage-D request (not executable)

- **Plan:** `artifacts/anchor_confirmation/plan_decide_ancD7_v7.txt` (`1f289ae3310c…`). Stage decide,
  8 cells, namespace `ancD7`. The admission renders both arms and requires them to differ in
  `axis_center` alone at all 8 coordinates.
- **Request:** `artifacts/anchor_confirmation/request_preview_ancD7_v7_carried.txt` (file
  `c482f59a7f2a…`). Canonical request SHA256
  **`6ef9d4ea9c6aa2a64f680a12a3147737b364459516cf023e5e3cee6b823367cc`**, recomputed from the
  printed JSON:
  - stage `decide`, mode `run`, namespace `ancD7`, 4 GPUs, no epoch override;
  - manifest `c0612963…`, selection `5cda7adb…`;
  - cells (declared = executed), anchors only:
    - CIFAR-10 N4, Flickr25K N4, NUS-WIDE N4 and MS-COCO N39;
    - each at seeds 43 and 44.
  - carried admission: the original ancS6 full-admission snapshot,
    `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json`,
    SHA256 `ef5a3e8d857a7a6cc22d1e978854c3f721470ed19f426a020873718199a2b8ff`. The smoke and S
    snapshots record carried admissions and are not used;
  - the four stage-1 seal pins, unchanged (`3e9bc7b9…`, `e44b363a…`, `6e9138a6…`, `aa1eaa5d…`);
  - result root `/home/yschoi/gdna_anchor4_result`.
- **The approval it needs:** a `stage-D-run` line naming the manifest, the selection and the request:
  `ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-D-run manifest=c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 selection=5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff request=6ef9d4ea9c6aa2a64f680a12a3147737b364459516cf023e5e3cee6b823367cc`.

## 3. The supervised command (for after that approval)

The same supervisor and ledger, with prior charge **14589.083378900774 s**, run from the anchor
worktree:

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancD7_v7 \
  env -C /data/yschoi/gdna_anchor_confirm_v1 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --stage stage-D-run --planned-cells 8 \
  --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm decide --namespace ancD7 --anchor-arms anchors \
  --anchor-selection artifacts/anchor_confirmation/ancS7_selected_n.json \
  --anchor-selection-sha256 5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff \
  --run --gpus <4 free GPUs> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal cifar10:stage1=/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --anchor-manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --admission-authority /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json \
  --anchor-approval-section <the approving ledger section>
```

- **Supervisor rules** (unchanged code, `b5eecf58…`):
  - stage `stage-D-run`: 4-h wall limit including admission, stop at limit − 130 s;
  - 4 GPUs, 8 planned cells, 520-s headroom, 1-s poll, 10-s watchdog, 120-s stop bound;
  - the cumulative ceiling of 45,000 s includes every earlier attempt.
- **Estimate from the S cells' own wall times.** Each stream's first S cell also paid the start-up
  load, so counting it for both seeds is conservative:

  | stream | per cell | 2 cells |
  |---|---:|---:|
  | CIFAR-10 N4 | 1188 s | 2376 s |
  | Flickr25K N4 | 274 s | 548 s |
  | NUS-WIDE N4 | 1493 s | 2986 s |
  | MS-COCO N39 | 1702 s | 3404 s (≈ 57 min, the longest stream) |

  Device time ≲ 9,315 s ≈ 2.6 GPU-h, against 30,410.9 s (8.45 GPU-h) remaining before the ceiling.
  The contract's planning range for D was 0.47–3.73 GPU-h.
- **Space:** 10 GiB + 8 × 0.75 GiB = 16 GiB at the first dispatch, 13 GiB live on four GPUs. `/` had
  380,165,808,128 bytes free (2026-09-29, after S), not reserved.
- **Roots:** both roots exist and hold the earlier attempts' evidence. They are neither cleared nor
  recreated. Ops/5 §2's pre-smoke sentence that neither root exists is stale, as §724.3 noted.

## 4. Not requested here

D execution before its own approval; the 12 probes, which follow D and need their own
`probe`-scope approval; L, R/T and the final freeze. Required TODO 13–15 still precede the freeze.
