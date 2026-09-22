# Stage 9 pre-registration (2026-09-22 18:46:36 +0900, written before any stage-9 cell ran)

Branch `arch-exp-2026-09`, code `7de5470`, worktree `/data/yschoi/gdna_wt_arms2`. Off-protocol screening,
Flickr25K stage-1 cell, seeds 42/43/44, `hash_target_mode siglip_cos`. Requested by the user on 2026-09-22
(levers 3 and 4 of the 2026-09-22 list; lever 5 refused, lever 6 last resort, lever 2 under review).

## Arms (each = the base command plus exactly these flags)

| arm | flags | tests |
|---|---|---|
| gate9 | none (seed 42 only) | the new code with every flag off must reproduce base_s42 log.csv 280/280 |
| **ancsoft** | `--axis_center anchors --global_gate_init_logit -3.0` | lever 3: axis-centred anchors + a weak global gate (starts at .047; the 2026-09-20 p3soft runs show it stays there) |
| **codsoft** | `--text_hash_ntxent_target axis_soft --text_hash_ntxent_target_tau 0.2` | lever 4: the axis-neighbour soft target at the CODON level (text-hash InfoNCE), instead of on the continuous slot token (stage-7 B) |
| **anccod** | both of the above sets minus the gate: `--axis_center anchors --text_hash_ntxent_target axis_soft --text_hash_ntxent_target_tau 0.2` | lever 4 on top of the anchors |

Comparators already on disk: base, p2anc (anchors), p3soft (weak gate alone, arch-exp-3), bc (stage-7 B).

## Endpoints (held-out val rows, deployment forward, no text)

mAP@R, unique, dead; supervised-dictionary reading, caption-path reading, CLIP-only reading (stage-2
reader); M1; codeword->own-axis (quant_gap); codon NMI between local slots and the joint label-decoding
curve (stage-6b analysis).

## Decision rule (3-seed means; descriptive, no test)

An arm is carried to NUS-WIDE and MS-COCO if ALL hold against **p2anc** (anchors), the current
loss-free candidate:
1. best text-free reading >= p2anc's supervised ceiling + .01 (p2anc .3460 -> >= .356), and >= p2anc on 3/4 axes;
2. mAP@R >= p2anc - .015 (p2anc .7513 -> >= .736);
3. unique >= .45 and dead <= .30.
For codsoft alone the comparator is base (.3469 / .7453).

Nothing here is edited after the first stage-9 cell starts; deviations are appended with a timestamp.
