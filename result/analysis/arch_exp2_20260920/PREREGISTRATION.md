# Pre-registration — slot-role mechanisms a-1 / a-2 / b-1 / c-1 (written 2026-09-20, before any arm was run)

Off-protocol, branch `arch-exp-2026-09`. Nothing here may enter `docs/paper_draft/`.

## Common setting
Flickr25K stage-1 selection cell of the 2026-09-15 `p3lamA` recipe (5 slots / 5 codebooks /
K=128 / 3 codons = 15 bases, epsilon horizon 5 with stop after epoch 4, `siglip_cos`,
`val_split_ratio 0.1`, official test split never read). Seeds 42/43/44. Every run is launched
off-protocol (no campaign binding) from code commit `8925504`.

**Baseline:** the same recipe with every new flag at its default, run in the same mode
(`base_s42/43/44`). The campaign-mode `p3lamA` cells are NOT used as the baseline: the two modes
diverge from epoch 1 (cause unidentified), so only same-mode runs are compared.
`base_s42` doubles as the entry gate: it must reproduce `gate1_incumbent_s42` in all 280 logged
values, or no arm is read.

## Arms
| arm | flags | runs |
|---|---|---|
| a-1 | `--routing_mass_alpha 0.5` (tau = pooled SD of centred cosines, from the optimisation rows) | always |
| a-2 | a-1 + `--use_null_centroid --null_free_marginal` | only if a-1 fails the rule below |
| b-1 | `--lambda_mec 1.0` (OVSegmentor's entity_weight) | always |
| b-1' | `--lambda_mec 0.1` | only if b-1 fails the retrieval half of the rule |
| c-1 | `--vq_bypass_epochs 2` (identity quantizer for epochs 0-1, k-means codebook at epoch 2) | always |

## Measurements (per run)
- Retrieval: terminal `eval_mAP_at_R` (held-out 10 % of train), dead and unique code ratios, from log.csv.
- `scripts/slot_role_probe.py` on the terminal checkpoint, deployment forward, held-out validation rows:
  M1 role advantage (primary), M2 dataset-label decoding, M3 deployment codebook perplexity, M4 slot mass.

## Operational rule (decides what runs next; NOT a significance test)
With n = 3 per arm, SD comparisons are not tests (F14); differences and SDs are reported as such.
An arm "passes" iff both hold:
1. retrieval non-inferior: mean mAP@R(arm) >= mean(base) - SD(base);
2. role gain: mean M1 advantage(arm) - mean M1 advantage(base) > SD of M1 advantage(base).
a-2 runs iff a-1 does not pass. b-1' runs iff b-1 fails condition 1.

## Known bias, declared in advance
Captions supervise every arm's training, and b-1 additionally trains on caption words, so M1 can
favour b-1 for reasons other than slot roles. M2 (dataset labels) and M3 do not use captions and
are reported alongside for exactly that reason.

## Decision log
- 2026-09-20 01:45: a-1 applied to the rule: retrieval Δ +.0023 (condition 1 holds), M1 advantage
  Δ −.0003 against a threshold of +.0053 (condition 2 fails) -> a-1 does not pass -> a-2 runs, as
  pre-registered. Mechanism check: per-image slot-mass max/min median rose 1.32 -> 1.79 with 0 %
  empty slots, so the mass targets did take effect; they did not change slot roles.
- 2026-09-20 01:52: b-1 applied to the rule: retrieval Δ +.0113 (condition 1 holds), M1 advantage
  Δ −.0044 (condition 2 fails) -> b-1 does not pass. b-1' (lambda 0.1) is NOT run: it was
  pre-registered to run only if b-1 failed condition 1, which it did not.
- 2026-09-20 01:58: c-1 applied to the rule: retrieval Δ −.0023 (condition 1 holds), M1 advantage
  Δ −.0054 (condition 2 fails) -> c-1 does not pass. Secondary (not part of the rule): deployment
  codebook min/median perplexity .94 ± .04 vs .58 ± .34, never-used/dead codes .05 vs .24, label
  decoding (M2) +.028.
