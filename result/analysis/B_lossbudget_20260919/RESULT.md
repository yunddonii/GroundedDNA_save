# (B) loss-budget rebalancing — Flickr25k, 3x3 grid, 2026-09-19

Off-protocol exploration. Base recipe = the 2026-09-15 `p3lamA` stage-1
selection cell (5 slots / 5 codebooks / K=128 / 3 codons = 15 bases, epsilon
schedule aligned via `sinkhorn_schedule_horizon=5` with `stop_after_epoch=4`,
`hash_target_mode=siglip_cos`). Single delta: `lambda_text_hash_ntxent`.
`selection_mode=select` + `val_split_ratio=0.1`: the official test split is
never read, and the trainer logs confirm it
(`[p0-stage1] SKIP official-test extraction/evaluation`).

5 cells were run here; 4 of the 9 grid points already existed.

| lambda_text_hash_ntxent | n | mAP@R mean | sd | dead mean | sd | anchor cos mean |
|---:|--:|---:|---:|---:|---:|---:|
| 0.05 (incumbent) | 3 | .7482 | .0139 | .2578 | .0539 | .1288 |
| 0.10 | 3 | .7446 | .0099 | .2328 | .0246 | .0546 |
| 0.20 | 3 | .7460 | .0082 | .2740 | .0480 | .0435 |

**Retrieval is flat.** The spread of the three means is .0036 while the
within-lambda seed SD is .008-.014, so no lambda is distinguishable on mAP@R.
Dead-codeword ratio is likewise inside seed noise. The "dead 0.013 -> 0.309
catastrophe" recorded for the 6-codebook v140 family does NOT reproduce at 5
codebooks.

**Anchor separation improves monotonically and is outside seed noise.** Mean
pairwise cosine between the four local text anchors falls .1288 -> .0546 ->
.0435; every 0.10/0.20 cell is below every 0.05 cell.

**But the routing plan does not move.** Centered slot cosine at the cost stage
(-.2944 -> -.3019) and at the plan (-.3034 -> -.3066) barely change, and the
centered metric is structurally pinned near -1/(M-1) = -.3333 anyway. Per-slot
transported mass is ~22% for every local slot regardless.

**Verdict: (B) does not achieve its goal.** Rebalancing the loss budget moves
the text side only. The failure this was meant to fix -- one codebook per run
collapsing to perplexity ~6 of 128, at a *different* axis each seed -- happens
downstream of routing, inside VQ/EMA, and no lambda tested reaches it.

## Limitation: the codebook-collapse axis was NOT measured on the deployment path

Per-codebook perplexity for all nine cells was computed with
`dna_utils.visualization._forward_text_routed`, which is the TEXT-routed path
(its own docstring: "training-style (text routing)"). All nine look healthy
(min/median perplexity .86-.99). That is not evidence about the collapse this
experiment cares about: the perplexity-6-of-128 collapse was measured from
`extract_db.npz`, which is produced by the DEPLOYMENT path, where the router
never sees text. The two numbers describe different forward passes and must not
be compared.

Stage-1 selection cells deliberately write no extraction
(`[p0-stage1] SKIP official-test extraction/evaluation`), so closing this gap
needs a separate image-only extraction over the train split. Until that is run,
this experiment says nothing about whether raising
`lambda_text_hash_ntxent` changes deployment-path codebook collapse.

## Qualification added after (A), 2026-09-19

The (A) experiment on this branch froze the codebook and, as a side effect,
produced the *best* anchor separation of anything measured here — .0189 and
.0299 against this baseline's .13, with cost-stage cosine .215 and .301 against
.84 — while being the worst arms on retrieval (mAP .633) and unique code ratio
(.027 against .530). With the codebook held still the text adapter moves
instead and over-separates the axes at the representation's expense.

So the monotone anchor-separation improvement reported above is **not by itself
evidence of a better model**. What (B) establishes is narrower: raising
`lambda_text_hash_ntxent` separates the text anchors at no measurable retrieval
or dead-codeword cost. Whether that separation is useful is not settled by this
experiment.
