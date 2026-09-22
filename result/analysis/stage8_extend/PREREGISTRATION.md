# Stage 8 pre-registration (2026-09-22 16:13:17 +0900, written before any stage-8 cell ran)

Branch `arch-exp-2026-09`, code `150deb2`, worktree `/data/yschoi/gdna_wt_arms`. Off-protocol, exploration only
(audit section 657: these concept / local-target paths are NOT admitted recipe changes and must not enter a D5 or
main campaign). Each dataset's stage-1 selection cell, seeds 42/43/44, `hash_target_mode siglip_cos`.
User decisions of 2026-09-22: A-prime's retrieval cost is acceptable, so extend it; run B2; include CIFAR-10.

## Part 1. A-prime + C on NUS-WIDE, MS-COCO, CIFAR-10 (replication of the Flickr25K stage-7 result)

- **ac2:** the dataset's base command, K 64, plus `--concept_codebook_npz /data/yschoi/gdna_stage7/concepts_<ds>.npz
  --lambda_concept 1.0 --concept_tau 0.5 --cibhash_local_target none --lambda_text_code_kl 0.0
  --lambda_text_hash_ntxent 0.0 --codon_joint_slots 0 --concept_label_ema`.
  - Concept files: opt rows only, k-means 64 per axis, seed 0.
  - sha256 digests: nuswide 11737cf6…, mscoco 33ce7a04…, cifar10 b01680df….
- **k64:** base plus K 64 only, on NUS-WIDE and MS-COCO. CIFAR-10's base is already K 64.

A dataset *replicates* the Flickr result if all three hold (3-seed means; descriptive, no test):

1. The supervised-dictionary reading is at least base + .02, and at least base on 3 of 4 axes.
2. The dictionary-free concept-name reading exceeds base's caption-path reading.
3. M1 exceeds base's M1 on at least 2 of 3 seeds.

**Retrieval** is reported, not gated (accepted by the user). A mAP@R loss larger than .09 (twice
Flickr's .045) is flagged for the user.

## Part 2. B2 on Flickr25K (screening)

- **bq:** base plus `--cibhash_local_target axis_soft --cibhash_local_target_tau 0.1 --cibhash_local_queue 4096`.
  Offline, this pair gives JS .283 nats between axis targets (in-batch B: .090) and keeps the
  image's own share at .58.
- **Decision rule:** the stage-7 rule. Best text-free reading at least .367 and at least base on 3/4
  axes; mAP@R at least .715; unique codes at least .45. Also reported against the stage-7 bc arm.

## Entry gate

`gate8_s42`: the Flickr base command on code 150deb2 must reproduce base_s42 log.csv 280/280.

Nothing here is edited after the first stage-8 cell starts; deviations are appended with a timestamp.

## Deviation 1 (2026-09-22 16:15:56 +0900, before any B2 result)

The first bq smoke cell (code 150deb2) failed at training step 2 with an autograd in-place error. The
FIFO was rewritten after the loss used it. Fixed in `ed911a5` by handing the loss a snapshot. That
path is reachable only with `--cibhash_local_queue > 0`. Part 1 cells are unaffected and keep
running on 150deb2 in `/data/yschoi/gdna_wt_arms`. B2 (bq_s42..44) runs on ed911a5 in
`/data/yschoi/gdna_wt_arms2`, with its own entry gate `gate8b_s42` (base_s42 280/280). The rule is
unchanged.
