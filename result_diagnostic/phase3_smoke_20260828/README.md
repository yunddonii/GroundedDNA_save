# Phase 3 launcher smoke, 2026-08-28. One cell, one epoch.

`scripts/phase3_selection_matrix.py --smoke --epochs 1 --only cifar10:4`.
Kept as evidence that the launcher asserts what the previous smoke did not:

```
args.txt          num_semantic_parts = num_codebooks = 5, codons = 3
run_identity.json num_slots 5, bases_per_slot 3, total_bases 15, total_bits 30
                  selection_mode = select        (not the default "refit")
                  val_split_ratio 0.1, seed 42, stop_after_epoch 4
use_gumbel_softmax = False, lambda_codeword_codon_sinkhorn = 0.0
no extract_*.npz  -- stage 1 never touched the official test split
selection         eval_mAP_at_R = 0.5775 at epoch 0, read from the best
                  checkpoint's sidecar, not from a log line
```

One epoch, so the horizons here are 1/1 rather than D2's 60 and N+1. It proves
the plumbing and the assertions, not a number. Diagnostic-only.
