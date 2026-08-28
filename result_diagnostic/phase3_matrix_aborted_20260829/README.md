# The first 16-cell matrix, aborted after two cells. Diagnostic-only.

Launched 2026-08-29 00:44 at commit a0a6b522, stopped at 00:5x with CIFAR-10
N4 and N9 recorded and the other three datasets mid-N4.

Stopped because CIFAR-10 N9 demonstrated, on a live production cell, the
defect the launcher was supposed to prevent:

```
selection value        0.8493928  at epoch 9   (log.csv, correct for D1)
best over the prefix   0.8938001  at epoch 4
final checkpoint       epoch 4
terminal_weights_preserved  false        <- recorded, and ACCEPTED
```

Without `--final_epoch_eval` -- which a selection run may not use, because it
evaluates the official test split -- the trainer replaces the final checkpoint
with the earlier best. So the cell's recorded score and its surviving weights
describe different epochs, and `assert_completed` noted that in a boolean and
admitted the cell anyway.

The metric itself is not wrong: 0.8493928 is epoch 9's row, written during
epoch 9's mid-eval, before the swap. D1 discards these weights anyway, since the
refit trains from scratch. But a cell whose two halves describe different epochs
is not self-consistent evidence, and accepting it silently is the pattern this
whole audit is about.

The fix is `--keep_final_checkpoint`, which skips the swap without triggering
the official-test evaluation, plus a refusal rather than a boolean when the
epochs disagree. The matrix is re-run from scratch under that; nothing here is
reused, because these two cells were produced without the flag.

Also kept here: `probe2_cifar_A_v4_N4_s42.json`, a one-epoch diagnostic that was
sitting in the record directory when the matrix started. The wrapper counted
every JSON in that directory against 16, so its presence alone made completion
unreachable -- the count is over the exact sixteen keys now.
