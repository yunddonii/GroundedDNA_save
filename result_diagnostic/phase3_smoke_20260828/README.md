# Phase 3 launcher smokes, 2026-08-28. Five runs, four of them superseded.

Diagnostic-only. Every one is a short run with the horizons collapsed, so no
number here is a candidate score; the records say `is_candidate_cell: false`.

| namespace | what it was for | outcome |
| --- | --- | --- |
| `p3smoke0828`  | first M5 run | passed, but the run identity said `selection_mode=refit` |
| `p3smoke0828b` | after `--selection_mode select` | passed; still read the metric from the best-over-prefix sidecar |
| `p3smoke0828c` | after the terminal-epoch reader | **REFUSED, correctly**: budget 2 with stop 4, so no row for epoch 4 |
| `p3smoke0828d` | after making smoke stops consistent | metric read correctly, then the launcher raised `KeyError` on a renamed field AFTER publishing the record -- record present, rc 1 |
| `p3smoke0828e` | after all of the above | the success path, end to end, no traceback |

What `p3smoke0828e` establishes, and the earlier four do not:

```
args.txt + run_identity.json   M=5, L=3, 15 bases, 30 bits
identity                       selection_mode=select, val 0.1/42, seed 42
                               stop 4, budget/horizons as actually run (5/5/5)
record                         copied from the manifest, not from the
                               launcher's constants
completion                     final checkpoint re-hashed against its sidecar,
                               terminal_weights_preserved: true, no active claim
selection                      eval_mAP_at_R = 0.8320 at epoch 4, from log.csv
no extract_*.npz               stage 1 never touched the official test split
launcher                       "1 of 1 cells complete", no traceback
```

`p3smoke0828d` is kept because it is the concrete case where a record existed
and the launcher had still failed -- "a record exists" and "the run succeeded"
are different states, and the record is published only after every check now.

The run trees and records here are untracked and the logs are ignored, so a
fresh clone gets this README and nothing else.
