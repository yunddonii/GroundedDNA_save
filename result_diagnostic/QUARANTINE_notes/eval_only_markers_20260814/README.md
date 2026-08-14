# Eval-only analysis markers, quarantined 2026-08-14

Nine `analysis_complete.json` files written by `scripts/phase2_recompute_metrics.sh`
between 20:38 and 21:0x KST, from committed source `8ada879`. They are moved
here rather than left in place because a consumer cannot tell them apart from a
finished cell, and they are not one.

They are inadmissible for two independent reasons, both raised as CRITICAL in
`docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md` §20:

* **§20.4 — the marker is published before the NMI runs.** `eval_cell_bioproj.py`
  seals the marker immediately after the bio-evaluation, so `metrics` holds only
  `map_at_R_bioproj`, `full_map_bioproj`, `full_map_pre_projection` and
  `dna_unique_db`. `mean_off_diag_nmi` is absent, and `pairwise_nmi.py` never
  updates or re-seals the marker. The aggregator reads the missing key with
  `.get()` and still counts the cell as paired, so a report of 15 paired cells
  with every NMI column blank was reachable. A file named `analysis_complete`
  must not be written by a stage that has not completed the analysis.

* **§20.7 — only one side, and only one of the two computations.** The loop
  iterates the fixed side and calls `eval_cell_bioproj.py` alone; it never calls
  the NMI, and it never recomputes the 15 legacy faces. §19.4 requires both
  sides of the delta to come from the same committed source, so a fixed side
  recomputed at `8ada879` against a legacy side computed by whatever ran in
  August is not a delta anyone can quote.

The `pairwise_nmi.json` files still sitting beside those cells are the older
19:55 artefacts and are unbound; they were not produced by this loop.

Digests as quarantined (first 16 hex):

| cell | analysis_complete.json |
| --- | --- |
| cifar10_N4 | 1524f997aa32345d |
| cifar10_N9 | 8cfa33455cc89544 |
| cifar10_N19 | 1c806792885eb71f |
| cifar10_N39 | 70680a9330fb111e |
| flickr25k_N4 | 412aa7514fc743c0 |
| flickr25k_N9 | 88ba2c8c64f10ee6 |
| flickr25k_N19 | d854cb6908984752 |
| mscoco_N19 | fefebf4fa10d3bee |
| mscoco_N39 | 1164538e9df31d26 |

The numbers inside them are not suspected of being wrong — the evaluator was the
committed one and the extraction binding validated. They are quarantined because
the *record* is incomplete, and an incomplete record that reads as complete is
the failure mode this whole audit is about. They are kept, not deleted, so the
recomputed values can be diffed against them.
