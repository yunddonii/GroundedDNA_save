# (A) mechanism smoke evidence, 2026-09-19

Recorded here because the smoke run directories are deleted afterwards to make
disk room for the 12-cell grid.

Two 1-epoch Flickr25k runs, identical seed (42) and identical k-means codebook
init, differing only in `--codebook_freeze_after_epoch`:

| | distinct cluster_size values | codebook std |
|---|---:|---:|
| `--codebook_freeze_after_epoch 0` (frozen) | **1** | 0.036098 |
| default `-1` (not frozen) | **264** | 0.037395 |

`max|cb_frozen - cb_notfrozen| = 1.497528`. In the frozen run
`max|codebooks - embed_avg/cluster_size| = 0.00000000` and every cluster_size
entry equals the 32.0 = N/K the initialiser wrote, i.e. the EMA never touched
the codebook.

Unit positive control (`SemanticCodebookQuantizer` directly, 5 steps):

| setting | max|delta codebook| |
|---|---:|
| freeze_after_epoch=-1 (default) | 2.167808 |
| freeze_after_epoch=0, epoch 0 | 0.000000 |
| freeze_after_epoch=3, epoch 0 | 2.167808 |
| freeze_after_epoch=3, epoch 3 | 0.000000 |

The two existing EMA tests in `tests/test_ablation_integrity.py` pass unchanged,
so the default path is unaffected.

**Per-codebook perplexity at 1 epoch is NOT interpretable.** Three 1-epoch
runs, same seed:

| run | per-codebook perplexity |
|---|---|
| kmeans init + frozen | 79.8 43.8 40.6 54.4 40.0 |
| kmeans init, not frozen | 5.4 5.5 4.3 3.9 4.1 |
| no init, not frozen (plain) | 17.2 12.9 13.8 13.7 12.2 |

The plain 1-epoch baseline sits at 12-17, and real 5-epoch cells sit at 60-88,
so every number here is a pre-convergence value. The frozen run's 40-80 is most
simply explained by the k-means init being spread by construction and staying
spread, not by freezing improving anything; and k-means-init-unfrozen being
*below* plain suggests the init actively disturbs early EMA. Nothing about the
mechanism's usefulness follows from these. The A4 arm (random init + freeze)
is what separates "text init helped" from "not moving helped".
