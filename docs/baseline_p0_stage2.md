# P0 stage 2 -- baselines (36-bit, CLIP features, unsupervised)

E* selected on held-out val (val_query vs opt-train DB, 10% carve, seed 42).
Cell = **test mAP@R of the 100%-train run at that E\***; (E*) in parentheses.

| Dataset | CIBHASH | CIMON | MLS3RDUH |
|---|---|---|---|
| Flickr25k | 0.8233 (E*=4) | 0.8288 (E*=49) | 0.7811 (E*=59) |
| MSCOCO | 0.8112 (E*=19) | 0.6716 (E*=59) | 0.6423 (E*=59) |
| NUSWIDE | 0.8152 (E*=4) | 0.7860 (E*=54) | 0.7746 (E*=59) |
| CIFAR10 | 0.9004 (E*=4) | 0.8367 (E*=59) | 0.5793 (E*=59) |

## Selected epochs (E*) and val mAP@R

| Method | Dataset | E* | val mAP@R | test mAP@R |
|---|---|---|---|---|
| cibhash | Flickr25k | 4 | 0.7101 | 0.8233 |
| cibhash | MSCOCO | 19 | 0.6265 | 0.8112 |
| cibhash | NUSWIDE | 4 | 0.7136 | 0.8152 |
| cibhash | CIFAR10 | 4 | 0.8681 | 0.9004 |
| cimon | Flickr25k | 49 | 0.7470 | 0.8288 |
| cimon | MSCOCO | 59 | 0.5610 | 0.6716 |
| cimon | NUSWIDE | 54 | 0.7186 | 0.7860 |
| cimon | CIFAR10 | 59 | 0.7750 | 0.8367 |
| mls3rduh | Flickr25k | 59 | 0.6834 | 0.7811 |
| mls3rduh | MSCOCO | 59 | 0.5295 | 0.6423 |
| mls3rduh | NUSWIDE | 59 | 0.7237 | 0.7746 |
| mls3rduh | CIFAR10 | 59 | 0.5060 | 0.5793 |

## Protocol notes

* E* is selected **only** on val_query vs opt-train DB. No test metric enters the selection.
* The val DB is the opt-train split, which is smaller than the official database. For Flickr25k (val DB = 4500 rows) the R=5000 cutoff never binds, so the selection metric degenerates to full mAP there. This is symmetric with our own model (same val DB), but it means the *selection* metric and the *reported* metric are not the identical statistic on Flickr25k.
* Absolute val mAP@R is not comparable to test mAP@R: different query set, much smaller DB, and a different effective cutoff.
