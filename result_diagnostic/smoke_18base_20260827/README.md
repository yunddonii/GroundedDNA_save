# The 2026-08-27 smoke ran at 18 bases, not 15. Quarantined.

Launched as a Phase 3 smoke and committed as "Phase 3 smoke passes" in
`521c5ae`. That was wrong, and the artefacts here say so directly:

```
num_semantic_parts = num_codebooks = 6      (the paper is 5)
total_bases = 18, total_bits = 36           (the paper is 15 / 30)
extract_db.npz  base_indices (59000, 18)
no_gumbel_softmax  unset                    (the S5 recipe sets it)
lambda_codeword_codon_sinkhorn = 0.1        (the S5 recipe uses 0.0)
```

`prompt_ablation_A_cell.sh` passes `NUM_CODONS=3` and nothing else about the
geometry, and M defaults to 6 in `config.py`, `model_siglip2.py` and
`dataloaders.py` alike. Its own header says "18-base" on line 2 and "the four
paper 15-base panel" twenty lines later. Nothing asserted the effective
geometry, so a run that was internally consistent at M=6 looked like a pass.

Run against the Phase 3 contract (M=5) the shared validator rejects it:
`db: manifest num_slots=6 but caller expects 5`. It is also incomplete as an
analysis: no `extract_train.npz`, no pairwise NMI, no `analysis_complete.json`,
no run completion marker.

What it *does* show, and the only thing it should be cited for: the three
handoff defences work. Stage 1 selected E* on held-out train validation and
skipped the official test, both stages carried distinct run manifests, the
caches resolved to the v6prov provenance root, and the claims were released at
exit. Those properties are geometry-independent.

Numbers here are diagnostic-only and must not enter any table.
