# r8 preparation tools (audits 797-810)

| File | Status | What it is |
|---|---|---|
| `build_lineage.py` | used once | built `artifacts/anchor_confirmation/anchor_t_recovery_lineage_v1.json` from JSON records and the settled R/T ledger (metadata only) |
| `sandboxed_pytest_r8.py` | **current test boundary** | OS boundary (bubblewrap): every test process sees only `/usr`, `/etc`, the conda env, the tree under test (read-only, data directories empty), a source-only git store and a private `/tmp`; no live data/result/ledger/claim path, no network, no GPU. A probe in a sandbox of the same configuration (a separate launch) checks this and the imported modules' identities before pytest |
| `guard_controls/test_sandbox_v4_controls.py` | current | the boundary's controls, run inside the sandbox (`--copy-in`) |
| `mutants_v14.py`, `../mutation_v14.py` | current | battery v14 (47 mutants), every declared test run inside the sandbox over the mutated sparse copy |
| `full_t_io_observations.json` | evidence | the four bounded 16:32–16:36 UTC I/O samples from the full-T run (audit 799 limits apply) |
| `guarded_pytest_r8.py`, `guard_site/`, `guard_controls/test_guard_v3_controls.py` | **superseded (failed development history)** | the Python audit-hook guard v1–v3; audits 802, 804, 805 and 808 found admission gaps (unbounded isolated fixtures, copied shells, a listing-history inference). Kept unchanged as history; not used for final evidence |
