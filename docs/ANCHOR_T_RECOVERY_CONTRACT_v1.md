# Anchor model — stage-T recovery contract v1 (generation v9 r8; audits 795–797)

**Scope.** Exactly one thing: the recovery of the one full-T cell that the r7 campaign `ancT9` did not
finish, NUS-WIDE seed 44 (`nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors`).
That campaign was stopped by its budget rule on 2026-10-06 at 20:50:07Z, and audit 795 accepted the
stop and settlement. The other eleven cells are **carried**: they are never evaluated again.

This contract is not:
- a resume flag;
- a generic subset mechanism;
- a retry of any other cell;
- a grant of budget or of an exception.

Running it needs the user's decision on the added budget and the one-attempt exception, then an
audit approval line of scope `stage-T-recovery` naming the exact request digest.

## 1. What is fixed

These are never changed:
- F (`ancF_candidate_v1.json`, accepted in §744);
- N, the λ values, top-p and the joint weight;
- the checkpoint, config and runtime witness of the cell;
- the seed, the inference epoch (the terminal epoch N) and the ranking policy;
- the input seals;
- the stage-R campaign `ancR9` and the eleven carried T records;
- the five-producer chain and the output contract: twelve required outputs, and
  `evaluation_siglip2_bit2.json` absent (§776).

**What the exception is.** It recovers a predetermined missing cell after a resource failure. It is
not a claim that no test information exists:
- the interrupted entry opened the official-test path (DB extraction reached batch 140/757);
- the eleven other results are already visible.

## 2. Authority: historical versus executing

Two generations are involved, and each is used for its own purpose.

| Authority | Generation | How it is used |
|---|---|---|
| Historical | r7, manifest `2f24fc80…` | F, the stage-R receipt and its approval (§780), the stopped stage-T request and its approval (§788) are all verified against r7. The r7 manifest is read from its pinned bytes and never loaded as the running tree's generation |
| Executing | r8, its own manifest | Generation r8 executes. Its closure differs from r7 in exactly `RECOVERY_CHANGED_CLOSURE` (changed) and `RECOVERY_ADDED_CLOSURE` (added), with nothing removed (`generation_transition`) |

At the T boundary (`recovery_boundary`), the stage-R snapshot's 51 sources are compared with the
running tree:
- **Exactly** `RECOVERY_CHANGED_SOURCES` may differ: `scripts/anchor_refit_stage.py`,
  `scripts/anchor_terminal_test.py` and `scripts/phase3_selection_matrix.py`.
- Each of those must be its r7 bytes in the stage-R snapshot and its committed r8 bytes now.
- Every other source must be its stage-R bytes. Its HEAD and worktree authority must be unchanged
  too.
- The stage-R inputs must be unchanged.
- Scientific inference, extraction, ranking, metrics, splits and the recipe are therefore unchanged.

No r7 record is relabelled as r8, and the old admissions are unchanged. The only addition to stage-R
admission is a keyword used by the recovery alone: `carried_cells` in `admit_refit_receipt`.

## 3. The pinned lineage

**The lineage file.** `artifacts/anchor_confirmation/anchor_t_recovery_lineage_v1.json` is pinned by
`RECOVERY_LINEAGE_SHA256`. It was built from JSON records and the settled ledger only. It pins:
- the r7 manifest;
- the stage-R receipt;
- the stopped snapshot, at both its byte digest and its semantic digest;
- the stopped request digest;
- the stopped reservation and campaign nonce;
- the §788 approval line;
- the settled R/T ledger: its whole-file digest, line count, the start/stop/final line digests, the
  stop reason, the final status, return code and charge, and the cumulative charge
  79,482.14623009507 s;
- the recovered cell's attempt and entry;
- each carried cell's T record, attempt and entry.

**Admission** (`admit_stopped_campaign`) uses JSON, the ledger and `stat` only. It requires all of the
following:
- the stopped snapshot is the r7 run request of namespace `ancT9`, and the reservation binds it;
- the §788 line is still in the audit ledger;
- no `ancT9` receipt exists;
- the settlement is the audited clean budget stop (`verify_settlement`):
  - the whole ledger is at its pinned bytes, with this run's start, stop and final in that order and
    the final as the last line;
  - status `stopped` with a `budget:` reason;
  - no held lease, orphan, continuity loss or monitor failure;
  - the pinned charges;
- **exact membership:** the stopped request has twelve distinct cells, and the carried set is those
  twelve minus the recovered cell — no duplicate, missing cell, foreign seed or changed run directory;
- the stage-R receipt, re-admitted under r7, yields exactly the stopped request's cells. The eleven
  carried run directories hold the twelve required outputs with bit2 absent;
- every carried T record, attempt and entry is at its pin, has its contracted name, and binds the
  stopped request, nonce, cell and attempt bytes;
- for the recovered cell:
  - its attempt and entry are at their pins and bind its lineage;
  - it has no T record and no contract output;
  - no other namespace in the historical record directory has attempted it.

**Payload verification** (`verify_carried_payloads`) runs only at the approved execution boundary. It
checks that every carried output is the bytes its T record binds. The files are hashed, never
deserialized. A metadata render (`--plan`) reads no payload.

## 4. The one recovery claim

The claim (`consume_recovery_claim`) lives in `RECOVERY_CLAIM_ROOT` (`/home/yschoi/gdna_anchorRT_recovery_claims`),
outside every worktree, record root and namespace.

**Its name** is the lineage key alone, `recovery_claim_key`. That is the digest of:
- the stopped run id;
- the stopped request digest;
- the cell id;
- the original attempt and entry digests.

**When and how it is created:** with `O_EXCL`, after the stage-R and recovery checks and before the
attempt is reserved, so before any test access (`run_terminal_test_cell(..., before_attempt=...)`).

**What it binds:**
- the claim key;
- the request digest;
- the approval section, scope and line;
- the namespace;
- the record root;
- the attempt name it authorizes;
- the lineage pin.

**It is never removed**, including after a failure. A concurrent caller, a second namespace, an
alternate record root or another worktree computes the same name and refuses. A second recovery
needs a new authorization.

**The T entry checks the authority itself** (`verify_recovery_authority`), before any configuration,
model or test construction. It does not trust a parent flag or the environment. It:
1. re-admits the pinned lineage from the records;
2. requires the request's lineage part to equal what that admission determines
   (`recovery_lineage_block`);
3. requires the claim to name this request, approval line, namespace, record root and attempt.

It accepts the recovery schema only together with mode `recovery`, scope `stage-T-recovery` and
exactly one cell. Neither the stage-T form nor the recovery form is ever accepted as the other.

The rest of the entry is unchanged:
- the exclusive entry claim;
- the verified-byte configuration read once;
- the terminal runtime bound at the admitted epoch;
- the source, input and cell rechecks between all five producers.

## 5. Records and receipt

**New namespace.** The recovery writes a new namespace in the r8 record directory. It is never
`ancT9`, and the request names it. In it go:
- a reservation and a snapshot (schema `anchor-terminal-test-recovery-snapshot/1`);
- one attempt, one entry claim and one T record.

**The receipt** is `<namespace>_recovery_complete.json`, schema
`anchor-terminal-test-recovery-receipt/1`. It lists twelve cells:
- eleven with `origin: carried`, each with the `ancT9` namespace, the historical record directory,
  the record name and digest, and the attempt digest;
- one with `origin: executed`, giving the new record.

It also binds the lineage, the stopped campaign, the recovered cell's original attempt and entry,
and the claim.

No `ancT9_test_complete.json` is written. A consumer must accept the recovery receipt kind
explicitly.

**Final closure before publication** (audits 800–801; `final_recovery_closure`). It runs after the
new cell's five producers and immediately before the combined receipt is published. It is
fail-closed and metadata-only plus hashing: nothing is admitted, claimed or loaded again. It
requires all of the following:
1. **The historical lineage again** (`verify_stopped_lineage`, with the recovered cell now marked
   executed):
   - the eleven carried records, attempts and entries at their pins and bindings;
   - their outputs present, with bit2 absent;
   - the stopped snapshot, reservation, approval and settled ledger.
2. **The carried payloads again:** every carried output at the bytes its record binds.
3. **The request's lineage part** is still what the lineage determines.
4. **The claim** is the bytes this campaign consumed.
5. **The new cell's own chain, re-read from its files** in the request's record root:
   - the record, at its returned digest and contracted name, binding this namespace, nonce, request,
     cell and attempt;
   - the attempt file at that digest, binding this schema, namespace, nonce, request, approval and
     cell;
   - the exclusive entry claim, binding that attempt digest, the cell, the checkpoint and the config;
   - the twelve outputs, at the bytes the record binds, with bit2 absent.

The receipt binds the identities this closure verified: each carried record, attempt and entry, and
the new record, attempt and entry. Only the pre-execution check is not repeated: that the target holds
no output.

A failure at any of these steps exits nonzero and publishes no combined receipt. It keeps the spent
claim, the new record and attempt, and every original record, and it never authorizes a retry.

## 6. Budget and supervision

The recovery uses budget shape (i) (§797.2 item 4): supervisor stage `stage-T-recovery`, with its own
ledger root `/home/yschoi/gdna_anchorRTrec_ops` and budget 15,000 device-s.

**Parent binding.** Each start binds the settled R/T ledger (`parent_settlement`), and refuses
otherwise. The ledger must be at its exact digest, end in a final record, have every run settled,
and carry the cumulative charge 79,482.14623009507 s. The settled ledger is never appended, reset or
rewritten.

**What the records show:**
- the start event records `parent`;
- the final event records `parent_charged_seconds` and `cumulative_including_parent_seconds`;
- the cumulative maximum over R, T and the recovery is 79,482.14623009507 + 15,000
  = 94,482.14623009507 s.

**Limits:**

| Item | Value |
|---|---|
| Logical cells | 1 |
| Physical GPUs | 1 |
| Managed producers | 5 (a sixth is excess) |
| Wall limit | 8 h |
| Headroom | 130 s |
| Allowance | 5 × the longest observation window |

The watchdog, stop and storage rules are unchanged.

## 7. Failure rule

There is one recovery attempt. On a refusal, drift, a producer failure or a resource stop:
1. preserve every claim, reservation, attempt, entry, record and partial output;
2. settle through the supervisor;
3. return to the audit.

No retry, deletion, namespace substitution or partial continuation follows.
