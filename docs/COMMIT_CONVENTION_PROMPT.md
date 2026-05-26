# GroundedDNA — GitHub Commit Convention Prompt (for Agents)

다음은 다른 에이전트가 이 repository (`yunddonii/GroundedDNA_save`) 에서 commit 할 때 따라야 할 규칙을 그대로 사용할 수 있도록 자체 포함된 프롬프트입니다.

---

## ROLE

You are committing to `GroundedDNA`, a SigLIP2-based compositional DNA hash retrieval research repo. Commits double as **lab-notebook entries**: the message body is the canonical record of what was tried, what happened, and what to do next. Treat every commit as something a future researcher (or paper reviewer) will read in isolation.

## HARD RULES (never violate)

1. **Never commit without the user explicitly asking.** When asked, follow the protocol below.
2. **Never commit `[in progress]` work.** Only completed experiments with final metrics, or finished analyses, get committed. Mid-run trajectories belong in PROJECT_LOG snippets, not as standalone commits.
3. **Always update `docs/PROJECT_LOG.md` in the SAME commit** as the experiment artifact / code. The log entry and the code change are atomic. After editing PROJECT_LOG, run `python scripts/reorder_project_log.py` so sections stay reverse-chronological with "Current state" pinned at top.
4. **Never `git add -A` / `git add .`.** Add named files only — `result/`, `params/`, `cache/`, `backup/`, `logs/`, and `slides_assets/` are `.gitignored` and must stay that way (large artefacts; only the `.json` analysis summaries inside `docs/` go in).
5. **Never use `--amend` or `--no-verify`** unless the user explicitly asks. If a hook fails, fix the cause and create a NEW commit.
6. **Never push to a branch other than `main`** unless told. After committing, push to `origin/main`.
7. **English in commit messages.** (Korean is fine in code comments / docs, but commit subject + body are English so external readers can parse them.)

## SUBJECT LINE FORMAT

Pattern: `<version-tag>[ + secondary tag]: <one-line outcome with the load-bearing number>`

The version tag is the experiment ID (`v62b`, `v78a`, `v79a/b/c/d`, `mscoco_v69a`, etc.). The outcome is one of these verdicts, ALWAYS present when applicable:

| Verdict | Meaning | When to use |
|---|---|---|
| `NEW SOTA` / `NEW <dataset> SOTA` | Beats the previous best mAP on that dataset | Headline result |
| `near-SOTA` | Within ~0.01 mAP of SOTA but with another metric improved (P@1, unique, NMI, etc.) | Paper-worthy trade-off |
| `DISCARDED` | Worse than baseline on the headline metric, won't be carried forward | Failed ablation |
| `<topic> analysis` | Post-hoc measurement (no new model) | NMI/drop/compositional studies |
| `<topic> fix` / `<file>: <fix>` | Bug or pipeline fix | Eval-script, cache, gitignore patches |
| `Plan doc:` / `Backlog:` | Documentation-only / cleanup | Use sparingly |

Always put the load-bearing number in the subject when there is one (e.g. `mAP 0.4856`, `P@1 +0.003`, `mAP -0.051`, `mAP 0.6778`). The subject must let a reader know "is this the new best, a failure, or housekeeping?" at a glance.

Examples from this repo:
- `v78a NEW MSCOCO SOTA (mAP 0.4856, P@1 0.6058); Flickr25k discarded`
- `v79a/b/c/d: 4 structural contributions attacks — v79c (hard routing) near-SOTA`
- `v78d Flickr25k K=64->96 + cosine VQ DISCARDED (mAP -0.051, worst v78)`
- `Compositional analysis: v79c hard routing halves codebook redundancy`
- `train_siglip2: wrap final extraction+evaluation in try/except`

## BODY STRUCTURE

The body is a mini lab-notebook entry. Use this skeleton — sections that don't apply can be dropped, but **never** drop the metric table for a new experiment.

```
<1–3 sentence motivation: what hypothesis / weakness / earlier finding triggered this>

<For experiments: setup table — one row per variant with the modification described concretely enough that a reader can reproduce it from the code paths>

| Tag | Modification | Implementation |
|---|---|---|
| v79a | Cross-codebook orthogonality loss λ=0.1 | `_loss_codebook_ortho` on z batch means |
| v79c | Hard routing via Gumbel-Softmax | gumbel_softmax(hard=True, τ=1.0) after Sinkhorn |

<Final results table — mAP / P@1 / P@10 / P@100 / unique / baseH / verdict. Use Δ vs the relevant baseline. Bold the SOTA, mark DISCARDED.>

| Run | mAP    | Δ vs v62b | P@1    | P@10   | unique | verdict |
|-----|-------:|----------:|-------:|-------:|-------:|---------|
| v62b| 0.6778 | —         | 0.7625 | 0.7587 | 0.075  | ★       |
| v79c| 0.6703 | -0.008    | 0.7655 | 0.7629 | 0.165  | near-SOTA |

<Key findings — bullet points, each one observational + a brief mechanism hypothesis. Why did it work / fail? Tie to NMI, redundancy, code-collapse, dynamic-τ, routing, etc.>

<For code changes: a "Code:" or "Adds:" block listing CLI flags / new files / new loss methods. Mention if defaults preserve legacy behaviour ("All default-off, bit-exact").>

<Per-dataset SOTA pairs (restate when changed):
  Flickr25k: v62b (0.6778)
  MSCOCO:    v78a MSCOCO (0.4856) >

<Suggested follow-ups (1–4 bullets): the next experiment(s) this result motivates>

<Optional: pointer to the detailed analysis doc (e.g. docs/ANALYSIS_compositional_contribution.md)>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

### Required content checklist
- Metric table with **numbers**, not adjectives. "mAP -0.012" not "slight regression".
- Concrete code anchors (function names, CLI flags, file paths in backticks).
- Verdict per variant (`SOTA / near-SOTA / DISCARDED / trade-off / weak / catastrophic`).
- Mechanism sentence: WHY the result happened, even if speculative — paper-future-you needs this.
- `Co-Authored-By: Claude Opus 4.X <noreply@anthropic.com>` trailer when an agent authored the work. Use the actual model ID currently running.

### Things NOT to include
- File diff stats (`git` shows those).
- Recap of what `git log` already shows ("this builds on v62b which was committed last week").
- Marketing words like "complex", "robust", "risk".
- TODO comments or `[in progress]` markers.
- Korean sentences in the commit message body (keep them in code/docs).

## COMMIT PROTOCOL (the actual flow)

When the user asks for a commit:

1. **Run in parallel**: `git status` (no `-uall`), `git diff` (staged + unstaged), `git log -5 --oneline`.
2. **Inspect** what's about to be staged. Refuse anything inside `result/`, `params/`, `cache/`, `backup/`, `slides_assets/`, or any `.env` / credentials file. If those show up unexpectedly, surface to the user.
3. **Ensure PROJECT_LOG.md was updated** for this change. If it wasn't and the change is non-trivial (new variant, ablation, dataset, cache, baseline, reverted decision, analysis), write the entry FIRST, then `python scripts/reorder_project_log.py`, THEN commit. This is a load-bearing project rule — verbatim from the user: "버전 관리는 무엇보다 중요한 과업이야."
4. **Stage named files only**: `git add <file1> <file2> ...`.
5. **Compose the message** following the structure above. Use a HEREDOC:
   ```bash
   git commit -m "$(cat <<'EOF'
   <subject>

   <body>

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   EOF
   )"
   ```
6. **Push**: `git push origin main`. Confirm the commit hash to the user.

## RELATIONSHIP TO `docs/PROJECT_LOG.md`

The commit message is the **summary** of the change. `docs/PROJECT_LOG.md` is the **full record**. Both must exist; neither replaces the other.

- PROJECT_LOG section header: `## YYYY-MM-DD — <version tag> (<one-line verdict>)`
- PROJECT_LOG body holds the detailed setup table, mid-eval trajectory tables (e.g. ep9/19/29/.../final), per-codebook breakdowns, drop ablations, qualitative notes, follow-up suggestions, and result-directory paths. The commit body is the compressed version.
- After editing PROJECT_LOG, ALWAYS run `python scripts/reorder_project_log.py` (idempotent). This re-sorts sections to reverse-chronological and re-pins the "Current state" header.

## ANTI-PATTERNS (observed mistakes to avoid)

- Committing an experiment without updating PROJECT_LOG.md (user has explicitly flagged this lapse before).
- Subject like `Update PROJECT_LOG.md` or `Add new experiment` — uninformative; replace with the variant tag + verdict + key number.
- Committing the `.npz` / `.pth` / training viz PNGs from `result/` — they're gitignored; if you find them staged, unstage.
- Forgetting the `Co-Authored-By` trailer when the agent did substantive work.
- Force-pushing or `--amend`-ing past commits without explicit instruction.
- Batching unrelated changes into one commit (e.g. a bug fix + a new experiment + a doc rewrite). Separate them.

## EXAMPLES TO STUDY

Run `git log --pretty=format:"%h %s" -30` for current style. For long-form body templates, study these representative commits:
- `e784ed8` — analysis-only commit (no new model)
- `790e0af` — multi-variant batch experiment with results table
- `510b85f` — single SOTA commit
- `880e03e` — DISCARDED commit (failure logged in equal detail to success)
- `c43dd8c` — cross-dataset generalisation commit with mechanism discussion

---

End of prompt. Hand this to any agent that will commit on behalf of the user, alongside the repo's `CLAUDE.md` / memory files.
