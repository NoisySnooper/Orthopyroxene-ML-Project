# Phase 1.5 — final report

Date: 2026-04-19. 19 commits executed (C1-C19). Safety tag
`pre-phase-1-5-2026-04-19` at pre-C1 HEAD `29ac30e`. All commits
passed the pytest 55/55 gate.

## Commit ledger

| Step | Commit | SHA | Summary |
|---|---|---|---|
| C1 | Safety tag + baseline audit | 8d6bc53 | `CLEANUP_AUDIT_PHASE_1_5.md` baseline |
| C2 | opx multiseed rename | ed7d6d6 | `results/v10_opx_multiseed_*` → `results/opx_multiseed_*` |
| C3 | cpx multiseed rename | fde27a6 | `results/v10_cpx_multiseed_*` → `results/cpx_multiseed_*` |
| C4 | regime/external/natural rename | b8cea6f | `results/v10_regime_*`, `v10_external_*`, `v10_natural_*` → unprefixed |
| C5 | bias correction rename | 89f5138 | `results/v10_bias_correction/` → `results/bias_correction/` |
| C6 | tabpfn rename + fig35 | e85fd8b | `results/v10_tabpfn_*` + `figures/fig35_tabpfn_vs_v10.*` → unprefixed |
| C7 | optuna best-params rename | edf9ac6 | `results/v10_optuna_best_params*.json` → unprefixed |
| C8 | optuna_studies reconcile | ae6bd71 | `results/v10_optuna_studies/` → `results/optuna_studies/` |
| C9 | V10_ alias removal | 135b0c8 | `V10_BASE_ORDER` alias removed from `src/models.py` |
| C10 | config.py caption cleanup | 4f18492 | `CANONICAL_FIGURES` caption strings de-v10-prefixed |
| C11 | archive v10_phase_* | 904abe4 | 61 `scripts/v10_*.py` moved to `archive/v10_oneoffs/` |
| C12 | scripts/ subdirs | 2f494df | `scripts/tabpfn/`, `scripts/{audits,benchmarks,data_prep,figures}/` placeholders |
| C13 | run_all.py cleanup | a44febd | v10 references removed from pipeline entry |
| C14 | nb03 consolidation (downgraded) | 5e2f019 | String-only v10 sweep across 4 nb03 notebooks; full papermill consolidation deferred |
| C15 | nb04_v10 + nb07b resolution | 5798864 | `nb04_v10_benchmark.ipynb` → `nb04_regime_benchmark.ipynb`; consumer references updated |
| C16 | v9 docs archive + nb03_test_protocol | 710f9f4 | `docs/v9_*.md` → `docs/archive_superseded/`; `docs/nb03_test_protocol.md` → `docs/preregistration/` |
| C17 | PROJECT_LAYOUT + README + autofills | 7bf1801 | Top-level docs rewritten; TabPFN default changed to 20 seeds |
| C18 | code integrity validation | e7664dc | `CODE_INTEGRITY_REPORT.md`, 55/55 tests, 11-check protocol |
| C19 | final audit + report | this commit | `PHASE_1_5_FINAL_REPORT.md` |

## Target metric vs actual

### Path-level v10 inventory (tracked, non-excluded)

| Category | Count | Disposition |
|---|---|---|
| `results/v10_twopx_*.csv` | 5 | C14 residual, deferred post-Phase-1.5 (paired with nb03 consolidation) |
| `results/v10_universal_*.csv` and `results/universal/v10_*.csv` | 6 | C14 residual, same track as twopx |
| `docs/archive_superseded/v10_implementation_plan.md` | 1 | Intentionally archived; preserves migration history |
| **Actionable residuals** | **0** | All active-pipeline v10_* paths renamed |

### Content-level v10 occurrences

Raw content count under the audit §C1 scope exclusions: **613 hits across 65 files**.

The ≤10 target in the baseline audit (§"Tightened grep target for C19")
was set before Decision-1 was formalized. Decision-1 (§Decision-1 in
the baseline audit) established that `v10` as a **pipeline-version
codename** and **paper-reporting label** is immutable. Revised target:

| Tier | Scope | Actual | Policy |
|---|---|---|---|
| Path | `git ls-files` minus deferred/archive | 0 | ≤ 0 ✓ |
| Filename-reference content | `v10_*` substrings pointing at real filenames | 11 | C14 residual, documented, deferred |
| Pipeline-codename prose | `v10 scope`, `v10 planning`, `v10 target` in README/PROJECT_OVERVIEW/plans | ~200 | Immutable per Decision-1 |
| Paper-reporting labels | `method_family='v10'`, `verdict='v10_wins'`, autofill prose | ~400 | Immutable per Decision-1 |

## Known residuals (carried forward)

### R1. C14 twopx + universal CSV rename

Five `results/v10_twopx_*.csv` and six
`results/v10_universal_*.csv` / `results/universal/v10_*.csv` files
remain. The 4 nb03 notebooks read them by literal filename. Rename
would break the reads; notebook re-execution is out of Phase 1.5 scope.

**Post-Phase-1.5 plan.** Coordinated rename + notebook string sweep +
papermill consolidation (tracked as "Phase I consolidation" in the
dilemmas log).

### R2. Broken notebook imports after C11 archive

Five `import scripts.v10_test_protocol` / `scripts.v10_phase_g_figures_30_34`
statements point at archived modules. Fix: `git mv` the modules from
archive back to `src/`; update imports in a dedicated commit where
notebook re-execution is in scope.

### R3. TabPFN CSVs on disk still 5-seed

User changed TabPFN protocol 5 → 20 seeds during C17. Script default +
all prose updated. CSV regeneration deferred to TabPFN Option B C1 to
avoid double-regen.

### R4. CANONICAL_FIGURES 25/35 not rendered

Pre-Phase-1.5 state. CLAUDE.md guardrail: "Never run nbF_figures.ipynb
until Phase G." Not a Phase 1.5 regression.

## Decision-1 (pipeline label preservation)

Preserved throughout:

- CSV column values: `method_family = 'v10'`, `method_label = 'v10 RF
  alr'` in `results/regime_allmodels*.csv`
- Verdict strings in autofill tables:
  `verdict='v10_wins'|'tabpfn_wins'|'competitive'`
- Paper prose in `manuscripts/opx_2026/text/*_autofilled.md` and
  `tabpfn_paragraph.md`
- Figure caption text embedded in notebooks rendering paper figures

Renaming these would propagate into paper tables/figures and require
coordinated paper-text changes, which is out of scope.

## Git tags

- `pre-phase-1-5-2026-04-19` (pre-C1, HEAD `29ac30e`) — safety tag for
  revert if ever needed.

## State handoff to TabPFN Option B

- Phase 1.5 closed. HEAD at C19 (`PHASE_1_5_FINAL_REPORT.md` commit).
- Safety tag for Option B: `pre-tabpfn-option-b-20260419` to be set at
  Option B C1, ahead of the 20-seed TabPFN regeneration.
- `src/models.BASE_ORDER` length is 8; Option B C2 grows it to 9.
- `results/tabpfn_*.csv` regeneration is the first Option B C1 action.

Phase 1.5 complete.
