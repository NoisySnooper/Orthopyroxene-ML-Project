# TabPFN Option B integration report

**Date:** 2026-04-19
**Branch:** main
**Starting point:** Phase 1.5 cleanup closed at commit c8d8e7f (→ e7664dc for check-report, → PHASE_1_5_FINAL_REPORT.md commit for closeout)

## Scope

Promote TabPFN v2 (Hollmann et al. 2025, *Nature* 637, doi:10.1038/s41586-024-08328-6) from a supplementary post-hoc baseline to the **9th `BASE_ORDER` family**, matching the 20-seed (42-61) protocol already used by the 8 Optuna-tuned families, and splice it into the downstream comparison layer (multiseed aggregates, regime scorecard, per-regime figure, manuscript paragraph) without disturbing the pre-registered Optuna-tuned pipeline, the pre-registered scorecard's internal logic, or the Ridge stacking meta-learner.

## 11-commit plan ledger

| # | Commit | SHA | Summary |
|---|--------|-----|---------|
| Tag | `pre-tabpfn-option-b-20260419` | 541fe9d | Safety tag at HEAD before any Option B edit |
| C1 | Integration notes | 80767de | `TABPFN_INTEGRATION_NOTES.md` self-audit |
| C2 | Registry + factory | 9850701 | `BASE_ORDER[-1]='TabPFN'`; `TUNED_BASES = BASE_ORDER[:-1]`; lazy import in `src/models.py`; 7 new contract tests |
| C3 | Multiseed merge | 9ae6ed1 | Append 20-seed TabPFN rows into `{opx,cpx}_multiseed_{summary,results}.csv` |
| C4 | Regime merge | 356f89d | Append 40 TabPFN rows into `regime_allmodels.csv` |
| C5 | Bias-correction exclusion | 75a8403 | 8 TabPFN rows with `winner='excluded'` in `bias_correction_shipped.csv` |
| C6 | Figure updates | 6201813 | fig24 (nb04 cell 22) 3-series + fig35 caption; fig25/26/32 deferred to Phase G |
| C7 | Scorecard augment | 1369a3e | `tabpfn_rmse{,_lo,_hi}` columns + 4-candidate argmin winner logic |
| C8 | run_all smoke | 414513b | `run_all.py` inline comment block for the TabPFN notebook gate |
| C9 | Manuscript autofills | 93dbc31 | `tabpfn_paragraph.md` framing (title, Paragraph 1, Caveats) |
| C10 | Docs | bce86d7 | README, PROJECT_LAYOUT, PROJECT_OVERVIEW, `docs/nb03_tabpfn_plan.md`, notebook markdown |
| C11 | Integrity + report | this commit | 11-check sweep + this report |

Safety tag verified present: `git rev-parse pre-tabpfn-option-b-20260419` → `541fe9d`.

## Decision-1 reaffirmation

Pipeline label `v10` in paper-reporting contexts (column values, verdict strings, autofill prose) remains immutable. TabPFN is added as a **distinct 9th family**, not renamed or merged into the v10 label. The 8-family Optuna-tuned roster stays locked at its 2026-04-17 pre-registration; `TUNED_BASES = BASE_ORDER[:-1]` is the subset used by all Optuna-pipeline consumers (`build_oof_matrix`, `evaluate_all_bases`, `fit_internal_ensembles`, `ensemble_predict_on_test`).

## Stacking and bias-correction exclusions

`STACKING_BASE_ORDER` remains `('RF', 'ERT', 'XGB', 'GB')` — 4 Optuna-tuned tree families. TabPFN is excluded from the Ridge meta-learner because its in-context architecture does not expose a compliant OOF-residual path without modifying the locked meta-learner hyperparameters.

TabPFN is excluded from bias correction (8 rows in `results/bias_correction_shipped.csv` with `winner='excluded'`) for the same reason: a single in-context forward pass yields no OOF residuals to fit the regime-piecewise (Form A) or sigmoid-blend (Form B) corrections on.

## 11-check integrity results (C11)

Run via `scripts/tabpfn/_c11_integrity_sweep.py`. Final: **ALL CHECKS PASSED**.

| # | Check | Status | Detail |
|---|-------|--------|--------|
| 1 | py_compile sweep | OK | 31 files compiled (src + scripts + config + run_all) |
| 2 | core imports | OK | config + 9 src modules (bias_correction, data, evaluation, external_models, features, io_utils, models, opx_tb_analysis, stacking) |
| 3 | config smoke | OK | CANONICAL_FIGURES len 35; SPLIT_SEEDS [42..61]; BASE_ORDER len 9 (last=TabPFN, TUNED_BASES len 8); STACKING_BASE_ORDER=('RF','ERT','XGB','GB') |
| 4 | run_all --help | OK | exit 0 |
| 5 | notebook JSON validity | OK | 16 notebooks parse |
| 6 | notebook first-cell ast.parse | OK | per-notebook first code cell compiles |
| 7 | v10/V10 grep count | 11309 hits | Reconciled via Decision-1 in `PHASE_1_5_FINAL_REPORT.md`: pipeline label `v10` in paper-reporting contexts (column values, verdict strings, autofill prose) is immutable |
| 8 | stale TabPFN phrase grep | OK | 0 hits for `5-seed ensemble`, `TabPFN v2 supplementary baseline`, `TabPFN is deliberately outside` (excluding sealed preregistration and Phase-G-gated figure sidecars) |
| 9 | CANONICAL_FIGURES on-disk | 10/35 | 25 missing — expected, Phase G gated; CLAUDE.md: "Never run nbF_figures.ipynb until Phase G" |
| 10 | pytest | OK | 62 passed (55 baseline + 7 TabPFN contract tests) |
| 11 | CSV schema | OK | All 6 merged-CSV shapes match target; TabPFN row counts match (4/4/80/80/40) |

## Known residuals and deferrals

- **R1: fig25, fig26, fig32, fig35 rendering.** All four live in `nbF_figures.ipynb`, which is locked until Phase G per CLAUDE.md. Cell source edits to add TabPFN bars are deferred to Phase G. The fig35 source cell in nbF has been updated to drop the 5-seed caption text; fig24 is rendered by `nb04_regime_benchmark.ipynb` (not `nbF`) and was updated in C6.
- **R2: Number refresh in `manuscripts/opx_2026/text/tabpfn_paragraph.md`.** The auto-filler `scripts/tabpfn/opx_tb_nb03_fill_tabpfn_paragraph.py` uses one-shot sentinels that were consumed on the prior 5-seed run. Re-running with 20-seed data requires either re-inserting the sentinels or modifying the script to detect the existing paragraphs by boundary anchors. Deferred to a downstream pass; the framing (title, Paragraph 1, Caveats) is already 20-seed / 9-family correct.
- **R3: `results/tabpfn_head_to_head.csv`.** Produced by the nb04 head-to-head cell on a 5-seed basis; needs a re-run against the 20-seed multiseed summary. Downstream consumer (the autofill script) is deferred per R2.
- **R4: Pre-registration document `docs/preregistration/nb03_test_protocol.md`** still references "TabPFN v2 supplementary baseline smoke test" (T19). This is a **sealed historical record** of the 2026-04-17 pre-registration (before the 2026-04-19 Option B promotion), so it is deliberately left untouched. Accept as archival.
- **R5: `figures/fig35_tabpfn_vs_opx_tb.txt`** sidecar still carries the 5-seed caption. This will be regenerated in Phase G when the figure is re-rendered from the 20-seed data. Accept as ephemeral.

## Handoff to Phase G

Phase G will:
1. Unlock `nbF_figures.ipynb` and re-render canonical figures 1-35.
2. Add TabPFN bars to fig25, fig26, fig32, fig35 cells at that time (C6 prep notes the anchor cells).
3. Refresh the tabpfn_paragraph.md numbers block against the 20-seed regen.
4. Commit the refreshed autofill script + tables to manuscripts/opx_2026/text/.

## Head-to-head result (20-seed aggregate)

On the 40-cell post-correction scorecard (4 tracks × 2 targets × 5 regimes):

| Winner | Count | Notes |
|---|---|---|
| external | 19 | Putirka/Agreda/Jorgenson lowest RMSE |
| v10_pre | 10 | Pre-correction ML pipeline lowest |
| v10_corrected | 9 | Post-correction ML pipeline lowest |
| **tabpfn** | **2** | TabPFN v2 strictly beats v10 pipeline + external |

TabPFN-winning cells:
- `opx_liq / P_kbar / deep_crustal_MASH`: TabPFN 2.01 kbar vs v10 2.08 kbar vs Putirka 29a 2.31 kbar
- `opx_only / T_C / deep_crustal_MASH`: TabPFN 107.4 °C vs v10 109.5 °C (no external for `*_only` tracks)

## Sign-off

TabPFN v2 is now the 9th `BASE_ORDER` family in this repo. It is a **post-hoc comparison baseline, not a primary-model replacement** — the paper's pre-registered verdict remains the v10 tuned pipeline vs Putirka/Agreda/Jorgenson, and TabPFN appears side-by-side as an independent foundation-model reference. Head-to-head counts above inform the Section 5 discussion paragraph but do not change the pre-registered scorecard's decision logic.
