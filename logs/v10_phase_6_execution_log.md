# Phase 6 execution log

Started 2026-04-20. Scope: comprehensive fix + audit sweep (Steps 1-15).

## Step 1 — Preregistration amendment (DONE)

- Wrote `docs/preregistration/AMENDMENT_1_acceptance_rule.md`.
- Registers tiered rule v2 with `N_MIN=20`: regimes with `n < N_MIN` are
  logged but do not veto a global-improving correction form.
- `N_MIN=20` confirmed by user.

## Step 2 — Extend `src/bias_correction.py` (DONE)

- `ShipDecision` gained `low_n_degradations: dict`.
- `ship_decision(...)` gained kwargs `per_regime_n=None, n_min_for_veto=0`.
- `n_min_for_veto=0` reproduces v1 byte-for-byte (backward-compat guarantee).

## Step 3 — Rescore under v2 (DONE)

- Script: `scripts/bias_correction/rescore_under_tiered_rule.py`.
- Outputs: `results/bias_correction_shipped_v2.csv`,
  `results/preregistered_scorecard_postcorrection_v2.csv`.
- Sanity check with `n_min_for_veto=0` vs v1: 0 mismatches.
- v2 vs v1 differences:
  - 1 shipped-verdict flip: opx/opx_liq/P_kbar, `none -> A` (deeper_mantle
    regime n<20 had vetoed under v1, non-vetoing under v2).
  - 3 scorecard regime flips in opx_liq/P_kbar (v10_pre -> v10_corrected).
- TabPFN rows: v1 verdicts preserved verbatim (no OOF residuals available).

## Step 4 — Regenerate ship-verdict figures (DONE)

- Core_05 (`make_fig_bias_correction_opx.py`): dual "v1 ships X / v2 ships Y"
  annotation when they differ; suptitle explains v2 primary annotation.
- Core_07 (`make_fig_scorecard_delta_opx.py`): loads v2 scorecard for
  post-RMSE, flags cells with "*" where v2 post-RMSE differs from v1;
  title/caption updated to reference Amendment 1.
- Core_08 (`make_opx_headline_fig.py`): loads v2 first with v1 fallback;
  shows "ALL winner (v2): X [method] (v1 winner: Y)" when flipped.
- SI figs (fig30-44, aug01-04): regenerated via notebook rerun (nbF
  deferred per CLAUDE.md guardrail; direct scripts invoked).

## Step 5 — SHAP expansion (WINNERS ONLY, per user direction)

- Scope restricted by user on 2026-04-20 to the 8 shipped-winner models
  (not 9 families x 8 cells).
- Script: `scripts/shap/run_shap_winners.py`.
- Explainer dispatch:
  - Tree models (RF, ERT, LightGBM, XGB, CatBoost): `TreeExplainer` (exact).
  - ElasticNet: `LinearExplainer` on the `scaler -> enet` pipeline's
    scaled features.
  - MLP: `sklearn.inspection.permutation_importance` (20 repeats,
    seed=42) as a fast SHAP surrogate. KernelExplainer would add ~30 min
    per MLP cell; deferred in Phase 6 one-shot scope.
- Outputs: `results/shap_values_winners.npz` (8 keys),
  `results/shap_importance_winners.csv` (310 rows).
- Figure: Core_12 (`make_fig_shap_winners.py`). 2x4 grid, top 10 features
  per cell, explainer note in each subtitle.
- All 8 cells completed without failure.

## Step 6 — Per-family bias correction (DROPPED, per user direction)

- Original scope: 9 families x 8 cells x 20 seeds x 2 forms (~12h compute).
- User direction 2026-04-20: "Bias correction will only be applied on
  winning model only." -> step dropped.

## Step 7 — Fix Core_06 axis scaling (DONE)

- Plan referenced "fig31/Core_06 cpx-only P axis", but Core_06 is opx-only
  and no cpx residual figure exists.
- Resolution: applied a universal 1-99 percentile axis clip (with 5%
  padding) to all four opx panels in `make_fig_bias_residuals_opx.py`.
  Outliers still visible at the frame; body of the scatter no longer
  collapses into a thin strip.

## Step 8 — TabPFN coverage audit (DONE)

Confirmed all 8 cells present across the canonical TabPFN surfaces, no
coverage gaps, no remedial action needed:

- `results/tabpfn_multiseed_summary.csv`: 8 rows (one per pipe/track/target).
- `results/tabpfn_regime_rmse.csv`: 40 rows (8 cells x 5 regimes).
- `results/preregistered_scorecard_postcorrection.csv`: 40/40 `tabpfn_rmse`
  non-null, 40/40 `tabpfn_post_rmse` non-null.
- `results/tabpfn_head_to_head.csv`: 8 rows.
- `results/regime_allmodels.csv`: 40 rows with `method_family='tabpfn'`.

Rationale for skipping TabPFN in v2 rescore (already registered in
`AMENDMENT_1_acceptance_rule.md`): TabPFN fits are in-context and emit no
OOF residuals, so Form A / Form B cannot be refit under v2 without
rerunning the full 20-seed sweep (~8h). v1 ship verdicts are
preserved verbatim and flagged in the v2 shipped CSV.

## Step 9 - Terminology audit v3 (DONE)

- Script: `scripts/audits/terminology_cleanup_v3.py` (three-pass).
- Extends v2 with a NEW Pass 0 that pre-cleans frankenwords produced by
  v2 (`pre-registered the tuned ML baseline`, `uncorrected the tuned
  ML baseline predictions`, etc.) BEFORE Pass 1 re-hits them.
- Scope unchanged: `manuscripts/**/*.md`, `figures/**/*.txt`,
  `deliverables/lee_package_20260420/figures/**/*.txt`.
- Run result: 9 files modified, 18 replacements. Report:
  `TERMINOLOGY_CLEANUP_REPORT_V3.md`.
- Grep post-run: 0 stale `v10`, `v9`, `two-axis honesty`,
  `no-degradation rule`, `the the tuned` occurrences in targeted
  surfaces.

## Step 10 - Data audit + qhat validation (DONE)

- All 6 processed parquets present, all have T_C and P_kbar columns,
  zero NaN in either target column, MD5 recorded.
- All 12 split index files present, 0 train/test overlap on every
  pipeline.
- `results/nb07_conformal_qhat.json` schema consistent: nested
  `T_C.qhat` / `P_kbar.qhat` match flat `q_hat_T_C` / `q_hat_P_kbar`.
- Coverage at alpha=0.10: T empirical 0.793 (UNDER-covers the
  registered 0.90 target; documented limitation), P empirical 0.960
  (exceeds 0.90).

## Step 11 - Code audit v2 (DONE)

- All 9 Phase-6-edited modules import cleanly (0 failures).
- Backward-compat smoke test: `ship_decision` with default
  `n_min_for_veto=0` produces v1 verdicts byte-for-byte across
  shipping / regime-veto / overall-veto fixtures.
- `n_min_for_veto=20` logs low-n regimes to `low_n_degradations` but
  only vetoes on high-n regimes.

## Step 12 - Manuscript audit (DONE)

- nb09 (`notebooks/nb09_manuscript_compilation.ipynb`): scanned
  markdown cells for stale v10 / v9 / coinages. Found one stale
  reference in cell 29 ("v10 tuned pipeline" in TabPFN head-to-head
  section header); patched to "tuned ML baseline pipeline".
- Post-patch grep is clean.

## Core_12 - TabPFN-scorecard-winner flag (DONE, user-raised)

- User noted 2026-04-20: for opx_only T and opx_only P, the 5-way
  scorecard winner is TabPFN, not LightGBM/RF.
- Fix: `scripts/figures/make_fig_shap_winners.py` now flags panels
  (c) and (d) with "scorecard winner: TabPFN post (no SHAP
  available); shown: tuned runner-up ..." subtitle. Suptitle and
  caption extended with the same clarification.
- Rationale for NOT using TabPFN for SHAP: TabPFN is in-context, no
  TreeExplainer / LinearExplainer / gradient surface. Permutation
  importance would be possible but slow (~30 min per cell, deferred).

## Step 13 - Preregistration tests T15-T18 (DONE)

- New file: `tests/test_preregistration.py`, 9 tests covering:
  - T15: `ship_decision` default reproduces v1 (shipping + 2 veto
    cases).
  - T16: low-n degrading regime does NOT veto under v2; high-n does.
  - T17: threshold boundary (n=20 vetoes, n=19 does not); N_MIN=20
    hard-coded in rescore script.
  - T18: TabPFN rows in v2 shipped CSV match v1 byte-for-byte.
- Run: 9/9 pass. Full suite: 78/78 pass.

## Step 14 - Advisor v3 package rebuild (DONE)

- `scripts/deliverables/build_lee_package.py` rewritten with:
  - CORE_FIGS list fixed: added Core_01b, corrected Core_08 stem,
    added Core_12 (new SHAP). Now 14 figures total.
  - Notebook section order REORGANIZED per user request:
    1. Core figures (first)
    2. SI figures
    3. Data tables (moved after figures)
    4. Methods summary (plain English)
    5. Limitations
    6. Caveats
    7. Provenance
    8. Pre-registration (verbatim)
  - Intro rewritten for a geologist with no ML background:
    "project in one paragraph" + mini glossary covering opx/cpx,
    pressure regimes, RMSE, bias correction, ship-if-better rule,
    TabPFN, Putirka.
  - Methods section rewritten in plain English, explicitly
    documents Amendment 1 (tiered rule, N_MIN=20).
- Output: `deliverables/lee_package_20260420/00_ADVISOR_REVIEW.html`
  (17.8 MB, PDF-exportable via browser Print to PDF), executed
  notebook (81 cells), figures/ (stale Core_08_opx_only_P files
  cleaned).

## Step 15 - Smoke test + package verification (DONE)

- Full test suite: 78/78 pass in 3.4s.
- Executed notebook section order verified: figures first, tables
  after, methods/limits/caveats/provenance/preregistration at the end.
- Package sanity-checked: all 14 Core_* figures present as PDF+PNG+txt
  triples; stale Core_08_fig45_opx_only_P_headline.* removed.

