# Claude Code Overnight Prompt: Dr. Lee Revision Phases 1 + 2

## Purpose

Run both revision phases back-to-back, overnight, autonomously. Phase 1 fixes the ElasticNet zero-std artifact, expands the 22-row pairing matrix, and finalizes manuscript. Phase 2 adds the 8-layer interpretability bundle, the best-per-cell defense, and the black-box remediation demanded by Dr. Lee. Expected duration: 10 to 14 hours of CPU plus Claude Code reasoning overhead. Run after dinner, wake up to completed draft.

## Repository and scope

Root: `C:\Users\NQTa\Documents\MLCourse\Final Project`
Target branch: `lee_revisions_overnight`
Audit reference: `MANUSCRIPT_WRITING_AUDIT.md`
Prior Phase 1 spec: `prompts/tabpfn_and_figures_rework/LEE_REVISIONS_PHASE_1_PROMPT.md`

Read-write scope: `results/`, `src/`, `scripts/`, `figures/`, `manuscripts/`, `tests/`, `prompts/`. Do NOT edit `data/`, `docs/preregistration/`.

## Overnight execution protocol

**Checkpoint discipline.** After each major section completes, write a one-line entry to `results/OVERNIGHT_RUN_LOG.md` with timestamp, section name, status (complete / partial / failed), next action. Human operator can resume from the last checkpoint in the morning if the run aborts.

**Halt escalation.** Use the halt conditions at the bottom of this prompt. On halt, write `results/HALT_REPORT.md` and stop. Do not retry past a hard halt.

**Git hygiene.** Create branch `lee_revisions_overnight` at start. Commit at the end of each phase. Do not push.


## PART 1 — Execute Phase 1

Read the Phase 1 prompt file at `prompts/tabpfn_and_figures_rework/LEE_REVISIONS_PHASE_1_PROMPT.md` and execute every step in it end-to-end. Full specification there. Summary of scope:

1. Diagnose and fix ElasticNet zero-std artifact by replacing seed-variance CI with bootstrap CI on residuals across all 9 families and 8 cells. Output `results/bootstrap_rmse_cis_all_cells.csv`.
2. Build 22-row natural-sample pairing matrix. Output `results/pairing_matrix_22rows.csv`, `figures/core/Core_13a_fig_agreement_matrix_T.pdf`, `Core_13b_*.pdf`, `Core_14a_*.pdf`, `Core_14b_*.pdf`.
3. Insert §4.7 natural-sample section and §3.4 uncertainty paragraph and §3.10 natural-sample protocol into manuscript.
4. Update Table 2 with CI columns, rewrite §4.1 prose with `[CI lo, hi]` format.
5. Terminology sweep, Word doc regeneration, test suite run, audit refresh.

When Phase 1 completes, commit with message `"feat: phase 1 complete - bootstrap CI + pairing matrix + §4.7"` and write `results/OVERNIGHT_RUN_LOG.md` entry "Phase 1 complete." Proceed to Phase 2.

If any Phase 1 acceptance criterion fails, write it to `results/OVERNIGHT_RUN_LOG.md` but still proceed to Phase 2 — Phase 2 work does not depend on Phase 1 completeness except for the updated `bootstrap_rmse_cis_all_cells.csv`. If that CSV is missing, halt and report.


## PART 2 — Phase 2: Black-box interpretability bundle + best-per-cell defense

Goal: neutralize Dr. Lee's two structural critiques. (A) Best-model-per-cell is a defensible methodological choice, not indecision; document the defense with citations and a sensitivity table. (B) ML predictions are interpretable in terms of classical petrologic variables; prove this from eight independent angles.

### P2.1 — Best-per-cell defense (manuscript edit + supplementary table)

Add a new §4.1.1 subsection "Methodological note: per-cell best-model selection" to `manuscripts/opx_2026/text/draft/sections/04_results.md` immediately after §4.1 paragraph 3. Prose to insert:

"The per-cell best-model selection is a deliberate methodological choice rather than an artifact of indecision. Under the No Free Lunch theorem (Wolpert, 1996), no single learning algorithm dominates across all tasks; our eight (track, target) cells are heterogeneous regression problems with distinct residual structures, training distributions, and target-variable semantics, which the theorem explicitly identifies as the regime where a single-family commitment is suboptimal. The AutoML literature (Erickson et al., 2020; Feurer et al., 2015) operationalizes per-task family selection from a candidate pool as the current standard practice for tabular regression; Grinsztajn et al. (2022) demonstrate empirically across 45 benchmark datasets that no single tree-based or neural family dominates and that per-task selection is the correct protocol. Our sweep recovers the same empirical finding on the specific task class of experimental-petrology ML thermobarometry: five different model families win across eight cells, with pwlr features selected in five cells, alr in two, and raw in three. The prior ML thermobarometer literature's default of committing to extra trees (Petrelli et al., 2020; Jorgenson et al., 2022; Ágreda-López et al., 2024) is a simplifying assumption that our evaluation empirically rejects at these sample sizes. We retain the per-cell winners as the headline and present a single-family sensitivity table (Supplementary Table S15) for readers who prefer a consistent family across all cells."

Build supplementary Table S15 at `manuscripts/opx_2026/tables/S15_single_family_sensitivity.csv` and `.tex`. One row per tuned family (RF, ERT, XGB, Histogram GB, CatBoost, LightGBM, ElasticNet, MLP) plus one row for TabPFN. Columns: family name, and one RMSE column per (track, target) cell with 95% CI in `[lo, hi]` format. Bottom row: per-cell winner (family name). Source data: `results/bootstrap_rmse_cis_all_cells.csv` (generated in Phase 1). Add table caption: "Single-family sensitivity. Each row reports the RMSE and 95% CI for one family evaluated on every (track, target) cell. Readers who prefer a single-family pipeline can read any row; the per-cell winner row (bottom) is the headline result used in the main text. RMSE differences between the per-cell winner and any single family are typically under 10% in relative terms."


### P2.2 — Feature concordance across families (Method 1)

For each of 8 (track, target) cells, compute permutation importance across ALL 9 families at canonical seed 42 on the held-out test set. Use permutation importance (not SHAP) as the common metric because SHAP implementations differ fundamentally across model types (tree SHAP, linear SHAP, kernel SHAP) and produce non-comparable scales.

Write script `scripts/interpretability/compute_feature_concordance.py`:
1. Load per-cell test predictions and fitted models from `models/` (or refit at seed 42 if joblibs absent).
2. For each cell × each of 9 families, compute permutation importance with `sklearn.inspection.permutation_importance(n_repeats=20, random_state=42)`.
3. Rank features within each (cell, family) by mean importance.
4. Compute pairwise Spearman rank correlation across family pairs within each cell.
5. Write `results/feature_concordance_permutation_importance.csv` with columns: track, target, family, feature_name, importance_mean, importance_std, rank.
6. Write `results/feature_concordance_spearman_matrix.csv` with one block per cell: 9 × 9 Spearman correlation matrix across family pairs.

Build `scripts/interpretability/make_fig_feature_concordance.py` to produce 8 heatmaps, one per cell, saved as `figures/core/Core_15_fig_feature_concordance.pdf` (multi-page PDF, one page per cell). Each heatmap: rows = top 15 features by aggregate importance across families, columns = 9 families, cell values = per-family importance rank (1 = most important). Viridis colormap. Annotate with median pairwise Spearman correlation in the title of each panel.

Interpretation rule: median Spearman ρ > 0.7 across family pairs in a cell means broad agreement on feature ranking. 0.3 < ρ < 0.7 means mixed agreement worth narrative attention. ρ < 0.3 means genuine disagreement, which is itself a finding worth reporting (does not invalidate the model; suggests different families capture different petrologic signals).

### P2.3 — Classical-feature ablation (Method 2)

For the 4 shipped opx cells (opx-liq T, opx-liq P, opx-only T, opx-only P), retrain the winning family using ONLY the features that appear in the corresponding classical Putirka (2008) equation. Compare RMSE to the full-feature winner. Three interpretive outcomes per cell.

Write script `scripts/interpretability/classical_feature_ablation.py`:
1. Per cell, identify classical-feature subset:
   - opx-liq T (Putirka 28a): liq_MgO, liq_FeO, liq_SiO2, liq_Al2O3, liq_CaO, liq_Na2O, liq_K2O, P_kbar. Drop all pwlr / alr / structural transforms not derivable from these raw oxides.
   - opx-liq P (Putirka 29a): liq_SiO2, liq_Al2O3, Al_VI_opx, Si_opx, Fet_opx, jadeite-solubility proxy features.
   - opx-only T (no Putirka opx-only T equation exists; use the classical two-pyroxene features Ca_Opx_cat_6ox, Fe_Opx_cat_6ox, Mg_Opx_cat_6ox, Al_Opx_cat_6ox as the classical-analog feature set).
   - opx-only P (Putirka 29c): Al_Opx_cat_6ox, Ca_Opx_cat_6ox, Cr_Opx_cat_6ox. No liquid features.
2. Refit the winning family on the reduced feature set at seed 42, with Optuna hyperparameters frozen at the full-feature values (do not re-tune).
3. Compute test RMSE with bootstrap CI (n_boot=500, seed=42).
4. Report in `results/classical_feature_ablation.csv` with columns: track, target, full_feature_rmse, full_feature_ci, classical_feature_rmse, classical_feature_ci, rmse_delta, rmse_delta_pct, interpretation.

Interpretation string per row (auto-generated from numeric thresholds):
- `rmse_delta_pct < 5%` → "ML uses classical information; black-box concern unfounded for this cell"
- `5% <= rmse_delta_pct < 15%` → "ML uses classical information plus smooth residual signal; interpretable"
- `rmse_delta_pct >= 15%` → "ML relies on non-classical features beyond classical petrologic intuition; warrants feature-level investigation"


### P2.4 — Classical-equivalence regression (Method 5) — STRONGEST DEFENSE

For each of 8 cells, fit a classical-form analytical expression to the ML predictions (NOT the true values). The R² of this fit is a direct quantitative measure of how much of the ML prediction is the classical formula plus smooth residuals. If R² > 0.9 for a cell, the manuscript can explicitly state that the ML IS the classical formula plus refinements, which is the strongest possible defense against the black-box critique.

Write script `scripts/interpretability/classical_equivalence_regression.py`:
1. For each cell, load test-set ML predictions from `results/per_seed_test_predictions.parquet` at seed 42 for the winning family.
2. Fit the following classical-form regressors via `sklearn.linear_model.LinearRegression` with the predicted T or P as the dependent variable and classical-form features as independent variables:
   - opx-liq T: `T_ML ~ liq_MgO + liq_FeO + ln(Mg_num_liq) + P + ln(liq_Al2O3)` — Putirka 28a shape.
   - opx-liq P: `P_ML ~ ln(jadeite_solubility) + T + ln(liq_SiO2) + liq_H2O` — Putirka 29a shape.
   - opx-only T: classical two-pyroxene expression `T_ML ~ Mg_num_opx + Ca_opx + Fe_opx + Al_opx + interaction_terms` — Brey-Köhler 1990 shape.
   - opx-only P: `P_ML ~ Al_opx + Ca_opx + Cr_opx + T + interaction_Al_Cr + interaction_Al_Ca` — Putirka 29c shape.
   - cpx-liq T/P, cpx-only T/P: corresponding Putirka cpx forms (30, 32a, 33).
3. Report per-cell R², coefficients, and 95% bootstrap CI on R².
4. Write `results/classical_equivalence_regression.csv` with columns: track, target, classical_form_name, r_squared, r_squared_ci_lo, r_squared_ci_hi, n_coefficients, interpretation.

Interpretation string per row:
- `r_squared > 0.90` → "ML prediction is well-approximated by classical functional form (R²=X.XX); ML is a flexible smoother of the classical relationship rather than a fundamentally opaque black box"
- `0.75 <= r_squared <= 0.90` → "ML prediction partially tracks classical functional form (R²=X.XX); departures from classical form encode non-linear residual structure the classical equation cannot express"
- `r_squared < 0.75` → "ML prediction diverges from classical functional form (R²=X.XX); the ML is doing something genuinely different from the classical equation at this task"

Also build a companion figure `figures/core/Core_16_fig_classical_equivalence.pdf`: 2 × 4 panel grid, one panel per cell, each panel is a scatter of ML-predicted vs classical-fit-predicted with R² annotated and identity line drawn.

### P2.5 — Partial dependence plots across families (Method 4)

For the top 3 SHAP features per cell, compute partial dependence across all 9 families on a common feature grid. Overlay all 9 family curves per feature on a single axis. Shows whether families use features the same way (curves cluster) or differently (curves diverge).

Write `scripts/interpretability/compute_partial_dependence_across_families.py`:
1. For each cell, identify top 3 SHAP features from `results/shap_importance_winners.csv` (cell already covered in existing infrastructure).
2. For each (cell, family, feature), compute partial dependence using `sklearn.inspection.partial_dependence(grid_resolution=30, method='brute')`.
3. Store curves in `results/partial_dependence_across_families.parquet` with columns: track, target, feature, family, grid_value, pd_value.

Build `scripts/interpretability/make_fig_partial_dependence.py` to produce `figures/core/Core_17_fig_partial_dependence.pdf`: 8 pages, one per cell, each page has 3 subpanels (one per top-3 feature). Each subpanel overlays 9 family curves on a shared x-axis (feature value) and y-axis (predicted T or P). Legend identifies families by color via the canonical Okabe-Ito palette in `scripts/figures/_model_palette.py`.

TabPFN permutation-importance treatment: if `sklearn.inspection.partial_dependence` fails on TabPFN (it likely will — TabPFN uses an in-context forward pass incompatible with sklearn's PD API), implement a manual partial-dependence calculation by iterating over the grid and running TabPFN inference at each grid point with other features held at training-median. Log if manual calculation is used.


### P2.6 — Surrogate decision trees (Method 6)

For each of 4 shipped opx cells, fit a depth-4 `DecisionTreeRegressor` to the ML predictions (not the true values). If R² > 0.80, the surrogate tree is a decision-procedure representation of what the ML is doing. This works for all families including TabPFN because it uses predictions, not model internals.

Write `scripts/interpretability/surrogate_decision_trees.py`:
1. For each of 4 shipped opx cells, load seed-42 test-set ML predictions.
2. Fit `DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)` with ML predictions as the target and raw oxide features + structural-formula features as inputs.
3. Compute test-set R² of the surrogate.
4. Export tree diagram via `sklearn.tree.plot_tree` to `figures/core/Core_18_fig_surrogate_trees.pdf` (one page per cell, 4 pages).
5. Write `results/surrogate_tree_r_squared.csv` with columns: track, target, surrogate_r_squared, surrogate_r_squared_ci_lo, surrogate_r_squared_ci_hi, n_leaves, interpretation.

Interpretation:
- `surrogate_r_squared >= 0.85` → "the ML prediction is well-approximated by a depth-4 decision tree (R²=X.XX), which can be inspected directly as a decision procedure; the ML is interpretable as a flowchart"
- `0.70 <= surrogate_r_squared < 0.85` → "surrogate tree approximates ML prediction with moderate fidelity (R²=X.XX); the ML uses smoothness or interactions the tree cannot capture, but the tree remains a useful approximate explanation"
- `surrogate_r_squared < 0.70` → "surrogate tree fails to approximate ML prediction (R²=X.XX); the ML relies on smooth non-tree structure, and this method is not a useful interpretability probe for this cell"

### P2.7 — Physics-informed consistency checks (Method 7)

For the natural-sample corpus (327 LEPR pairs), verify that ML predictions satisfy thermodynamic consistency constraints independent of training data.

Write `scripts/interpretability/physics_consistency_checks.py`:
1. **Fe-Mg exchange equilibrium (Kd) check.** For each natural sample where our opx-liq pipeline predicts T and where a coexisting liquid composition is available in LEPR, compute Kd_Fe-Mg_opx_liq from the predicted T via the Fe-Mg exchange thermometer relation (Putirka 2008, §5.2) and compare to the observed Kd from the measured opx-liq pair. Fraction of samples where Kd falls in the equilibrium window (0.23 to 0.35; Roeder and Emslie 1970) is the consistency-check pass rate.
2. **Al-P trend check.** For opx-only P predictions on the LEPR natural corpus, verify that samples with higher Al_Opx_cat_6ox are predicted at higher P (Gasparik 1987 relationship). Compute Spearman rank correlation between Al_Opx_cat_6ox and predicted P. Expect ρ > 0.5 if the ML has learned the Gasparik relationship.
3. Write `results/physics_consistency_checks.csv` with columns: check_name, n_samples, pass_rate_or_correlation, expected_range, interpretation.

### P2.8 — Gasparik 1987 agreement analysis (Method 8)

Verify that our opx-only P winning model's top features include Al-bearing ratios, as predicted by Gasparik (1987). If yes, the manuscript can cite Gasparik 1987 as the theoretical predecessor that the ML empirically rediscovers.

Write `scripts/interpretability/gasparik_agreement.py`:
1. Load top 10 features by mean |SHAP| for opx-only P (RF/pwlr, from `results/shap_importance_winners.csv`).
2. Identify features involving Al (any pwlr or alr log-ratio whose numerator or denominator contains Al, or any raw Al feature).
3. Compute the fraction of top-10 features that are Al-bearing.
4. Write `results/gasparik_agreement.csv` with columns: track, target, n_al_features_in_top_10, fraction_al, top_3_al_features, interpretation.

Interpretation:
- `fraction_al >= 0.4` → "opx-only P top features are substantially Al-bearing (N of 10), consistent with Gasparik (1987) aluminum-solubility barometry; ML empirically rediscovers the classical opx-only barometer"
- `0.2 <= fraction_al < 0.4` → "opx-only P top features partially overlap with Gasparik (1987) aluminum-solubility predictors (N of 10); ML uses aluminum information alongside other diagnostics"
- `fraction_al < 0.2` → "opx-only P top features do not prominently include aluminum-based ratios; ML extracts pressure signal from features outside the classical Gasparik (1987) framework"


### P2.9 — Rewrite §4.5 Feature attribution as unified interpretability bundle

Replace the existing §4.5 in `manuscripts/opx_2026/text/draft/sections/04_results.md` with a unified interpretability narrative that integrates SHAP, feature concordance, classical-feature ablation, classical-equivalence regression, partial dependence, surrogate trees, physics consistency checks, and Gasparik agreement. Structure (one paragraph per subtopic):

- Paragraph 1 (overview): summarize the eight-layer interpretability bundle and the questions each layer answers.
- Paragraph 2 (SHAP + feature concordance): report canonical SHAP top features per shipped cell, then report median pairwise Spearman correlation across 9 families from `results/feature_concordance_spearman_matrix.csv` and interpret.
- Paragraph 3 (classical-feature ablation): report the 4 opx cells' RMSE delta when trained on classical-feature-only subsets, cite `results/classical_feature_ablation.csv`.
- Paragraph 4 (classical-equivalence regression): report per-cell R² of the classical-form fit to ML predictions, cite `results/classical_equivalence_regression.csv`. THIS IS THE HEADLINE interpretability result — lead with the highest R² cell.
- Paragraph 5 (partial dependence): qualitative interpretation of Figure Core_17 with per-cell commentary on feature-agreement across families.
- Paragraph 6 (surrogate decision trees): report surrogate R² for 4 shipped opx cells, cite `results/surrogate_tree_r_squared.csv`, reference Figure Core_18.
- Paragraph 7 (physics consistency + Gasparik agreement): report Fe-Mg Kd pass rate, Al-P Spearman correlation on natural samples, and the Al-feature fraction in opx-only P top 10. Cite `results/physics_consistency_checks.csv` and `results/gasparik_agreement.csv`.
- Paragraph 8 (synthesis): "We addressed the black-box concern from eight independent angles [list them]. All eight pass for the shipped cells. The ML predictions are interpretable in terms of classical petrologic variables, with R² of classical-equivalence fit exceeding 0.X across the shipped cells, surrogate-tree R² exceeding 0.X, and Spearman feature concordance exceeding 0.X across 9 families."

Use actual numbers from the 8 new CSVs. Do not leave placeholders.

### P2.10 — New §5.X Discussion paragraph on interpretability

Add a new subsection §5.5 "On the black-box concern" to `manuscripts/opx_2026/text/draft/sections/05_discussion.md` immediately after the existing §5.4 on TabPFN. Content:

"A common reservation about ML thermobarometers is that they substitute opaque non-linear regressors for transparent classical calibrations, trading interpretability for accuracy in a way that impedes petrologic reasoning from model output. We address this concern from eight independent angles in §4.5: SHAP feature attribution, inter-family feature concordance (Spearman ρ across 9 families), classical-feature ablation (retraining winners on Putirka-form features only), classical-equivalence regression (fitting classical functional forms to ML predictions), partial dependence overlays across families, surrogate decision trees, physics-informed Fe-Mg exchange equilibrium checks on natural samples, and Gasparik (1987) aluminum-solubility consistency analysis. The headline result is that classical-equivalence R² exceeds [value from CSV] across the shipped cells, which means the ML predictions are well-approximated by the classical functional form of the corresponding Putirka (2008) or Gasparik (1987) equation plus a smooth residual structure that the classical equation cannot capture. The ML is not a fundamentally opaque regressor; it is a flexible smoother of the classical relationship. Reviewers and end-users who require a petrologic interpretation of ML output can read the Putirka-form classical-equivalence regression as the interpretable approximation and use the ML residual structure as a data-driven refinement of the classical calibration. The eight-layer bundle is a deliberate over-provisioning of interpretability evidence; we provide it in full because the ML thermobarometry literature has not to date responded to the black-box concern with this level of quantitative support, and we believe the field should raise its baseline interpretability reporting standard."


Renumber existing §5.5, §5.6, §5.7 (if any) to §5.6, §5.7, §5.8. Verify references to §5.X elsewhere in the manuscript still resolve after renumbering.

### P2.11 — Bibliography additions for Phase 2

Add the following citations to the bibliography queue at `manuscripts/opx_2026/text/draft/sections/SUBMISSION_CHECKLIST.md` (or create the file if missing). These are Phase 2 additions on top of Phase 1's list:

- Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. Neural Computation, 8(7), 1341-1390.
- Erickson, N., et al. (2020). AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv:2003.06505.
- Feurer, M., et al. (2015). Efficient and robust automated machine learning. NeurIPS 2015.
- Grinsztajn, L., et al. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022.

## PART 3 — Unified finalization

Run after Phase 2 completes (all 11 P2 subtasks).

### 3.1 Regenerate all figures including new Core_15 through Core_18

Producer scripts: `scripts/interpretability/make_fig_feature_concordance.py`, `scripts/interpretability/make_fig_classical_equivalence.py` (embedded in `classical_equivalence_regression.py`), `scripts/interpretability/make_fig_partial_dependence.py`, `scripts/interpretability/make_fig_surrogate_trees.py` (embedded in `surrogate_decision_trees.py`).

Final figure inventory check:
- Core_01 through Core_12 (existing)
- Core_13a, Core_13b (Phase 1, method agreement matrices)
- Core_14a, Core_14b (Phase 1, pairing panels)
- Core_15 (Phase 2, feature concordance)
- Core_16 (Phase 2, classical equivalence)
- Core_17 (Phase 2, partial dependence)
- Core_18 (Phase 2, surrogate trees)

Update `figures/core/AUDIT.md` with final inventory and producer-script mapping.

### 3.2 Terminology sweep

Re-run the terminology sweep from Phase 1 across the updated sections (§4.1.1, §4.5 rewrite, §5.5 new). Flag any new instances of forbidden strings and repair.

### 3.3 Test suite

Run `python -m pytest tests/ -v --tb=short` covering preregistration tests, evaluation tests, interpretability tests (if any added). Log failures to `results/test_suite_failure_log.md`. Do not roll back on failure.

### 3.4 Audit v3

Generate `MANUSCRIPT_WRITING_AUDIT_v3.md` at project root with all 15 sections plus a new Section 16 "Interpretability bundle inventory":
- Classical-equivalence R² per cell
- Surrogate-tree R² per shipped opx cell
- Feature concordance median Spearman per cell
- Physics-consistency pass rate (Fe-Mg Kd, Al-P Spearman)
- Gasparik agreement fraction for opx-only P
- Bibliography additions count

### 3.5 Word doc regeneration

Run `python build_manuscript_docx.py` from project root. Regenerates `manuscripts/opx_2026/opx_ml_thermobarometer_draft.docx` with all updated sections including §4.1.1, §4.5 rewrite, §4.7, §5.5. If build script fails, log and continue.

### 3.6 Delta report

Write `results/LEE_REVISIONS_OVERNIGHT_SUMMARY.md` with the full run summary: Phase 1 outcomes, Phase 2 outcomes, known issues, test suite status, time spent, git SHAs at each checkpoint, next recommended action for morning review.

### 3.7 Commit

Commit all Phase 2 work with message `"feat: phase 2 complete - interpretability bundle + best-per-cell defense"`. Leave branch `lee_revisions_overnight` local.


## Acceptance criteria (combined Phase 1 + Phase 2)

Execution is complete when ALL of the following hold:

**From Phase 1:**
1. `results/bootstrap_rmse_cis_all_cells.csv` exists with ≥ 72 rows.
2. `results/pairing_matrix_22rows.csv` exists with 22 rows.
3. `figures/core/Core_13a_fig_agreement_matrix_T.pdf`, `Core_13b_*.pdf`, `Core_14a_*.pdf`, `Core_14b_*.pdf` exist.
4. `manuscripts/opx_2026/text/draft/sections/03_methods.md` contains new §3.4 (uncertainty) and §3.10 (natural-sample protocol).
5. `manuscripts/opx_2026/text/draft/sections/04_results.md` contains §4.7 with 9 paragraphs using real numbers and §4.1 uses `[CI lo, hi]` format.
6. `manuscripts/opx_2026/tables/T2_model_roster.csv` exists with CI columns.

**From Phase 2:**
7. `manuscripts/opx_2026/text/draft/sections/04_results.md` contains new §4.1.1 (best-per-cell defense) and rewritten §4.5 (8-layer interpretability bundle).
8. `manuscripts/opx_2026/text/draft/sections/05_discussion.md` contains new §5.5 (black-box remediation).
9. `manuscripts/opx_2026/tables/S15_single_family_sensitivity.csv` and `.tex` exist.
10. `results/feature_concordance_permutation_importance.csv`, `results/feature_concordance_spearman_matrix.csv` exist.
11. `results/classical_feature_ablation.csv` exists with 4 opx rows.
12. `results/classical_equivalence_regression.csv` exists with 8 cells.
13. `results/partial_dependence_across_families.parquet` exists.
14. `results/surrogate_tree_r_squared.csv` exists with 4 opx rows.
15. `results/physics_consistency_checks.csv` exists.
16. `results/gasparik_agreement.csv` exists.
17. `figures/core/Core_15_fig_feature_concordance.pdf`, `Core_16_fig_classical_equivalence.pdf`, `Core_17_fig_partial_dependence.pdf`, `Core_18_fig_surrogate_trees.pdf` exist.

**From Unified Finalization:**
18. `MANUSCRIPT_WRITING_AUDIT_v3.md` exists at project root.
19. `results/LEE_REVISIONS_OVERNIGHT_SUMMARY.md` exists with full delta report.
20. `manuscripts/opx_2026/opx_ml_thermobarometer_draft.docx` is regenerated (or failure logged).
21. Git branch `lee_revisions_overnight` exists with Phase 1 and Phase 2 commits.
22. `results/OVERNIGHT_RUN_LOG.md` shows checkpoint entries for every major section with timestamps.


## Halt-and-report conditions (unified)

Stop execution and write a halt report to `results/HALT_REPORT.md` if ANY of the following occur:

1. `git status --short` at start shows more than 500 dirty files.
2. Any Phase 1 or Phase 2 script silently returns without producing expected output AND no error is raised.
3. Bootstrap CI computation shows all-zero CI widths for any family × cell (likely a bug).
4. 22-row pairing matrix produces n_pairs different from 327 for rows involving our opx-only or opx-liq.
5. Classical-equivalence regression R² is negative for any cell (impossible if design matrix is valid; indicates a bug in feature construction).
6. Permutation-importance computation takes longer than 30 minutes per cell (likely stuck; TabPFN inference may be the culprit).
7. `data/LEPR.xlsx` SHA256 prefix has drifted from `c96ffea4` (natural-sample results invalid without correct input).
8. Any Phase 2 script requires a file from `models/` that does not exist AND cannot be regenerated from existing predictions.
9. Disk space on the working drive falls below 10 GB during execution.
10. Claude Code hits a 4-hour wall-clock timeout on any single subtask.

Halt report contents: git SHA, phase and subtask where halt occurred, last successful checkpoint, full error traceback (if any), list of completed acceptance criteria, list of remaining acceptance criteria, recommended next action for the human operator.

## Overnight execution protocol details

1. **Start clock:** record UTC timestamp to `results/OVERNIGHT_RUN_LOG.md` at first command execution.
2. **Environment check:** verify Python 3.13.13, scikit-learn 1.8.0, shap 0.51.0, Thermobar 1.0.70, tabpfn present in `.venv-tabpfn`. Log version mismatches.
3. **Branch create:** `git checkout -b lee_revisions_overnight`.
4. **Phase 1 execution:** read `LEE_REVISIONS_PHASE_1_PROMPT.md`, execute, commit at end. Write `"Phase 1 complete"` checkpoint to log.
5. **Phase 2 execution:** execute P2.1 through P2.11 in order. Commit at end of each of P2.2 / P2.5 / P2.8 / P2.11 for incremental progress. Log each subtask completion.
6. **Finalization:** execute Part 3 steps in order. Commit at end.
7. **End clock:** write total elapsed time, files produced count, and summary status to `results/OVERNIGHT_RUN_LOG.md` and `results/LEE_REVISIONS_OVERNIGHT_SUMMARY.md`.

**Expected duration breakdown** (for checkpoint-budgeting purposes):
- Phase 1: 3-5 hours (bootstrap CI is the main cost; 500 resamples x 20 seeds x 9 families x 8 cells ≈ 50 CPU-min total; pairing matrix ≈ 30 min).
- P2.1 (best-per-cell defense): 20 min (mostly prose + supp table).
- P2.2 (feature concordance): 90 min (permutation importance x 9 families x 8 cells x 20 repeats).
- P2.3 (classical-feature ablation): 60 min (4 refits with Optuna params frozen).
- P2.4 (classical-equivalence regression): 30 min (OLS fits are fast).
- P2.5 (partial dependence): 120 min (30-point grid x 9 families x 3 features x 8 cells; TabPFN is slow).
- P2.6 (surrogate trees): 20 min (sklearn fits are fast).
- P2.7 (physics checks): 30 min.
- P2.8 (Gasparik analysis): 10 min.
- P2.9 + P2.10 (manuscript rewrites): 30 min (prose + number injection).
- P2.11 (bibliography): 5 min.
- Part 3 (finalization): 60 min (figure regen + audit + Word doc).

Total expected: 10 to 14 hours. Budget for 16 hours overnight.


## Execution constraints (unified)

- **Tone:** caveman for all internal logs, commit messages, TODOs. Manuscript prose stays formal research style.
- **Citations:** every numeric claim in manuscript prose must cite a CSV + column. No fabrication.
- **Missing inputs:** if a required CSV does not exist, flag in `results/finalization_log.md` with "NOT FOUND: expected at X, impact Y." Do not invent values.
- **TabPFN environment:** if `.venv-tabpfn` is unavailable, skip TabPFN-specific refits and use existing predictions. Log the skip. Do NOT block other work.
- **LEPR integrity:** if `data/LEPR.xlsx` SHA256 prefix has drifted from `c96ffea4`, halt and report. Do not proceed with wrong input data.
- **No pushes:** all commits are local. Human operator reviews branch `lee_revisions_overnight` in the morning.
- **Autonomy:** do not ask for confirmation between steps. Log decision points to `results/finalization_log.md` for retrospective audit.
- **Checkpoint every major section:** write one-line entry to `results/OVERNIGHT_RUN_LOG.md` with timestamp, section, status, next action.

## Morning review checklist (for the human operator)

When you wake up, open these files in order:
1. `results/OVERNIGHT_RUN_LOG.md` — see what completed and where it stopped.
2. `results/HALT_REPORT.md` — if present, read first to understand why execution halted.
3. `results/LEE_REVISIONS_OVERNIGHT_SUMMARY.md` — the full delta report from the run.
4. `results/test_suite_failure_log.md` — if present, review any failed tests.
5. `MANUSCRIPT_WRITING_AUDIT_v3.md` — the refreshed audit reflecting Phase 1 + Phase 2 state.
6. `manuscripts/opx_2026/opx_ml_thermobarometer_draft.docx` — the Word doc to skim before sending to Dr. Lee.

Run this in a terminal to verify acceptance criteria quickly:
```
cd "C:\Users\NQTa\Documents\MLCourse\Final Project"
ls results/bootstrap_rmse_cis_all_cells.csv results/pairing_matrix_22rows.csv results/classical_equivalence_regression.csv results/surrogate_tree_r_squared.csv
ls figures/core/Core_13a*.pdf figures/core/Core_14a*.pdf figures/core/Core_15*.pdf figures/core/Core_16*.pdf figures/core/Core_17*.pdf figures/core/Core_18*.pdf
git log --oneline -20
```

## What is NOT in this prompt

Explicitly deferred to a hypothetical Phase 3 (not part of this overnight run):
- Canonical-family pivot (switching manuscript from best-per-cell to single ERT throughout). Not executed because Phase 2 P2.1 provides the defense for keeping best-per-cell; pivot is only needed if Dr. Lee rejects that defense.
- Companion paper on natural-sample validation with extended LEPR corpus.
- Web-app deployment (Streamlit/Flask).
- Out-of-training-domain validation on non-ExPetDB experimental runs.

If Claude Code encounters partially-implemented work on any of the above, leave untouched.

## Final note

Execute all of Part 1 + Part 2 + Part 3 autonomously overnight. Log everything. Commit at checkpoint boundaries. Do not push. Do not ask questions. Halt and report if any of the 10 halt conditions trigger. Wake up to a draft ready for Dr. Lee review.
