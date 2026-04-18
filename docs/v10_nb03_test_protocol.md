# v10 NB03 test-first rebuild protocol

**Status:** canonical methodology for Phase C
**Author:** NQTa
**Date:** 2026-04-16 (expanded per v10 master plan scope)
**Companion to:** `v10_master_plan.md`

---

## Premise

Every design change in NB03 gets a pre-registered empirical test. If the test passes on a defined acceptance criterion, the change ships to canonical models. If it fails, the change is ablated — code is preserved under `src/ablations/` for reproducibility, but canonical inference bypasses it.

This replaces the v7 / v8 / v9 pattern of "implement, hope for the best, remove if reviewer objects." Pre-registration makes the manuscript defensible. Reviewers cannot accuse us of garden-of-forking-paths when every decision is logged with its test.

**v10 scope expansion.** The protocol now runs independently per pipeline: opx_only, opx_liq, cpx_only, cpx_liq, twopx, and (isolated) universal. Each pipeline runs its own T01-T12 suite. Results logged to `results/v10_nb03_test_log.csv` with a `pipeline` column. A test passing on opx does not ship the change to cpx; each pipeline decides its own ship/ablate per empirical evidence.

---

## General protocol

For each test T0X run in each pipeline:

1. **Pre-register.** In this doc, write the hypothesis, design, and accept criterion before running any code. Do not tweak the criterion after seeing the data.

2. **Implement.** In the per-pipeline NB03 notebook (`nb03_opx_baseline_models.ipynb`, `nb03_cpx_baseline_models.ipynb`, `nb03_twopx_baseline_models.ipynb`, or `nb03_universal_exploration.ipynb`), add a markdown cell titled `## Test T0X: [hypothesis]` followed by the code cell that computes the observed metric.

3. **Run.** Under the multi-seed 20-split protocol (`SPLIT_SEEDS = range(42, 62)`) where applicable, or single-seed on canonical train/test split where the test is itself a variance quantification.

4. **Log.** Append one row to `results/v10_nb03_test_log.csv` with columns (pipeline, test_id, hypothesis, metric, observed, threshold, passed, shipped, timestamp, commit_sha).

5. **Decide.** If passed: implement in canonical path for that pipeline. If failed: move implementation code to `src/ablations/<pipeline>/` and bypass in canonical for that pipeline.

6. **Document.** Add a 1-2 sentence commentary in the NB03 markdown cell explaining the result, especially if it is surprising.

---

## Test specifications

### T01: Boosted is the best primary model on ArcPL

**Hypothesis.** The boosted family (XGB, GB, CatBoost, LightGBM) has the smallest |T bias| and RMSE on ArcPL n=197 compared to forest (RF, ERT) and non-tree families (ElasticNet, MLP).

**Design.** Fit all 8 base models with their winning feature sets (from v9 Optuna params if unchanged for opx, re-optimized for cpx/twopx/universal). Predict on ArcPL n=197 (opx/cpx) or pipeline-specific external validation set (twopx/universal). Report T bias, T RMSE, P bias, P RMSE per model.

**Accept criterion.** A single family has both the smallest |T bias| AND RMSE within 5% of the best among all 8 models.

**Expected result** (opx pipeline, based on v9 re-verification).
- Forest T bias +48.51 C, boosted T bias +20.95 C. Boosted wins |T bias|.
- Forest T RMSE 71.60, boosted T RMSE 63.68. Boosted wins T RMSE.

**Ship decision if passed.** Declare boosted as primary, forest as alternative. Update manuscript Methods to reflect this. Per pipeline.

**Ship decision if failed.** Drop the primary-model framing; report all families as equally valid. Should not happen given v9 numbers for opx but may happen for cpx where Agreda-Lopez-style ET regressors perform differently.

---

### T02: Ensemble method shootout (4-way comparison)

**Hypothesis.** Among the 4 ensemble methods (Ridge stacking, two-level Ridge, greedy Caruana, AutoGluon), one method dominates on ArcPL T RMSE without catastrophic test T RMSE regression.

**Design.** Fit all 4 ensemble methods on the 8-base-model OOF matrix per `docs/v10_ensemble_methods_plan.md`. Compute test RMSE and ArcPL RMSE per method. Multi-seed (20 splits).

**Accept criterion.** Winning method has:
- ArcPL T RMSE <= best base T RMSE minus 1 C (mean across seeds), AND
- Test T RMSE >= best base T RMSE minus 5 C (ensemble is not catastrophically worse on test)

Ship the method that wins most panels (test T, test P, ArcPL T, ArcPL P). Report all four as ablation.

**Expected result** (v9 single-seed, Ridge only).
- ArcPL T stacked 53.59 vs boosted 56.24. Ridge wins by 2.65 C — passes.
- Test T stacked 89.4 vs RF 84.9. Ridge worse by 4.5 — within 5 C tolerance — passes.
- With 8 bases instead of 4, Ridge alpha may not peg at 100; greedy may outperform Ridge.

**Ship decision if passed.** Winning ensemble ships as an OOD-robustness option, documented in manuscript as secondary model for distribution shift. Not the primary-reported model on in-distribution benchmarks. Per pipeline.

**Ship decision if failed.**
- If all 4 methods fail ArcPL (stacking no longer helps OOD): ablate ensemble entirely, ship base models only
- If all 4 methods fail test (ensemble catastrophically worse): ablate
- If one method is borderline: report as null result, omit from canonical

---

### T03: Tempered resampling hurts

**Hypothesis.** P-T tempered resampled models have higher test and ArcPL RMSE than non-resampled models for 5 or more of 8 base models (v10 expands from 4 to 8 bases).

**Design.** Re-run resampled training (Phase 3.9 from v9, unchanged). Compare test RMSE and ArcPL RMSE for resampled vs non-resampled per model. Multi-seed.

**Accept criterion.** At least 5 of 8 models show resampled RMSE >= non-resampled RMSE on test with 95% bootstrap CI excluding zero.

**Expected result** (v9 handoff, 4 models). 6/8 configs worse on test, 4/4 worse on ArcPL. Passes for opx. May differ for cpx given broader P-T distribution in cpx training data.

**Ship decision if passed.** Resampling is ablated: move resampled joblibs to `models/ablation/<pipeline>/resampled/`, update per-pipeline `nb03_per_family_winners.json` with `ablation_tested: [resampled]` metadata.

**Ship decision if failed.** Resampling unexpectedly helped. Reconsider canonical status in favor of resampled models. Low probability for opx; moderate probability for cpx.

---

### T04: N_AUG=1 beats N_AUG=5 on test RMSE

**Hypothesis.** Disabling Agreda-Lopez EPMA noise augmentation (N_AUG=1) gives lower test T and P RMSE than N_AUG=5 (the v6 / v7 default).

**Design.** Fit with N_AUG=1 and N_AUG=5 using same Optuna params and same split. Compare test RMSE for both targets, both tracks. Single-seed is sufficient for a sensitivity check.

**Accept criterion.** N_AUG=1 test RMSE <= N_AUG=5 test RMSE for both targets, both tracks. Inequality can be small.

**Expected result** (v9 sensitivity test for opx). N_AUG=1 wins. Passes.

**Ship decision if passed.** N_AUG=1 is canonical for that pipeline. `augment_dataframe` still exists in `src/features.py` for future users but is not called in the canonical path.

**Ship decision if failed.** Revisit N_AUG default for that pipeline. Would require re-running with augmentation enabled.

---

### T05: Feature set winners stable

**Hypothesis.** The winning feature set per (model, target, track) combo is stable between v9 opx and v10 opx Optuna runs, OR between v10 cpx and v10 twopx Optuna runs (each pipeline tested independently).

**Design.** For opx: load `nb03_optuna_best_params.json` from v9. Re-fit each combo at the frozen params with 8-model roster. Compute multi-seed test RMSE. Identify winning feature set per combo. Compare to v9 winners.

For cpx, twopx, universal: run Optuna from scratch (~2.5 h each), then test whether raw/alr/pwlr winners are stable across re-runs with different seeds (compare seed 42 Optuna vs seed 43 Optuna).

**Accept criterion.** Winning feature set matches in >= 10 of 12 combos (allow 2 ties where means are within 1 std).

**Expected result.** Should match exactly for opx if Optuna seed and data are unchanged. Cpx/twopx/universal expected to show similar stability.

**Ship decision if passed.** v9 Optuna params are canonical for v10 opx. No re-optimization needed for opx. For cpx/twopx/universal, frozen params ship.

**Ship decision if failed.** Investigate data or Optuna version drift. Re-run Optuna for the unstable pipeline.

---

### T06: Mineral-only ML is worse than paired-phase ML (opx/cpx pipelines only)

**Hypothesis.** opx-only ML has negative R^2 on ArcPL; cpx-only ML may be negative or barely positive. Confirms mineral-only models cannot be primary.

**Design.** Run opx-only / cpx-only ML predictions on ArcPL Kd-eq subset. Compare R^2 to Putirka 2-px eq36/39 and to their liquid-paired counterparts.

**Accept criterion.** Mineral-only R^2 < 0.3 AND paired-phase R^2 > mineral-only R^2.

**Expected result** (v9 ArcPL metrics for opx). opx-only stacked T R^2 = -0.567, P R^2 = -0.244. Passes.

**Ship decision if passed.** Mineral-only relegated to supplementary. Manuscript primary results use liq-paired or two-pyroxene tracks. Framing: "mineral-only is for cases where liquid composition is unavailable (phenocrysts, xenoliths) and is fundamentally limited."

**Ship decision if failed.** Unexpectedly useful mineral-only; promote to primary. Rewrite framing.

**Not applicable to twopx or universal pipelines.**

---

### T07: Composition-conditional bias correction improves ArcPL T

**Hypothesis.** A ridge regression of ArcPL T residual on liq_Na2O, H2O_Liq, liq_TiO2, liq_K2O, Mg_num fitted by in-domain GroupKFold (not on ArcPL) reduces ArcPL T RMSE by >=5 C.

**Design.**
1. Compute in-domain train-set residuals via 5-fold GroupKFold CV on training data.
2. Fit `ridge_corr = RidgeCV().fit(X_composition, residuals)` where X_composition is the 5 features above, standardized.
3. Apply correction to ArcPL predictions: `T_corrected = T_predicted - ridge_corr.predict(X_arcpl_composition)`.
4. Report ArcPL T RMSE before and after correction.
5. Bootstrap 95% CI on the RMSE delta.

**Accept criterion.** ArcPL T RMSE decreases by >= 5 C with 95% CI excluding zero, using boosted as the base model.

**Expected result** (probe7 forest numbers for opx). Linear composition correction: 17.7% in-domain improvement, significant. Should translate to ~10 C ArcPL improvement on boosted (bigger absolute bias, similar fractional).

**Ship decision if passed.** Correction is canonical in NB07. Manuscript reports both raw and corrected numbers per probe7 recommendation. Correction is flagged as "in-domain trained, applied out-of-domain" — a reviewer-transparent design.

**Ship decision if failed.** Do not correct. Report raw +20.95 C ArcPL T bias with caveat about distribution shift and composition sensitivity. Flag liq_Na2O / H2O as OOD indicators in NB10.

**Note: this test lives in NB07, not NB03.** Decision logged to the per-pipeline NB03 test log for single-source-of-truth purposes.

---

### T08: P piecewise bias correction improves test P

**Hypothesis.** The nb07 piecewise correction reduces test P RMSE beyond sampling noise (replicating v9 finding).

**Design.** Re-run nb07 piecewise correction on test set. Bootstrap 95% CI on RMSE delta.

**Accept criterion.** Test P RMSE decreases with 95% CI excluding zero.

**Expected result** (v9 `nb07_bias_correction_null_result.csv`). Delta = -0.27 kbar, CI excludes zero. Passes for opx.

**Ship decision if passed.** P piecewise correction ships in the canonical P inference pipeline. Manuscript Methods describes both the raw and corrected P. Correct the handoff claim of "NULL on both targets" — P is SIGNIFICANT.

**Ship decision if failed.** Unexpected deviation from v9. Investigate data shift.

**Note: this test lives in NB07, not NB03.**

---

### T09: Fix stacking distributional mismatch (CV-predict at test time)

**Hypothesis.** Using `cross_val_predict(cv=5)` to generate base predictions at test time (not just training time) gives lower test T RMSE than using full-model predictions, by matching the distributional properties the meta model was trained on.

**Design.**
1. Legacy stacking: meta fit on OOF, test-time base predictions from full-data-fit base models. Record test RMSE.
2. Revised stacking: meta fit on OOF, test-time base predictions from `cross_val_predict(n_splits=5)`. Record test RMSE.
3. Compare.

**Accept criterion.** Revised test T RMSE < legacy test T RMSE by >= 1 C, OR revised test T RMSE within 1 C of legacy and revised ArcPL T RMSE no worse than legacy.

**Expected result.** Plausible small improvement on test; unknown effect on ArcPL. Low-confidence test.

**Ship decision if passed.** Revised stacking is canonical. Update `src/stacking.py` with `predict_with_cv` function. Update `docs/stacking_strategy.md` with the fix.

**Ship decision if failed.** Legacy stacking stays. Document the distributional mismatch in stacking_strategy.md as an acknowledged limitation.

**Note: this test lives in NB04 (benchmark), decision logged to per-pipeline NB03 log.**

---

### T10: IsolationForest OOD flag correlates with residual magnitude

**Hypothesis.** Samples flagged OOD by IsolationForest have systematically larger absolute residuals than in-distribution samples.

**Design.** Compute IsolationForest OOD score on ArcPL n=197. Split ArcPL into 5 quintile bins by OOD score. Compute mean absolute T residual per bin. Test for monotonic trend via Spearman rank correlation.

**Accept criterion.** Spearman r >= 0.2 with p < 0.05 between bin rank and mean |T residual|.

**Expected result** (v9 `nb10_ood_isoforest.csv` for opx). OOD subset (n=24) has RMSE 61.9 vs in-domain (n=370) RMSE 64.6 — OOD is slightly BETTER. Unusual result; v9 data may not support this hypothesis.

**Ship decision if passed.** OOD flag is canonical for user warnings. Include in predictions CSV output.

**Ship decision if failed.** OOD flag is an informational diagnostic, not a reliability indicator. Document honestly in manuscript: "IsolationForest flags compositional outliers but does not strongly predict residual magnitude in our test. Report for transparency."

**Note: this test lives in NB10, decision logged to per-pipeline NB03 log.**

---

### T11: New boosted/tree models (CatBoost, LightGBM) beat the v9 4-model roster

**Hypothesis.** CatBoost or LightGBM beats the best of (RF, ERT, XGB, GB) on pipeline-specific test T RMSE by at least 1% with 95% bootstrap CI excluding zero.

**Design.** Fit CatBoost and LightGBM under the same Optuna protocol (50 trials, TPE, seed 42). Compare multi-seed test RMSE (20 seeds) to best of v9 4-model roster.

**Accept criterion.** Either CatBoost or LightGBM achieves test T RMSE <= best_of_v9_4_models * 0.99, with 95% bootstrap CI on the delta excluding zero. Pipeline-specific.

**Expected result.** Likely to pass on at least one pipeline. CatBoost often wins on imbalanced tabular regression; LightGBM is faster but may not beat XGB.

**Ship decision if passed.** The winning model joins the canonical 8-model roster. If both pass, both ship.

**Ship decision if failed.** The new model is ablated for that pipeline but still trained and evaluated (kept in the 8-base ensemble shootout for T02). If both fail across all pipelines, consider dropping from pool entirely.

---

### T12: Non-tree models (ElasticNet, MLP) contribute ensemble diversity

**Hypothesis.** ElasticNet and MLP contribute to ensemble performance even if they lose head-to-head to trees, because they provide prediction diversity for the stacked meta-model. Specifically: removing either from the 8-base pool increases ensemble test T RMSE.

**Design.** Train all 8 bases. Refit Ridge stacking over: (a) all 8, (b) 7 excluding ElasticNet, (c) 7 excluding MLP. Compare test T RMSE across the three ensembles.

**Accept criterion.** Ensemble (a) test T RMSE < both (b) and (c), with 95% bootstrap CI excluding zero on at least one comparison.

**Expected result.** Uncertain. ElasticNet/MLP may be too correlated with trees to add diversity, OR may provide the heterogeneity the v9 4-tree Ridge was missing. T12 tests this empirically.

**Ship decision if passed.** Non-tree bases ship in the canonical ensemble. Linear-SHAP on the Ridge meta (per NB06) reports which bases contribute.

**Ship decision if failed.** Non-tree bases are evaluated and reported for transparency but excluded from canonical ensemble. Document honestly: "ElasticNet and MLP were trained and evaluated but did not improve ensemble performance over the 6-tree or 4-tree configurations."

---

## Test ordering

Per pipeline: run T01 through T12 in numeric order. Each test can fail without blocking downstream tests (they are independent). T07, T08, T09, T10 have execution in NB04/NB07/NB10 but decision logging in per-pipeline NB03 test log.

All results land in `results/v10_nb03_test_log.csv` with schema:
```
pipeline, test_id, hypothesis, metric, observed, threshold, passed, shipped, timestamp, commit_sha
```

---

## What this protocol is NOT

- Not a substitute for the existing multi-seed evaluation (20 seeds still computed for variance quantification)
- Not a substitute for LOSO / Cluster-KFold / TargetBin / LeaveOneRegionOut generalization (NB05 still runs all four)
- Not a gate on manuscript submission — failed tests become ablations, not show-stoppers, unless T01 fails (would require rethinking primary model)

## Reviewer-facing value

By the time the manuscript reaches peer review, every pre-registered test has a logged outcome per pipeline. A reviewer who asks "why did you ship stacking but not resampling for opx but the opposite for cpx?" gets the answer: T02 passed for opx on its pre-registered criterion; T03 failed. For cpx the opposite happened. The criteria were set before data inspection, logged in this doc, committed to git before Phase C ran.

This is defensive, but it is the correct discipline for ML-in-petrology where the field has a history of post-hoc cherry-picking.

---

## 9. NB03 structure per pipeline

Each of the four NB03 variants follows this structure:

```
# Phase 3.0: Setup and imports
# Phase 3.1: Data loading
# Phase 3.2: Feature engineering (per feature set)
# Phase 3.3: Optuna hyperparameter search (T05 preregistered test)
# Phase 3.4: Multi-seed final training (20 seeds)
# Phase 3.5: Canonical winner selection
# Phase 3.6: Test T01 (primary model)
# Phase 3.7: Test T04 (N_AUG)
# Phase 3.8: Test T05 (feature stability)
# Phase 3.9: Test T03 (resampling ablation)
# Phase 3.10: Test T02 (ensemble methods)
# Phase 3.11: Test T06 (mineral-only supplementary; opx/cpx only)
# Phase 3.12: Tests T11, T12 (new model families)
# Phase 3.13: Serialize canonical models
# Phase 3.14: Write per_family_winners.json and manifests
```

Phase numbers harmonize with v9 structure; new tests add new phases rather than restructuring.

---

## 10. Cross-pipeline tests (separate)

Tests T07, T08, T09, T10 involve ArcPL / test-set inference per pipeline. They live in NB07 (bias), NB10 (OOD), NB04 (benchmark), not in NB03. But their pass/fail decisions are logged with NB03 test log per pipeline.

This keeps NB03 focused on training-time decisions and NB04/NB07/NB10 focused on inference-time decisions, while maintaining one test log per pipeline.

---

## 11. Universal-only tests T13 and T14

The universal masking model has two additional tests that do not apply to the opx/cpx/twopx pipelines because they depend on the masking architecture. Both are fully specified in `docs/v10_universal_model_exploration.md`:

| Test | Name | Hypothesis | Pass condition |
|---|---|---|---|
| T13 | Graceful degradation | Universal R^2 scales monotonically with number of phases present (1 phase < 2 phases < 3 phases) | All three degradation tiers show RMSE ordering in expected direction; no catastrophic drop when phases are masked |
| T14 | All-phase outperforms subset | A sample with opx+cpx+liq all filled gets lower RMSE from universal than the best specialized opx-liq or cpx-liq or twopx model on the same sample | Universal wins on >=60% of all-phase ArcPL samples by RMSE delta > 5 C (T) or > 0.3 kbar (P) |

Both tests run ONLY in `nb03_universal_exploration.ipynb`. Neither test runs in opx, cpx, or twopx pipelines. Results logged to `results/universal/v10_nb03_test_log.csv` (separate from the primary test log).

**Decision tree:**

- If T13 fails: universal architecture is ablated and the exploration ends.
- If T13 passes but T14 fails: universal model is kept as an exploration artifact with honest negative framing ("tested but specialized models remain superior").
- If both pass: universal model becomes the lead novelty claim in the cpx paper.

See `docs/v10_universal_model_exploration.md` Sections 5-7 for full rationale, evaluation protocol, and decision tree details.

---

## 12. T15: pre-registered regime-stratified pass condition (opx-liq pressure)

**Scope:** opx pipeline only (T15 for primary pipelines; T13 and T14 remain reserved for the universal exploration track — see Section 11).

**Hypothesis:** on the pre-registered four-bin P partition (edges `[0, 5, 15, 30, 100]` kbar, registered 2026-04-17 in `docs/v10_p_regime_preregistration.md`), at least one regime with n ≥ 20 shows a directional v10-outperforms-Putirka claim for the `P_kbar` target at the **two-axis honesty bar** (see v2 below).

**Rationale:** the opx paper's pre-registered headline claim is that the v10 thermobarometer is better calibrated in at least one geologically meaningful P regime, not globally. Section G.1c-extended produces the all-models regime CSV; the opx-liq subset of that CSV, filtered to the pre-registered bins, drives this test.

**Pass condition (v2, revised 2026-04-18 in Chunk C):** `results/v10_opx_per_regime_claims_audit_robust.csv` contains at least one row matching `target == 'P_kbar'` AND `n >= 20` AND `axis1_nonoverlap == True` AND `axis2_nonoverlap == True` (equivalently `robust_outperforms == True`). Both axes must pass: axis 1 (test-set bootstrap CI) quantifies sampling noise; axis 2 (20-seed RMSE spread) quantifies model-fit stochasticity. If both pass, the audit row counts as a "pre-registered headline claim passes, robustly." Otherwise the test fails and the manuscript's headline claim is reframed as calibration-domain characterization only (no directional claim).

**Pass condition (v1, deprecated):** the original single-axis pass condition read `results/v10_opx_per_regime_claims_audit.csv` and required only non-overlapping bootstrap 95% CI on the residual axis. Chunk C (2026-04-18) found that for non-deterministic cells (MLP/raw, CatBoost/raw) a single-seed point estimate could fall on the favorable tail of the across-seed RMSE distribution, inflating apparent performance. The robust audit adds the seed-axis check; v2 is the canonical pass condition for all future manuscript claims.

**Source of truth:** `results/v10_opx_per_regime_claims_audit_robust.csv` (produced by `scripts/v10_phase_g_chunkC_robust_audit.py`, which consumes `results/v10_opx_per_regime_benchmark.csv` from nb04 and `results/v10_chunkC_perseed_regime_rmse.csv` from `scripts/v10_phase_g_chunkC_seed_regime_probe.py`). This test does NOT re-run any training; it is a pure consistency check between the headline claim and the robust audit table.

**Log format:** results appended to `results/v10_nb03_test_log.csv` with `test_id=T15`, `pipeline=opx`, `target=P_kbar`, `track=opx_liq`. The `details` JSON field carries the winning regime, its v10 and Putirka RMSEs with CIs, seed-axis CI, and n.

**Execution:** `scripts/v10_nb03_test_t15.py` (see Appendix). One-shot script; no side effects beyond appending one row to the test log. After Chunk C the script reads the robust audit CSV.
