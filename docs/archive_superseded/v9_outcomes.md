# v9 outcomes report

**Status:** retrospective, locked
**Date:** 2026-04-16
**Purpose:** record what each v9 experiment tested, what the result was, and
what decision followed. This document is the backing store for the v10 plan's
"narrative corrections" and the stacking/resampling methodology docs'
post-hoc outcome blocks.

---

## Cascade summary

23/23 pipeline health checks PASS as of 2026-04-16 08:44.

- Phase 2 Optuna: 48 studies complete, 2h 9min on i7-1265U, zero failures
- Phase 3.5-3.12: per-family winners, resampling variants, stacking meta-models
- NB04-NB10: external benchmarks, validation, SHAP, bias correction
- NB09: manuscript table compilation

---

## Experiment ledger

### E1. Optuna TPE replaces HalvingRandomSearchCV

**Hypothesis:** TPE with 50 trials yields stable best_params that beat or match
HalvingRandomSearchCV while being more defensible for peer review.

**Test:** 48 studies (4 models x 2 targets x 2 tracks x 3 feature sets) x 50
trials each.

**Outcome:** All studies completed, no NaN, stable frozen best_params saved to
`results/nb03_optuna_best_params.json`.

**Decision:** KEEP. Methodology improvement. Doc block in `optuna_strategy.md`.

### E2. Ridge stacking (test-set claim)

**Hypothesis:** Stacking 4 base models via RidgeCV improves test-set T RMSE by
1-4 C over best single base.

**Test:** 5-fold GroupKFold OOF generation per base; RidgeCV on (N, 4) OOF
matrix; evaluate on held-out test set (same 20-seed protocol as bases).

**Outcome:** HURT on test set for all 4 targets:

| Target | Best base | Base RMSE | Stacked RMSE | Delta |
|---|---|---|---|---|
| opx_liq T_C | RF | 84.89 | 89.36 | +4.5 C worse |
| opx_liq P_kbar | GB | 4.74 | 4.77 | +0.03 kbar tied |
| opx_only T_C | RF | 152.04 | 158.56 | +6.5 C worse |
| opx_only P_kbar | RF | 10.33 | 10.69 | +0.36 kbar worse |

All alpha_ values pegged at 100 (endpoint of search range), signaling
collapse: base OOF correlations 0.86-0.99 leave the meta-model no uncorrelated
error to exploit. Negative weights on RF for 3 of 4 targets.

**Decision:** Test-set hypothesis FALSIFIED. See E3 for the external
validation angle that rescues part of the story.

### E3. Ridge stacking (distribution-shift claim)

**Hypothesis:** Stacking is protective under distribution shift because
individual base models overfit training-distribution structure in different
ways; averaging those biases attenuates each.

**Test:** Stacked predictions vs best base on ArcPL n=96 Kd-equilibrated
subset (head-to-head scope, defensible apples-to-apples against Putirka).

**Outcome:** HELP on T, marginal on P.

| Target | Best base ArcPL | Stacked ArcPL | Delta |
|---|---|---|---|
| opx_liq T_C | Boosted 56.24 | 53.59 | -2.65 C better |
| opx_liq P_kbar | Forest 2.98 | 3.30 | +0.32 kbar worse |

**Decision:** Partial support. Stacking is the right call for T under
distribution shift. Stacking is the wrong call for P. Asymmetric: keep
stacked as optional OOD-robust T variant, report forest as primary P
recommendation.

Manuscript must honestly acknowledge test-set loss for stacking even though
ArcPL T wins. Framing: "Stacking protects T predictions against distribution
shift at the cost of in-distribution performance. For P, base models already
generalize, and stacking does not help."

### E4. P-T tempered resampling

**Hypothesis:** Tempered resampling (halfway between empirical and uniform
over occupied P-T cells) reduces regression-to-mean on tail P-T regions,
improving external ArcPL RMSE by 2-5 C on T and modestly on P.

**Test:** Training-only resampling; outer test + ArcPL evaluated on original
distribution.

**Outcome:** HURT on 6 of 8 test configurations; hurt on ArcPL T for forest
(+5.8 C worse) and P for forest (+0.64 kbar worse).

**Decision:** Hypothesis FALSIFIED. Move 8 resampled joblibs to
`models/ablation/resampled/`. Do not claim in manuscript. Keep as negative-
result ablation for reviewer transparency.

### E5. N_AUG=1 (augmentation disabled)

**Hypothesis:** 15x EPMA-noise augmentation per Agreda-Lopez improves RMSE
because it regularizes tree models against measurement noise.

**Test:** NB03 appendix sensitivity test: N_AUG in {1, 3, 5, 10, 15}, same
Optuna-tuned models.

**Outcome:** Augmentation hurt RF performance; XGB marginal. N_AUG=1 is the
winner.

**Decision:** N_AUG=1 default locked in `config.py`. Augmentation code
retained in `src/features.py` for Monte Carlo inference-time error
propagation (NB10).

### E6. Per-base feature_set in stacking

**Hypothesis:** Letting each of the 4 base models use its own winning
feature_set (rather than forcing all to use the same set) diversifies the
meta-input and improves blended predictions.

**Test:** Compare stacking with per-base winning feature_set vs stacking
with single global feature_set.

**Outcome:** Marginal help on test set. Base OOF correlations still 0.96+
after diversification, so the ceiling on blending gains stayed low.

**Decision:** KEEP per-base feature_set as the default. Matches deployment
behavior (each base canonical uses its own set). Low-risk, low-reward.

### E7. Bias correction (test set)

**Hypothesis:** OOF-fit linear or piecewise correction reduces test-set
RMSE by 1-3%.

**Test:** 5-fold GroupKFold OOF residuals, fit linear and piecewise
(hinge-at-median) correctors, bootstrap CI on delta-RMSE.

**Outcome:** Asymmetric.

| Target | Delta RMSE | Bootstrap CI | Interpretation |
|---|---|---|---|
| T_C | +0.75 C | [-0.89, +2.32] | NULL (within sampling noise) |
| P_kbar | -0.27 kbar | [-0.53, -0.06] | SIGNIFICANT (CI excludes zero) |

**Decision:** P piecewise correction is significant; T is null. Manuscript
must report both, not just "null." Update `nb07_bias_correction_null_result.csv`
interpretation string.

Handoff labeled both as null. This is a narrative correction for v10.

### E8. ArcPL bias probe (NB07b)

**Hypothesis:** ArcPL T residual has a recoverable composition-conditional
structure that can be corrected without trainig to the test set.

**Test:** Probes 1-7 on forest predictions on ArcPL:
- Probe 1: signed bias stats
- Probe 2: extreme residuals by prediction
- Probe 3: bias by quintile of prediction
- Probe 4: residual-composition correlations
- Probe 5: ArcPL vs test distribution comparison
- Probe 6: in-domain OOF bias-correction validation
- Probe 7: composition-conditional correction recommendation

**Outcome:**

| Metric | Forest | Boosted |
|---|---|---|
| Mean T residual (ArcPL) | +48.51 C | +20.95 C |
| Test mean T residual | -1.26 C | small |
| Distribution-shift gap | +49.77 | ~+22 |
| Pearson(pred, residual) ArcPL | +0.077 | weak |
| In-domain GroupKFold const-correction | +17.8% RMSE reduction | - |
| In-domain GroupKFold linear-correction | +17.7% RMSE reduction | - |
| Top composition predictors (forest) | liq_Na2O -0.38, H2O_Liq +0.37, liq_TiO2 -0.34 | similar |

Probe7 recommendation: CHOICE D (composition-conditional correction) for T;
CHOICE A (report uncorrected with caveat) for P.

**Decision:** Implement probe7 recommendation in v10. Forest stays supported
as the "interpretable residual-by-composition" model. Boosted becomes the
primary low-bias recommendation.

### E9. Opx-only on natural samples (NB08)

**Hypothesis:** Opx-only model has negative R^2 on external ArcPL and natural
samples, confirming that opx alone is a weak thermobarometer.

**Test:** 327 Lin 2023 NE China peridotite natural samples, compare ML
opx-only to Jorgenson cpx-only to Putirka 2-px.

**Outcome:**

| Method | T RMSE | T bias | P RMSE | P bias |
|---|---|---|---|---|
| ML opx-only (ours) | 114.9 C | +80.2 C | 5.53 kbar | +2.46 kbar |
| Jorgenson cpx-only | 74.9 C | +42.1 C | 1.84 kbar | +0.38 kbar |
| Putirka 2-px eq36/39 | 77.6 C | +36.0 C | 5.75 kbar | -2.33 kbar |

**Decision:** Hypothesis CONFIRMED. Opx-only ML is the worst of three on
natural samples. Paper framing: opx-only ML is a supplementary method for
phenocrysts and xenoliths where liquid is unavailable; do not claim it as a
primary result.

### E10. NB10 two-pyroxene benchmark

**Hypothesis:** Two-pyroxene comparison using ML opx-only + Jorgenson cpx-only
on paired samples gives cross-mineral validation signal.

**Test:** NB10 cell intended to produce
`results/nb10_two_pyroxene_benchmark.csv`.

**Outcome:** File is empty. Either cell failed silently or was not re-run in
v9 cascade.

**Decision:** v10 must actually execute this cell or drop the claim from the
pipeline. Current handoff-claimed status is false.

---

## Summary table

| ID | Experiment | v9 outcome | v10 action |
|---|---|---|---|
| E1 | Optuna TPE | HELP | Keep, document |
| E2 | Stacking on test | HURT | Honest narrative in paper |
| E3 | Stacking on ArcPL | HELP (T only) | Keep as OOD-robust variant |
| E4 | Tempered resampling | HURT | Ablation, document negative result |
| E5 | N_AUG=1 | HELP | Lock in config |
| E6 | Per-base feature_set | MARGINAL | Keep, low-risk |
| E7 | Bias correction test | ASYMMETRIC | Fix narrative: P is significant |
| E8 | ArcPL bias probe | SIGNAL | Implement probe7 recommendation |
| E9 | Opx-only natural | CONFIRMED WEAK | Supplementary framing |
| E10 | NB10 2-px | BROKEN | Re-run or drop |
