# v10 twopx pipeline plan

**Status:** canonical spec for `nb03_twopx_baseline_models.ipynb`
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`

Two-pyroxene model: classical approach where opx and cpx are required in the same rock (assumed to have equilibrated). Ships in cpx_2026 paper alongside cpx-only, cpx-liq, and universal.

---

## 1. Scope

Single track: **twopx**. Input = opx composition + cpx composition (both required). Output = single P, T (assumed equilibrated).

Primary benchmark: Putirka 2008 two-pyroxene thermometers (eq36, eq37, eq39), applied to ArcPL samples where both minerals are present.

Novelty claim: first ML two-pyroxene thermobarometer. Putirka 2008 eq36/39 is the classical benchmark, no published ML two-px model as of 2026.

---

## 2. Training data

### 2.1 Primary source: ExPetDB experiments with opx + cpx pairs

Filter criteria:
- Same experiment ID (same charge)
- Both opx and cpx analyzed (both have passing oxide total + cation sum)
- Fe-Mg equilibrium between opx and cpx (Putirka 2008 KD_Fe-Mg(opx-cpx) in 1.09 ± 0.14 per Brey & Köhler 1990 calibration)
- Wo_opx < 5 AND Wo_cpx > 20 (pyroxene quad filter on both)

Expected n: 500-1,000 pairs after filtering.

### 2.2 Secondary source: LEPR pyroxene pairs

LEPR records with both `_Opx` and `_Cpx` column groups populated.

Expected: 1,000-1,500 additional pairs.

### 2.3 Final training n

Target: 1,500-2,500 twopx pairs after dedup.

---

## 3. Features

### 3.1 Base feature sets (three, same pattern)

- **raw**: 9 opx oxides + 9 cpx oxides + engineered ratios from both minerals = ~25 features
- **alr**: additive log-ratio, SiO2 denominator for each mineral = ~16 features
- **pwlr**: pairwise log-ratio for opx pairs + cpx pairs + cross-mineral pairs (oxide_opx / oxide_cpx) = ~60-80 features (many!)

### 3.2 Twopx-specific engineered features (NEW)

From `src/twopx_features.py` (new module):

- **KD_Fe-Mg(opx-cpx)**: per-sample Fe-Mg exchange coefficient between opx and cpx
- **Ca_opx_cpx_ratio**: Ca_opx / Ca_cpx (partitioning proxy)
- **Al_opx_cpx_ratio**: Al_opx / Al_cpx
- **Na_opx_cpx_ratio**: Na_opx / Na_cpx
- **Mg_num_opx** and **Mg_num_cpx** (already exist)
- **Delta_Mg_num**: Mg_num_cpx - Mg_num_opx (equilibrium indicator)
- Both minerals' **En**, **Fs**, **Wo** fractions
- **T_BKN90_initial**: Brey-Köhler 1990 T estimate as a feature (priors)
- **P_CA_eq30_initial**: Putirka eq30 P estimate as a feature

### 3.3 Feature set configurations

Four feature sets tested:
- raw
- alr
- pwlr (cross-mineral pairs included)
- twopx_components (structural + KD features)

Test T05 determines winners per (model, target).

---

## 4. Models

Same 8 base models per master plan Section 3. Same 4 ensemble methods.

---

## 5. Optuna setup

- 8 models × 4 feature sets × 2 targets = 64 Optuna studies (single track, no opx_only/cpx_only split)
- 50 trials each
- Estimated compute at ~150 s/study (smaller training n): **~2.7 hours**

---

## 6. Test protocol T01-T12 applied to twopx

### T01 twopx — Boosted is primary

Hypothesis: boosted family has smallest |T bias| and lowest T RMSE on external twopx benchmark.

### T02 twopx — Ensemble shootout

Hypothesis: at least one ensemble method beats best base.

### T03 twopx — Resampling hurts

Hypothesis: resampling hurts 6+ of 8 combos.

### T04 twopx — N_AUG=1 wins

Hypothesis: N_AUG=1 <= N_AUG=5.

### T05 twopx — Feature set stability

Hypothesis: winners stable across seeds.

Specifically compare twopx_components (structural formulas + KD) vs raw/alr/pwlr. If twopx_components wins, manuscript has a nice "physical feature engineering still matters" story.

### T06 twopx — N/A

No mineral-only version; twopx requires both minerals by definition.

### T07 twopx — Composition-conditional T correction

Hypothesis: correction reduces ArcPL T RMSE by >=5 C.

### T08 twopx — P piecewise correction

Hypothesis: correction reduces test P RMSE.

### T09 twopx — CV-predict stacking

Hypothesis: CV-predict improves test RMSE.

### T10 twopx — OOD flag

Hypothesis: flag correlates with residual magnitude.

### T11 twopx — CatBoost/LightGBM

Hypothesis: beats XGB.

### T12 twopx — MLP/ElasticNet

Hypothesis: beats tree winner.

---

## 7. "Beat Putirka 2-px eq36/39" gate

Per gate (c). To claim beat:

1. Lower RMSE on ArcPL twopx subset (T AND P)
2. Lower RMSE on LOSO (T AND P)
3. Better natural-sample agreement (using Lin 2023 + additional sources where available)

All three with 95% bootstrap CI excluding Putirka's published numbers.

Putirka 2008 two-pyroxene thermometer calibration error is reported as ~50-60 C for T, ~2-3 kbar for P. We aim to improve by >=10 C and >=1 kbar while preserving coverage.

---

## 8. Diagnostics and figures

Mirror cpx figure set with twopx subdirs:

- `figures/twopx/`
- Multi-seed boxplot
- Optuna convergence
- Stacking diagnostics
- External benchmark vs Putirka eq36/39
- SHAP T and P
- Cross-pipeline heatmap: twopx vs (opx+cpx inferred separately)

Specific key figure: **`fig_nb04_twopx_vs_separate_inference.png`** — shows twopx model predictions vs opx-only and cpx-only models' predictions averaged, on samples where all three can run. Tests hypothesis that twopx model is strictly better than averaging separate inferences.

---

## 9. Natural sample cross-validation for twopx

`nb08_natural_samples.ipynb` will include samples where both opx and cpx are present and the twopx model runs alongside opx-only, cpx-only, and the universal model.

This is the main empirical test of twopx: does forcing the model to use both minerals simultaneously give better predictions than either alone?

---

## 10. Config constants (add to config.py)

```python
# Two-pyroxene pipeline
TWOPX_KD_FEMG_MIN = 0.95
TWOPX_KD_FEMG_MAX = 1.23
TWOPX_FEATURE_METHODS = ('raw', 'alr', 'pwlr', 'twopx_components')
TWOPX_COMPONENTS = (
    'KD_FeMg_opx_cpx', 'Ca_ratio', 'Al_ratio', 'Na_ratio',
    'Mg_num_opx', 'Mg_num_cpx', 'Delta_Mg_num',
    'En_opx', 'Fs_opx', 'Wo_opx',
    'En_cpx', 'Fs_cpx', 'Wo_cpx',
    'T_BKN90_initial', 'P_CA_eq30_initial'
)
```

---

## 11. Timeline

- 2 days active
- ~3 h compute (64 Optuna studies at ~150 s each)
- Per Phase E of master plan

---

## 12. Deliverables

- `nb03_twopx_baseline_models.ipynb`
- `results/twopx/` populated
- `models/canonical/twopx/`
- `results/twopx/v10_nb03_test_log.csv`
- All twopx figures per `v10_figure_audit.md`

---

## 13. Special concern: Fe-Mg equilibrium filter

Putirka 2008 KD_Fe-Mg(opx-cpx) = 1.09 ± 0.14 per Brey & Köhler 1990.

In Phase D / E, we log:
- How many experimental pairs fail this filter
- Distribution of KD values in training data

If a significant fraction fails (say >25%), we revisit whether experimental cpx-opx pairs are genuinely at equilibrium. Thermobar has built-in `eq_tests=True` for this that we use in NB04 benchmark runs.

---

## 14. Stretch goal: two-pyroxene + liquid

If compute budget allows, add `twopx_liq` sub-track. Training data: same pairs but with liquid also present. This would be the most information-rich input and might have the best absolute performance.

Explicitly stretch — NOT in v10 baseline. Explored if Phase E finishes under budget.
