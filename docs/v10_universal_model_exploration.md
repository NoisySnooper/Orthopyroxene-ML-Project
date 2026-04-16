# v10 universal model exploration

**Status:** SIDE PROJECT for cpx_2026 paper. Isolated diagnostics per user decision.
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`

---

## 0. Isolation principle

Per user decision: **universal model diagnostics are completely separated from opx / cpx / twopx primary results.** No figures, tables, or results from the universal model appear in opx_2026 or the cpx_2026 main-text narrative. Universal gets:

- Its own notebook `nb03_universal_exploration.ipynb`
- Its own results dir `results/universal/`
- Its own figures dir `figures/universal/`
- Its own models dir `models/canonical/universal/`
- Its own section in manuscript (or separate paper if results warrant)

This isolation is to prevent novelty contamination: main-paper cpx claims compare like-for-like with published specialists (Agreda, Jorgenson). The universal story is genuinely separate.

---

## 1. Architecture

### 1.1 The novelty

No published ML thermobarometer accepts a variable set of phases. Agreda is cpx-only or cpx-liq — separate models. Jorgenson is cpx-only or cpx-liq — separate models. Our universal model takes ANY combination and predicts P, T.

Use cases:
- Sample has opx only → use universal, get P-T
- Sample has cpx only → use universal, get P-T
- Sample has opx+cpx (twopx) → use universal
- Sample has opx+liq → use universal
- Sample has cpx+liq → use universal
- Sample has all three (opx+cpx+liq) → use universal

Single model, single prediction, graceful degradation as information is removed.

### 1.2 Input specification

Fixed-length feature vector: **~45 features + 3 presence bits = 48 input dimensions.**

```
[
  # opx oxides (9)
  SiO2_opx, TiO2_opx, Al2O3_opx, Cr2O3_opx, FeO_total_opx, MnO_opx, MgO_opx, CaO_opx, Na2O_opx,
  
  # opx engineered (7)
  Mg_num_opx, Al_IV_opx, Al_VI_opx, En_frac_opx, Fs_frac_opx, Wo_frac_opx, MgTs_opx,
  
  # cpx oxides (9)
  SiO2_cpx, TiO2_cpx, Al2O3_cpx, Cr2O3_cpx, FeO_total_cpx, MnO_cpx, MgO_cpx, CaO_cpx, Na2O_cpx,
  
  # cpx engineered (7)
  Mg_num_cpx, Jd, Di, Aeg, CaTs, En_cpx, Fs_cpx,
  
  # liquid oxides (8)
  liq_SiO2, liq_TiO2, liq_Al2O3, liq_FeO, liq_MgO, liq_CaO, liq_Na2O, liq_K2O,
  
  # liquid engineered (2)
  liq_Mg_num, H2O_Liq,
  
  # presence mask bits (3)
  opx_present, cpx_present, liq_present
]
```

### 1.3 Missing-phase encoding

When a phase is absent:
- Set its oxide values to 0.0
- Set its engineered features to 0.0
- Flip presence bit to 0

When a phase is present:
- Fill its values from data
- Set presence bit to 1

### 1.4 Why this encoding

**Pros:**
- Fixed input dim → single model, no per-combination retraining
- Presence bits give the model an explicit signal to condition on
- Missing-phase zeros don't corrupt training if the model learns to mask via presence bits

**Cons:**
- Model must learn to read presence bits (may fail)
- Tree models can split on presence bits naturally
- MLP may need careful training to use the mask signal

---

## 2. Training data

### 2.1 Sources (union of all four pipelines)

- opx-only samples: ExPetDB opx (no cpx, no liq) → fill cpx+liq slots with zeros, opx_present=1, cpx=0, liq=0
- opx-liq samples: ExPetDB opx+liq → fill cpx with zeros, opx=1, cpx=0, liq=1
- cpx-only samples: ExPetDB cpx → fill opx+liq with zeros, cpx=1 only
- cpx-liq samples: ExPetDB cpx+liq → fill opx with zeros, cpx=1, liq=1
- twopx samples: ExPetDB opx+cpx pairs → fill liq with zeros, opx=1, cpx=1, liq=0
- twopx+liq samples (if ExPetDB has them): all present, opx=1, cpx=1, liq=1

### 2.2 Expected training set n

Union of:
- ~1,100 opx-only
- ~600 opx-liq
- ~2,000 cpx-only
- ~1,500 cpx-liq
- ~500 twopx
- ~200 twopx-liq

Total: ~5,900 unique experiments. After deduplication (same experiment counted once even if appearing in multiple source subsets), probably ~3,500-4,500.

### 2.3 Group awareness

Each sample keeps its Citation for GroupKFold purposes. LOSO splits stay valid.

---

## 3. Models

Same 8 base models per master plan. No architectural changes.

MLP might perform better on the universal model than on specialized pipelines because the higher-dimensional fixed-length input is more NN-amenable.

---

## 4. Optuna setup

- 8 models × 3 feature sets (raw only for universal — no alr/pwlr because log-ratios break on zero values, pwlr already has zero-handling but gets weird with many zero phases)
- Actually: **1 feature set** (universal_raw) since feature set variation is less meaningful when half the vector is zero for many samples
- 2 targets
- **16 Optuna studies**
- 50 trials each
- Estimated compute ~200 s/study: **~55 min**

Low compute compared to pipelines. Advantage of single-feature-set approach.

---

## 5. Test protocol (subset of T01-T12)

### T01 universal — Primary model selection

Applied per standard.

### T02 universal — Ensemble shootout

Applied.

### T03 universal — Resampling hurts

Applied. Universal training set is the most imbalanced (combines 6 sub-populations of different sizes), so resampling could plausibly help here even if it hurts elsewhere.

### T04 universal — N_AUG=1 vs N_AUG=5

Applied.

### T05 universal — Feature set stability

Only one feature set (universal_raw), so this test becomes "best hyperparameters stable across Optuna seeds."

### T06 universal — N/A

No mineral-only vs mineral+liq comparison within universal; that's the WHOLE POINT of universal.

### T07, T08, T09, T10, T11, T12 — applied per standard

### NEW T13 universal — Graceful degradation

**Hypothesis.** Universal model on a sample with fewer phases present has higher RMSE than the same model on a sample with more phases present, but the gap is smaller than the gap between specialized models.

**Design.**
1. Predict with universal model on ArcPL opx-only subset (filter to samples where only opx filled).
2. Predict with universal model on ArcPL opx-liq subset.
3. Predict with universal model on ArcPL cpx-liq subset.
4. Predict with universal model on ArcPL twopx subset.
5. Predict with universal model on ArcPL all-present subset.
6. For each subset, compare to the specialized model's RMSE on the same subset.

**Accept criterion.** Universal RMSE on X-subset is within 10% of specialized-model RMSE on X-subset for 4+ of 5 subsets.

If passes: universal is a viable single-model alternative.
If fails: universal degrades significantly; keep specialized models primary.

### NEW T14 universal — All-phase inference outperforms any subset

**Hypothesis.** A sample with all three phases filled (opx + cpx + liq) gets lower universal RMSE than the same model on samples with fewer phases.

**Design.** Subset ArcPL to samples where all three phases known. Predict with universal. Compare RMSE to ArcPL cpx-liq-only and ArcPL opx-liq-only.

**Accept criterion.** All-phase RMSE is lowest of the three.

If passes: "more data = better prediction" is empirically confirmed. Manuscript highlight.
If fails: universal is degenerate; doesn't exploit extra information. Known failure mode worth discussing.

---

## 6. "Beat specialized" gate for universal

To claim universal model is useful:

1. T13 passes (graceful degradation)
2. T14 passes (all-phase is best)
3. T01 shows a model family that reliably wins the universal pipeline

Three-of-three required to call universal "viable." One-of-three: document as "exploratory, full performance requires specialized models."

---

## 7. Diagnostics and figures (ISOLATED)

All in `figures/universal/`:

- `fig_nb03_universal_multiseed_rmse.png`
- `fig_nb03_universal_optuna_convergence.png`
- `fig_nb03_universal_stacking_diagnostics.png`
- `fig_nb03_universal_graceful_degradation.png` — 5-panel plot: universal RMSE per phase-subset scope
- `fig_nb03_universal_all_phase_vs_subset.png` — shows all-phase < subset prediction
- `fig_nb03_universal_vs_specialized_delta.png` — bar chart: universal RMSE vs specialized RMSE per scope
- `fig_nb06_universal_shap.png` — SHAP feature importance, showing role of presence bits
- `fig_nb08_universal_natural.png` — universal predictions on natural samples, compared to specialized

**Explicitly NOT combined with opx/cpx/twopx figures.** Isolated presentation per user decision.

---

## 8. Interpretability angle

The presence bits give a natural interpretability story. Via SHAP:
- If `opx_present` bit has high SHAP magnitude, model uses the mask signal
- If the bit has low SHAP, model isn't exploiting the mask (just averaging zeros)

This tells us whether the universal architecture is working as designed.

---

## 9. Manuscript placement

The universal exploration becomes a Results subsection in the cpx_2026 paper:

> "Section 4.5: Universal pyroxene thermobarometer. Motivated by the observation that [...], we trained a single masking-based model on the union of all 4 phase combinations and evaluated its graceful degradation across phase subsets. [...] The universal model matches specialized cpx-liq performance within 8% on the all-phase ArcPL subset but degrades to 15% on opx-only samples (Table SI.X), confirming that specialized models remain preferred when all relevant phases are known but a single universal model is viable for heterogeneous datasets."

If universal performs exceptionally well: spin into a third paper, method-focused.

If universal performs poorly: honest null result documented, still a contribution.

---

## 10. Config constants (add to config.py)

```python
# Universal pipeline
UNIVERSAL_INPUT_DIM = 48
UNIVERSAL_MASK_BITS = ('opx_present', 'cpx_present', 'liq_present')
UNIVERSAL_FEATURE_METHODS = ('universal_raw',)
```

---

## 11. Timeline

- 3-4 days active per master plan Phase F
- ~55 min Optuna compute (low)
- ~2 h for 20-seed final + ensemble + SHAP

---

## 12. Deliverables (all isolated)

- `nb03_universal_exploration.ipynb`
- `results/universal/` — all outputs
- `figures/universal/` — all figures
- `models/canonical/universal/` — joblibs
- `results/universal/v10_nb03_test_log.csv` — T01-T14 results

Reminder: NONE of these appear in `results/opx/`, `results/cpx/`, or `results/twopx/`. Universal is its own track.

---

## 13. Risk register for universal

| # | Risk | Mitigation |
|---|---|---|
| 1 | Universal worse than specialized across the board | Report honestly; still a contribution ("we tested, doesn't help") |
| 2 | MLP unable to learn mask signal | Expected per small-data NN limitation; compensate with tree models |
| 3 | Training set deduplication error | Careful sample_id tracking across sources |
| 4 | Presence bits not actually used by the model | SHAP diagnostic T13/T14 catches |
| 5 | Reviewer asks "why not use separate specialists and pick at inference time" | Answer in Methods: specialized requires knowing which phases are present before selecting model; universal handles heterogeneous data where that is not known a priori |
