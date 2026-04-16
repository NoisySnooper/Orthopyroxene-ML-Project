# v10 external models audit

**Status:** Phase A verification plan for external training data sources
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`

Per user: "check before you start" if other papers use the same dataset. This doc specifies the audit protocol before any external training data download.

---

## 1. Purpose

Before Phase D (cpx rebuild), confirm which external ML thermobarometers use LEPR / ExPetDB vs private datasets. If they use LEPR and we already have it, we:

1. Don't need to download external training data (we have it)
2. Can claim a genuinely like-for-like comparison (same training source, different methodology)

If they use private datasets:

1. We note the training-source asymmetry in Methods
2. We use their released pre-trained models directly for benchmarking (not re-train on our data)

---

## 2. External models to audit

### 2.1 Ágreda-López et al. 2024 (cpx-liq)

**Paper:** Ágreda-López, M., Parodi, V., Musu, A., Bonadiman, C., Perinelli, C., Aulinas, M., & Petrelli, M. (2024). "Enhancing machine learning thermobarometry for clinopyroxene-bearing magmas." *Computers & Geosciences.*

**Published training source:** LEPR (per paper Section 2.1). Specifically, cpx analyses from LEPR with paired melt compositions.

**Released artifacts:** GitHub repo likely at `MaurizioPetrelli/ML_thermobarometry` or similar. Released `.onnx` files + training data CSV.

**We have:** `data/external/agreda_lopez_2024/` (137 MB) — already in repo.

**Phase A audit check:**
- Confirm repo files include training CSV (not just pre-trained models)
- Compare training sample citations to our LEPR cpx subset
- Report overlap percentage: `n_overlap / n_our_cpx`

**Expected overlap:** High (both are LEPR-based).

### 2.2 Jorgenson et al. 2022 (cpx-only + cpx-liq)

**Paper:** Jorgenson, C., Higgins, O., Petrelli, M., Bégué, F., & Caricchi, L. (2022). "A machine learning-based approach to clinopyroxene thermobarometry: model optimization and distribution for use in Earth sciences." *JGR: Solid Earth.*

**Published training source:** LEPR (per paper Section 3.1). Explicit: "We used the Library of Experimental Phase Relations (LEPR)..."

**Released artifacts:** Python package `Thermobar` includes calls to their ExtraTreesRegressor models. Training CSVs may be in Thermobar's examples directory.

**We have:** `data/external/thermobar_examples/Thermobar/` (1.1 GB) — already in repo.

**Phase A audit check:**
- Find Jorgenson training CSV inside Thermobar examples
- If not present in Thermobar repo, check Zenodo link in paper SI
- If unavailable, use their published pre-trained model directly (ship our benchmark against theirs)

**Expected:** training data publicly available via Thermobar or Zenodo.

### 2.3 Wang et al. 2021 (cpx-liq)

**Paper:** Wang, X., Hou, T., Wang, M., Zhang, C., Zhang, Z., Pan, R., Marxer, F., & Zhang, H. (2021). "A new clinopyroxene thermobarometer for mafic to intermediate magmatic systems." *European Journal of Mineralogy.*

**Published training source:** LEPR (per paper). Compilation of cpx-melt experiments from LEPR with filters.

**Released artifacts:** Paper includes empirical equations (not ML). Not an ML model per se — classical regression-based thermobarometer.

**Phase A action:** Check if publicly released or only equations in paper. Implement as classical thermobarometer in `src/external_models.py`.

### 2.4 Petrelli et al. 2020 (cpx-only + cpx-liq)

**Paper:** Petrelli, M., Caricchi, L., & Perugini, D. (2020). "Machine learning thermo-barometry: Application to clinopyroxene-bearing magmas." *JGR: Solid Earth.*

**Published training source:** LEPR-based. Same lab group as Jorgenson 2022 (follow-up).

**Released artifacts:** ExtraTreesRegressor models available via Thermobar. Earlier version of Jorgenson's work.

**Phase A action:** Verify Thermobar has Petrelli 2020 model accessible. Use via Thermobar API.

### 2.5 Putirka 2008 (cpx-liq, opx-liq, two-pyroxene)

**Paper:** Putirka, K. D. (2008). "Thermometers and barometers for volcanic systems." *Reviews in Mineralogy and Geochemistry.*

**Training source:** Compilation of experiments circa pre-2008. Classical multi-parameter regression, not ML. Overlaps heavily with LEPR data.

**Released artifacts:** Equations in paper; implemented in Thermobar.

**Phase A action:** Use via Thermobar calls. No training data download needed.

---

## 3. Training data overlap quantification

Per Phase A audit script:

```
For each external model M:
  1. Load M's training CSV (if publicly released)
  2. Extract unique (citation, sample_name) tuples
  3. Compare to our LEPR cpx subset
  4. Compute:
     - n_overlap = tuples shared
     - n_M_only = tuples only in M's data
     - n_ours_only = tuples only in ours
     - overlap_pct = n_overlap / n_M_training
  5. Log to results/v10_external_model_training_overlap.csv
```

Expected outcomes:

| Model | Our LEPR coverage of their training | Action |
|---|---|---|
| Agreda 2024 | 80-95% | Use our LEPR + extract diff; retrain comparison allowed |
| Jorgenson 2022 | 85-95% | Same |
| Petrelli 2020 | 90-98% | Same |
| Wang 2021 | 70-90% | Use empirical equations from Thermobar |

If overlap < 70% for any model: consider whether to match their training set by downloading their data.

---

## 4. Decision matrix

| Audit finding | Action |
|---|---|
| Training CSV publicly available AND overlaps our LEPR by >80% | Use our LEPR only. Note in Methods: "Our cpx training data is >80% overlapping with Agreda-Lopez 2024's. Performance differences reflect methodology, not training source." |
| Training CSV available but overlap <80% | Download their CSV. Compare our model vs theirs both on their training data and ours. |
| Training CSV not publicly available | Use their pre-trained model directly via `src/external_models.py`. Note asymmetric comparison in Methods. |
| Pre-trained model and training both unavailable | Skip that benchmark. Report in Methods as limitation. |

---

## 5. Pre-trained model status check

For each external model, verify that our copy runs:

### Agreda 2024

`data/external/agreda_lopez_2024/` contains `.onnx` files. Verify:
```python
from src.external_models import predict_agreda_cpx_liq
preds = predict_agreda_cpx_liq(X_arcpl_cpx_liq)
# Should return T_C, P_kbar arrays without error
```

### Jorgenson 2022

```python
import Thermobar as pt
preds = pt.calculate_cpx_only_press_temp(cpx_comps=df, equationP='P_Jorgenson22', equationT='T_Jorgenson22')
```

### Petrelli 2020

Via Thermobar similarly.

### Putirka 2008

Via Thermobar: `equationT='T_Put2008_eq33'`, `equationP='P_Put2008_eq32c'`, etc.

If any fail: update `src/external_models.py` with fallback or note unavailability.

---

## 6. Data SHA256 pinning

After Phase A audit finalizes:

```
scripts/v10_phase_a_cleanup.py (section generating data/hashes.json):

for path in recursive walk of data/:
  compute SHA256
  write to data/hashes.json:
    {
      "path": "data/external/agreda_lopez_2024/model.onnx",
      "sha256": "ab12cd34...",
      "source": "https://github.com/user/repo/releases/download/...",
      "retrieved_date": "2026-04-XX",
      "notes": "Downloaded per v10_external_models_audit.md Section 2.1"
    }
```

This pins reproducibility. Reviewer can re-download from source and verify match.

---

## 7. Citation discipline

Every external model gets:

1. Full paper citation in Methods (authors, year, journal, DOI)
2. Repository URL in SI (link to GitHub or Zenodo)
3. Model file SHA256 in `data/hashes.json`
4. Version notation in code comments: `# Agreda-Lopez 2024 v1.0, commit abc123, retrieved 2026-04-16`

---

## 8. Audit script

`scripts/v10_external_models_audit.py`:

```
1. For each model in {agreda_2024, jorgenson_2022, wang_2021, petrelli_2020}:
   a. Load training data if present
   b. Load pre-trained model; test prediction on dummy input
   c. Compute overlap with our LEPR (if training data available)
   d. Write audit row to results/v10_external_model_audit.csv
2. Write final summary:
   - Which models have training data
   - Which models run via pre-trained artifacts
   - Which models need empirical equation implementation
   - Overlap percentages
3. Print actionable recommendations per Decision matrix Section 4
```

---

## 9. Pre-Phase-D prerequisites

Before Phase D (cpx rebuild) starts:

- [ ] `scripts/v10_external_models_audit.py` has run and reported all 4 external models
- [ ] All external pre-trained models verified to run on dummy input
- [ ] Training data overlap percentages logged
- [ ] Decision per model per Section 4 made and documented
- [ ] `data/hashes.json` populated
- [ ] `src/external_models.py` updated with any new model wrappers (Wang 2021 equations, Petrelli 2020 Thermobar calls)

Gate: user approves "external models audit clean, proceed to Phase D" in chat.

---

## 10. Expected Phase A runtime

- Audit script: 5-15 min
- Training data downloads (if needed): 10-30 min
- Decision documentation: 30 min active

Total Phase A portion for external models: under 1 hour.
