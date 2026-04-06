# REVISED MASTER PROMPT: ML Thermobarometer for Orthopyroxene

## ROLE AND CONTEXT

You are helping me build a machine learning thermobarometer for orthopyroxene (opx) as an academic paper targeting peer-reviewed publication. I am a first-class cadet at the United States Coast Guard Academy working under PI Dr. Kanani K.M. Lee in mineral physics. I have experience with high-pressure diamond anvil cell experiments, optical absorption spectroscopy of iron-bearing silicates, and Python/scikit-learn from coursework.

## PROJECT OVERVIEW

**Goal:** Build the first standalone ML thermobarometer for orthopyroxene. Benchmark against Putirka (2008) equations. Provide interpretable feature importance via SHAP. Package as open-source Python tool with trained model weights.

**Working title:** Machine Learning Thermobarometry for Orthopyroxene: Benchmarking, Interpretability, and Calibrated Uncertainty

**Target journals (ranked):**
1. JGR: Machine Learning and Computation (AGU, launched Dec 2023, explicitly seeks ML geoscience applications)
2. Computers & Geosciences (Elsevier, IF ~4.2, published Ágreda-López et al. 2024 cpx ML enhancement)
3. JGR: Solid Earth (AGU, IF ~3.9, published Petrelli 2020, Jorgenson 2022, Li & Zhang 2022)

**Timeline:** 5–6 weeks from data cleaning to submission-ready manuscript draft.

**Compute:** Kaggle free tier (CPU only). Tree-based models on ~1,500 samples train in seconds.

---

## THE GAP (confirmed via literature search, April 2026)

No peer-reviewed standalone ML thermobarometer exists for orthopyroxene. Confirmed via systematic search across Google Scholar, JGR:SE, C&G, CMP, J. Petrology, Am. Min., Lithos, G-Cubed, ESSOAr, and EarthArXiv. The Thermobar package (Wieser et al. 2022) includes ML models for cpx but only conventional Putirka equations for opx.

Two tangential publications do NOT fill the gap:
- Petrelli (2023, Springer textbook chapter): uses opx as a pedagogical example, not a validated tool
- Qin et al. (2024, GRL): trains on multi-mineral pairs (garnet-opx), not standalone opx chemistry

**Published ML thermobarometers by mineral phase:**

| Mineral | Key papers | n (experiments) | Best method |
|---|---|---|---|
| Clinopyroxene | Petrelli 2020; Jorgenson 2022; Chicchi 2023; Ágreda-López 2024 | 2,000–5,600 | ERT |
| Amphibole | Higgins et al. 2022 | ~1,900 | Random Forest |
| Biotite | Li & Zhang 2022 | 839 | ERT |
| Garnet | Zhang et al. 2025 | 1,308 | XGBoost |
| Melt/liquid | Weber & Blundy 2024 | 2,545 | ERT |
| Plagioclase | Cutler et al. 2024 | — | RF-based |
| **Orthopyroxene** | **None** | **~1,500–1,800 available** | — |

No published ML thermobarometer for any mineral uses MC-dropout Bayesian uncertainty or multi-task P-T prediction. Petrelli (2024, J. Petrology review) explicitly identifies these as "future directions."

**Two contributions this paper makes:**
1. First standalone ML thermobarometer for orthopyroxene
2. Open-source deployable tool with trained model weights

---

## TRADITIONAL BENCHMARKS (Putirka 2008)

These are the error bars to beat. Source: Putirka (2008) Rev. Min. Geochem. vol. 69, pp. 61–120.

**Opx-liquid models:**

| Equation | Type | SEE | Notes |
|---|---|---|---|
| Eq. 28a | T (P-dependent) | ±48°C | Requires P input |
| Eq. 28b | T (opx sat. surface) | ±26°C | Best T, but near analytical floor |
| Eq. 29a | P (T-dependent) | ±2.6 kbar | Requires T input |
| Eq. 29b | P (T-dependent) | ±2.8 kbar | Requires T input |
| Eq. 29c | P (opx-only, T-dep.) | ±3.2 kbar | Uses Al, Ca, Cr cations only |

**Two-pyroxene models:**

| Equation | Type | SEE | Notes |
|---|---|---|---|
| Eq. 36 | T (P-dependent) | ±56°C (all); ±38°C (Mg#>0.75) | |
| Eq. 37 | T (P-dependent) | ±60°C (all); ±48°C (Mg#>0.75) | |
| Eq. 38 | P (T-independent) | ±3.7 kbar | Soft target, no T input needed |
| Eq. 39 | P (T-dependent) | ±2.8 kbar | |

**Strategic focus:** Opx-only models have the most room for improvement and the highest practical utility (no equilibrium liquid required). Opx-liquid models are secondary because Putirka 28b is already at ±26°C, near the analytical precision floor (~16°C per Weber & Blundy 2024).

**Realistic ML targets based on published analogues:**

| Model type | T target (RMSE) | P target (RMSE) | Basis |
|---|---|---|---|
| Opx-only | 50–70°C | 2.5–3.5 kbar | Comparable to cpx-only (Jorgenson 2022: 72°C, 3.2 kbar) |
| Opx-liquid | 30–50°C | 2.0–2.8 kbar | Comparable to cpx-liq (Jorgenson 2022: 45°C, 2.7 kbar) |

---

## DATA SOURCE AND INITIAL CENSUS

**ExPetDB** (formerly LEPR), downloaded 2025-07-21. Orthopyroxene sheet.

| Metric | Count |
|---|---|
| Total opx analyses | 1,796 |
| Unique experiments | 1,552 |
| Unique publications | 164 |
| T range | 550–1,975°C |
| P range | 0.001–200 kbar (0–20 GPa) |
| Median T | 1,300°C |
| Median P | 12 kbar |
| Opx-liquid pairs | 1,383 rows (1,171 experiments) |

**Oxide completeness:**

| Oxide | Non-null | Coverage |
|---|---|---|
| SiO2 | 1,773 | 98.7% |
| Al2O3 | 1,707 | 95.0% |
| MgO | 1,745 | 97.2% |
| FeO | 1,539 | 85.7% |
| CaO | 1,510 | 84.1% |
| TiO2 | 1,200 | 66.8% |
| Na2O | 1,120 | 62.4% |
| Cr2O3 | 1,023 | 57.0% |
| MnO | 1,005 | 56.0% |
| Fe2O3 | 4 | 0.2% |

**Missing data strategy (two-model approach):**
- **Core model (5 oxides):** SiO2, Al2O3, FeO, MgO, CaO. ~1,484 rows with all 5 present. This is the "general" tool.
- **Full model (9 oxides):** Add TiO2, Na2O, Cr2O3, MnO. Subset with all 9 present (~800–900 rows estimated). This is the "enhanced" tool for well-characterized samples.
- Present both. Compare performance. If the full model is only marginally better, recommend the core model for general use.

---

## DATA CLEANING PROTOCOL

### Step 1: Extract and merge
- Join Orthopyroxene sheet with Experiment sheet on Experiment ID
- Extract: opx oxide values, T (°C), P (GPa → kbar by ×10), Citation, DOI
- Also extract Liquid sheet oxides for opx-liquid models

### Step 2: Iron handling
- Fe2O3 is reported for only 4 of 1,796 rows. Treat FeO as FeO_total throughout.
- Document this decision. It is standard practice (Putirka 2008; Jorgenson et al. 2022).

### Step 3: Oxide total filter
- Compute sum of all reported oxides per analysis
- **Retain 95–102 wt%** for anhydrous minerals (standard filter per Neave et al. 2017)
- Flag and examine outliers outside this range before dropping

### Step 4: Pigeonite filter
- Recalculate end-member components: En, Fs, Wo
- **Remove analyses with Wo > 5 mol%** (these are pigeonite or contaminated; flagged by Frontiers in Geochemistry 2025 review)
- Log removed entries for supplementary table

### Step 5: Cation recalculation (6-oxygen basis, Morimoto 1988)
- Convert wt% oxides → moles → cation fractions per 6 oxygens
- Quality filter: **cation sum = 4.00 ± 0.05** (true pyroxene stoichiometry)
- Remove analyses outside this range

### Step 6: Equilibrium filter (opx-liquid models only)
- KD(Fe-Mg)_opx-liq = (Fe/Mg)_opx / (Fe/Mg)_liq
- Expected range: **0.29 ± 0.06** (Putirka 2008)
- Remove out-of-equilibrium pairs

### Step 7: Feature engineering
Computed from cation fractions:
- **Mg#** = Mg/(Mg+Fe_total)
- **Al_IV** = min(2 - Si, Al_total) — tetrahedral aluminum
- **Al_VI** = Al_total - Al_IV — octahedral aluminum
- **En** = Mg/(Mg+Fe+Ca) — enstatite component
- **Fs** = Fe/(Mg+Fe+Ca) — ferrosilite component
- **Wo** = Ca/(Mg+Fe+Ca) — wollastonite component
- **MgTs** = Al_IV (Mg-Tschermak proxy)

For opx-liquid models, also include:
- Liquid oxide wt% (SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O)
- Liquid Mg# = MgO_liq / (MgO_liq + FeO_liq) (molar)

### Step 8: Provenance tracking
- Every row retains: Experiment ID, Citation, DOI
- This enables leave-one-study-out (LOSO) validation
- Save as `opx_clean.csv` (core) and `opx_clean_full.csv` (9-oxide)

---

## MODEL ARCHITECTURES (4 models for comparison)

All four models predict P and T as separate regression targets (separate model per target).

### 1. Random Forest (scikit-learn)
- Baseline ensemble method
- `RandomForestRegressor` with 30-iteration `RandomizedSearchCV`, 5-fold CV
- Hyperparameter space: n_estimators [100, 500, 1000], max_depth [10, 20, 30, None], min_samples_split [2, 5, 10], min_samples_leaf [1, 2, 4], max_features ['sqrt', 'log2', 0.5]

### 2. Extremely Randomized Trees (ERT, scikit-learn)
- Expected best performer based on all published analogues (Jorgenson 2022, Li & Zhang 2022, Weber & Blundy 2024)
- `ExtraTreesRegressor` with same search space as RF
- Key difference from RF: random split thresholds, further reducing variance

### 3. XGBoost
- Gradient boosting baseline
- `XGBRegressor` with 30-iteration `RandomizedSearchCV`
- Hyperparameter space: n_estimators [100, 500, 1000], max_depth [3, 6, 9, 12], learning_rate [0.01, 0.05, 0.1, 0.2], subsample [0.7, 0.8, 0.9, 1.0], colsample_bytree [0.5, 0.7, 0.9, 1.0], reg_alpha [0, 0.1, 1], reg_lambda [1, 5, 10]
- Early stopping on validation set, patience=50

### 4. Gradient Boosting (scikit-learn)
- Second gradient boosting variant for completeness
- `GradientBoostingRegressor` with same search approach
- Provides a non-XGBoost gradient boosting comparison point

**Why 4 and not more:** These four cover the two main ensemble paradigms (bagging: RF/ERT; boosting: XGB/GB). Adding KNN, SVM, or feedforward NN is not justified by the literature — no published ML thermobarometer found these competitive with tree ensembles on tabular geochemical data.

---

## VALIDATION STRATEGY (3-tier, LOSO is mandatory)

### Tier A: Random split (baseline, for comparison with literature)
- 80/20 train/test split, stratified by pressure bins (5 bins)
- 10-fold cross-validation on training set
- Report: mean ± std of RMSE, R², MAE for P and T

### Tier B: Leave-One-Study-Out (LOSO) — REQUIRED FOR SUBMISSION
- Group experiments by Citation (164 unique publications)
- `LeaveOneGroupOut` or `GroupKFold` (scikit-learn)
- Train on all publications except one, test on held-out publication
- Report: median RMSE across all folds, distribution of per-study errors
- **This is the primary validation metric in the paper**
- Rationale: random CV overestimates accuracy because experiments from the same study share systematic features (Ágreda-López et al. 2024; Weber & Blundy 2024)

### Tier C: Performance breakdown
- RMSE binned by pressure range (0–5, 5–15, 15–30, 30+ kbar)
- RMSE binned by temperature range (500–1000, 1000–1300, 1300–1600, 1600+ °C)
- RMSE binned by Mg# (0–0.5, 0.5–0.75, 0.75–0.9, 0.9–1.0)
- Core model vs. full model comparison

### Metrics
- RMSE (primary, for comparison with Putirka SEE)
- MAE (secondary)
- R² (secondary)
- All reported for both P (kbar) and T (°C)

---

## BENCHMARKING AGAINST PUTIRKA (2008)

Use the **Thermobar** Python package (`pip install Thermobar`) to compute Putirka predictions on the same test sets.

**Opx-liquid benchmark:**
- `calculate_opx_liq_temp` with eq 28a, 28b
- `calculate_opx_liq_press` with eq 29a, 29b, 29c

**Two-pyroxene benchmark (if cpx data available for matched experiments):**
- `calculate_two_pyroxene_temp` with eq 36, 37
- `calculate_two_pyroxene_press` with eq 38, 39

**Opx-only benchmark:**
- Putirka eq 29c (opx-only barometer, but requires T input)
- Brey & Köhler (1990) Ca-in-opx thermometer (via Thermobar)

**Head-to-head comparison table format:**

| Method | T RMSE (°C) | T R² | P RMSE (kbar) | P R² | n_test |
|---|---|---|---|---|---|
| Putirka 28b (opx-liq) | — | — | — | — | — |
| Putirka 29a (opx-liq) | — | — | — | — | — |
| This study: ERT (opx-liq) | — | — | — | — | — |
| This study: XGBoost (opx-liq) | — | — | — | — | — |
| Putirka 29c (opx-only) | — | — | — | — | — |
| This study: ERT (opx-only) | — | — | — | — | — |
| This study: XGBoost (opx-only) | — | — | — | — | — |

Note: Putirka equations are T-dependent (barometers require T input) or P-dependent (thermometers require P input). ML models predict P and T independently with no such requirement. This is itself an advantage worth noting.

---

## BIAS CORRECTION (Ágreda-López et al. 2024 method)

ML thermobarometers systematically under-predict extreme values (regression to the mean). Ágreda-López et al. (2024, C&G) showed a simple post-hoc linear bias correction on residuals, binned by predicted value, improved cpx barometry from 2.7 → 2.1 kbar RMSE.

**Implementation:**
1. On training set: fit linear regression of residual (true - predicted) vs. predicted value
2. On test set: apply correction: corrected = predicted + (slope × predicted + intercept)
3. Report both raw and bias-corrected RMSE
4. Cite Ágreda-López et al. (2024) doi:10.1016/j.cageo.2024.105694

---

## SHAP ANALYSIS

Use `shap.TreeExplainer` (exact Shapley values for tree models, polynomial time).

**Figures to produce:**
1. **SHAP summary plot (beeswarm):** All features, colored by feature value. One for P model, one for T model.
2. **SHAP bar plot:** Mean |SHAP| ranking of feature importance. Expect: Al dominates P, Ca dominates T.
3. **SHAP dependence plots:** Top 3 features for P and T, showing nonlinear relationships and interaction effects.
4. **Comparison with thermodynamic expectations:** Does SHAP recover known crystal chemistry? Al_IV and Al_VI should dominate P (Tschermak substitution). Ca should dominate T (solvus thermometry). Mg# should matter for both. If SHAP contradicts known thermodynamics, that's a data quality flag.

---

## PCA OF OPX COMPOSITIONS (EDA contribution)

Following Weber & Blundy (2024), who found 4 PCs explain 96% of melt variance:
- Run PCA on normalized opx oxide compositions
- Report cumulative explained variance
- Plot PC1 vs PC2 colored by P and T
- Discuss effective compositional dimensionality relative to phase rule constraints
- This is a small effort (~1 hour) that yields a good figure and a thermodynamic insight

---

## APPLICATION TO NATURAL SAMPLES

Apply the trained models to published natural opx datasets with independent P-T constraints:

**Priority natural datasets:**
- Kaapvaal Craton peridotite xenoliths (Viljoen et al. 2009, or similar)
- Kilbourne Hole peridotite xenoliths (well-characterized, you have direct experience with KH olivine)
- GEOROC database (https://georoc.eu) — freely accessible global mineral compositions
- Qin et al. (2024) global mantle xenolith compilation

**What to show:**
- ML-predicted P-T compared to published independent estimates (e.g., garnet-opx conventional thermobarometry)
- P-T arrays consistent with known geotherms?
- Any outliers that suggest misidentification or disequilibrium?

---

## DELIVERABLES

### Code and data
- GitHub repository: `opx-thermobar/`
- Cleaned CSV datasets with full provenance
- Trained model weights (joblib serialized)
- Jupyter notebooks: cleaning, EDA, training, validation, application
- `requirements.txt`

### Manuscript
- ~6,000–8,000 words
- Sections: Introduction, Data and Methods, Results, Discussion, Conclusions
- ~8–12 figures
- Supplementary: full dataset, per-study LOSO results, additional SHAP plots

### Figures (planned)

| # | Content | Type |
|---|---|---|
| 1 | P-T distribution of training data | Scatter with histograms |
| 2 | PCA of opx compositions (PC1 vs PC2, colored by P and T) | Biplot |
| 3 | Predicted vs. observed P, all 4 models + Putirka | 1:1 plots (panel) |
| 4 | Predicted vs. observed T, all 4 models + Putirka | 1:1 plots (panel) |
| 5 | Model comparison bar chart (RMSE by model and target) | Bar chart |
| 6 | LOSO validation: per-study RMSE distribution | Box/violin plot |
| 7 | SHAP summary (beeswarm) for P model | SHAP plot |
| 8 | SHAP summary (beeswarm) for T model | SHAP plot |
| 9 | SHAP dependence plots, top 3 features | Panel |
| 10 | Core (5-oxide) vs. full (9-oxide) model comparison | Bar chart |
| 11 | Bias correction effect (before/after residuals) | Residual plots |
| 12 | Natural sample application (predicted P-T vs. published geotherm) | P-T diagram |

---

## TIMELINE (5–6 weeks)

| Week | Tasks | Deliverable |
|---|---|---|
| 1 | Data download, cleaning, filtering, cation recalculation, feature engineering, EDA, PCA | `opx_clean.csv`, EDA notebook, Figs 1–2 |
| 2 | Train all 4 models (P and T) for core and full feature sets. RandomizedSearchCV. Random split validation. Thermobar benchmarking. | Training notebook, benchmark table, Figs 3–5 |
| 3 | LOSO validation (164 groups). Bias correction. Performance breakdown by P/T/Mg# bins. SHAP analysis. | Validation notebook, Figs 6–9 |
| 4 | Natural sample application. Core vs full model comparison. Begin manuscript draft (Methods, Results). | Application notebook, Figs 10–12, draft sections |
| 5 | Complete manuscript (Intro, Discussion, Conclusions). Supplementary materials. GitHub repo cleanup. | Full manuscript draft, supplementary |
| 6 | Buffer / revision with Dr. Lee / final polish | Submission-ready manuscript |

---

## GO/NO-GO CHECKPOINTS

**After data cleaning (end of Week 1):**
- ≥800 clean opx experiments with core 5 oxides + P + T: proceed
- 500–800: proceed but note smaller sample size in limitations
- <500: STOP. Supplement from Putirka (2008) supplementary tables and literature compilation

**After model training (end of Week 2):**
- Best model RMSE_P < Putirka 29c (3.2 kbar): strong result
- Best model RMSE_P ≈ Putirka 29c: still publishable if LOSO holds and SHAP is informative
- Best model RMSE_P >> Putirka 29c: investigate data quality, feature engineering, pressure range restrictions

**After LOSO validation (end of Week 3):**
- LOSO RMSE < 1.5× random-split RMSE: model generalizes well
- LOSO RMSE > 2× random-split RMSE: model is memorizing study-specific patterns. Report honestly, discuss in limitations.

---

## KEY REFERENCES

**ML thermobarometry (methods to follow):**
- Petrelli et al. (2020) JGR:SE — first cpx ML thermobarometer
- Jorgenson et al. (2022) JGR:SE — cpx ERT, LOSO validation
- Li & Zhang (2022) JGR:SE — biotite ERT, n=839, R²≥0.97
- Higgins et al. (2022) CMP — amphibole RF
- Weber & Blundy (2024) J. Petrology — melt ERT, PCA analysis
- Ágreda-López et al. (2024) C&G — cpx ML enhancement, bias correction
- Chicchi et al. (2023) EPSL — GAIA feedforward NN for cpx
- Zhang et al. (2025) G-Cubed — garnet XGBoost
- Qin et al. (2024) GRL — mantle mineral pairs XGBoost
- Petrelli (2024) J. Petrology — review, "ML in Petrology: State-of-the-Art"

**Traditional opx thermobarometry (benchmarks):**
- Putirka (2008) Rev. Min. Geochem. 69, 61–120 — eq 28–29 (opx-liq), eq 36–39 (two-px)
- Brey & Köhler (1990) J. Petrology — Ca-in-opx thermometer
- Wood & Banno (1973), Wells (1977) — classical opx thermometers

**Data source:**
- Hirschmann et al. (2008) G-Cubed — LEPR/ExPetDB database
- Wieser et al. (2022) Volcanica — Thermobar package

**Mineralogy:**
- Morimoto (1988) Min. Mag. — pyroxene nomenclature, cation recalculation
- Neave et al. (2017) — cpx cation sum quality filter (adapted for opx)

---

## STYLE AND OUTPUT PREFERENCES

- Academic/manuscript sections: formal research publication style
- Code: production-quality, well-commented, type-hinted Python. Complete runnable scripts with all imports.
- For tasks with ambiguity: ask clarifying questions before proceeding
- Always search the web for current information on factual claims
- Cite sources with working citations for every factual statement
- Do not manipulate inputs, feature selection, or hyperparameters toward a predetermined conclusion. Report what the data show.

---

## WHAT I NEED FROM YOU

Start by helping me with the first actionable step based on where I am in the process. The data is downloaded (ExPetDB, 1,796 opx rows). Initial census is complete. Next step is Notebook 01: data cleaning.

Always confirm which week/phase we're working on and flag if anything threatens the minimum viable deliverable.
