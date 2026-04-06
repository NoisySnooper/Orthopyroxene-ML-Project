# Opx ML Thermobarometer: Execution Roadmap

This document is the step-by-step guide from where you are now (Notebook 01 complete, prototype Notebook 03 run) to a submission-ready manuscript. It covers what to do, how to produce every figure, the analyses I did not do, and how to interpret the final results.

---

## Where you are right now

Completed:
- Notebook 01 (data cleaning): 1,148-row core dataset, 525-row full dataset, both saved as CSVs
- Notebook 02 (EDA + PCA): all figures produced (fig01, fig02, correlation heatmap, distributions)
- Notebook 03 prototype: rough performance ceiling established (opx-only R² ~0.34, opx-liquid R² ~0.83 for T)

Not yet done:
- Rigorous hyperparameter tuning with proper cross-validation
- Putirka benchmark via Thermobar on identical test sets
- KD equilibrium filter on opx-liquid pairs
- LOSO validation (leave-one-study-out)
- SHAP analysis
- Bias correction
- Natural sample application
- Manuscript

---

## Notebook-by-notebook plan

### Notebook 01: Data cleaning — DONE

Output files already exist:
- `opx_clean_core.csv` (1,148 rows, 5 oxides required)
- `opx_clean_full.csv` (525 rows, 9 oxides required)
- `cleaning_log.txt`

Script: `nb01_data_cleaning.py`

What to verify before proceeding:
1. Open both CSVs. Row counts match (1,148 and 525).
2. Spot-check 3 random rows against the original Excel file using Experiment ID.
3. Confirm all cation sums fall in 3.95-4.05.
4. Confirm no Wo > 5% survives in either CSV.
5. Confirm no P > 100 kbar survives.

### Notebook 02: Exploratory Data Analysis — DONE (code exists)

Script: `nb02_eda_pca.py`

What it produces:
- `fig01_pt_distribution.png` — P-T scatter with marginal histograms, colored by Mg#
- `fig02_pca_biplot.png` — two-panel PCA (colored by T and by P) with loading vectors
- `fig_eda_correlation.png` — feature-target correlation heatmap
- `fig_eda_distributions.png` — histograms of all core oxides + Mg#

How to run it:
```bash
cd [your_working_directory]
python nb02_eda_pca.py
```

Dependencies: pandas, numpy, matplotlib, scikit-learn. That's it.

How each figure is built:

**Fig 1 (P-T distribution):** Uses matplotlib `GridSpec` with 3 panels — main scatter (T on x, P on y, colored by Mg#), marginal histograms top and right. Y-axis is inverted so pressure increases downward (depth convention). The scatter uses `c=df['Mg_num']` with `cmap='viridis'`. Size 12, alpha 0.6 for visibility of dense regions.

**Fig 2 (PCA biplot):** Uses `sklearn.preprocessing.StandardScaler` to z-score the 5 core oxides, then `sklearn.decomposition.PCA()`. Two subplots side by side: panel A colored by T, panel B colored by P. Both show the same PC1-PC2 score space. Loading vectors are drawn as arrows using `ax.annotate()` with `arrowprops`, scaled by 3.0 for visibility. Feature labels are placed at arrow tips.

**Correlation heatmap:** `df.corr()` on selected columns, then `ax.imshow()` with `cmap='RdBu_r'`, `vmin=-1, vmax=1`. Each cell gets a text annotation of the correlation value, colored white if |r| > 0.5 else black.

**Distributions:** Grid of histograms, one per feature, with mean/std in the subplot titles.

You already ran this. All four PNGs should be in your output directory.

### Notebook 03: Baseline models (PROPER version, not the prototype)

This is the rigorous version of what I prototyped. It must include everything below or the paper fails review.

**3.1 Setup:**
```python
import pandas as pd, numpy as np, joblib, copy
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(42)
```

**3.2 Load data and apply KD equilibrium filter (for opx-liquid model):**

The KD filter removes opx-liquid pairs where the crystal and melt were not in equilibrium. Formula:
```
KD(Fe-Mg)_opx-liq = (Fe/Mg)_opx_molar / (Fe/Mg)_liq_molar
Expected: 0.29 ± 0.06 (Putirka 2008)
```

Code pattern:
```python
df = pd.read_csv('opx_clean_core.csv')
liq = load_liquid_data(...)  # same as prototype
df_liq = df.merge(liq, on='Experiment', how='inner')

# Molar Fe/Mg in opx (already have FeO_total, MgO)
df_liq['FeMg_opx'] = (df_liq['FeO_total']/71.844) / (df_liq['MgO']/40.304)
df_liq['FeMg_liq'] = (df_liq['liq_FeO']/71.844) / (df_liq['liq_MgO']/40.304)
df_liq['KD_FeMg'] = df_liq['FeMg_opx'] / df_liq['FeMg_liq']

# Filter: keep 0.23-0.35 (0.29 ± 0.06)
n_before = len(df_liq)
df_liq = df_liq[(df_liq['KD_FeMg'] >= 0.23) & (df_liq['KD_FeMg'] <= 0.35)]
print(f"KD filter: {n_before} -> {len(df_liq)}")
```

Expected: you'll lose 20-35% of opx-liquid pairs. This is the filter I skipped in the prototype.

**3.3 Train/test split:** 80/20 stratified by pressure quintile, same as prototype. Save indices for reproducibility.

**3.4 Hyperparameter tuning with RandomizedSearchCV:**

The prototype used fixed hyperparameters because RandomizedSearchCV timed out in my sandbox. On your machine (or Kaggle), run the full search:

```python
param_grids = {
    'RF': {
        'n_estimators': [200, 500, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, 0.8],
    },
    # ERT same shape as RF
    # XGB with learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
    # GB with learning_rate, subsample, min_samples_split
}

for name, model in base_models.items():
    search = RandomizedSearchCV(
        model, param_grids[name],
        n_iter=50, cv=5,
        scoring='neg_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    print(f"{name} best params: {search.best_params_}")
```

Expected runtime on your laptop: 10-30 minutes per target per model type. Total: 2-4 hours for all 8 model fits. On Kaggle with more cores, faster.

**3.5 Report results:**

For each of 8 models (4 algorithms × T/P), report:
- Train RMSE, test RMSE, test MAE, test R²
- Train/test ratio (overfitting check)
- 5-fold CV mean ± std on training set

Build this as a single table that becomes Table 2 in the paper.

**3.6 Figures to produce:**

**Fig 3 (pred vs obs, P):** 2×2 grid, one panel per model (RF, ERT, XGB, GB). Each panel: scatter of observed vs predicted P on test set, 1:1 line in black dashed. Title includes RMSE and R². Same x/y limits, equal aspect ratio.

**Fig 4 (pred vs obs, T):** identical format to Fig 3 but for temperature.

**Fig 5 (model comparison bar chart):** 1×2 panel. Left: T RMSE for all 4 models. Right: P RMSE for all 4 models. Add horizontal dashed lines showing Putirka benchmarks (48°C and 56°C for T; 3.2 and 3.7 kbar for P). This is the money figure for the abstract.

Code for Fig 5:
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
models = ['RF','ERT','XGB','GB']
colors = ['#2c7bb6','#1a9641','#d7191c','#fdae61']

t_rmses = [results_df[(results_df['target']=='T_C')&(results_df['model']==m)]['rmse_test'].values[0] for m in models]
p_rmses = [results_df[(results_df['target']=='P_kbar')&(results_df['model']==m)]['rmse_test'].values[0] for m in models]

axes[0].bar(range(4), t_rmses, color=colors, edgecolor='black')
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(models)
axes[0].axhline(48, color='gray', ls='--', label='Putirka 28a')
axes[0].axhline(56, color='gray', ls=':', label='Putirka 36')
axes[0].set_ylabel('RMSE (°C)')
axes[0].legend()

axes[1].bar(range(4), p_rmses, color=colors, edgecolor='black')
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(models)
axes[1].axhline(3.2, color='gray', ls='--', label='Putirka 29c')
axes[1].axhline(3.7, color='gray', ls=':', label='Putirka 38')
axes[1].set_ylabel('RMSE (kbar)')
axes[1].legend()
```

### Notebook 04: Putirka benchmarking via Thermobar

This is the analysis I did not do. The prototype compared to Putirka's published SEE values, which were computed on his calibration dataset, not ours. For the paper you must run Putirka's equations on the identical test set.

**4.1 Install and verify:**
```bash
pip install Thermobar
```

**4.2 Format data for Thermobar:**

Thermobar expects specific column names. Documentation: https://thermobar.readthedocs.io

```python
import Thermobar as pt

# Opx dataframe
opx_for_tb = pd.DataFrame({
    'SiO2_Opx': df_test['SiO2'],
    'TiO2_Opx': df_test['TiO2'].fillna(0),
    'Al2O3_Opx': df_test['Al2O3'],
    'FeOt_Opx': df_test['FeO_total'],
    'MgO_Opx': df_test['MgO'],
    'CaO_Opx': df_test['CaO'],
    'MnO_Opx': df_test['MnO'].fillna(0),
    'Cr2O3_Opx': df_test['Cr2O3'].fillna(0),
    'Na2O_Opx': df_test['Na2O'].fillna(0),
})

# Liquid dataframe (similar structure, _Liq suffix)
```

**4.3 Run Putirka equations:**
```python
# Opx-liquid thermobarometry
T_putirka_28a = pt.calculate_opx_liq_temp(equationT='T_Put2008_eq28a', opx_comps=opx_for_tb, liq_comps=liq_for_tb, P=P_input_kbar)
P_putirka_29a = pt.calculate_opx_liq_press(equationP='P_Put2008_eq29a', opx_comps=opx_for_tb, liq_comps=liq_for_tb, T=T_input_C)

# Opx-only
P_putirka_29c = pt.calculate_opx_only_press(equationP='P_Put2008_eq29c', opx_comps=opx_for_tb, T=T_input_C)
```

Note the circular dependency: most Putirka equations require P input for T prediction or T input for P prediction. Options:
1. Iterate to convergence (Thermobar has built-in iterative solvers)
2. Use the true P from the database as input when computing T (upper bound on Putirka T performance)
3. Use the ML-predicted P when computing Putirka T (tests combined pipeline)

Report all three for completeness. Option 2 gives Putirka its best-case performance, which is the fairest comparison.

**4.4 Comparison table:**
```python
comparison = pd.DataFrame({
    'Method': ['Putirka 28a', 'Putirka 28b', 'ML-RF opx-liq', 'ML-ERT opx-liq'],
    'T_RMSE_C': [...],
    'T_MAE_C': [...],
    'T_R2': [...],
    'n_test': [...],
})
```

This becomes Table 3 in the paper.

### Notebook 05: LOSO validation (the critical one)

Leave-one-study-out is what reviewers will demand. Without it, the paper gets rejected.

**5.1 The logic:**

Random 80/20 split leaks information because experiments from the same study (same starting material, same lab, same analytical conditions) get distributed across train and test. The model learns study-specific patterns that don't generalize. LOSO fixes this by holding out all rows from one publication at a time.

**5.2 Implementation:**

```python
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor

groups = df['Citation'].values  # one group per publication
logo = LeaveOneGroupOut()

loso_results = []
for train_idx, test_idx in logo.split(X, y, groups=groups):
    if len(test_idx) < 3:
        continue  # skip single-row publications
    
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    held_out_study = df.iloc[test_idx]['Citation'].iloc[0]
    
    loso_results.append({
        'study': held_out_study,
        'n_test': len(test_idx),
        'rmse': rmse,
        'mean_P_held_out': np.mean(y[test_idx]),
    })

loso_df = pd.DataFrame(loso_results)
print(f"LOSO median RMSE: {loso_df['rmse'].median():.2f}")
print(f"LOSO mean RMSE:   {loso_df['rmse'].mean():.2f}")
print(f"LOSO IQR: {loso_df['rmse'].quantile(0.25):.2f} - {loso_df['rmse'].quantile(0.75):.2f}")
```

Runtime: 126 folds × 8 models = 1,008 fits. For RF/ERT with 500 trees, about 30-60 seconds per fit. Total: 8-16 hours on a laptop. Run overnight.

**5.3 Figure 6: LOSO box plot:**

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Panel A: box plot of per-study RMSE for each model
data_T = [loso_df_by_model[m]['rmse_T'] for m in ['RF','ERT','XGB','GB']]
axes[0].boxplot(data_T, labels=['RF','ERT','XGB','GB'])
axes[0].axhline(random_split_median_T, color='red', ls='--', label='Random-split median')
axes[0].set_ylabel('Per-study RMSE (°C)')
axes[0].set_title('LOSO: Temperature')

# Panel B: same for pressure
```

**5.4 Expected result:**

LOSO RMSE will be 1.2-1.8× higher than random-split RMSE. If it's exactly the same, something is wrong. If it's 3× higher, the model is memorizing study-level patterns and doesn't generalize.

For interpretation: honestly report the LOSO numbers as the primary result. The random-split numbers become the "upper bound" shown for literature comparison.

### Notebook 06: SHAP analysis

**6.1 Setup:**
```python
import shap
best_model_P = joblib.load('model_RF_P_kbar_opx_liq.joblib')  # whichever wins
explainer = shap.TreeExplainer(best_model_P)
shap_values_P = explainer.shap_values(X_test)
```

**6.2 Figures:**

**Fig 7 (SHAP beeswarm, P):**
```python
shap.summary_plot(shap_values_P, X_test, feature_names=FEATURES, show=False)
plt.savefig('fig07_shap_P.png', dpi=200, bbox_inches='tight')
plt.close()
```

**Fig 8 (SHAP beeswarm, T):** same code, T model.

**Fig 9 (SHAP dependence plots, top 3 features):**
```python
top3 = np.argsort(np.abs(shap_values_P).mean(0))[-3:][::-1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feat_idx in enumerate(top3):
    shap.dependence_plot(feat_idx, shap_values_P, X_test, feature_names=FEATURES, ax=axes[i], show=False)
```

**6.3 What to look for in interpretation:**

For the pressure model, Al_IV and Al_VI should dominate. This is because aluminum enters opx via the Tschermak substitution (MgSi ↔ AlAl), which is strongly pressure-sensitive. If Al doesn't dominate, either your model is wrong or your feature engineering missed something.

For the temperature model, Ca (Wo fraction) should dominate for opx-only, and liquid SiO2/MgO should dominate for opx-liquid. Ca solubility in opx increases with T (basis of the two-pyroxene thermometer). If SHAP says something else, report it honestly and discuss.

### Notebook 07: Bias correction

From Ágreda-López et al. (2024). ML models regress to the mean: they under-predict high values and over-predict low values.

**7.1 Fit correction on training set:**
```python
from sklearn.linear_model import LinearRegression

# On training predictions
train_pred = best_model.predict(X_train)
residuals_train = y_train - train_pred  # positive = under-predicted

# Fit linear correction: residual = a * predicted + b
bias_model = LinearRegression()
bias_model.fit(train_pred.reshape(-1, 1), residuals_train)

a, b = bias_model.coef_[0], bias_model.intercept_
print(f"Correction: residual = {a:.4f} * pred + {b:.2f}")
```

**7.2 Apply to test set:**
```python
test_pred = best_model.predict(X_test)
test_pred_corrected = test_pred + (a * test_pred + b)

rmse_raw = np.sqrt(mean_squared_error(y_test, test_pred))
rmse_corrected = np.sqrt(mean_squared_error(y_test, test_pred_corrected))
print(f"Raw: {rmse_raw:.2f}, Corrected: {rmse_corrected:.2f}")
```

**7.3 Figure 11: residual plots before/after:**

Two-panel figure. Left: residuals vs predicted, raw. Right: residuals vs predicted, corrected. The slope should be near zero after correction. If correction makes things worse, skip it and document why.

### Notebook 08: Natural sample application

**8.1 Find datasets:**

Priority sources:
1. GEOROC database: https://georoc.eu (free, needs account)
2. Viljoen et al. (2009) Kaapvaal peridotite xenoliths
3. Lin et al. (2023) NE China spinel peridotites (Mendeley Data, open)
4. Qin et al. (2024) GRL supplementary data

Download as CSVs, standardize column names to match your training features.

**8.2 Run predictions:**
```python
natural_df = pd.read_csv('kaapvaal_peridotites.csv')
# Apply same cleaning: pigeonite filter, oxide total filter, cation recalc
natural_clean = clean_data(natural_df)
# Apply same feature engineering
natural_features = engineer_features(natural_clean)

X_natural = natural_features[FEATURE_LIST].values
T_pred = model_T.predict(X_natural)
P_pred = model_P.predict(X_natural)
```

**8.3 Figure 12: predicted P-T on geotherm:**

Plot T on x-axis, P on y-axis (inverted). Overlay your predictions as points. Add published geotherms (40 mW/m² and 45 mW/m² for cratons, using the Hasterok & Chapman 2011 formulation). Check whether your predictions fall along the expected geotherm. If they scatter wildly or all cluster at one point, your model has a problem on natural samples.

### Notebook 09: Manuscript figures and tables

Compile every figure in publication quality: 300 dpi, sans-serif font (Arial or Helvetica), consistent color palette, clear axis labels, panel labels (a, b, c) for multi-panel figures.

Tables:
- Table 1: Dataset summary (counts, ranges, coverage)
- Table 2: Hyperparameter search results
- Table 3: Putirka vs ML comparison on test set
- Table 4: LOSO validation results (median, IQR, per-study breakdown for top 10 studies)
- Table 5: SHAP feature importance ranking

---

## How to interpret the final results

Interpretation means translating numbers into scientific claims you can defend. Here is the framework for each result type.

**If opx-only RMSE is 150-200°C / 10-12 kbar (like the prototype showed):**

This is a negative-but-informative result. It means opx chemistry alone does not carry enough information to determine P-T crystallization conditions to geologically useful precision. This is not a failure of the model — it is a measurement of the information content of opx composition. Publishable as: "We quantify the fundamental limits of single-phase opx thermobarometry. The observed R² of ~0.34 establishes a performance ceiling that explains why no traditional opx-only thermobarometer exists in the literature."

**If opx-liquid RMSE is ~80-100°C / 7-9 kbar overall:**

This is comparable to Putirka's published SEE for two-pyroxene equations (±56-60°C, 3.7 kbar). Weaker than Putirka's opx-liquid equations in absolute terms, but your model has advantages: (1) no T input required for P prediction, (2) no P input required for T prediction, (3) single model across all compositions, (4) MC-dropout-ready for uncertainty quantification.

**If pressure-binned performance shows ML beats Putirka at 0-15 kbar but loses at 30+ kbar:**

This is the most interesting and publishable finding. Frame as: "ML excels in the compositional range where most natural opx-bearing rocks form (crustal and shallow mantle, 0-15 kbar), but both ML and traditional methods struggle at deep mantle conditions where training data is sparse. We recommend ML for crustal applications and flag the need for more high-P experimental data."

**If LOSO RMSE is 1.2-1.5× the random-split RMSE:**

Model generalizes well to new studies. Report LOSO as the primary metric. This is what reviewers want to see.

**If LOSO RMSE is 2-3× the random-split RMSE:**

Model is memorizing study-specific patterns. Honest options:
1. Report both numbers and discuss in limitations
2. Add a section explaining which compositional regions are "safe" for prediction
3. Recommend users check whether their sample's composition falls within the training distribution before trusting predictions

**If SHAP identifies expected features (Al for P, Ca for T):**

Write: "SHAP analysis recovers the expected thermodynamic controls on opx composition. Al content dominates pressure prediction, consistent with the Tschermak substitution [MgSi ↔ AlAl] being pressure-sensitive. Calcium content dominates temperature prediction, consistent with the well-known increase in Ca solubility in opx with temperature [Brey & Köhler 1990]."

**If SHAP identifies unexpected features dominating:**

Do NOT hide this. Investigate: is it a genuine finding, or an artifact of training data bias? Could the model be using Fe content as a proxy for something else? If you cannot explain it, report honestly: "SHAP analysis identifies [feature X] as the dominant predictor, which is inconsistent with classical thermodynamic expectations. We hypothesize this reflects [possible explanation] but note it warrants further investigation."

**If Putirka beats ML on specific equations:**

Report it. The paper is not "ML wins everything." The paper is "here is the first ML opx thermobarometer, here is how it compares to Putirka in each regime, and here are the conditions under which each method is preferable." Honest comparison builds reviewer trust and makes the paper more useful to the community.

**If natural samples give sensible geotherms:**

Your model works in the wild. Write: "Application to [N] natural peridotite xenoliths from [locality] yields P-T estimates consistent with a [40 mW/m²] conductive geotherm, validating the model on independent data."

**If natural samples give nonsense:**

Investigate why. Compositional extrapolation? Wrong rock type (mantle vs cumulate)? Report the failure honestly and discuss limits of applicability.

---

## Timeline assuming prototype is done

- **Week 1:** Re-run Notebook 01 and 02 on your machine to verify reproducibility. Fix any issues. Run Notebook 03 with proper RandomizedSearchCV and KD filtering. (5 days)

- **Week 2:** Notebook 04 (Thermobar benchmark) and Notebook 05 (LOSO). LOSO is the big time sink — plan for overnight runs. (5 days)

- **Week 3:** Notebook 06 (SHAP), Notebook 07 (bias correction), initial interpretation of LOSO results. (5 days)

- **Week 4:** Notebook 08 (natural samples), Notebook 09 (final figures). Begin manuscript draft. (5 days)

- **Week 5:** Complete manuscript draft. Send to Dr. Lee. (5 days)

- **Week 6:** Incorporate feedback. Format for journal. Submit. (5 days)

This assumes no major data quality surprises. If KD filtering drops your opx-liquid dataset below 500 rows, add a week to search for supplementary data from Putirka (2008) and Weber & Blundy (2024) supplementary tables.

---

## Files already in your output directory

- `revised_master_prompt_opx_thermobar.md` — the project plan
- `nb01_data_cleaning.py` — Notebook 01 script
- `opx_clean_core.csv` — cleaned core dataset (1,148 rows)
- `opx_clean_full.csv` — cleaned full dataset (525 rows)
- `nb02_eda_pca.py` — Notebook 02 script
- `fig01_pt_distribution.png`
- `fig02_pca_biplot.png`
- `fig_eda_correlation.png`
- `fig_eda_distributions.png`
- `nb03_baseline_models.py` — Notebook 03 prototype script
- `nb03_results_all.csv` — prototype results
- `fig03_pred_vs_obs_combined.png` — prototype comparison figure

All scripts are standalone. Clone them to your machine and run in order.
