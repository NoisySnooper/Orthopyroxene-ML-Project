# v10 figure audit

**Status:** canonical spec for every figure in both papers
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`

The big one. User's top concern is figures — legibility, completeness, comparative metrics, visual appeal. This doc enforces per-figure standards, per-NB inventory, and checklist validation.

---

## 1. Global standards (apply to every figure)

### 1.1 Palette

**Okabe-Ito colorblind-safe palette** (enforced via `src/plot_style.py`):

| Color | Hex | Usage |
|---|---|---|
| Black | `#000000` | 1:1 line, Putirka classical reference |
| Orange | `#E69F00` | External cpx models (Agreda, Wang, Petrelli) |
| Sky blue | `#56B4E9` | Putirka 2008 (classical reference) |
| Bluish green | `#009E73` | Stacked family |
| Yellow | `#F0E442` | Jorgenson 2022 cpx-only |
| Blue | `#0072B2` | Forest family (RF, ERT) |
| Vermillion | `#D55E00` | Boosted family (XGB, GB) |
| Reddish purple | `#CC79A7` | Universal model |

**New v10 additions:**

| Color | Hex | Usage |
|---|---|---|
| Teal | `#44AA99` | CatBoost |
| Indigo | `#332288` | LightGBM |
| Olive | `#999933` | ElasticNet |
| Rose | `#882255` | MLP |

Every figure uses the same method → color mapping across all NBs. Enforced via a central `FAMILY_COLORS` dict in `src/plot_style.py`.

### 1.2 Dimensions (JGR ML & Computation)

- Single column: 90 mm wide (354 pt)
- 1.5 column: 140 mm wide (551 pt)
- Double column: 190 mm wide (748 pt)

Default: 1.5 column for benchmark figures, single for scatter, double for matrix/heatmap.

### 1.3 Output format

Every figure produces:
- **PDF vectorized** with embedded fonts (submission)
- **PNG 300 DPI** (screens, quick inspection)
- **TXT caption** separately with full figure legend text

### 1.4 Legibility requirements

- Minimum font size: 8 pt in figure, 10 pt in axis labels
- No overlapping text labels. If labels are dense (e.g., method benchmark bars), use programmatic collision detection (matplotlib's `adjust_text` library) OR rotate labels 45° OR spread via y-offsets
- Axis labels always filled, never "x" or "y" alone
- Legends always present when multiple series, never hidden
- Figure title concise (<80 chars), subtitle allowed (<100 chars)

### 1.5 Metrics on figure (NOT only in caption)

For every scatter plot or comparison, display directly on the figure:
- **RMSE** (and 95% CI)
- **R²**
- **Slope** (fitted line)
- **Intercept**
- **n** (sample size)
- **1:1 deviation** statistic if applicable

Placement: upper-left or upper-right corner box, 8-10 pt font, semi-transparent background.

### 1.6 Data coverage requirement

Every benchmark or comparison figure MUST include:
- All 8 base models per pipeline
- All 4 ensemble methods per pipeline (or winner + second-place)
- All external models (Agreda, Jorgenson, Wang, Petrelli, Putirka) where applicable
- Stacked model per pipeline per `v10_stacking_propagation_audit.md`

Missing a method = figure fails audit.

### 1.7 Caption template

Every figure caption includes:
1. One-sentence description
2. What's on each axis
3. What each color/marker represents (if not obvious)
4. Key statistic
5. Sample size
6. Reference to source notebook (e.g., "Produced by nb04_benchmark.ipynb cell 27")
7. Reference to associated test (e.g., "Supports T01")

---

## 2. Figure inventory

Full list by notebook. Audited during Phase G.

### 2.1 NB02 EDA figures

Per pipeline (opx, cpx, twopx, universal). 4 pipelines × 6 figures = 24 figures.

| Figure | Purpose |
|---|---|
| `fig_nb02_{pipeline}_pca_variance.png` | PCA cumulative variance |
| `fig_nb02_{pipeline}_pca_biplot.png` | PC1/PC2 with samples colored by T or P |
| `fig_nb02_{pipeline}_clusters.png` | k-means clusters in PC1/PC2 space |
| `fig_nb02_{pipeline}_pt_distribution.png` | Training P-T distribution, hex bin density |
| `fig_nb02_{pipeline}_correlation_matrix.png` | Oxide correlation heatmap |
| `fig_nb02_{pipeline}_feature_distributions.png` | Per-oxide histograms + target distributions |

### 2.2 NB03 (per pipeline) training figures

Per pipeline × per test. ~15 figures per pipeline × 4 pipelines = 60 figures.

Per pipeline:

| Figure | Purpose |
|---|---|
| `fig_nb03_{pipeline}_multiseed_rmse.png` | Boxplot of 20-seed RMSE per (model, feature_set) — each panel per target/track. Per Q10(a) style. |
| `fig_nb03_{pipeline}_optuna_convergence.png` | Best-so-far curves per study, faceted by model |
| `fig_nb03_{pipeline}_hp_importance.png` | Hyperparameter importance bars per model |
| `fig_nb03_{pipeline}_ensemble_comparison.png` | 4 panels: test T RMSE, test P RMSE, external T, external P. Bars per method (A/B/C/D) + best single base reference. |
| `fig_nb03_{pipeline}_stacking_weights.png` | Ridge coefficients per (target × track) per base model |
| `fig_nb03_{pipeline}_stacking_oof_correlation.png` | Heatmap of base model OOF correlations |
| `fig_nb03_{pipeline}_T01_primary_model.png` | Primary model selection bar chart with CIs |
| `fig_nb03_{pipeline}_T03_resampling.png` | Resampling vs non-resampled per combo |
| `fig_nb03_{pipeline}_T04_naug.png` | N_AUG=1 vs N_AUG=5 comparison |
| `fig_nb03_{pipeline}_T05_feature_set_stability.png` | Winning feature set across seeds |
| `fig_nb03_{pipeline}_T11_catboost_lightgbm.png` | New model family comparison |
| `fig_nb03_{pipeline}_T12_nn_linear.png` | MLP and ElasticNet vs trees |

**Universal-specific extras per `v10_universal_model_exploration.md`:**

| Figure | Purpose |
|---|---|
| `fig_nb03_universal_graceful_degradation.png` | RMSE per phase-subset scope |
| `fig_nb03_universal_all_phase_vs_subset.png` | All-phase lowest RMSE check |
| `fig_nb03_universal_vs_specialized.png` | Bar delta per scope |

### 2.3 NB04 benchmark figures

Combined benchmark notebook (merged from old NB04 + NBM).

Per pipeline and cross-pipeline. Key figures:

| Figure | Purpose |
|---|---|
| `fig_nb04_{pipeline}_method_benchmark_arcpl_kdEq.png` | Bar chart. All methods (8 base + 4 ensemble + external + Putirka). T RMSE panel + P RMSE panel + coverage panel. CIs. ArcPL Kd-eq subset. **HEADLINE FIGURE.** |
| `fig_nb04_{pipeline}_method_benchmark_arcpl_full.png` | Same but full n=197/equivalent. Supplementary. |
| `fig_nb04_{pipeline}_arcpl_scatter.png` | Grid of predicted vs observed scatters. Rows = methods (8 base + stacked + external = ~12 rows). Columns = T_C, P_kbar. Each panel: 1:1 line, fit line, metrics box (RMSE, R², slope, intercept, n). Panel letters. |
| `fig_nb04_{pipeline}_diagnostic_encyclopedia_T.png` | Multi-panel: residual histograms, residual vs predicted, residual vs composition, for every method. |
| `fig_nb04_{pipeline}_diagnostic_encyclopedia_P.png` | Same for P. |
| `fig_nb04_{pipeline}_model_heatmap_boxplot.png` | Per user Q10(a). Panels per track (opx_only, opx_liq for opx pipeline etc.). Each panel: x=model, y=RMSE, boxes=20-seed variance, colored by feature_set. |
| `fig_nb04_{pipeline}_model_heatmap_matrix.png` | Per user Q10(b). Rows=method (all 12+), columns=eval scope (test, LOSO, Cluster-KFold, ArcPL full, ArcPL Kd-eq, LeaveOneRegionOut). Cells = RMSE with color + numeric annotation. Separate heatmap for T and P. |
| `fig_nb04_{pipeline}_h2o_sensitivity.png` | Per-track RMSE at different H2O bins (if liq track) |
| `fig_nb04_{pipeline}_coverage.png` | Coverage percentage per method (important because our ML = 100%, Putirka may be less) |

Cross-pipeline comparisons (NEW):

| Figure | Purpose |
|---|---|
| `fig_nb04_cross_pipeline_heatmap.png` | 4 pipelines × 2 targets × 2 tracks = matrix of best-model RMSE. Visualizes which pipeline wins what. |
| `fig_nb04_external_model_comparison.png` | Side-by-side: our best opx-liq, our best cpx-liq, our best twopx vs Agreda, Jorgenson, Petrelli, Putirka. ArcPL same samples. |

### 2.4 NB05 generalization figures

Per pipeline + cross-pipeline.

| Figure | Purpose |
|---|---|
| `fig_nb05_{pipeline}_loso.png` | LOSO pooled RMSE per model. Error bars from per-fold distribution. |
| `fig_nb05_{pipeline}_loso_per_fold.png` | Distribution of per-fold RMSE per model (violin/boxplot) |
| `fig_nb05_{pipeline}_cluster_kfold.png` | Same for chemical_cluster groups |
| `fig_nb05_{pipeline}_target_bin.png` | Same for T bins (extrapolation) |
| `fig_nb05_{pipeline}_leave_one_region.png` | NEW: tectonic setting as group |
| `fig_nb05_{pipeline}_generalization_summary.png` | 4-strategy × N-models matrix of RMSE. Colored by rank. |

### 2.5 NB06 SHAP figures

Per pipeline. ~10 figures per pipeline.

| Figure | Purpose |
|---|---|
| `fig_nb06_{pipeline}_shap_T_beeswarm.png` | Tree-SHAP beeswarm for T prediction, best tree model |
| `fig_nb06_{pipeline}_shap_P_beeswarm.png` | Same for P |
| `fig_nb06_{pipeline}_shap_T_dependence.png` | Top feature's dependence plot |
| `fig_nb06_{pipeline}_shap_P_dependence.png` | Same for P |
| `fig_nb06_{pipeline}_stacked_shap_T.png` | Base-weighted SHAP from stacked ensemble |
| `fig_nb06_{pipeline}_stacked_shap_P.png` | Same for P |
| `fig_nb06_{pipeline}_mlp_kernel_shap.png` | MLP KernelSHAP (100 samples, supplementary) |
| `fig_nb06_{pipeline}_meta_weights_linear_shap.png` | Linear-SHAP on Ridge meta, which base contributed most |
| `fig_nb06_{pipeline}_robustness_proxy_check.png` | SHAP after ablating liq_SiO2 + liq_MgO |
| `fig_nb06_{pipeline}_robustness_correlation.png` | Feature corr heatmap for proxy diagnosis |

### 2.6 NB07 bias correction (merged with nb07b)

Per pipeline.

| Figure | Purpose |
|---|---|
| `fig_nb07_{pipeline}_arcpl_residual_hist.png` | Histogram of T residuals on ArcPL per model |
| `fig_nb07_{pipeline}_bias_vs_prediction_T.png` | Residual T vs predicted T, per model |
| `fig_nb07_{pipeline}_bias_vs_prediction_P.png` | Same for P |
| `fig_nb07_{pipeline}_bias_vs_composition_T.png` | Residual T vs [liq_Na2O, H2O, liq_TiO2, Mg_num] (4 panel) |
| `fig_nb07_{pipeline}_bias_vs_composition_P.png` | Same for P |
| `fig_nb07_{pipeline}_piecewise_P_correction.png` | Before/after P correction, RMSE delta + CI |
| `fig_nb07_{pipeline}_composition_conditional_T.png` | Ridge corr fit, before/after ArcPL T RMSE |
| `fig_nb07_{pipeline}_conformal_coverage.png` | Conformal interval empirical coverage |
| `fig_nb07_{pipeline}_qrf_interval_width.png` | QRF IQR distribution |
| `fig_nb07_{pipeline}_residual_test_vs_arcpl.png` | Test vs ArcPL residual distribution comparison (distribution shift) |

### 2.7 NB08 natural samples (merged with nb08b, KEY user concern)

**This is where user wants maximum completeness.** All 2-way comparisons (a, b, c, d per Q12).

Per pipeline combination. Key figures:

| Figure | Purpose |
|---|---|
| `fig_nb08_cross_mineral_1to1_grid.png` | **9×9 panel grid** of all-vs-all 1:1 scatters for every method on natural samples where both minerals present. Our opx stacked, our cpx stacked, our twopx stacked, Agreda cpx-liq, Jorgenson cpx-only, Wang cpx-liq, Petrelli cpx-liq, Putirka 2-px eq36/39, Putirka opx-liq eq28a. |
| `fig_nb08_our_best_opx_vs_our_best_cpx.png` | Key 1:1 scatter. Full metrics overlay. |
| `fig_nb08_our_vs_agreda_cpx.png` | Our cpx vs Agreda cpx on same samples. |
| `fig_nb08_our_vs_jorgenson_cpx.png` | Our cpx vs Jorgenson cpx on same samples. |
| `fig_nb08_our_vs_putirka_twopx.png` | Our twopx vs Putirka eq36/39. |
| `fig_nb08_divergence_map.png` | Scatter of ΔT and ΔP between opx and cpx predictions, colored by Kd equilibrium test result. |
| `fig_nb08_opx_world_map_static.png` | **Static world map**. Robinson projection, opx samples colored by tectonic setting, sized by sample density per hex bin. Locality names annotated for famous sites. |
| `fig_nb08_opx_world_map_predicted_T.png` | **Second panel (user Q34):** same map, colored by our best model's predicted T (hex-bin mean). |
| `fig_nb08_cpx_world_map_static.png` | Same for cpx |
| `fig_nb08_cpx_world_map_predicted_T.png` | Predicted T map for cpx |
| `fig_nb08_twopx_world_map_static.png` | Same for twopx |
| `fig_nb08_twopx_world_map_predicted_T.png` | Predicted T map for twopx |
| `fig_nb08_opx_world_map_interactive.html` | Folium HTML, clustered markers, popup per sample |
| `fig_nb08_cpx_world_map_interactive.html` | Same |
| `fig_nb08_twopx_world_map_interactive.html` | Same |
| `fig_nb08_locality_stratified_rmse.png` | Per-locality RMSE heatmap |
| `fig_nb08_convergence_by_kd.png` | ΔT(opx-cpx) binned by Fe-Mg equilibrium test |

### 2.8 NB10 extended analyses

| Figure | Purpose |
|---|---|
| `fig_nb10_{pipeline}_ood_vs_residual.png` | IsolationForest score vs residual magnitude |
| `fig_nb10_{pipeline}_mc_vs_iqr.png` | Monte Carlo intervals vs QRF IQR intervals |
| `fig_nb10_{pipeline}_mc_uncertainty.png` | MC-derived uncertainty distribution |
| `fig_nb10_{pipeline}_h2o_dependence.png` | Residuals vs H2O, stratified |
| `fig_nb10_{pipeline}_analytical_uncertainty.png` | EPMA noise propagation effect on predictions |
| `fig_nb10_twopx_benchmark.png` | Populate the empty CSV and make the figure. Putirka eq36 vs our twopx. |

### 2.9 NBF manuscript figures (per paper)

Per user: separate per paper. Opx paper selects subset; cpx paper selects subset.

**Opx paper figure list (~15 figures):**

| F# | Source | Purpose |
|---|---|---|
| 1 | NB02 | Training data P-T distribution + geographic origin (from natural world map) |
| 2 | NB02 | PCA biplot with clusters (opx) |
| 3 | NB03 opx | Multi-seed RMSE boxplot |
| 4 | NB03 opx | Ensemble method comparison |
| 5 | NB04 opx | ArcPL method benchmark bar chart (**HEADLINE**) |
| 6 | NB04 opx | ArcPL scatter grid (condensed to 6 panels: our best base, our stacked, Putirka variants) |
| 7 | NB04 opx | Model heatmap matrix |
| 8 | NB05 opx | Generalization summary (LOSO, Cluster-KFold, TargetBin, LeaveOneRegionOut) |
| 9 | NB06 opx | SHAP T beeswarm |
| 10 | NB06 opx | SHAP P beeswarm |
| 11 | NB07 opx | Bias correction residuals (before/after) |
| 12 | NB08 opx | Natural samples: our opx vs external methods 1:1 scatter |
| 13 | NB08 opx | World map static (opx, tectonic setting) + predicted T panel |
| 14 | NB10 opx | Conformal coverage |
| 15 | NB10 opx | OOD score vs residual |

**Cpx paper figure list (~25 figures):** similar structure × 3 pipelines + cross-pipeline comparisons + universal exploration (isolated).

---

## 3. Figure audit checklist (applied in Phase G)

For each figure:

- [ ] PDF + PNG both produced, paths match canonical list in `config.CANONICAL_FIGURES`
- [ ] Okabe-Ito palette used (via `src/plot_style.FAMILY_COLORS`)
- [ ] Font sizes >= 8 pt anywhere, >= 10 pt on axis labels
- [ ] No overlapping text (auto-check via pixel inspection or manual)
- [ ] Legend present if multiple series
- [ ] Metrics overlay (RMSE, R², slope, intercept, n) on scatter/comparison figures
- [ ] 1:1 line on every prediction vs observed scatter
- [ ] Stacked model appears where required per `v10_stacking_propagation_audit.md`
- [ ] All 8 base + 4 ensemble + external appear where applicable
- [ ] Caption in separate TXT file, follows template
- [ ] JGR-MLC dimension compliance (90/140/190 mm)
- [ ] Per-paper subsetting enforced (opx paper figure doesn't include cpx results)

Automated check: `scripts/v10_figure_audit_checker.py` validates the figure inventory against the expected list and does basic sanity checks (dimensions, DPI, file presence).

---

## 4. Known v9 figure problems — explicit fixes

Per user's question in earlier chat:

| v9 problem | v10 fix |
|---|---|
| `fig_nb04_arcpl_opx_liq_scatter` only shows forest + boosted | Include stacked + 4 ensemble + all external; 6-9 panel grid |
| `fig_nb04_diagnostic_encyclopedia` incomplete | Include all 12+ methods per pipeline |
| `fig_nb04_method_benchmark_paired` label overlap | Programmatic adjust_text + rotate 45° + spread labels |
| CANONICAL_FIGURES list stale | Rewrite entirely per this doc |
| No model heatmap | NEW: both styles (boxplot per Q10a AND matrix per Q10b) |
| No world map | NEW: static + interactive per mineral |
| Stacking missing from downstream figures | Enforced per `v10_stacking_propagation_audit.md` |
| No per-paper subsetting | NBF outputs to `figures/opx/` and `figures/cpx/` separately |
| Caption not rigorous | Use `scripts/v10_figure_audit_checker.py` |

---

## 5. Implementation notes

### 5.1 `src/plot_style.py` changes

```python
# v10 extension
FAMILY_COLORS = {
    'forest': '#0072B2',
    'boosted': '#D55E00',
    'stacked': '#009E73',
    'catboost': '#44AA99',
    'lightgbm': '#332288',
    'elasticnet': '#999933',
    'mlp': '#882255',
    'universal': '#CC79A7',
    'agreda_cpx_liq': '#E69F00',
    'agreda_cpx_only': '#FBD07A',
    'jorgenson_cpx_only': '#F0E442',
    'jorgenson_cpx_liq': '#FFEE8C',
    'wang_cpx_liq': '#D55E00',   # shared with boosted — consider alt
    'petrelli_cpx_liq': '#DD8888',
    'putirka_cpx_liq': '#56B4E9',
    'putirka_opx_liq': '#89C8E9',
    'putirka_twopx_eq36': '#0072B2',  # reuse forest
    'onetoone': '#000000',
}

def apply_jgr_mlc_style():
    """Set matplotlib rcParams for JGR-MLC requirements."""
    ...

def metrics_textbox(ax, rmse, r2, slope, intercept, n, loc='upper left'):
    """Standardized metrics overlay on a scatter plot."""
    ...
```

### 5.2 `src/io_utils.py` extensions

```python
def save_figure(fig, path_stem, dpi=300, formats=('pdf', 'png'), caption=None):
    """Save figure in multiple formats with caption TXT."""
    ...
```

---

## 6. Phase G execution

1. Ensure every NB produces its figure inventory per this doc
2. Run `scripts/v10_figure_audit_checker.py`
3. Review any FAILED checks, fix
4. Re-run until all PASS
5. User visual spot-check: open the 15 headline figures, confirm visual appeal

Approval gate: Phase H starts only after figure audit returns zero FAILED.

---

## 7. Figure counts summary

| Notebook | Figures per pipeline | Pipelines | Cross-pipeline | Total |
|---|---|---|---|---|
| NB02 | 6 | 4 | 0 | 24 |
| NB03 | ~12-15 | 4 | 0 | ~54 |
| NB04 | ~10 | 4 | 2 | ~42 |
| NB05 | 6 | 4 | 0 | 24 |
| NB06 | 10 | 4 | 0 | 40 |
| NB07 | 10 | 4 | 0 | 40 |
| NB08 | ~15 (incl world maps) | 4 (with pipeline-specific logic) | 2 | ~20 |
| NB10 | 6 | 4 | 0 | 24 |
| NBF opx paper | — | — | — | 15 |
| NBF cpx paper | — | — | — | 25 |
| **Total** | | | | **~308** |

Plus ~10 interactive HTML figures.

Big number. Manageable with Okabe-Ito enforcement and programmatic audit.
