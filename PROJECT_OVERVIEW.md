# Project overview

This document sketches the data flow and module topology of the opx ML
thermobarometer pipeline. For operational instructions (install, run order),
see [README.md](README.md).

## Data flow

```mermaid
flowchart TD
    RAW["ExPetDB + LEPR raw xlsx"] --> NB01[nb01_data_cleaning]
    NB01 --> CORE["opx_clean_core.parquet<br/>opx_clean_core.csv<br/>opx_clean_full.parquet"]
    CORE --> NB02[nb02_eda_pca]
    NB02 --> CLUSTERED["opx_clean_core_with_clusters.parquet<br/>(adds chemical_cluster)"]
    CLUSTERED --> NB03[nb03_baseline_models]
    NB03 --> LIQ["opx_clean_opx_liq.parquet<br/>opx_clean_opx_only.parquet"]
    NB03 --> SPLITS["data/splits/*.npy"]
    NB03 --> WIN["nb03_winning_configurations.json<br/>nb03_multi_seed_results.csv<br/>canonical models (joblib)"]

    WIN --> NB04[nb04_putirka_benchmark]
    WIN --> NB04B[nb04b_lepr_arcpl_validation]
    WIN --> NB05[nb05_loso_validation]
    WIN --> NB06[nb06_shap_analysis]
    WIN --> NB07[nb07_bias_correction]
    WIN --> NB08[nb08_natural_samples]
    WIN --> NB10[nb10_extended_analyses]

    NB04 --> FIG[nbF_figures]
    NB04B --> FIG
    NB05 --> FIG
    NB06 --> FIG
    NB07 --> FIG
    NB08 --> FIG
    NB10 --> FIG

    NB04 --> NB09[nb09_manuscript_compilation]
    NB05 --> NB09
    NB06 --> NB09
    NB10 --> NB09
    NB09 --> T[results/table1..10]
    FIG --> F[figures/fig01..14 PNG+PDF]
```

## Module topology

```mermaid
flowchart LR
    CONFIG[config.py<br/>paths + constants + seeds]
    FEATURES[src.features<br/>raw / alr / pwlr / augment]
    MODELS[src.models<br/>BASE_MODELS / predict_median / predict_iqr]
    DATA[src.data<br/>load_opx_* / load_splits / load_winning_config]
    EVAL[src.evaluation<br/>compute_metrics / loso_splits / cluster_kfold]
    GEO[src.geotherm<br/>hasterok_chapman_geotherm]
    IO[src.io_utils<br/>save_figure / save_table / with_progress]
    PLOT[src.plot_style<br/>Tol palette / apply_style / panel_label]

    CONFIG --> DATA
    CONFIG --> IO
    DATA --> EVAL
    DATA --> PLOT
    MODELS --> EVAL
    IO --> PLOT

    NB[notebooks/*] --> FEATURES
    NB --> MODELS
    NB --> DATA
    NB --> EVAL
    NB --> GEO
    NB --> PLOT
```

## Invariants downstream of NB03

- `nb03_winning_configurations.json` names the global-winner feature set.
- `canonical_model_filename(model, target, track, RESULTS)` returns the
  correct joblib filename without requiring the caller to know which feature
  set won.
- `load_canonical_model(...)` loads it. No downstream notebook hardcodes
  `alr`, `pwlr`, or `raw` anywhere.
- `data/splits/test_indices_opx_liq.npy` and `test_indices_opx.npy` are the
  only test-set mapping used by every figure and metric.

## Robustness checks in NB06 (appendix)

Each check targets the concern that SHAP's dominant features might be proxies
for laboratory experimental design rather than genuine physicochemical
signal.

| Test                       | Expected outcome if model is sound            |
|----------------------------|-----------------------------------------------|
| Ablation of `liq_SiO2`+`liq_MgO`  | RMSE rises; magnitude bounds proxy risk       |
| Liquid-oxide vs target scatter | Monotonic trends indicate proxy risk       |
| Feature correlation heatmap | Strong cross-corr with target supports proxy |
| Y-randomization             | `R2 <= 0` after shuffle (sanity)             |
| Dummy regressor             | Baseline to beat                             |
| Perfect-signal injection    | Unconstrained model should nail it; constrained one may not |

## Validation strategies in NB05

| Strategy       | Group column          | Intuition                                       |
|----------------|-----------------------|-------------------------------------------------|
| LOSO           | `Citation`            | Generalization across laboratories / studies    |
| Cluster-KFold  | `chemical_cluster`    | Generalization across composition regions       |
| Gridded-PT     | `pt_grid` (T x P bin) | Generalization across the P-T plane             |

All three use pooled out-of-fold RMSE as the primary metric; per-fold RMSE is
saved for distribution diagnostics.
