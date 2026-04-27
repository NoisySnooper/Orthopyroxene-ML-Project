# Project layout

This document is the canonical directory tree and module topology for the opx ML thermobarometer repository accompanying the JGR: Machine Learning and Computation submission. For operating instructions see [README.md](README.md). For data flow and module relationships see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). Tree depth capped at three levels for readability.

## Top level

```
Final Project/
├── README.md                            # install, run order, project description
├── PROJECT_OVERVIEW.md                  # data flow, module topology, locked constants
├── PROJECT_LAYOUT.md                    # this file
├── config.py                            # paths, seeds, pre-registered constants
├── run_all.py                           # papermill pipeline driver (nb01 to nbF)
├── requirements.txt                     # core dependencies
├── requirements-tabpfn.txt              # TabPFN v2 add-on (separate venv)
├── .gitignore
│
├── data/                                # read-only training and evaluation data
├── docs/                                # locked pre-registration only
├── notebooks/                           # papermill-driven analysis notebooks
├── src/                                 # library code
├── scripts/                             # pipeline scripts
├── tests/                               # reviewer-runnable test suite
├── models/                              # canonical model joblibs + external artifacts
├── results/                             # canonical CSVs and JSONs
├── figures/                             # core and SI figures
├── tables/                              # manuscript tables (Table 4 + S8 series + regime tables)
├── manuscripts/                         # JGR:MLC submission package
├── archive/                             # one retained pre-v10 snapshot
└── logs/                                # gitignored execution logs (.gitkeep tracked)
```

## Detailed structure

```
data/
├── raw/                                 # ExPetDB source XLSX, ArcPL, LEPR sources
├── processed/                           # cleaned per-track parquet (opx_liq, opx_only)
├── splits/                              # canonical citation-grouped test-index NPYs
├── external/                            # Agreda, Jorgenson, Wang model artifacts (comparison only)
├── natural/                             # ArcPL natural-sample inputs
├── optuna_studies/                      # Optuna hyperparameter-search SQLite DBs
└── hashes.json                          # SHA256 manifest cited in manuscript Section 7

docs/preregistration/
├── p_regime_preregistration.md          # locked 2026-04-17, regime bins + ship rule
└── nb03_test_protocol.md                # locked 2026-04-16, T1-T22 test specifications

src/
├── __init__.py
├── bias_correction.py                   # Form A regime-piecewise OLS, Form B sigmoid, ship_decision
├── calibration.py                       # split conformal prediction
├── data.py                              # per-track loaders (opx-liq, opx-only)
├── ensembles.py                         # Ridge stacking meta-learner
├── evaluation.py                        # bootstrap CIs, regime assignment, dual robustness check
├── external_models.py                   # Putirka 2008, Agreda, Jorgenson, Wang wrappers (comparators)
├── features.py                          # opx feature construction (raw, alr, pwlr)
├── geotherm.py                          # Fe-Mg equilibrium helpers
├── io_utils.py                          # CSV, JSON, joblib helpers
├── models.py                            # 8 tuned families (RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP)
├── opx_tb_analysis.py                   # opx benchmark utilities, prepare_train_test
├── plot_style.py                        # publication-style matplotlib defaults
├── prepare_train_test.py                # citation-grouped split builder
├── resampling.py                        # tempered class resampling
├── stacking.py                          # Ridge stacking implementation
├── thermobar_adapter.py                 # Thermobar 1.0.70 K to C contract
├── ablations/                           # ablation experiments (classical-feature ablation)
└── external/                            # external-model adapters (ArcPL helpers)

scripts/
├── ablations/                           # feature ablation drivers
├── audits/                              # repo audit utilities, robust_audit driver
├── benchmarks/                          # external-method benchmark runners
├── bias_correction/
│   └── rescore_under_v3_rule.py         # canonical scorecard regenerator
├── data_prep/                           # nb01 cleaning drivers
├── evaluation/                          # multi-seed eval, scorecard build
├── external_eval/                       # external-method evaluation on ArcPL and ExPetDB
├── figures/                             # core and SI figure builders
│   ├── _model_palette.py                # 9-family Okabe-Ito palette
│   ├── _style.py                        # shared figure style
│   ├── make_fig_dataset_map.py          # Core_01
│   ├── make_fig_dataset_map_holdout.py  # Core_01b
│   ├── make_fig_citation_split.py       # Core_02
│   ├── make_fig_methods_flowchart.py    # Core_03
│   ├── make_fig_opx_heatmap.py          # Core_04
│   ├── make_fig_bias_correction_opx.py  # Core_05/06 family
│   ├── make_fig_bias_residuals_opx.py   # Core_06
│   ├── make_fig_scorecard_delta_opx.py  # Core_07
│   ├── make_opx_headline_fig.py         # Core_08
│   ├── make_opx_only_P_headline_fig.py  # Core_08 supplement
│   ├── make_fig_opx_regime_families.py  # Core_09a
│   ├── make_fig_opx_overall_families.py # Core_09b
│   ├── make_fig_opx_four_combos.py      # Core_09 four-panel companion
│   ├── make_fig_best_vs_putirka.py      # Core_10
│   ├── make_fig_arcpl_bias_corrected_vs_putirka.py  # Core_10b
│   ├── make_fig_shipped_vs_putirka_expetdb.py       # Core_10c
│   ├── make_fig_shipped_vs_putirka_arcpl.py         # Core_10d
│   ├── make_fig_shap_winners.py         # Core_12
│   └── make_tabpfn_bias_scoreboard_fig.py            # SI scoreboard
├── interpretability/                    # SHAP, classical-equivalence regression, surrogate trees
├── manuscript/
│   └── build_manuscript_docx.py         # docx export from sections/*.md
├── pairing/
│   ├── _pairing_figure.py               # pairing matrix rendering helper
│   ├── compute_pairing_matrix.py        # LEPR pairing comparison computation
│   ├── make_fig_method_agreement_matrix.py  # Core_13a/b
│   └── make_fig_pairing_opx.py          # Core_14a opx-side pairing matrix
├── shap/                                # SHAP attribution drivers
└── tabpfn/                              # TabPFN inference and bias-correction drivers

notebooks/
├── nb01_data_cleaning.ipynb             # ExPetDB ingest, filtering, parquet writes
├── nb02_eda_pca.ipynb                   # exploratory PCA, regime-stratified summaries
├── nb03_opx_baseline_models.ipynb       # 8 tuned families, Optuna search, 20-seed refit
├── nb03_tabpfn_baseline.ipynb           # TabPFN v2 ninth-family evaluation
├── nb04_putirka_benchmark.ipynb         # Putirka 28a/29a/29c head-to-head
├── nb04_regime_benchmark.ipynb          # regime-stratified scorecard generation
├── nb04b_aug_test.ipynb                 # 15x Gaussian augmentation sensitivity
├── nb05_loso_validation.ipynb           # leave-one-citation-out validation
├── nb06_shap_analysis.ipynb             # SHAP attribution per cell winner
├── nb07_bias_correction.ipynb           # Form A and Form B fitting + ship decision
├── nb07b_arcpl_bias_probe.ipynb         # ArcPL natural-sample bias probe
├── nb09_manuscript_compilation.ipynb    # numeric autofill for manuscript prose
└── nbF_figures.ipynb                    # Core figure regeneration entrypoint

tests/
├── test_augmentation.py                 # nb04b augmentation sensitivity tests
├── test_bias_correction.py              # Form A and Form B unit tests
├── test_bootstrap_ci.py                 # bootstrap_rmse_ci verification
├── test_evaluation_regime.py            # assign_p_regime + regime-aware metrics
├── test_prepare_train_test_parity.py    # opx legacy-vs-unified prep parity
├── test_preregistration.py              # T1-T22 ship rule and constant assertions
└── test_tabpfn_registry.py              # TabPFN model build script verification

models/
├── canonical/opx/                       # 8 tuned families x 4 cells joblibs (canonical seed 42)
├── ablation/                            # ablation checkpoints (classical-feature subset)
└── external/                            # third-party model artifacts (Agreda, Wang)

results/
├── bias_correction_shipped.csv          # canonical v3 ship verdicts (4 opx cells)
├── bias_correction_per_seed.csv         # 20-seed per-cell ship decisions
├── preregistered_scorecard_postcorrection.csv  # canonical scorecard
├── opx_multiseed_results.csv            # 20-seed RMSE per family / feature_set / cell
├── opx_multiseed_summary.csv            # 20-seed mean and std summary
├── bootstrap_rmse_cis_all_cells.csv     # bootstrap CIs at canonical seed 42
├── tabpfn_*.csv                         # TabPFN evaluation outputs
├── pairing_matrix_22rows.csv            # LEPR pairing comparison data
├── opx_per_regime_*.csv                 # regime-stratified RMSE tables
└── ...

figures/
├── core/                                # 23 manuscript-bound figures + AUDIT.md
└── SI/                                  # supplementary figures

manuscripts/opx_2026/
├── text/draft/sections/                 # 00 abstract through 09 cover letter
├── figures/                             # publication-bound figure copies
├── tables/                              # publication-bound table copies
├── arxiv_submission/manuscript.docx     # built docx
└── SUBMISSION_CHECKLIST.md              # submission gate checklist

archive/
├── README.md
└── pre_v10_rebuild_2026_04_16/          # nb07 q_hat_P=10.0 re-derivation source

logs/
└── .gitkeep                             # gitignored content; placeholder so directory exists
```

## Trimmed and renamed during the pre-submission audit

The repository was trimmed to opx-only scope before submission. The cleanup audit removed clinopyroxene and two-pyroxene model artifacts, the parallel `manuscripts/cpx_2026/` companion-paper directory, the GEOROC natural-sample work and its 152k-sample inference outputs, all internal AI scaffolding files, all phase execution logs, and most planning and audit documents. Pre-registration amendments 1 and 2 were folded into the single locked `p_regime_preregistration.md`. Versioned scorecard CSVs (`_v2`, `_v3` suffixes) were consolidated into canonical names. Branch state at `submission-ready-2026-04-27` is the result.
