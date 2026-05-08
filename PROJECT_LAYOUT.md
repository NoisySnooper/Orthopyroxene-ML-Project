# Project layout

This document is the canonical directory tree for the opx ML thermobarometer repository accompanying the JGR: Machine Learning and Computation submission. For run instructions see [README.md](README.md). For data flow and module relationships see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Top level

```
Final Project/
├── README.md                            # install, run order, project description
├── PROJECT_OVERVIEW.md                  # data flow, module topology, locked constants
├── PROJECT_LAYOUT.md                    # this file
├── config.py                            # paths, seeds, pre-registered constants, figure roster
├── run_all.py                           # papermill driver (currently stale; see README)
├── requirements.txt                     # core dependencies
├── requirements-tabpfn.txt              # TabPFN v2 add-on (separate venv)
├── .gitignore
│
├── data/                                # read-only training and evaluation data
├── docs/                                # locked pre-registration documents
├── notebooks/                           # 13 analysis notebooks
├── src/                                 # library code (notebooks and scripts both import here)
├── scripts/                             # heavy-compute drivers and figure/manuscript builders
├── tests/                               # reviewer-runnable test suite (T1-T22)
├── models/                              # canonical model joblibs + external artifacts
├── results/                             # CSVs, JSONs, NPZs, Optuna SQLite, checkpoints
├── figures/                             # publication figures (opx_only/) and pre-rename archive (legacy/)
├── tables/                              # manuscript tables (Table 4 + S8 series + per-regime tables)
├── manuscripts/                         # JGR:MLC submission package
├── archive/                             # one retained pre-v10 snapshot
└── logs/                                # gitignored execution logs (.gitkeep tracked)
```

## Detailed structure

```
data/
├── raw/
│   ├── ExPetDB_download_ExPetDB-2025-07-21.xlsx   # primary training corpus
│   └── external/                        # LEPR XLSX + external reference data
├── processed/                           # cleaned per-track parquet (and CSVs)
│   ├── opx_clean_opx_liq.parquet        # opx-liq training matrix (canonical)
│   ├── opx_clean_opx_only.parquet       # opx-only training matrix (canonical)
│   ├── opx_clean_core.parquet           # superset, pre-track split
│   ├── opx_clean_core_with_clusters.parquet
│   ├── opx_clean_full.parquet
│   ├── cpx_clean_cpx_liq.parquet        # retained for cpx-side comparator wiring
│   ├── cpx_clean_cpx_only.parquet       # retained for cpx-side comparator wiring
│   ├── twopx_clean.parquet              # retained for LEPR pairing matrix
│   └── universal_clean.parquet          # retained for v10_universal_* sensitivity outputs
├── splits/                              # citation-grouped train/test index NPYs
│   ├── {train,test}_indices_opx_liq.npy
│   ├── {train,test}_indices_opx_only.npy
│   ├── {train,test}_indices_opx.npy
│   ├── {train,test}_indices_cpx_liq.npy        # comparator wiring
│   ├── {train,test}_indices_cpx_only.npy       # comparator wiring
│   ├── {train,test}_indices_twopx.npy          # LEPR pairing
│   └── {train,test}_indices_universal.npy      # universal sensitivity track
├── external/                            # vendored upstream repos and artifacts
│   ├── agreda_lopez_2024/repo/          # Agreda-Lopez 2024 ML thermobarometer
│   ├── jorgenson_2022/                  # Jorgenson 2022 model
│   └── thermobar_examples/              # Thermobar 1.0.70 reference package
├── natural/                             # ArcPL natural-sample inputs + GEOROC dumps
│   ├── natural_opx_cleaned.csv          # ArcPL processed opx
│   ├── natural_opx_with_coords.csv
│   ├── 2024-12-GEOROC_CLINOPYROXENES.csv         # gitignored (regenerable)
│   ├── 2024-12-SGFTFN_ORTHOPYROXENES.csv         # gitignored
│   ├── georoc_cache/                    # gitignored
│   └── natural_sample_prep_script.py
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
├── external_models.py                   # Putirka 2008, Agreda, Jorgenson, Wang wrappers
├── features.py                          # opx feature construction (raw, alr, pwlr)
├── geotherm.py                          # Fe-Mg equilibrium and geotherm helpers
├── io_utils.py                          # CSV, JSON, joblib helpers
├── models.py                            # 8 tuned families (RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP)
├── optuna_search.py                     # Optuna study driver used by nb03
├── opx_tb_analysis.py                   # opx benchmark utilities, prepare_train_test
├── plot_style.py                        # publication-style matplotlib defaults
├── prepare_train_test.py                # citation-grouped split builder
├── resampling.py                        # tempered class resampling
├── stacking.py                          # Ridge stacking implementation
├── thermobar_adapter.py                 # Thermobar 1.0.70 K-to-C contract
├── ablations/
│   ├── __init__.py
│   └── augmentation.py                  # 15x Gaussian augmentation utilities
└── external/
    ├── __init__.py
    └── arcpl_opx.py                     # ArcPL helper (natural-sample evaluation)

scripts/
├── ablations/                           # augmentation ablation drivers (nb04b)
│   ├── run_augmentation_ablation_opx.py         # 1920-fit sweep (~3.3h)
│   ├── run_augmentation_oof_bias.py             # OOF residuals + bias-correction fits
│   ├── make_augmentation_figures.py
│   └── make_augmentation_report.py
├── audits/
│   └── terminology_cleanup_v3.py        # one-shot terminology cleanup (final pass)
├── benchmarks/                          # (empty)
├── bias_correction/
│   └── rescore_under_v3_rule.py         # canonical scorecard regenerator
├── data_prep/
│   └── build_per_family_winners.py      # per-family winners JSON
├── evaluation/
│   ├── build_S15_single_family_sensitivity.py
│   ├── build_T2_model_roster.py
│   ├── compute_bootstrap_cis_all_cells.py
│   └── diagnostic_seed_variance_check.py
├── external_eval/
│   └── eval_arcpl_opx_corrected.py      # ArcPL evaluation under shipped corrections
├── figures/                             # production figure builders (write to figures/opx_only/)
│   ├── _model_palette.py                # 9-family Okabe-Ito palette
│   ├── _style.py                        # shared figure style
│   ├── make_fig_dataset_map.py
│   ├── make_fig_dataset_map_holdout.py
│   ├── make_fig_citation_split.py
│   ├── make_fig_methods_flowchart.py
│   ├── make_fig_opx_heatmap.py
│   ├── make_fig_bias_correction_opx.py
│   ├── make_fig_bias_residuals_opx.py
│   ├── make_fig_scorecard_delta_opx.py
│   ├── make_opx_headline_fig.py
│   ├── make_opx_only_P_headline_fig.py
│   ├── make_fig_opx_regime_families.py
│   ├── make_fig_opx_overall_families.py
│   ├── make_fig_opx_four_combos.py
│   ├── make_fig_best_vs_putirka.py
│   ├── make_fig_arcpl_bias_corrected_vs_putirka.py
│   ├── make_fig_shipped_vs_putirka_expetdb.py
│   ├── make_fig_shipped_vs_putirka_arcpl.py
│   ├── make_fig_shap_winners.py
│   └── make_tabpfn_bias_scoreboard_fig.py
├── interpretability/                    # SHAP, classical-equivalence regression, surrogate trees
│   ├── classical_equivalence_regression.py
│   ├── classical_feature_ablation.py
│   ├── compute_feature_concordance.py
│   ├── gasparik_agreement.py
│   ├── make_fig_feature_concordance.py
│   ├── partial_dependence_across_families.py
│   ├── physics_consistency_checks.py
│   └── surrogate_decision_trees.py
├── manuscript/
│   ├── __init__.py
│   ├── build_manuscript_docx.py         # assembles manuscripts/opx_2026/arxiv_submission/manuscript.docx
│   ├── figure_inventory.txt             # main_fig_* / supp_fig_* roster
│   ├── generate_table_2.py              # Table 2 per-cell winners
│   └── rename_figures_phase13.py        # one-shot Core_NN -> main_fig/supp_fig migration
├── pairing/
│   ├── _pairing_figure.py               # rendering helper
│   ├── compute_pairing_matrix.py        # LEPR pairing computation
│   ├── make_fig_method_agreement_matrix.py
│   └── make_fig_pairing_opx.py
├── shap/
│   └── run_shap_winners.py
└── tabpfn/                              # TabPFN inference and bias-correction drivers
    ├── opx_tb_nb03_tabpfn_baseline.py
    ├── opx_tb_nb03_tabpfn_oof.py
    ├── opx_tb_nb03_tabpfn_bias_correction.py
    ├── opx_tb_nb03_tabpfn_postcorrection_extend.py
    ├── opx_tb_nb03_merge_tabpfn_canonical.py
    ├── opx_tb_nb03_augment_scorecard.py
    ├── opx_tb_nb03_apply_part2_cells.py
    ├── opx_tb_nb03_fill_tabpfn_paragraph.py
    ├── opx_tb_nb03_test_tabpfn_smoke.py
    ├── _c11_integrity_sweep.py
    └── _patch_nb04_fig24.py

notebooks/
├── nb01_data_cleaning.ipynb             # ExPetDB ingest, filtering, parquet writes
├── nb02_eda_pca.ipynb                   # exploratory PCA, regime-stratified summaries
├── nb03_opx_baseline_models.ipynb       # 8 tuned families, Optuna search, 20-seed refit
├── nb03_tabpfn_baseline.ipynb           # TabPFN v2 ninth-family evaluation (shells to .venv-tabpfn)
├── nb04_putirka_benchmark.ipynb         # Putirka 28a/29a/29c head-to-head
├── nb04_regime_benchmark.ipynb          # regime-stratified scorecard generation
├── nb04b_aug_test.ipynb                 # 15x Gaussian augmentation sensitivity (reads ablation CSVs)
├── nb05_loso_validation.ipynb           # leave-one-citation-out validation
├── nb06_shap_analysis.ipynb             # SHAP attribution per cell winner
├── nb07_bias_correction.ipynb           # Form A and Form B fitting + ship decision
├── nb07b_arcpl_bias_probe.ipynb         # ArcPL natural-sample bias probe
├── nb09_manuscript_compilation.ipynb    # numeric autofill for manuscript prose
└── nbF_figures.ipynb                    # legacy figure entrypoint (superseded by scripts/figures/)

tests/
├── test_augmentation.py                 # nb04b augmentation sensitivity tests
├── test_bias_correction.py              # Form A and Form B unit tests
├── test_bootstrap_ci.py                 # bootstrap_rmse_ci verification
├── test_evaluation_regime.py            # assign_p_regime + regime-aware metrics
├── test_prepare_train_test_parity.py    # opx legacy-vs-unified prep parity
├── test_preregistration.py              # T1-T22 ship rule and constant assertions
└── test_tabpfn_registry.py              # TabPFN model build script verification

models/
├── canonical/
│   ├── opx/                             # 8 tuned families x 4 cells joblibs (canonical seed 42)
│   └── universal/                       # universal-track sensitivity models (retained)
├── ablation/                            # ablation checkpoints (classical-feature subset)
└── external/                            # third-party model artifacts (Agreda, Wang)

results/
├── bias_correction_shipped.csv          # canonical v3 ship verdicts (4 opx cells)
├── bias_correction_per_seed.csv         # 20-seed per-cell ship decisions
├── bias_correction_summary.csv
├── bias_correction_edge_sensitivity.csv
├── bias_correction_form_b_stability.csv
├── bias_correction/checkpoints/         # per-seed bias-correction checkpoints
├── preregistered_scorecard_postcorrection.csv  # canonical scorecard
├── opx_multiseed_results.csv            # 20-seed RMSE per family / feature_set / cell
├── opx_multiseed_summary.csv
├── opx_per_cell_results.csv
├── opx_per_regime_*.csv                 # regime-stratified RMSE tables
├── opx_liq_*.csv                        # opx-liq specific outputs (residuals, generalization, SHAP)
├── opx_liq_shap_values.npz
├── opx_ensemble_results.csv
├── bootstrap_rmse_cis_all_cells.csv     # bootstrap CIs at canonical seed 42
├── tabpfn_*.csv                         # TabPFN multiseed, head-to-head, bias correction
├── tabpfn_oof_predictions.csv
├── tabpfn_predictions.csv
├── tabpfn_checkpoints/                  # TabPFN per-seed model checkpoints
├── pairing_matrix_22rows.csv            # LEPR pairing comparison
├── augmentation_ablation_opx_*.{csv,pkl}        # nb04b 15x augmentation outputs
├── classical_equivalence_regression.csv
├── classical_feature_ablation.csv
├── feature_concordance_*.csv
├── partial_dependence_across_families.parquet
├── physics_consistency_checks.csv
├── surrogate_tree_r_squared.csv
├── shap_importance_winners.csv
├── shap_values_winners.npz
├── core11_extended_predictions.csv
├── canonical_cells_h0b.{csv,json}       # H0b regime claim canonical cells
├── arcpl_opx_corrected_per_regime.{csv,json}
├── natural_opx_*.csv                    # ArcPL natural-sample inference outputs
├── nb07_conformal_qhat.json
├── nb03_per_family_winners.json         # canonical per-family roster
├── optuna_studies/                      # Optuna SQLite DBs
│   └── {cpx,opx,twopx,universal}/
├── universal/                           # universal-track sensitivity outputs
├── v10_universal_multiseed_*.csv        # universal-track multiseed sensitivity
├── regime_allmodels.csv
├── regime_allmodels_postcorrection.csv
├── regime_benchmark.csv
├── external_benchmark.csv
├── gasparik_agreement.csv
├── phase_h2_locality_coverage.csv
├── phase_h3a_regime_summary.csv
└── nb08_natural_predictions.csv

figures/
├── opx_only/                            # current publication figures (post phase-13 rename)
│   ├── main_fig_1..14.{pdf,png,txt}     # 14 main figures (8 split into 8a/8b -> 15 panels)
│   └── supp_fig_*.{pdf,png,txt}         # 28 supplementary figures
└── legacy/                              # pre-rename Core_01..Core_18 PDFs/PNGs (kept for reference)
    └── AUDIT.md

tables/
├── table_4_bias_correction_summary.{csv,md,tex}     # main-text Table 4
├── table_2_per_cell_winners.csv                     # main-text Table 2
├── S8_5_1..S8_5_4_regime_*_opx_liq.{csv,md}         # opx-liq per-regime supp tables
├── S8_6_generalization_opx_liq.{csv,md}
├── S8_7_shap_top_features_opx_liq.{csv,md}
├── S8_8_bias_correction_opx_liq.{csv,md}
├── S8_10_bias_correction_regimewise.{csv,md}
├── regime_{all,combined,opx,cpx}_{T,P}_{T_C,P_kbar}.{csv,md}   # cross-tabbed regime stats
└── _archived/                                       # superseded supp tables (S8_9, S8_11, S8_12)

manuscripts/opx_2026/
├── text/draft/sections/                 # 00_abstract, 00a_front_matter, 01..09 sections
├── figures/                             # publication-bound figure copies
├── tables/                              # publication-bound table copies
├── arxiv_submission/manuscript.docx     # built docx (output of build_manuscript_docx.py)
└── SUBMISSION_CHECKLIST.md

archive/
├── README.md
└── pre_v10_rebuild_2026_04_16/          # nb07 q_hat_P=10.0 re-derivation source

logs/
└── .gitkeep                             # gitignored content; placeholder so directory exists
```

## Notes on current state

**Figure naming.** Phase-13 renamed all production figures from `Core_NN_*` to `main_fig_N` / `supp_fig_N`. The new names are in `figures/opx_only/` and are the ones referenced by `manuscripts/opx_2026/arxiv_submission/manuscript.docx`. The original `Core_NN_*` PDFs and PNGs are preserved in `figures/legacy/` for reviewer back-reference. Source builders in `scripts/figures/` retain their `make_fig_*.py` names; the rename happens at output time.

**Repository scope vs retained data.** The submission claims and figures are opx-only, but `data/processed/`, `data/splits/`, `models/canonical/universal/`, and `results/` retain cpx, two-pyroxene, and "universal" track artifacts. These are kept because the cpx-side external comparators (Putirka 2008 cpx-liq and cpx-only, Agreda-Lopez 2024, Jorgenson 2022, Wang 2021) need cpx test data for the LEPR pairing matrix, and the universal track is referenced by the `v10_universal_*` sensitivity outputs cited in supplementary text.

**`run_all.py` is stale.** It references `nb03_cpx_baseline_models`, `nb03_twopx_baseline_models`, and `nb08_natural_twopx`, which no longer exist; it omits `nb04_regime_benchmark` and `nb07b_arcpl_bias_probe`, which do. See README for the working notebook order.

**Pre-submission audit history.** The repository was trimmed to opx-only scope before submission. The cleanup audit removed clinopyroxene and two-pyroxene model artifacts from the *manuscript*, the parallel `manuscripts/cpx_2026/` companion-paper directory, the GEOROC natural-sample inference workflow, all internal AI scaffolding files, and most planning documents. Pre-registration amendments 1 and 2 were folded into the single locked `p_regime_preregistration.md`. Versioned scorecard CSVs (`_v2`, `_v3` suffixes) were consolidated into canonical names. Branch state at `submission-ready-2026-04-27` is the result.
