# Project layout

Single-page tree view of the opx ML thermobarometer repository, with a
one-line purpose per entry. For operating instructions, see
[README.md](README.md). For the active plan, see
[docs/master_plan.md](docs/master_plan.md).

```
Final Project/
├── README.md                           # install, run order, design decisions, TabPFN baseline
├── PROJECT_OVERVIEW.md                 # status log + mermaid data flow + module topology
├── PROJECT_LAYOUT.md                   # this file
├── CLAUDE.md                           # session guardrails for Claude Code
├── CLEANUP_AUDIT.md                    # 2026-04-19 cleanup audit (Stage A artifact)
├── config.py                           # single source of truth: paths, seeds, figure registry
├── run_all.py                          # papermill pipeline driver (nb01 → nbF)
├── requirements.txt                    # core dependencies
├── requirements-tabpfn.txt             # TabPFN v2 add-on
│
├── data/                               # read-only; CLAUDE.md hard rule
│   ├── raw/                            # ExPetDB, LEPR, external train sets
│   ├── processed/                      # cleaned per-track parquet files
│   ├── splits/                         # canonical test-index NPYs
│   ├── external/                       # Agreda, Jorgenson, Wang artifacts
│   └── natural/                        # GEOROC 2024-12 SGFTFN opx + cpx
│
├── src/
│   ├── __init__.py
│   ├── bias_correction.py              # Form A (per-regime OLS) + Form B (piecewise)
│   ├── calibration.py                  # split conformal
│   ├── cpx_features.py                 # cpx feature construction
│   ├── data.py                         # per-track loaders
│   ├── ensembles.py                    # greedy Caruana, two-level, AutoGluon wrappers
│   ├── evaluation.py                   # metrics, LOSO/Cluster/TargetBin CV
│   ├── external_models.py              # Putirka/Agreda/Jorgenson/Wang wrappers
│   ├── features.py                     # raw/alr/pwlr + engineered + EPMA augment
│   ├── geotherm.py                     # Hasterok & Chapman 2011
│   ├── io_utils.py                     # save_figure, save_table
│   ├── models.py                       # 8-model factory + predict_median/iqr
│   ├── optuna_search.py                # TPE + median pruning
│   ├── opx_tb_analysis.py              # BASE_ORDER + helpers (was v10_phase_c_analysis)
│   ├── plot_style.py                   # Okabe-Ito palette + save_figure wrapper
│   ├── prepare_train_test.py           # unified dispatcher for opx/cpx/twopx/universal
│   ├── resampling.py                   # P-T tempered resampling
│   ├── stacking.py                     # Ridge meta-model
│   ├── thermobar_adapter.py            # Thermobar 1.0.70 API bridge
│   ├── twopx_features.py               # two-pyroxene features
│   └── universal_features.py           # masking-architecture features
│
├── notebooks/
│   ├── nb01_data_cleaning.ipynb        # ExPetDB + LEPR cleaning for all tracks
│   ├── nb02_eda_pca.ipynb              # PCA + per-track k-means clusters
│   ├── nb03_opx_baseline_models.ipynb  # opx_only + opx_liq, 8 models, T01-T12
│   ├── nb03_cpx_baseline_models.ipynb  # cpx_only + cpx_liq, 8 models
│   ├── nb03_twopx_baseline_models.ipynb # twopx, 8 models
│   ├── nb03_universal_exploration.ipynb # universal masking (isolated side project)
│   ├── nb03_tabpfn_baseline.ipynb      # TabPFN v2 supplementary baseline driver
│   ├── nb04_putirka_benchmark.ipynb    # Putirka 2008 + Thermobar benchmarks + head-to-head
│   ├── nb04_v10_benchmark.ipynb        # v10 per-regime + cross-pipeline heatmaps
│   ├── nb05_loso_validation.ipynb      # LOSO + cluster + TargetBinKFold + LeaveOneRegion
│   ├── nb06_shap_analysis.ipynb        # tree-SHAP + linear-SHAP on stack
│   ├── nb07_bias_correction.ipynb      # composition-conditional T correction (Form A/B)
│   ├── nb07b_arcpl_bias_probe.ipynb    # ArcPL bias probe
│   ├── nb08_natural_twopx.ipynb        # GEOROC world maps + cross-mineral convergence
│   ├── nb09_manuscript_compilation.ipynb # per-paper table subsets (S8_* + table_4_*)
│   └── nbF_figures.ipynb               # canonical figure regen (CANONICAL_FIGURES 1-35)
│
├── scripts/
│   ├── opx_tb_nb03_tabpfn_baseline.py       # TabPFN multi-seed driver (was v10_nb03_*)
│   ├── opx_tb_nb03_fill_tabpfn_paragraph.py # manuscript paragraph auto-fill
│   ├── opx_tb_nb03_apply_part2_cells.py     # nbF/nb04/nb09 TabPFN cell applicator
│   ├── opx_tb_nb03_test_tabpfn_smoke.py     # smoke test for TabPFN wiring
│   ├── v10_phase_g_*.py                     # bias-correction mini-project drivers (Phase G)
│   ├── v10_phase_h*.py                      # Phase H natural-samples drivers
│   ├── v10_phase_c_driver.py / opx_runner.py # opx baseline runners
│   ├── v10_phase_d_cpx_*.py                 # cpx baseline drivers
│   ├── v10_phase_e_twopx_*.py               # twopx baseline drivers
│   ├── v10_phase_f_universal_*.py           # universal baseline drivers
│   ├── v10_pull_georoc_cpx.py               # GEOROC cpx download (pending GRO 503)
│   ├── v10_georoc_cpx_schema_diff.py        # cpx/opx header diff diagnostic
│   ├── v10_external_models_audit.py         # external-model training-data audit
│   ├── v10_test_protocol.py / v10_audit.py  # test orchestration / repo audit
│   └── v10_nb03_test_t{15,16_18}.py         # T15-T18 tests for Phase G + TabPFN
│
├── tests/                              # pytest: 55 tests
│   ├── test_bias_correction.py
│   ├── test_evaluation_regime.py
│   └── test_prepare_train_test_parity.py
│
├── results/                            # CSVs/JSONs consumed by notebooks
│   ├── v10_opx_*                       # opx multiseed, per-regime, generalization
│   ├── v10_cpx_* / v10_twopx_* / v10_universal_*
│   ├── v10_tabpfn_*                    # TabPFN multiseed, predictions, head-to-head
│   ├── v10_tabpfn_checkpoints/         # 40 per-seed pickled TabPFN fits
│   ├── v10_bias_correction/            # Phase G.7 per-seed checkpoints
│   ├── v10_optuna_studies/             # Optuna study joblibs
│   ├── v10_regime_*                    # pre-/post-correction scorecards
│   └── v10_natural_opx_*               # Phase H opx inference + uncertainty
│
├── figures/                            # PDF + PNG + TXT sidecar per entry
│   ├── fig24-fig35                     # canonical (registered in CANONICAL_FIGURES)
│   ├── fig_nb02_* / fig_nb03_* / fig_nb04_* # legacy registered figures
│   ├── fig_h5ac_opx_world_map.png      # Phase H.5a static world map
│   └── fig_h5b_opx_interactive.html    # Phase H.5b folium interactive
│
├── tables/
│   ├── S8_5_*, S8_6_*, S8_7_*, S8_8_*, S8_9_* # supplementary tables
│   ├── S8_10_*, S8_11_*                # bias-correction supplementary
│   ├── S8_12_tabpfn_benchmark.{csv,md,tex}   # TabPFN head-to-head
│   └── table_4_bias_correction_summary.*     # main-body table
│
├── manuscripts/
│   └── opx_2026/
│       ├── figures/                    # selected canonical figures
│       ├── tables/                     # selected canonical tables
│       ├── text/                       # paragraph auto-fills + draft
│       │   ├── bias_correction_autofilled.md
│       │   ├── bias_correction_numbers.md
│       │   ├── regime_results_autofilled.md
│       │   └── tabpfn_paragraph.md     # auto-filled via opx_tb_nb03_fill_tabpfn_paragraph
│       └── arxiv_submission/
│
├── models/                             # 1.5 GB; models/external/ is read-only
│   ├── canonical/                      # opx, cpx, twopx, universal winners
│   ├── ablation/                       # resampled ablation fits
│   └── external/                       # Agreda/Jorgenson/Wang ONNX + joblib + json
│
├── logs/                               # per-phase execution logs
│
├── docs/
│   ├── master_plan.md                  # unified v10+v11 plan (was v10_master_plan)
│   ├── cleanup_manifest.md             # cleanup file list
│   ├── figure_audit.md                 # per-figure spec + checklist
│   ├── cpx_pipeline_plan.md / twopx_pipeline_plan.md / universal_model_exploration.md
│   ├── ensemble_methods_plan.md / stacking_propagation_audit.md
│   ├── external_models_audit.md / markdown_template.md
│   ├── nb03_test_protocol.md           # T01-T18 tests
│   ├── nb03_tabpfn_plan.md             # TabPFN integration plan (was v10_nb03_tabpfn_plan)
│   ├── natural_worldwide_plan.md       # GEOROC re-integration plan
│   ├── notebooks_compatibility_audit.md
│   ├── optuna_strategy.md / resampling_strategy.md / stacking_strategy.md
│   ├── opx_paper_regime_additions_v1.md / codebase_consistency_audit_optionB.md
│   ├── preregistration/                # sealed: p_regime_preregistration.md
│   ├── archive_superseded/             # v9_* + optionB preflight + v10_implementation_plan
│   └── v9_* / putirka_inconsistency_audit / putirka_kd_filter_lookup
│
└── archive/
    ├── pre_consolidation_2026_04_18/   # 2026-04-19 cleanup archive
    │   ├── figures/                    # orphan v10_regime_* + fig_nb04_per_regime dups
    │   ├── notebooks/                  # nb03_baseline_models_pre_v10 (pre-track-split)
    │   ├── scripts/phase_legacy/       # audit_*, run_phase*
    │   └── logs/
    ├── pipeline_v1_legacy/             # very early notebooks
    ├── pre_v9_rebuild_2026_04_15/      # pre-v9 snapshot
    ├── pre_v10_rebuild_2026_04_16/     # pre-v10 snapshot
    ├── v5_reports/                     # historical audit reports
    └── v7_preparation_20260414_164844/ # v7 staging
```

## Data flow at a glance

```
raw/processed → nb01 → per-track parquet
per-track parquet + natural → nb02 clusters → nb03_{opx,cpx,twopx,universal}_baseline_models
                                            → nb03_tabpfn_baseline (supplementary)
8-model winners → nb04 (Putirka + ArcPL) → nb05 (LOSO/Cluster/TargetBin) →
nb06 (SHAP) → nb07 (bias correction) → nb08 (natural/world maps) →
nb09 (tables) + nbF (figures) → manuscripts/opx_2026/
```

## Conventions

- **Paths.** Import from `config.py` only. Never hardcode.
- **Seeds.** `SEED_SPLIT`, `SEED_MODEL`, `SEED_NOISE_AUG`, `SEED_KMEANS`,
  `SEED_BOOTSTRAP` all default to 42; multi-seed protocol uses
  `SPLIT_SEEDS = list(range(42, 62))` (20 seeds).
- **Figure registry.** `CANONICAL_FIGURES` in `config.py` is the source
  of truth; entries 1-35. Orphans live in `archive/`.
- **Symbol naming.** `BASE_ORDER` is canonical; `V10_BASE_ORDER` is a
  deprecated alias kept one release for backward compat.
- **Filename prefix.** New artifacts use `opx_tb_` (opx thermobarometer).
  Existing `v10_*` result-file names retained until a coordinated pass
  renames producers and consumers together.
