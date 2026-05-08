# Project overview

This is the orthopyroxene machine-learning thermobarometer pipeline that accompanies the JGR: Machine Learning and Computation submission. For install and run instructions, see [README.md](README.md). For the full directory tree, see [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md).

## What this repository contains

A reproducible pipeline that trains, evaluates, bias-corrects, and benchmarks an opx-only and opx-liq machine-learning thermobarometer against the classical Putirka (2008) opx calibrations. Eight tuned model families (RF, ERT, XGB, Histogram GB, CatBoost, LightGBM, ElasticNet, MLP) plus the pretrained TabPFN v2 foundation model are evaluated under an identical 20-seed citation-grouped cross-validation protocol on the ExPetDB experimental corpus. Per-regime bias correction (Form A regime-piecewise OLS; Form B Agreda-Lopez 2024 quantile-thresholded sigmoid) is fit on out-of-fold residuals and accepted under a pre-registered tolerance-band ship rule.

The natural-sample evaluation is the ArcPL Kd-equilibrated subset (n = 96 samples with literature P-T per sample). External-method comparators (Agreda-Lopez 2024, Jorgenson 2022, Wang 2021) are present in `src/external_models.py`. The cpx-side comparator code paths are retained because the LEPR pairing-matrix work uses them as classical-or-external baselines, even though no cpx model is shipped or claimed in the submission.

## Module topology

The pipeline has three layers and one configuration root.

**`config.py`** is the single source of truth for paths, seeds, the locked P-regime bins, ship-rule tolerances, the model-family roster, and the canonical figure manifest. Every notebook does `sys.path.insert(0, str(Path.cwd().parent))` then imports from `config`. No paths are hardcoded elsewhere.

**`src/`** holds 17 library modules (~3,800 LOC). Notebooks and `scripts/` both import from here. Public surface includes `data.py` (per-track loaders), `features.py` (raw / alr / pwlr feature construction), `models.py` (the 8 tuned families), `optuna_search.py` (hyperparameter sweep driver), `stacking.py` and `ensembles.py` (Ridge meta-learner), `evaluation.py` (bootstrap CIs, regime assignment, dual robustness check), `bias_correction.py` (Form A regime-piecewise OLS, Form B sigmoid, ship-decision logic), `external_models.py` (Putirka 2008, Agreda, Jorgenson, Wang wrappers), `thermobar_adapter.py` (Thermobar 1.0.70 K-to-C contract), and `plot_style.py` (publication style + winning-config loader).

**`notebooks/`** orchestrate. Thirteen notebooks (nb01 through nbF). Each one consumes parquet/CSV inputs, calls into `src/`, and writes CSV/JSON outputs to `results/` and joblibs to `models/canonical/`.

**`scripts/`** contain heavy-compute drivers and figure/manuscript builders that don't fit cleanly in a notebook: the augmentation ablation (`scripts/ablations/`, ~3.3h sweep), Optuna-driven scorecard regenerations, SHAP attribution, the publication figure builders (`scripts/figures/`, all writing to `figures/opx_only/`), and the docx assembler (`scripts/manuscript/build_manuscript_docx.py`).

## Data flow

```
data/raw/ExPetDB_*.xlsx                  data/external/{agreda_lopez_2024,
data/raw/external/LEPR_*.xlsx             jorgenson_2022,thermobar_examples}
    │                                       │
    ▼                                       ▼
nb01_data_cleaning  ─────────►  data/processed/{opx_liq,opx_only}.parquet
                                            │
                                            ▼
                              src/prepare_train_test.py
                              (citation-grouped 20-seed splits in data/splits/)
                                            │
                                            ▼
                              src/optuna_search.py  ─►  results/optuna_studies/opx/
                              src/models.py + src/stacking.py
                              nb03_opx_baseline_models  +  nb03_tabpfn_baseline (v2)
                                            │
                                            ▼
                              nb04_putirka_benchmark   nb04_regime_benchmark
                              nb04b_aug_test (reads scripts/ablations/ outputs)
                              nb05_loso_validation     nb06_shap_analysis
                              nb07_bias_correction     nb07b_arcpl_bias_probe
                                            │
                                            ▼
                              results/  (multiseed RMSE, scorecard, ship verdicts,
                                         pairing matrix, bootstrap CIs, SHAP NPZs)
                                            │
                                            ▼
                              scripts/figures/make_fig_*.py
                              scripts/pairing/make_fig_*.py
                              scripts/shap/run_shap_winners.py
                                            │
                                            ▼
                              figures/opx_only/main_fig_{1..14}.{pdf,png}
                              figures/opx_only/supp_fig_*.{pdf,png}
                              tables/table_4_*, S8_*, regime_*
                                            │
                                            ▼
                              nb09_manuscript_compilation (numeric autofill)
                              scripts/manuscript/build_manuscript_docx.py
                                            │
                                            ▼
                              manuscripts/opx_2026/arxiv_submission/manuscript.docx
```

## Pre-registered constants (locked 2026-04-17)

Locked in [docs/preregistration/p_regime_preregistration.md](docs/preregistration/p_regime_preregistration.md):

- Pressure-regime bin edges (kbar): `[0, 5, 15, 30, 100]`
- Sample-size floor: `N_MIN_FOR_VETO = 20`
- Ship-rule absolute tolerances: `T_ABS_T = 10.0` °C; `T_ABS_P = 1.0` kbar
- Ship-rule relative tolerance: `T_REL = 0.10`
- Numerical floor: `SHIP_TOL = 1e-6`

These constants are asserted by [tests/test_preregistration.py](tests/test_preregistration.py) (T15-T22) against the locked document and against the canonical scorecard `results/bias_correction_shipped.csv`. The full T1-T22 protocol is locked in [docs/preregistration/nb03_test_protocol.md](docs/preregistration/nb03_test_protocol.md) (locked 2026-04-16).

## Key results at a glance

- **Headline.** Opx-only P bias-corrected (Random Forest with pairwise log-ratio features + Form A correction) reduces aggregate RMSE from 10.35 to 6.05 kbar (41.9 % reduction), unanimous across 20 seeds, beating Putirka (2008) equation 29c by 54.7 % and winning every pre-registered pressure regime.
- **Bias-correction acceptance.** Three of four opx cells accept Form A under the locked tolerance-band rule (opx-liq P, opx-only T, opx-only P). Opx-liq T accepts Form B at canonical seed only and fails the 20-seed majority vote; reported as a null against filtered Putirka 28a.
- **Foundation-model verdict.** TabPFN v2 (with leave-one-citation-out pseudo-OOF residuals) wins the post-correction scorecard for opx-only P at 5.94 kbar vs tuned RF/pwlr-corrected at 6.05 kbar by a margin within per-seed spread; loses on the other three opx cells.
- **Honest nulls.** Opx-liq T is effectively a null against filtered Putirka 28a (77.06 vs 71.67 °C aggregate). Form B does not transfer to opx data and is not rescued by the 15x Gaussian augmentation protocol that enables it on Agreda-Lopez et al. (2024) cpx data. The opx-liq deeper-mantle regime has only n = 8 samples in the test split, below the n >= 20 honesty bar.

## Reproducibility

All input data files have SHA256 hashes recorded in [data/hashes.json](data/hashes.json) and cited in manuscript Section 7. The reviewer-runnable test suite at [tests/](tests/) verifies pre-registration adherence and canonical-CSV integrity. The notebook execution order is described in [README.md](README.md); end-to-end wall clock is approximately two CPU days on a workstation-class machine without GPU.

## Repository scope

The repository ships an opx-only pipeline. Earlier development states included clinopyroxene and two-pyroxene tracks alongside opx; those tracks were trimmed from the final submission package because the JGR:MLC manuscript scope is opx-specific. External-method wrappers in `src/external_models.py` retain the cpx-side comparators (Agreda-Lopez, Jorgenson, Wang, Putirka cpx-liq and cpx-only) because the LEPR-corpus pairing matrix uses them as classical-or-external baselines. Some cpx and two-pyroxene data products (`data/processed/cpx_*.parquet`, `data/processed/twopx_clean.parquet`, `data/splits/{train,test}_indices_{cpx,twopx,universal}.npy`, `models/canonical/universal/`, `results/v10_universal_*.csv`) remain in-tree to keep those comparator paths functional. No clinopyroxene or two-pyroxene model is shipped, evaluated, or claimed in this submission.
