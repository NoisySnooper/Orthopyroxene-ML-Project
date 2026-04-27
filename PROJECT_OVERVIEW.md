# Project overview

This is the orthopyroxene machine-learning thermobarometer pipeline that accompanies the JGR: Machine Learning and Computation submission. For install and run instructions, see [README.md](README.md). For the full directory tree and file-level inventory, see [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md).

## What this repository contains

A reproducible pipeline that trains, evaluates, bias-corrects, and benchmarks an opx-only and opx-liq machine-learning thermobarometer against the classical Putirka (2008) opx calibrations. Eight tuned model families (RF, ERT, XGB, Histogram GB, CatBoost, LightGBM, ElasticNet, MLP) plus the pretrained TabPFN v2 foundation model are evaluated under an identical 20-seed citation-grouped cross-validation protocol on the ExPetDB experimental corpus. Per-regime bias correction (Form A regime-piecewise OLS; Form B Ágreda-López 2024 quantile-thresholded sigmoid) is fit on out-of-fold residuals and accepted under a pre-registered tolerance-band ship rule.

The natural-sample evaluation is the ArcPL Kd-equilibrated subset (n = 96 samples with literature P-T per sample). External-method comparators (Ágreda-López 2024, Jorgenson 2022, Wang 2021) are present in `src/external_models.py` as classical and ML baselines for the LEPR-corpus comparison work archived alongside the manuscript.

## Data flow

```
data/raw/                              data/external/
    │                                       │
    ▼                                       ▼
nb01_data_cleaning  ─────────►  data/processed/{opx_liq,opx_only}.parquet
                                            │
                                            ▼
                              src/prepare_train_test.py
                              (citation-grouped 20-seed splits in data/splits/)
                                            │
                                            ▼
                              src/models.py  +  nb03_tabpfn_baseline (TabPFN v2)
                              src/optuna_search.py  +  src/stacking.py
                                            │
                                            ▼
                              nb04_putirka_benchmark   nb04_regime_benchmark
                              nb05_loso_validation     nb04b_aug_test
                              nb06_shap_analysis       nb07_bias_correction
                              nb07b_arcpl_bias_probe
                                            │
                                            ▼
                              results/  (multiseed RMSE, scorecard, ship verdicts,
                                         pairing matrix, bootstrap CIs)
                                            │
                                            ▼
                              nbF_figures   scripts/figures/*
                                            │
                                            ▼
                              figures/core/Core_*.{pdf,png}   tables/*.{csv,md,tex}
                                            │
                                            ▼
                              nb09_manuscript_compilation
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

These constants are asserted by [tests/test_preregistration.py](tests/test_preregistration.py) (T15-T22) against the locked document and against the canonical scorecard `results/bias_correction_shipped.csv`.

## Key results at a glance

- **Headline.** Opx-only P bias-corrected (Random Forest with pairwise log-ratio features + Form A correction) reduces aggregate RMSE from 10.35 to 6.05 kbar (41.9 % reduction), unanimous across 20 seeds, beating Putirka (2008) equation 29c by 54.7 % and winning every pre-registered pressure regime.
- **Bias-correction acceptance.** Three of four opx cells accept Form A under the locked tolerance-band rule (opx-liq P, opx-only T, opx-only P). Opx-liq T accepts Form B at canonical seed only and fails the 20-seed majority vote; reported as a null against filtered Putirka 28a.
- **Foundation-model verdict.** TabPFN v2 (with leave-one-citation-out pseudo-OOF residuals) wins the post-correction scorecard for opx-only P at 5.94 kbar vs tuned RF/pwlr-corrected at 6.05 kbar by a margin within per-seed spread; loses on the other three opx cells.
- **Honest nulls.** Opx-liq T is effectively a null against filtered Putirka 28a (77.06 vs 71.67 °C aggregate). Form B does not transfer to opx data and is not rescued by the 15× Gaussian augmentation protocol that enables it on Ágreda-López et al. (2024) cpx data. The opx-liq deeper-mantle regime has only n = 8 samples in the test split, below the n ≥ 20 honesty bar.

## Reproducibility

All input data files have SHA256 hashes recorded in [data/hashes.json](data/hashes.json) and cited in manuscript Section 7. The reviewer-runnable test suite at [tests/](tests/) verifies pre-registration adherence and canonical-CSV integrity. The papermill driver [run_all.py](run_all.py) executes the thirteen notebooks in dependency order; total wall-clock cost is approximately two CPU days on a workstation-class machine without GPU.

## Repository scope

The repository ships an opx-only pipeline. Earlier development states included clinopyroxene and two-pyroxene tracks alongside opx; those tracks were trimmed from the final submission package because the JGR:MLC manuscript scope is opx-specific. External-method wrappers in `src/external_models.py` retain the cpx-side comparators (Ágreda-López, Jorgenson, Wang, Putirka cpx-liq and cpx-only) because the LEPR-corpus pairing matrix uses them as classical-or-external baselines. No clinopyroxene or two-pyroxene model is shipped, evaluated, or claimed in this submission.
