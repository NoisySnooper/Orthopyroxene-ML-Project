# Open Research, Data Availability, and Code Availability

## Data Availability

All training and test data used in this study derive from the Experimental Petrology Database (ExPetDB) snapshot dated 2025-07-21. The SHA256 hashes of the preprocessed parquet files and of the source XLSX export are committed to the repository at `data/hashes.json` (hash file dated 2026-04-16). The first eight hexadecimal characters of each hash are reproduced below for quick reviewer verification:

| File | SHA256 prefix (first 8 characters) |
|---|---|
| `data/splits/opx_liq.parquet` | `0c15cc4f` |
| `data/splits/opx_only.parquet` | `c38fa83e` |
| `data/raw/ExPetDB.xlsx` | `8aebe073` |

The ExPetDB source data are distributed by Hirschmann et al. (2008) and subsequent database curators under open licensing terms; our snapshot is archived with the manuscript's Zenodo record at DOI [to be assigned at acceptance] to ensure exact-byte reproducibility against the hashes above. The LEPR natural-sample compilation used for out-of-training-domain validation (reported in the companion work) has SHA256 prefix `c96ffea4` and is similarly archived.

Users who wish to reproduce our results from scratch may regenerate the preprocessed splits by running `scripts/preprocessing/build_splits.py` at the tagged commit `opx-submission-YYYYMMDD` with the archived ExPetDB source as input; the script verifies the resulting hashes against `data/hashes.json` before proceeding. Direct downloads of the preprocessed splits are available from the Zenodo record for users who prefer to skip the preprocessing step.

## Code Availability

All analysis code, pre-registration documents, 20-seed per-seed results CSVs, trained model joblibs, SHAP value files, and figure-generation scripts are committed to the main branch at the tagged commit `opx-submission-YYYYMMDD` and archived at Zenodo DOI [to be assigned at acceptance]. The repository structure follows a conventional layout:

- `src/`: library code, including `bias_correction.py::ship_decision` (the pre-registered acceptance-rule implementation) and `evaluation.py::assign_p_regime` (the regime-bin assignment helper)
- `scripts/`: runnable pipeline scripts including preprocessing, Optuna tuning, 20-seed training, bias-correction rescoring, and figure generation
- `notebooks/`: per-pipeline analysis notebooks (nb01 through nb12)
- `docs/preregistration/`: pre-registration documents, committed with lock dates visible in git history
- `results/`: per-seed result CSVs, aggregated summaries, and scorecard files
- `figures/opx_only/`: the 15 main figures and 14 supplementary figures referenced in the manuscript (PDF + PNG + caption TXT each)
- `manuscripts/opx_2026/`: manuscript source files (Markdown), tables (CSV + LaTeX), and supplementary material
- `tests/`: pre-registration test assertions including T15 (headline claim pass condition) and T19-T22 (bias-correction rule constants)

Key software dependencies and versions are pinned in `requirements.txt` (main Python environment) and `requirements-tabpfn.txt` (TabPFN v2 environment, kept separate to avoid scikit-learn 1.8.0 dependency conflicts). Python 3.13.13, scikit-learn 1.8.0, pandas 3.0.2, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10, SHAP 0.51.0, Thermobar 1.0.70, and TabPFN >= 2.0 < 2.5. Optuna study databases (Bayesian hyperparameter search state) are archived in `data/optuna_studies/` for full reproducibility of the tuning results.

## Pre-registration and Reviewer Materials

The pre-registration documents governing this study are committed to the main branch with visible lock dates:

- `docs/preregistration/p_regime_preregistration.md` (locked 2026-04-17, before any bias-correction fitting) fixes the four-bin pressure-regime partition, the dual robustness check, and the n >= 20 honesty bar.
- `docs/preregistration/nb03_test_protocol.md` (locked 2026-04-16) fixes the 20-seed evaluation protocol, the Optuna freeze-then-refit discipline, the ship-if-better rule with tolerance-band constants, and the 22-test pre-registered empirical-test suite (T01 through T22). Test T15 is the pre-registered pass condition for the opx-liq P headline claim; it passes (`results/opx_per_regime_claims_audit_robust.csv`).

Reviewers are invited to clone the repository at the submission-tagged commit and run `python -m pytest tests/test_preregistration.py` to verify that the tolerance-band acceptance-rule constants (T_ABS_T = 10.0 C, T_ABS_P = 1.0 kbar, T_REL = 0.10, N_MIN_FOR_VETO = 20, SHIP_TOL = 1e-6) are consistent across `src/bias_correction.py`, `scripts/bias_correction/rescore.py`, and the pre-registration documents. Any silent drift in these constants between the pre-registration date and the submission date would cause one or more of T19 through T22 to fail.
