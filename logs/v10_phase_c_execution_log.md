# Phase C Execution Log

**Date:** 2026-04-16 to 2026-04-17
**Status:** COMPLETE
**Scope:** opx Optuna TPE search (96 studies) + refit/ensemble runner (12 cells)

---

## 1. Grid

- 8 models: RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP
- 2 targets: T_C, P_kbar
- 2 tracks: opx_only, opx_liq
- 3 feature sets: raw, alr, pwlr
- Total: **96 studies** x 50 trials each

## 2. Execution

- `scripts/v10_phase_c_opx_driver.py` (Optuna TPE + GroupKFold(3) on Citation)
- `scripts/v10_phase_c_opx_runner.py` (refit all bases + build OOF + 3 ensembles + evaluate test)
- Inner CV parallelism: `OPTUNA_N_JOBS_INNER` started at 4, escalated to 5, then 10, then 12

## 3. Failures and fixes

- **MLP `n_layers` ValueError (all cells)**: Optuna trial params `n_layers`/`layer_size` are synthetic keys, not valid sklearn keys. Fixed in `src/models.py::build_model` by popping the two keys and computing `hidden_layer_sizes` tuple before sklearn construction.
- **Phase C runner relaunched** after fix. All 12 cells completed cleanly.

## 4. Best per cell

| Target | Track | Winning base | Feature set | Test RMSE |
|---|---|---|---|---|
| T_C | opx_only | LightGBM | alr | 147.98 |
| T_C | opx_liq  | ElasticNet | raw | 77.06 |
| P_kbar | opx_only | XGB | pwlr | 10.20 |
| P_kbar | opx_liq  | MLP | raw | 3.82 |

## 5. Artifacts

- `results/v10_optuna_studies/opx/*.joblib` (96 studies)
- `results/v10_optuna_best_params_opx.json` + partial
- `results/v10_opx_per_cell_results.csv` (96 base rows)
- `results/v10_opx_ensemble_results.csv` (36 ensemble rows: ridge / two_level / greedy)
- `models/canonical/opx/base_*.joblib` + `ens_*.joblib`
- `logs/v10_phase_c_opx_driver.log`, `logs/v10_phase_c_opx_runner.log`

## 6. Commits

- `3c62901` Phase C: opx 96 Optuna studies + runner complete
- `fd18274` v10 Phase C analysis module + build_model Pipeline-param fix
- `9384a96` v10 Phase C opx runner + thin nb03 orchestrator

## 7. Tests logged

T01 (boosted primary), T02 (ensemble beats base), T11 (CatLGB beats XGB), T12 (MLP/ENet beats trees). Rows in `results/v10_nb03_test_log.csv`.
