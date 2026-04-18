# Phase E Execution Log

**Date:** 2026-04-17
**Status:** COMPLETE
**Scope:** twopx (paired opx+cpx) Optuna TPE + refit/ensemble runner

---

## 1. Grid

- 8 models x 2 targets x 1 track (twopx) x 4 feature sets (raw, alr, pwlr, twopx_components) = **64 studies**

## 2. Execution

- `scripts/v10_phase_e_twopx_driver.py` + `scripts/v10_phase_e_twopx_runner.py`
- Ran in parallel with Phase D driver (5 cores each initially, escalated later).
- Completed early; compute reallocated to Phase D runner.

## 3. Best per cell

| Target | Track | Winning base | Feature set | Test RMSE |
|---|---|---|---|---|
| T_C | twopx | ElasticNet | raw | 79.54 |
| P_kbar | twopx | XGB | alr | 4.26 |

## 4. Notes

- Twopx paired data produces the lowest P_kbar RMSE of any opx/cpx track (4.26 kbar) — as expected from the information gain of co-equilibrated phases.
- Consistent with Phase F's universal result where `twopx_liq` is the best-performing phase scope.

## 5. Artifacts

- `results/v10_optuna_studies/twopx/*.joblib` (64 studies)
- `results/v10_optuna_best_params_twopx.json` + partial
- `results/v10_twopx_per_cell_results.csv`, `results/v10_twopx_ensemble_results.csv`
- `models/canonical/twopx/base_*.joblib` + `ens_*.joblib`
- `logs/v10_phase_e_twopx_driver.log`, `logs/v10_phase_e_twopx_runner.log`

## 6. Commits

- `041e3d3` Phase E: twopx 64 Optuna studies + runner complete
- `827234c` Phase E: twopx data pipeline + driver + runner + notebook

## 7. Tests logged

T01/T02/T11/T12 rows in `results/v10_nb03_test_log.csv`.
