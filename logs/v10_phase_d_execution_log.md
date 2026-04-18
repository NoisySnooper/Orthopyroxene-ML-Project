# Phase D Execution Log

**Date:** 2026-04-17
**Status:** COMPLETE
**Scope:** cpx Optuna TPE search (96 studies) + refit/ensemble runner (12 cells)

---

## 1. Grid

Same shape as Phase C, with cpx tracks:
- 8 models x 2 targets x 2 tracks (cpx_only, cpx_liq) x 3 feature sets = **96 studies**

## 2. Execution

- `scripts/v10_phase_d_cpx_driver.py` + `scripts/v10_phase_d_cpx_runner.py`
- Ran in parallel with Phase E driver for wall-time savings. Cores reallocated to D from E after E completed.
- Final driver run at `OPTUNA_N_JOBS_INNER = 12` (full machine).
- Runner ran at `V10_MAX_JOBS=12`.

## 3. Driver wall time

- Started 2026-04-17 morning, finished 16:08:17 (~24 studies at 12 cores in final stretch). Zero failures.
- Driver: `Phase D cpx driver DONE completed=96 failed=0`
- Runner: all 12 cells in ~12 min, 96 base rows + 36 ensemble rows.

## 4. Best per cell

| Target | Track | Winning base | Feature set | Test RMSE |
|---|---|---|---|---|
| T_C | cpx_only | ERT | pwlr | 127.35 |
| T_C | cpx_liq  | CatBoost | pwlr | 71.57 |
| P_kbar | cpx_only | MLP | alr | 13.47 |
| P_kbar | cpx_liq  | LightGBM | pwlr | 6.54 |

## 5. Ensemble highlights

- cpx_liq P_kbar: LightGBM pwlr (6.54) edged by ridge ensemble (6.52).
- cpx_only T_C: ERT pwlr best (127.35) versus ensembles near 128.

## 6. Artifacts

- `results/v10_optuna_studies/cpx/*.joblib` (96 studies)
- `results/v10_optuna_best_params_cpx.json` + partial
- `results/v10_cpx_per_cell_results.csv`, `results/v10_cpx_ensemble_results.csv`
- `models/canonical/cpx/base_*.joblib` + `ens_*.joblib`
- `logs/v10_phase_d_cpx_driver.log`, `logs/v10_phase_d_cpx_runner.log`

## 7. Commits

- `8ae813c` Phase D: cpx 96 Optuna studies + runner complete
- `e35d0da` Phase D: cpx Optuna driver + runner + notebook (smoke-tested)
- `cce0917` Phase D: cpx data pipeline (cpx_only + cpx_liq parquet + splits)

## 8. Tests logged

T01/T02/T11/T12 rows appended to `results/v10_nb03_test_log.csv`.
