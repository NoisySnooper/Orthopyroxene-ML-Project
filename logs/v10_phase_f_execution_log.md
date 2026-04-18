# Phase F Execution Log

**Date:** 2026-04-17
**Status:** COMPLETE
**Scope:** Universal (single-model, mask-aware) pipeline: 16 Optuna studies + runner with T13/T14 tests

---

## 1. Grid

- 8 models x 2 targets x 1 track (universal) x 1 feature set (universal_raw) = **16 studies**
- Single feature set by construction: log-ratio transforms are invalid on zero-filled absent phases.
- 45-dim fixed feature vector: 9 opx ox + 7 opx eng + 9 cpx ox + 7 cpx eng + 8 liq ox + 2 liq eng + 3 presence bits.

## 2. Execution

- `scripts/v10_phase_f_universal_prep.py` (union of opx_only/cpx_only/opx_liq/cpx_liq parquets; zero-fill missing; GroupShuffleSplit on Citation).
- `scripts/v10_phase_f_universal_driver.py` (`--skip-existing` to preserve cached smoke-test study).
- `scripts/v10_phase_f_universal_runner.py` (all 12 cores, 2 cells, ~2.5 min total).

Driver started 16:51:23, DONE 18:00:50 (~1h 10m). Runner ~2.5 min.

## 3. Best per cell

| Target | Winning base | Feature set | Test RMSE |
|---|---|---|---|
| T_C | RF | universal_raw | 143.71 |
| P_kbar | MLP | universal_raw | 16.62 |

Winning ensembles:
- T_C: **greedy** (147.07)
- P_kbar: **two_level** (16.36)

## 4. Phase-scope RMSE (P_kbar, two_level ensemble)

| Scope | n | RMSE (kbar) |
|---|---|---|
| twopx_liq | 144 | **4.07** |
| opx_liq | 77 | 6.06 |
| cpx_liq | 581 | 9.60 |
| twopx | 21 | 11.47 |
| opx_only | 95 | 19.86 |
| cpx_only | 87 | 44.21 |

## 5. Test outcomes (T13/T14)

- **T14 pass** for P_kbar: `twopx_liq` has the lowest RMSE (4.07), confirming the all-phase subset beats every partial subset.
- **T13 mixed**: tier medians ordered correctly for 3-phase vs 2-phase (4.07 < 9.60 median) but single-phase tier (opx_only/cpx_only) is much worse than expected. Consistent with expected model degradation on single-phase inputs that had been zero-filled during training.

T01/T02/T11/T12/T13/T14 rows appended to `results/universal/v10_nb03_test_log.csv`.

## 6. Artifacts

- `data/processed/universal_clean.parquet` (3229 rows, 45 feat)
- `data/splits/train_indices_universal.npy` / `test_indices_universal.npy`
- `results/v10_optuna_studies/universal/*.joblib` (16 studies)
- `results/v10_optuna_best_params_universal.json` + partial
- `results/universal/v10_universal_per_cell_results.csv` (16 base rows)
- `results/universal/v10_universal_ensemble_results.csv` (6 ensemble rows)
- `results/universal/v10_universal_scope_rmse.csv` (12 scope rows)
- `models/canonical/universal/base_*.joblib` + `ens_*.joblib`
- `logs/v10_phase_f_universal_driver.log`, `logs/v10_phase_f_universal_runner.log`

## 7. Commits

- `63d94a3` Phase F: universal 16 Optuna studies + runner complete
- `22d4112` Phase F: universal (masking) pipeline + T13/T14 tests

## 8. Phase boundary

All v10 training phases (C/D/E/F) now complete. Next: Phase G (figures, manuscript compilation).
