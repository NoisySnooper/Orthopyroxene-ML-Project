# Table 4. Bias correction acceptance scorecard (opx, canonical seed 42, v3 ship rule)

| Pipeline | Track | Target | Model | Shipped form | Pre-RMSE test | Post-RMSE test | ΔRMSE | % reduction |
|---|---|---|---|---|---|---|---|---|
| opx | opx-liq  | T_C    | ElasticNet/raw | B (marginal) | 77.06  | 76.42  | 0.65  | 0.8 % |
| opx | opx-liq  | P_kbar | MLP/raw        | A            | 4.40   | 3.17   | 1.23  | 28.0 % |
| opx | opx-only | T_C    | LightGBM/alr   | A            | 146.63 | 122.06 | 24.56 | 16.8 % |
| opx | opx-only | P_kbar | RF/pwlr        | A            | 10.35  | 6.05   | 4.30  | 41.5 % |

Acceptance under the pre-registered tolerance-band rule (`docs/preregistration/p_regime_preregistration.md` §6): a correction form ships if aggregate test-set RMSE improves beyond `SHIP_TOL = 1e-6` and no pre-registered regime with `n ≥ N_MIN_FOR_VETO = 20` degrades by more than `max(SHIP_TOL, T_ABS, T_REL × pre_rmse_r)` with `T_ABS_T = 10 °C`, `T_ABS_P = 1 kbar`, `T_REL = 0.10`.

Form A is regime-piecewise OLS; Form B is the Ágreda-López et al. (2024) quantile-thresholded piecewise sigmoid. Opx-liq T accepts Form B at canonical seed only and fails the 20-seed majority vote (9 of 20 seeds accept), so it is flagged "marginal" and reported in §4.7 as effectively a null against filtered Putirka 28a (71.67 °C aggregate). The other three opx cells accept Form A under canonical and 20-seed majority; opx-only P is unanimous (20 of 20 seeds).

Pre-RMSE values are 20-seed mean test-set RMSE. Post-RMSE values are pre minus the v3 `ship_decision.overall_delta` for the shipped form. % reduction = ΔRMSE / pre-RMSE × 100. Source: `results/bias_correction_shipped.csv` and `results/bootstrap_rmse_cis_all_cells.csv`.
