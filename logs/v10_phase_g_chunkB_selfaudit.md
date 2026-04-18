# Phase G Chunk A + B self-audit

Summary: 24/24 checks pass.

| # | Check | Pass | Detail |
|---|---|---|---|
| 1 | 1. pre-registration doc exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/docs/v10_p_regime_preregistration.md |
| 2 | 2. P_REGIME_BIN_EDGES_KBAR == [0, 5, 15, 30, 100] | PASS | [0.0, 5.0, 15.0, 30.0, 100.0] |
| 3 | 3. four pre-registered labels present | PASS | ['shallow_crustal', 'deep_crustal_MASH', 'lithospheric_mantle', 'deeper_mantle'] |
| 4 | 4. registration date = 2026-04-17 | PASS | 2026-04-17 |
| 5 | 5. honesty-bar floor n >= 20 | PASS | 20 |
| 6 | 6. src.evaluation regime helpers importable | PASS |  |
| 7 | 7. v10_regime_allmodels.csv has all 6 tracks | PASS | tracks=['cpx_liq', 'cpx_only', 'opx_liq', 'opx_only', 'twopx', 'universal'] |
| 8 | 8. all four pre-registered labels in allmodels CSV | PASS | ['deep_crustal_MASH', 'deeper_mantle', 'lithospheric_mantle', 'shallow_crustal'] |
| 9 | 9. exploratory finer bins retained in allmodels CSV | PASS | ['10<=P<20', '20<=P<40', '5<=P<10', 'P<5', 'P>=40'] |
| 10 | 10. bootstrap CI columns present | PASS | ['mae_hi', 'mae_lo', 'rmse_hi', 'rmse_lo'] |
| 11 | 11. opx per-regime benchmark CSV has 8 rows (4 bins x 2 targets) | PASS | rows=8 |
| 12 | 12. honesty-bar verdict rule enforced in claims audit | PASS | all rows OK |
| 13 | 13. figure artifact fig24_per_regime_rmse_opx_liq.pdf | PASS | bytes=35343 |
| 14 | 13. figure artifact fig24_per_regime_rmse_opx_liq.png | PASS | bytes=164042 |
| 15 | 13. figure artifact fig25_per_regime_residual_violins_opx_liq.pdf | PASS | bytes=54644 |
| 16 | 13. figure artifact fig25_per_regime_residual_violins_opx_liq.png | PASS | bytes=352674 |
| 17 | 14. SI table artifact S8_5_1_regime_rmse_opx_liq.md | PASS | bytes=842 |
| 18 | 14. SI table artifact S8_5_1_regime_rmse_opx_liq.csv | PASS | bytes=440 |
| 19 | 14. SI table artifact S8_5_2_regime_rmse_ci_opx_liq.md | PASS | bytes=894 |
| 20 | 14. SI table artifact S8_5_2_regime_rmse_ci_opx_liq.csv | PASS | bytes=505 |
| 21 | 14. SI table artifact S8_5_3_regime_claims_audit_opx_liq.md | PASS | bytes=1755 |
| 22 | 14. SI table artifact S8_5_3_regime_claims_audit_opx_liq.csv | PASS | bytes=935 |
| 23 | 15. regime_results_autofilled.md exists | PASS | bytes=9064 |
| 24 | 16. T15 log row consistent with claims audit | PASS | t15_pass=True audit_hit=True |
