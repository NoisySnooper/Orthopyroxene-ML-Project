# Table S8.8: opx-liq in-sample piecewise bias correction

Per-regime OLS correction fit on 10-fold StratifiedGroupKFold out-of-fold training-set predictions, evaluated on the held-out test split. `pre` = raw model prediction RMSE; `post` = corrected RMSE; both with bootstrap 95% CI (500 resamples). A useful correction shows post CI strictly below pre CI on at least one regime and no worse than pre CI elsewhere.

| Cell | Regime | n | pre RMSE [95% CI] | post RMSE [95% CI] | Δ |
|---|---|---|---|---|---|
| ElasticNet/raw/T_C | deep_crustal_MASH | 61 | 78.46 [30.40, 111.64] °C | 76.92 [33.27, 107.70] °C | +1.55 °C |
| ElasticNet/raw/T_C | deeper_mantle | 8 | 106.00 [78.46, 130.08] °C | 61.54 [52.23, 70.11] °C | +44.45 °C |
| ElasticNet/raw/T_C | lithospheric_mantle | 58 | 93.70 [79.13, 108.67] °C | 93.58 [79.00, 108.80] °C | +0.12 °C |
| ElasticNet/raw/T_C | shallow_crustal | 47 | 35.33 [27.96, 43.56] °C | 54.01 [47.26, 62.60] °C | -18.68 °C |
| ElasticNet/raw/T_C | ALL | 174 | 77.06 [61.81, 92.73] °C | 77.17 [63.23, 91.64] °C | -0.11 °C |
| MLP/raw/P_kbar | deep_crustal_MASH | 61 | 2.08 [1.67, 2.52] kbar | 1.95 [1.58, 2.29] kbar | +0.13 kbar |
| MLP/raw/P_kbar | deeper_mantle | 8 | 4.72 [3.62, 5.95] kbar | 7.87 [4.03, 10.75] kbar | -3.15 kbar |
| MLP/raw/P_kbar | lithospheric_mantle | 58 | 5.38 [4.01, 6.85] kbar | 2.64 [1.99, 3.17] kbar | +2.74 kbar |
| MLP/raw/P_kbar | shallow_crustal | 47 | 2.96 [1.48, 3.96] kbar | 0.80 [0.67, 0.93] kbar | +2.17 kbar |
| MLP/raw/P_kbar | ALL | 174 | 3.82 [3.08, 4.54] kbar | 2.58 [2.03, 3.20] kbar | +1.23 kbar |
| ElasticNet/raw/P_kbar | deep_crustal_MASH | 61 | 3.15 [2.61, 3.71] kbar | 1.73 [1.36, 2.06] kbar | +1.42 kbar |
| ElasticNet/raw/P_kbar | deeper_mantle | 8 | 18.36 [15.21, 21.14] kbar | 9.59 [0.65, 13.54] kbar | +8.77 kbar |
| ElasticNet/raw/P_kbar | lithospheric_mantle | 58 | 5.56 [4.21, 6.94] kbar | 3.36 [2.57, 4.12] kbar | +2.20 kbar |
| ElasticNet/raw/P_kbar | shallow_crustal | 47 | 2.66 [2.27, 3.06] kbar | 1.02 [1.01, 1.02] kbar | +1.65 kbar |
| ElasticNet/raw/P_kbar | ALL | 174 | 5.59 [4.54, 6.74] kbar | 3.05 [2.23, 3.83] kbar | +2.53 kbar |