# Regime-stratified RMSE · scope=opx · bin=T · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · P_kbar (binned by T)

| Regime | n | our_best | Putirka |
| --- | --- | --- | --- |
| 800<=T<1000 | 4 | **0.18 [0.07, 0.28] (RF/pwlr)** | 4.74 [2.90, 6.04] (Putirka 29a) |
| 1000<=T<1200 | 31 | **2.85 [2.34, 3.33] (RF/alr)** | 3.85 [2.17, 5.30] (Putirka 29a) |
| 1200<=T<1400 | 109 | **3.44 [2.65, 4.30] (MLP/raw)** | 3.82 [3.31, 4.32] (Putirka 29b) |
| T>=1400 | 30 | 5.43 [3.34, 7.69] (MLP/raw) | **4.92 [3.00, 6.73] (Putirka 29b)** |
| ALL | 174 | **3.82 [3.10, 4.49] (MLP/raw)** | 4.75 [3.81, 5.75] (Putirka 29a) |

### opx_only · P_kbar (binned by T)

| Regime | n | our_best | Putirka |
| --- | --- | --- | --- |
| 800<=T<1000 | 22 | 5.87 [4.68, 7.10] (CatBoost/alr) | **2.79 [1.70, 3.79] (Putirka 29c)** |
| 1000<=T<1200 | 78 | 7.13 [5.85, 8.29] (XGB/alr) | **4.68 [3.74, 5.53] (Putirka 29c)** |
| 1200<=T<1400 | 63 | **4.93 [3.74, 6.13] (RF/pwlr)** | 13.63 [7.76, 18.30] (Putirka 29c) |
| T>=1400 | 27 | **20.23 [7.44, 31.87] (XGB/pwlr)** | 25.07 [13.05, 35.79] (Putirka 29c) |
| ALL | 190 | **10.33 [7.01, 14.10] (RF/pwlr)** | 13.34 [9.32, 17.34] (Putirka 29c) |
