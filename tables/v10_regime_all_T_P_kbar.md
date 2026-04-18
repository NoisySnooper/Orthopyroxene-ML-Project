# Regime-stratified RMSE · scope=all · bin=T · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · P_kbar (binned by T)

| Regime | n | v10 best | Putirka |
| --- | --- | --- | --- |
| 800<=T<1000 | 4 | **0.18 [0.07, 0.28] (RF/pwlr)** | 4.74 [2.90, 6.04] (Putirka 29a) |
| 1000<=T<1200 | 31 | **2.85 [2.34, 3.33] (RF/alr)** | 3.85 [2.17, 5.30] (Putirka 29a) |
| 1200<=T<1400 | 109 | **3.44 [2.65, 4.30] (MLP/raw)** | 3.82 [3.31, 4.32] (Putirka 29b) |
| T>=1400 | 30 | 5.43 [3.34, 7.69] (MLP/raw) | **4.92 [3.00, 6.73] (Putirka 29b)** |
| ALL | 174 | **3.82 [3.10, 4.49] (MLP/raw)** | 4.75 [3.81, 5.75] (Putirka 29a) |

### opx_only · P_kbar (binned by T)

| Regime | n | v10 best | Putirka |
| --- | --- | --- | --- |
| 800<=T<1000 | 22 | 5.87 [4.68, 7.10] (CatBoost/alr) | **2.79 [1.70, 3.79] (Putirka 29c)** |
| 1000<=T<1200 | 78 | 7.13 [5.85, 8.29] (XGB/alr) | **4.68 [3.74, 5.53] (Putirka 29c)** |
| 1200<=T<1400 | 63 | **4.93 [3.74, 6.13] (RF/pwlr)** | 13.63 [7.76, 18.30] (Putirka 29c) |
| T>=1400 | 27 | **20.23 [7.44, 31.87] (XGB/pwlr)** | 25.07 [13.05, 35.79] (Putirka 29c) |
| ALL | 190 | **10.33 [7.01, 14.10] (RF/pwlr)** | 13.34 [9.32, 17.34] (Putirka 29c) |

### cpx_liq · P_kbar (binned by T)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| T<800 | 7 | **2.94 (ElasticNet/alr)** | 6.82 [0.61, 11.35] (Putirka 30) | 4.62 [3.17, 5.62] (Agreda 2024) |
| 800<=T<1000 | 52 | 3.28 [2.63, 3.96] (XGB/alr) | 6.36 [5.48, 7.23] (Putirka 30) | **3.18 [1.83, 4.26] (Agreda 2024)** |
| 1000<=T<1200 | 267 | 4.29 [3.79, 4.84] (LightGBM/pwlr) | 4.85 [4.44, 5.29] (Putirka 30) | **3.77 [3.25, 4.33] (Agreda 2024)** |
| 1200<=T<1400 | 242 | **3.25 [2.90, 3.62] (RF/pwlr)** | 3.50 [3.04, 4.04] (Putirka 31) | 4.82 [3.66, 5.87] (Agreda 2024) |
| T>=1400 | 86 | 13.64 [5.15, 22.13] (MLP/pwlr) | **8.13 [4.83, 11.13] (Putirka 30)** | 23.29 [13.58, 33.58] (Agreda 2024) |
| ALL | 654 | 6.54 [4.40, 8.90] (LightGBM/pwlr) | **5.48 [4.75, 6.32] (Putirka 30)** | 9.31 [5.85, 12.86] (Agreda 2024) |

### cpx_only · P_kbar (binned by T)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| T<800 | 14 | 25.70 [18.82, 31.67] (GB/raw) | 13.97 [9.79, 17.07] (Putirka 32b) | **6.96 [4.85, 8.69] (Agreda 2024)** |
| 800<=T<1000 | 93 | **9.08 [5.68, 12.80] (XGB/raw)** | 10.25 [6.09, 14.26] (Putirka 32a) | 13.04 [8.71, 16.89] (Agreda 2024) |
| 1000<=T<1200 | 229 | 7.99 [4.66, 11.01] (LightGBM/raw) | **6.40 [4.50, 8.51] (Putirka 32a)** | 11.81 [6.64, 16.53] (Agreda 2024) |
| 1200<=T<1400 | 301 | 8.39 [6.41, 10.64] (ElasticNet/pwlr) | **8.19 [4.83, 12.09] (Putirka 32a)** | 10.79 [7.38, 14.11] (Agreda 2024) |
| T>=1400 | 147 | **13.07 [10.10, 15.46] (GB/alr)** | 19.12 [12.80, 24.82] (Putirka 32a) | 34.68 [26.25, 42.47] (Agreda 2024) |
| ALL | 784 | 13.39 [11.54, 15.31] (MLP/alr) | **11.83 [9.53, 13.93] (Putirka 32a)** | 18.22 [14.72, 21.27] (Agreda 2024) |

### twopx · P_kbar (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| 800<=T<1000 | 1 | **0.86 (MLP/twopx_components)** |
| 1000<=T<1200 | 30 | **3.56 [2.34, 4.83] (CatBoost/pwlr)** |
| 1200<=T<1400 | 15 | **2.47 [1.50, 3.31] (MLP/twopx_components)** |
| T>=1400 | 2 | **8.94 (MLP/raw)** |
| ALL | 48 | **4.27 [3.00, 5.30] (XGB/alr)** |

### universal · P_kbar (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| T<800 | 7 | **27.77 [22.29, 34.25] (ERT/universal_raw)** |
| 800<=T<1000 | 97 | **7.79 [5.53, 9.61] (RF/universal_raw)** |
| 1000<=T<1200 | 443 | **5.58 [4.93, 6.22] (RF/universal_raw)** |
| 1200<=T<1400 | 317 | **7.51 [6.31, 8.73] (ERT/universal_raw)** |
| T>=1400 | 141 | **34.47 [26.23, 42.76] (MLP/universal_raw)** |
| ALL | 1005 | **14.64 [11.92, 17.54] (MLP/universal_raw)** |
