# Regime-stratified RMSE · scope=all · bin=P · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · P_kbar (binned by P)

| Regime | n | v10 best | Putirka |
| --- | --- | --- | --- |
| shallow_crustal | 47 | **2.66 [2.25, 3.13] (ElasticNet/raw)** | 3.89 [3.24, 4.47] (Putirka 29a) |
| deep_crustal_MASH | 61 | **2.08 [1.62, 2.54] (MLP/raw)** | 2.31 [1.72, 2.85] (Putirka 29a) |
| lithospheric_mantle | 58 | **4.54 [3.49, 5.56] (CatBoost/raw)** | 4.66 [3.25, 5.96] (Putirka 29a) |
| deeper_mantle | 8 | **4.56 [3.19, 5.75] (MLP/alr)** | 8.52 [4.44, 11.14] (Putirka 29b) |
| P<5 | 47 | **2.66 [2.25, 3.13] (ElasticNet/raw)** | 3.89 [3.24, 4.47] (Putirka 29a) |
| 5<=P<10 | 11 | 2.61 [2.04, 3.16] (MLP/alr) | **2.39 [2.01, 2.70] (Putirka 29a)** |
| 10<=P<20 | 81 | **2.76 [2.25, 3.27] (RF/pwlr)** | 3.44 [2.51, 4.32] (Putirka 29a) |
| 20<=P<40 | 32 | 5.25 [3.67, 6.85] (XGB/alr) | **3.72 [2.93, 4.53] (Putirka 29b)** |
| P>=40 | 3 | **0.77 [0.43, 1.01] (MLP/alr)** | 12.79 [11.29, 13.79] (Putirka 29b) |
| P<20 (Agreda range) | 139 | **2.97 [2.45, 3.45] (MLP/raw)** | 3.53 [3.00, 4.08] (Putirka 29a) |
| ALL | 174 | **3.82 [3.10, 4.49] (MLP/raw)** | 4.75 [3.81, 5.75] (Putirka 29a) |

### opx_only · P_kbar (binned by P)

| Regime | n | v10 best | Putirka |
| --- | --- | --- | --- |
| shallow_crustal | 48 | 7.19 [5.67, 8.76] (ERT/raw) | **2.20 [1.69, 2.72] (Putirka 29c)** |
| deep_crustal_MASH | 76 | 5.21 [4.48, 5.80] (CatBoost/alr) | **4.68 [3.92, 5.36] (Putirka 29c)** |
| lithospheric_mantle | 37 | **4.03 [2.63, 5.27] (ElasticNet/pwlr)** | 5.07 [3.83, 6.15] (Putirka 29c) |
| deeper_mantle | 29 | **19.65 [7.48, 30.43] (XGB/pwlr)** | 30.92 [21.45, 38.86] (Putirka 29c) |
| P<5 | 48 | 7.19 [5.67, 8.76] (ERT/raw) | **2.20 [1.69, 2.72] (Putirka 29c)** |
| 5<=P<10 | 20 | 6.45 [2.85, 9.25] (MLP/pwlr) | **5.09 [3.25, 6.96] (Putirka 29c)** |
| 10<=P<20 | 84 | **3.96 [3.52, 4.37] (ElasticNet/pwlr)** | 4.88 [4.13, 5.60] (Putirka 29c) |
| 20<=P<40 | 23 | **7.48 [6.08, 8.90] (XGB/alr)** | 9.71 [5.71, 13.12] (Putirka 29c) |
| P>=40 | 15 | **25.87 [6.36, 41.75] (XGB/pwlr)** | 44.03 [35.62, 53.35] (Putirka 29c) |
| P<20 (Agreda range) | 152 | 6.04 [5.27, 6.82] (CatBoost/alr) | **4.11 [3.58, 4.65] (Putirka 29c)** |
| ALL | 190 | **10.33 [7.01, 14.10] (RF/pwlr)** | 13.34 [9.32, 17.34] (Putirka 29c) |

### cpx_liq · P_kbar (binned by P)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| shallow_crustal | 268 | 4.97 [4.24, 5.68] (LightGBM/pwlr) | **4.18 [3.90, 4.44] (Putirka 30)** | 4.44 [3.90, 4.95] (Agreda 2024) |
| deep_crustal_MASH | 187 | 2.78 [2.32, 3.28] (LightGBM/pwlr) | 2.93 [2.61, 3.29] (Putirka 31) | **1.99 [1.73, 2.26] (Agreda 2024)** |
| lithospheric_mantle | 143 | 4.12 [3.58, 4.69] (XGB/raw) | 5.50 [4.24, 6.63] (Putirka 31) | **3.15 [2.42, 3.77] (Agreda 2024)** |
| deeper_mantle | 56 | 16.80 [5.29, 25.22] (GB/alr) | **10.23 [6.16, 13.95] (Putirka 30)** | 29.67 [16.54, 41.40] (Agreda 2024) |
| P<5 | 268 | 4.97 [4.24, 5.68] (LightGBM/pwlr) | **4.18 [3.90, 4.44] (Putirka 30)** | 4.44 [3.90, 4.95] (Agreda 2024) |
| 5<=P<10 | 46 | 3.24 [2.46, 4.01] (ElasticNet/raw) | 2.64 [2.01, 3.18] (Putirka 30) | **2.54 [1.81, 3.29] (Agreda 2024)** |
| 10<=P<20 | 210 | 2.55 [2.27, 2.81] (RF/pwlr) | 3.10 [2.78, 3.45] (Putirka 31) | **1.78 [1.53, 2.00] (Agreda 2024)** |
| 20<=P<40 | 115 | **4.48 [3.75, 5.25] (MLP/pwlr)** | 6.06 [4.83, 7.19] (Putirka 31) | 5.00 [3.64, 6.44] (Agreda 2024) |
| P>=40 | 15 | 31.94 [16.37, 45.97] (GB/alr) | **22.49 [11.39, 31.17] (Putirka 30)** | 56.35 [36.92, 74.92] (Agreda 2024) |
| P<20 (Agreda range) | 524 | 4.12 [3.68, 4.56] (LightGBM/pwlr) | 4.10 [3.83, 4.37] (Putirka 31) | **3.45 [3.09, 3.77] (Agreda 2024)** |
| ALL | 654 | 6.54 [4.40, 8.90] (LightGBM/pwlr) | **5.48 [4.75, 6.32] (Putirka 30)** | 9.31 [5.85, 12.86] (Agreda 2024) |

### cpx_only · P_kbar (binned by P)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| shallow_crustal | 236 | 8.33 [7.08, 9.55] (CatBoost/raw) | **2.84 [2.50, 3.22] (Putirka 32b)** | 3.84 [3.32, 4.26] (Agreda 2024) |
| deep_crustal_MASH | 193 | 3.28 [2.05, 4.73] (ElasticNet/alr) | 2.42 [2.14, 2.68] (Putirka 32a) | **2.35 [1.91, 2.70] (Agreda 2024)** |
| lithospheric_mantle | 213 | 8.19 [5.53, 10.82] (ERT/raw) | 5.97 [5.16, 6.80] (Putirka 32b) | **4.19 [3.55, 4.84] (Agreda 2024)** |
| deeper_mantle | 142 | **18.43 [14.60, 22.15] (GB/alr)** | 24.71 [19.00, 30.32] (Putirka 32a) | 42.12 [34.28, 49.79] (Agreda 2024) |
| P<5 | 236 | 8.33 [7.08, 9.55] (CatBoost/raw) | **2.84 [2.50, 3.22] (Putirka 32b)** | 3.84 [3.32, 4.26] (Agreda 2024) |
| 5<=P<10 | 43 | **2.53 [1.97, 3.01] (ElasticNet/alr)** | 3.27 [2.66, 3.85] (Putirka 32a) | 2.96 [2.31, 3.44] (Agreda 2024) |
| 10<=P<20 | 296 | 5.54 [3.67, 7.56] (RF/raw) | 4.44 [3.64, 5.18] (Putirka 32b) | **2.85 [2.32, 3.42] (Agreda 2024)** |
| 20<=P<40 | 148 | 8.27 [6.01, 10.25] (LightGBM/pwlr) | **7.85 [6.92, 8.78] (Putirka 32b)** | 8.32 [7.23, 9.55] (Agreda 2024) |
| P>=40 | 61 | **27.51 [21.70, 32.67] (RF/alr)** | 38.35 [30.97, 45.10] (Putirka 32a) | 63.21 [54.50, 72.75] (Agreda 2024) |
| P<20 (Agreda range) | 575 | 6.79 [5.92, 7.60] (CatBoost/raw) | 3.87 [3.41, 4.38] (Putirka 32b) | **3.30 [2.98, 3.63] (Agreda 2024)** |
| ALL | 784 | 13.39 [11.54, 15.31] (MLP/alr) | **11.83 [9.53, 13.93] (Putirka 32a)** | 18.22 [14.72, 21.27] (Agreda 2024) |

### twopx · P_kbar (binned by P)

| Regime | n | v10 best |
| --- | --- | --- |
| shallow_crustal | 9 | **4.93 [2.53, 7.44] (XGB/alr)** |
| deep_crustal_MASH | 30 | **2.38 [2.04, 2.72] (ElasticNet/raw)** |
| lithospheric_mantle | 8 | **3.42 [1.47, 5.22] (LightGBM/raw)** |
| deeper_mantle | 1 | **9.18 (MLP/raw)** |
| P<5 | 9 | **4.93 [2.53, 7.44] (XGB/alr)** |
| 5<=P<10 | 4 | **2.86 [1.47, 3.90] (ElasticNet/raw)** |
| 10<=P<20 | 33 | **2.60 [2.25, 2.94] (ElasticNet/alr)** |
| 20<=P<40 | 2 | **8.94 (MLP/raw)** |
| P<20 (Agreda range) | 46 | **3.40 [2.54, 4.22] (CatBoost/pwlr)** |
| ALL | 48 | **4.27 [3.00, 5.30] (XGB/alr)** |

### universal · P_kbar (binned by P)

| Regime | n | v10 best |
| --- | --- | --- |
| shallow_crustal | 318 | **4.64 [4.11, 5.08] (RF/universal_raw)** |
| deep_crustal_MASH | 286 | **4.02 [3.57, 4.48] (ERT/universal_raw)** |
| lithospheric_mantle | 272 | **7.98 [6.55, 9.38] (ERT/universal_raw)** |
| deeper_mantle | 129 | **37.99 [29.17, 46.44] (MLP/universal_raw)** |
| P<5 | 318 | **4.64 [4.11, 5.08] (RF/universal_raw)** |
| 5<=P<10 | 106 | **4.09 [3.33, 4.86] (ERT/universal_raw)** |
| 10<=P<20 | 278 | **5.91 [4.47, 7.31] (ERT/universal_raw)** |
| 20<=P<40 | 213 | **8.80 [7.24, 10.35] (ERT/universal_raw)** |
| P>=40 | 90 | **46.03 [35.36, 56.11] (MLP/universal_raw)** |
| P<20 (Agreda range) | 702 | **5.53 [4.68, 6.41] (RF/universal_raw)** |
| ALL | 1005 | **14.64 [11.92, 17.54] (MLP/universal_raw)** |
