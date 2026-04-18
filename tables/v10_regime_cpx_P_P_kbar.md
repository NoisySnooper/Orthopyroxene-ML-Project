# Regime-stratified RMSE · scope=cpx · bin=P · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

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
