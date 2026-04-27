# Regime-stratified RMSE · scope=cpx · bin=P · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### cpx_liq · P_kbar (binned by P)

| Regime | n | our_best | Putirka | Agreda | Jorgenson | Wang |
| --- | --- | --- | --- | --- | --- | --- |
| shallow_crustal | 268 | 4.97 [4.24, 5.68] (LightGBM/pwlr) | **4.18 [3.90, 4.44] (Putirka 30)** | 4.44 [3.90, 4.95] (Agreda 2024) | 4.66 [4.20, 5.10] (Jorgenson 2022) | 6.73 [5.76, 7.77] (Wang 2021) |
| deep_crustal_MASH | 187 | 2.78 [2.32, 3.28] (LightGBM/pwlr) | 2.93 [2.61, 3.29] (Putirka 31) | **1.99 [1.73, 2.26] (Agreda 2024)** | 2.08 [1.88, 2.28] (Jorgenson 2022) | 3.66 [2.97, 4.39] (Wang 2021) |
| lithospheric_mantle | 143 | 4.12 [3.58, 4.69] (XGB/raw) | 5.50 [4.24, 6.63] (Putirka 31) | 3.15 [2.42, 3.77] (Agreda 2024) | **3.08 [2.43, 3.65] (Jorgenson 2022)** | 8.04 [6.55, 9.64] (Wang 2021) |
| deeper_mantle | 56 | 16.80 [5.29, 25.22] (GB/alr) | **10.23 [6.16, 13.95] (Putirka 30)** | 29.67 [16.54, 41.40] (Agreda 2024) | 28.87 [16.05, 40.52] (Jorgenson 2022) | 29.23 [20.44, 40.03] (Wang 2021) |
| P<5 | 268 | 4.97 [4.24, 5.68] (LightGBM/pwlr) | **4.18 [3.90, 4.44] (Putirka 30)** | 4.44 [3.90, 4.95] (Agreda 2024) | 4.66 [4.20, 5.10] (Jorgenson 2022) | 6.73 [5.76, 7.77] (Wang 2021) |
| 5<=P<10 | 46 | 3.24 [2.46, 4.01] (ElasticNet/raw) | 2.64 [2.01, 3.18] (Putirka 30) | 2.54 [1.81, 3.29] (Agreda 2024) | **2.49 [2.05, 3.01] (Jorgenson 2022)** | 3.10 [1.97, 4.42] (Wang 2021) |
| 10<=P<20 | 210 | 2.55 [2.27, 2.81] (RF/pwlr) | 3.10 [2.78, 3.45] (Putirka 31) | **1.78 [1.53, 2.00] (Agreda 2024)** | 1.91 [1.64, 2.16] (Jorgenson 2022) | 3.89 [3.38, 4.46] (Wang 2021) |
| 20<=P<40 | 115 | **4.48 [3.75, 5.25] (MLP/pwlr)** | 6.06 [4.83, 7.19] (Putirka 31) | 5.00 [3.64, 6.44] (Agreda 2024) | 5.07 [4.22, 5.93] (Jorgenson 2022) | 11.08 [9.69, 12.65] (Wang 2021) |
| P>=40 | 15 | 31.94 [16.37, 45.97] (GB/alr) | **22.49 [11.39, 31.17] (Putirka 30)** | 56.35 [36.92, 74.92] (Agreda 2024) | 54.67 [34.46, 73.86] (Jorgenson 2022) | 53.16 [35.96, 69.22] (Wang 2021) |
| P<20 (Agreda range) | 524 | 4.12 [3.68, 4.56] (LightGBM/pwlr) | 4.10 [3.83, 4.37] (Putirka 31) | **3.45 [3.09, 3.77] (Agreda 2024)** | 3.62 [3.29, 3.89] (Jorgenson 2022) | 5.41 [4.87, 6.04] (Wang 2021) |
| ALL | 654 | 6.54 [4.40, 8.90] (LightGBM/pwlr) | **5.48 [4.75, 6.32] (Putirka 30)** | 9.31 [5.85, 12.86] (Agreda 2024) | 9.14 [5.68, 12.63] (Jorgenson 2022) | 10.46 [8.02, 12.74] (Wang 2021) |

### cpx_only · P_kbar (binned by P)

| Regime | n | our_best | Putirka | Agreda | Jorgenson |
| --- | --- | --- | --- | --- | --- |
| shallow_crustal | 236 | 8.33 [7.08, 9.55] (CatBoost/raw) | **2.84 [2.50, 3.22] (Putirka 32b)** | 3.84 [3.32, 4.26] (Agreda 2024) | 3.19 [2.76, 3.53] (Jorgenson 2022) |
| deep_crustal_MASH | 193 | 3.28 [2.05, 4.73] (ElasticNet/alr) | 2.42 [2.14, 2.68] (Putirka 32a) | 2.35 [1.91, 2.70] (Agreda 2024) | **1.39 [1.05, 1.67] (Jorgenson 2022)** |
| lithospheric_mantle | 213 | 8.19 [5.53, 10.82] (ERT/raw) | 5.97 [5.16, 6.80] (Putirka 32b) | 4.19 [3.55, 4.84] (Agreda 2024) | **3.83 [3.23, 4.46] (Jorgenson 2022)** |
| deeper_mantle | 142 | **18.43 [14.60, 22.15] (GB/alr)** | 24.71 [19.00, 30.32] (Putirka 32a) | 42.12 [34.28, 49.79] (Agreda 2024) | 41.36 [33.82, 48.98] (Jorgenson 2022) |
| P<5 | 236 | 8.33 [7.08, 9.55] (CatBoost/raw) | **2.84 [2.50, 3.22] (Putirka 32b)** | 3.84 [3.32, 4.26] (Agreda 2024) | 3.19 [2.76, 3.53] (Jorgenson 2022) |
| 5<=P<10 | 43 | 2.53 [1.97, 3.01] (ElasticNet/alr) | 3.27 [2.66, 3.85] (Putirka 32a) | 2.96 [2.31, 3.44] (Agreda 2024) | **1.69 [1.17, 2.14] (Jorgenson 2022)** |
| 10<=P<20 | 296 | 5.54 [3.67, 7.56] (RF/raw) | 4.44 [3.64, 5.18] (Putirka 32b) | 2.85 [2.32, 3.42] (Agreda 2024) | **2.30 [1.82, 2.80] (Jorgenson 2022)** |
| 20<=P<40 | 148 | 8.27 [6.01, 10.25] (LightGBM/pwlr) | 7.85 [6.92, 8.78] (Putirka 32b) | 8.32 [7.23, 9.55] (Agreda 2024) | **7.37 [6.40, 8.30] (Jorgenson 2022)** |
| P>=40 | 61 | **27.51 [21.70, 32.67] (RF/alr)** | 38.35 [30.97, 45.10] (Putirka 32a) | 63.21 [54.50, 72.75] (Agreda 2024) | 62.29 [53.73, 71.70] (Jorgenson 2022) |
| P<20 (Agreda range) | 575 | 6.79 [5.92, 7.60] (CatBoost/raw) | 3.87 [3.41, 4.38] (Putirka 32b) | 3.30 [2.98, 3.63] (Agreda 2024) | **2.67 [2.39, 2.95] (Jorgenson 2022)** |
| ALL | 784 | 13.39 [11.54, 15.31] (MLP/alr) | **11.83 [9.53, 13.93] (Putirka 32a)** | 18.22 [14.72, 21.27] (Agreda 2024) | 17.82 [14.38, 20.81] (Jorgenson 2022) |
