# Regime-stratified RMSE · scope=all · bin=P · target=T_C

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · T_C (binned by P)

| Regime | n | v10 best | Putirka |
| --- | --- | --- | --- |
| shallow_crustal | 47 | **23.47 [15.89, 30.40] (ERT/pwlr)** | 25.52 [16.32, 33.87] (Putirka 28a) |
| deep_crustal_MASH | 61 | **78.46 [29.85, 113.07] (ElasticNet/raw)** | 87.48 [34.61, 120.83] (Putirka 28a) |
| lithospheric_mantle | 58 | 93.70 [77.31, 111.17] (ElasticNet/raw) | **79.65 [54.89, 107.41] (Putirka 28a)** |
| deeper_mantle | 8 | 105.60 [75.14, 127.73] (ElasticNet/pwlr) | **85.50 [14.78, 123.50] (Putirka 28a)** |
| P<5 | 47 | **23.47 [15.89, 30.40] (ERT/pwlr)** | 25.52 [16.32, 33.87] (Putirka 28a) |
| 5<=P<10 | 11 | **27.00 [18.19, 33.91] (ElasticNet/raw)** | 43.05 [25.11, 61.83] (Putirka 28a) |
| 10<=P<20 | 81 | **88.40 [59.90, 114.05] (ElasticNet/raw)** | 97.54 [66.44, 124.69] (Putirka 28a) |
| 20<=P<40 | 32 | 92.14 [75.43, 108.58] (ElasticNet/raw) | **41.76 [25.17, 59.66] (Putirka 28a)** |
| P>=40 | 3 | **20.43 (MLP/raw)** | 138.78 [61.86, 173.83] (Putirka 28a) |
| P<20 (Agreda range) | 139 | **70.95 [50.29, 88.99] (ElasticNet/raw)** | 75.04 [53.39, 96.25] (Putirka 28a) |
| ALL | 174 | 77.06 [62.56, 91.65] (ElasticNet/raw) | **71.67 [53.75, 88.74] (Putirka 28a)** |

### opx_only · T_C (binned by P)

| Regime | n | v10 best |
| --- | --- | --- |
| shallow_crustal | 48 | **108.26 [90.90, 122.97] (GB/raw)** |
| deep_crustal_MASH | 76 | **109.47 [86.56, 132.73] (LightGBM/pwlr)** |
| lithospheric_mantle | 37 | **70.43 [57.66, 84.52] (ERT/raw)** |
| deeper_mantle | 29 | **247.61 [167.18, 313.44] (ElasticNet/raw)** |
| P<5 | 48 | **108.26 [90.90, 122.97] (GB/raw)** |
| 5<=P<10 | 20 | **136.73 [95.58, 174.79] (MLP/alr)** |
| 10<=P<20 | 84 | **87.91 [74.08, 105.38] (LightGBM/pwlr)** |
| 20<=P<40 | 23 | **108.01 [69.23, 152.09] (LightGBM/alr)** |
| P>=40 | 15 | **321.18 [205.15, 406.97] (ElasticNet/raw)** |
| P<20 (Agreda range) | 152 | **112.02 [100.43, 124.72] (XGB/raw)** |
| ALL | 190 | **147.98 [122.71, 172.97] (LightGBM/alr)** |

### cpx_liq · T_C (binned by P)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| shallow_crustal | 268 | 41.96 [35.70, 48.41] (RF/pwlr) | 42.38 [36.79, 48.46] (Putirka 34) | **30.09 [25.87, 34.36] (Agreda 2024)** |
| deep_crustal_MASH | 187 | 49.19 [39.10, 61.23] (ERT/raw) | 53.67 [46.58, 61.71] (Putirka 34) | **47.53 [35.68, 61.14] (Agreda 2024)** |
| lithospheric_mantle | 143 | 74.90 [65.11, 85.35] (XGB/pwlr) | 79.12 [64.68, 93.50] (Putirka 33) | **42.58 [36.47, 48.45] (Agreda 2024)** |
| deeper_mantle | 56 | 131.04 [65.02, 195.72] (MLP/pwlr) | **66.12 [57.79, 74.98] (Putirka 33)** | 243.67 [138.75, 326.77] (Agreda 2024) |
| P<5 | 268 | 41.96 [35.70, 48.41] (RF/pwlr) | 42.38 [36.79, 48.46] (Putirka 34) | **30.09 [25.87, 34.36] (Agreda 2024)** |
| 5<=P<10 | 46 | **41.79 [31.36, 52.86] (GB/raw)** | 49.05 [35.73, 62.62] (Putirka 34) | 42.38 [27.75, 55.71] (Agreda 2024) |
| 10<=P<20 | 210 | 53.25 [44.58, 63.64] (ERT/raw) | 63.07 [54.95, 72.41] (Putirka 34) | **44.41 [33.89, 57.25] (Agreda 2024)** |
| 20<=P<40 | 115 | 64.55 [53.38, 75.07] (MLP/alr) | 78.25 [61.37, 94.16] (Putirka 33) | **45.28 [38.31, 51.89] (Agreda 2024)** |
| P>=40 | 15 | 279.05 [146.33, 382.94] (CatBoost/pwlr) | **96.32 [66.13, 129.84] (Putirka 33)** | 467.19 [264.94, 611.82] (Agreda 2024) |
| P<20 (Agreda range) | 524 | 48.46 [42.61, 55.05] (RF/pwlr) | 52.19 [47.51, 57.06] (Putirka 34) | **37.57 [32.02, 44.08] (Agreda 2024)** |
| ALL | 654 | 71.57 [59.35, 85.84] (CatBoost/pwlr) | **63.43 [57.58, 69.45] (Putirka 33)** | 80.61 [49.84, 109.05] (Agreda 2024) |

### cpx_only · T_C (binned by P)

| Regime | n | v10 best | Putirka | Agreda |
| --- | --- | --- | --- | --- |
| shallow_crustal | 236 | 82.18 [71.97, 92.69] (GB/pwlr) | 118.75 [106.63, 129.39] (Putirka 32d) | **67.52 [55.91, 80.33] (Agreda 2024)** |
| deep_crustal_MASH | 193 | 86.67 [73.63, 101.93] (RF/pwlr) | 120.34 [102.39, 139.79] (Putirka 32d) | **64.29 [51.51, 77.63] (Agreda 2024)** |
| lithospheric_mantle | 213 | 134.81 [110.74, 162.91] (MLP/pwlr) | 186.63 [157.15, 213.47] (Putirka 32d) | **100.55 [76.69, 122.25] (Agreda 2024)** |
| deeper_mantle | 142 | **165.99 [142.54, 187.92] (MLP/pwlr)** | 285.24 [233.00, 329.41] (Putirka 32d) | 202.59 [163.44, 238.62] (Agreda 2024) |
| P<5 | 236 | 82.18 [71.97, 92.69] (GB/pwlr) | 118.75 [106.63, 129.39] (Putirka 32d) | **67.52 [55.91, 80.33] (Agreda 2024)** |
| 5<=P<10 | 43 | 85.57 [69.60, 100.70] (GB/pwlr) | 168.31 [139.88, 192.92] (Putirka 32d) | **74.46 [60.55, 89.37] (Agreda 2024)** |
| 10<=P<20 | 296 | 105.25 [89.12, 121.81] (XGB/pwlr) | 134.40 [114.07, 154.42] (Putirka 32d) | **71.79 [54.79, 89.45] (Agreda 2024)** |
| 20<=P<40 | 148 | 129.31 [90.55, 177.09] (MLP/pwlr) | 171.30 [131.73, 206.28] (Putirka 32d) | **108.60 [85.19, 132.34] (Agreda 2024)** |
| P>=40 | 61 | **221.65 [187.45, 254.88] (MLP/pwlr)** | 436.82 [365.37, 490.89] (Putirka 32d) | 293.93 [244.83, 350.49] (Agreda 2024) |
| P<20 (Agreda range) | 575 | 96.14 [87.01, 105.54] (GB/pwlr) | 131.15 [119.84, 143.52] (Putirka 32d) | **70.28 [60.96, 79.60] (Agreda 2024)** |
| ALL | 784 | 127.35 [114.37, 140.51] (ERT/pwlr) | 177.92 [161.01, 192.93] (Putirka 32d) | **112.12 [98.68, 127.50] (Agreda 2024)** |

### twopx · T_C (binned by P)

| Regime | n | v10 best |
| --- | --- | --- |
| shallow_crustal | 9 | **72.22 [27.99, 109.34] (CatBoost/pwlr)** |
| deep_crustal_MASH | 30 | **82.75 [59.66, 102.34] (ElasticNet/alr)** |
| lithospheric_mantle | 8 | **54.93 [30.76, 73.83] (ElasticNet/twopx_components)** |
| deeper_mantle | 1 | **6.56 (ElasticNet/twopx_components)** |
| P<5 | 9 | **72.22 [27.99, 109.34] (CatBoost/pwlr)** |
| 5<=P<10 | 4 | **53.81 [27.72, 65.22] (ElasticNet/alr)** |
| 10<=P<20 | 33 | **80.40 [60.16, 99.94] (ElasticNet/alr)** |
| 20<=P<40 | 2 | **4.80 (ElasticNet/twopx_components)** |
| P<20 (Agreda range) | 46 | **78.91 [63.15, 93.42] (ElasticNet/alr)** |
| ALL | 48 | **79.54 [65.19, 93.31] (ElasticNet/raw)** |
