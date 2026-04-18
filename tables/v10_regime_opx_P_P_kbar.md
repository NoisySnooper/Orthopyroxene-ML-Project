# Regime-stratified RMSE · scope=opx · bin=P · target=P_kbar

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
