# Regime-stratified RMSE · scope=combined · bin=P · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

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
