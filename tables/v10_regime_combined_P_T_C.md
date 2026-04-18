# Regime-stratified RMSE · scope=combined · bin=P · target=T_C

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

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

### universal · T_C (binned by P)

| Regime | n | v10 best |
| --- | --- | --- |
| shallow_crustal | 318 | **62.50 [54.79, 70.30] (MLP/universal_raw)** |
| deep_crustal_MASH | 286 | **76.93 [63.92, 89.50] (CatBoost/universal_raw)** |
| lithospheric_mantle | 272 | **95.74 [83.91, 106.59] (MLP/universal_raw)** |
| deeper_mantle | 129 | **314.35 [271.25, 352.24] (ElasticNet/universal_raw)** |
| P<5 | 318 | **62.50 [54.79, 70.30] (MLP/universal_raw)** |
| 5<=P<10 | 106 | **63.37 [48.90, 78.33] (CatBoost/universal_raw)** |
| 10<=P<20 | 278 | **82.84 [65.79, 97.99] (XGB/universal_raw)** |
| 20<=P<40 | 213 | **100.49 [89.58, 112.34] (MLP/universal_raw)** |
| P>=40 | 90 | **363.91 [310.54, 414.01] (ElasticNet/universal_raw)** |
| P<20 (Agreda range) | 702 | **73.78 [65.67, 83.12] (GB/universal_raw)** |
| ALL | 1005 | **143.71 [127.78, 159.81] (RF/universal_raw)** |
