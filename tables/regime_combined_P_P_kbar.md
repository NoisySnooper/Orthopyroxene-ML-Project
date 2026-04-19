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
