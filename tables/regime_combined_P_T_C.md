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
