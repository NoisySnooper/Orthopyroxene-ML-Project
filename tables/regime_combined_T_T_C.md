# Regime-stratified RMSE · scope=combined · bin=T · target=T_C

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### twopx · T_C (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| 800<=T<1000 | 1 | **0.14 (ElasticNet/twopx_components)** |
| 1000<=T<1200 | 30 | **89.58 [68.83, 107.62] (ElasticNet/raw)** |
| 1200<=T<1400 | 15 | **50.04 [36.25, 63.65] (CatBoost/alr)** |
| T>=1400 | 2 | **4.80 (ElasticNet/twopx_components)** |
| ALL | 48 | **79.54 [65.19, 93.31] (ElasticNet/raw)** |
