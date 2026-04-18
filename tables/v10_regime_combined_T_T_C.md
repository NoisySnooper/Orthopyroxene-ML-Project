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

### universal · T_C (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| T<800 | 7 | **303.32 [274.43, 325.74] (MLP/universal_raw)** |
| 800<=T<1000 | 97 | **99.87 [74.49, 126.08] (LightGBM/universal_raw)** |
| 1000<=T<1200 | 443 | **71.76 [65.19, 78.19] (CatBoost/universal_raw)** |
| 1200<=T<1400 | 317 | **77.27 [68.51, 85.84] (CatBoost/universal_raw)** |
| T>=1400 | 141 | **301.70 [261.93, 339.06] (ElasticNet/universal_raw)** |
| ALL | 1005 | **143.71 [127.78, 159.81] (RF/universal_raw)** |
