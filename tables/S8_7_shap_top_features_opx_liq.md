# Table S8.7: opx-liq SHAP top features per canonical cell

Top 10 features by mean absolute SHAP value per canonical cell. Linear cells use `shap.LinearExplainer` (exact for linear models); tree cells use `shap.TreeExplainer`; the MLP cell uses `shap.KernelExplainer` with a k=50 KMeans background and a random 80-sample subset of the test set (seed = SEED_BOOTSTRAP).

| Cell | Rank | Feature | Mean abs SHAP |
|---|---|---|---|
| ElasticNet/raw/T_C | 1 | raw_liq_MgO | 64.1832 |
| ElasticNet/raw/T_C | 2 | raw_liq_FeO | 23.9922 |
| ElasticNet/raw/T_C | 3 | raw_FeO_total | 21.1280 |
| ElasticNet/raw/T_C | 4 | Al_VI | 20.2011 |
| ElasticNet/raw/T_C | 5 | Wo_frac | 14.0651 |
| ElasticNet/raw/T_C | 6 | raw_Na2O | 11.0766 |
| ElasticNet/raw/T_C | 7 | raw_liq_Al2O3 | 8.4950 |
| ElasticNet/raw/T_C | 8 | raw_liq_K2O | 7.6334 |
| ElasticNet/raw/T_C | 9 | raw_liq_Na2O | 6.2638 |
| ElasticNet/raw/T_C | 10 | raw_TiO2 | 5.7234 |
| ElasticNet/raw/P_kbar | 1 | raw_CaO | 11.3941 |
| ElasticNet/raw/P_kbar | 2 | Wo_frac | 9.6582 |
| ElasticNet/raw/P_kbar | 3 | Al_VI | 3.9818 |
| ElasticNet/raw/P_kbar | 4 | raw_liq_MgO | 3.3405 |
| ElasticNet/raw/P_kbar | 5 | raw_liq_SiO2 | 3.1458 |
| ElasticNet/raw/P_kbar | 6 | raw_liq_FeO | 2.2712 |
| ElasticNet/raw/P_kbar | 7 | liq_Mg_num | 2.0577 |
| ElasticNet/raw/P_kbar | 8 | raw_liq_CaO | 1.8008 |
| ElasticNet/raw/P_kbar | 9 | raw_TiO2 | 1.7154 |
| ElasticNet/raw/P_kbar | 10 | raw_liq_Al2O3 | 1.6173 |
| MLP/raw/P_kbar | 1 | raw_liq_SiO2 | 2.1709 |
| MLP/raw/P_kbar | 2 | raw_liq_Al2O3 | 1.7562 |
| MLP/raw/P_kbar | 3 | Al_VI | 1.6143 |
| MLP/raw/P_kbar | 4 | raw_Al2O3 | 1.1291 |
| MLP/raw/P_kbar | 5 | raw_liq_FeO | 0.9887 |
| MLP/raw/P_kbar | 6 | liq_Mg_num | 0.7674 |
| MLP/raw/P_kbar | 7 | raw_TiO2 | 0.7604 |
| MLP/raw/P_kbar | 8 | raw_liq_MgO | 0.7312 |
| MLP/raw/P_kbar | 9 | raw_Cr2O3 | 0.7231 |
| MLP/raw/P_kbar | 10 | raw_Na2O | 0.6986 |
| CatBoost/raw/P_kbar | 1 | raw_liq_SiO2 | 1.4915 |
| CatBoost/raw/P_kbar | 2 | raw_liq_MgO | 1.0240 |
| CatBoost/raw/P_kbar | 3 | raw_Na2O | 0.9728 |
| CatBoost/raw/P_kbar | 4 | raw_liq_Al2O3 | 0.8938 |
| CatBoost/raw/P_kbar | 5 | MgTs | 0.7112 |
| CatBoost/raw/P_kbar | 6 | Al_VI | 0.7099 |
| CatBoost/raw/P_kbar | 7 | raw_liq_CaO | 0.6384 |
| CatBoost/raw/P_kbar | 8 | raw_Al2O3 | 0.5605 |
| CatBoost/raw/P_kbar | 9 | raw_liq_K2O | 0.5042 |
| CatBoost/raw/P_kbar | 10 | Mg_num | 0.4551 |
| XGB/alr/T_C | 1 | alr_liq_MgO | 69.1233 |
| XGB/alr/T_C | 2 | alr_Na2O | 16.3464 |
| XGB/alr/T_C | 3 | alr_Cr2O3 | 8.8343 |
| XGB/alr/T_C | 4 | Wo_frac | 6.5515 |
| XGB/alr/T_C | 5 | alr_liq_FeO | 6.2348 |
| XGB/alr/T_C | 6 | alr_liq_CaO | 5.7262 |
| XGB/alr/T_C | 7 | alr_liq_Al2O3 | 5.7120 |
| XGB/alr/T_C | 8 | alr_liq_Na2O | 5.0490 |
| XGB/alr/T_C | 9 | alr_TiO2 | 4.7746 |
| XGB/alr/T_C | 10 | alr_FeO_total | 4.4798 |