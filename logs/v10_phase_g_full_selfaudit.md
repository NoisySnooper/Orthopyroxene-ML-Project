# Phase G full self-audit (G.2 - G.6)

Summary: 30/30 checks pass.

| # | Check | Pass | Detail |
|---|---|---|---|
| 1 | G.2a. generalization CSV exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/results/v10_opx_liq_generalization.csv |
| 2 | G.2b. generalization has >= 12 rows (3 cells x 4 strategies) | PASS | rows=12 |
| 3 | G.2c. generalization has required columns | PASS | [] |
| 4 | G.2d. all 4 strategies present | PASS | [] |
| 5 | G.2e. generalization predictions CSV non-trivial | PASS | 789657B |
| 6 | G.3a. SHAP importance CSV exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/results/v10_opx_liq_shap_importance.csv |
| 7 | G.3b. SHAP importance has >= 5 unique cells | PASS | cells=['CatBoost/raw/P_kbar', 'ElasticNet/raw/P_kbar', 'ElasticNet/raw/T_C', 'MLP/raw/P_kbar', 'XGB/alr/T_C'] |
| 8 | G.3c. SHAP values npz non-trivial | PASS | 96158B |
| 9 | G.3d. Table S8.7 markdown exists | PASS | 2862 |
| 10 | G.3e. Table S8.7 csv exists | PASS | 3322 |
| 11 | G.4a. bias-correction CSV exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/results/v10_opx_liq_bias_correction.csv |
| 12 | G.4b. bias-correction has required columns | PASS | [] |
| 13 | G.4c. bias-correction has ALL-regime rows | PASS | n_all=3 |
| 14 | G.4d. MLP P_kbar ALL improves (improvement > 0) | PASS | improvement=1.2330538725764004 |
| 15 | G.4e. ElasticNet P_kbar ALL improves (improvement > 0) | PASS | improvement=2.5325668635464957 |
| 16 | G.4f. bias-correction params CSV non-trivial | PASS | 1034B |
| 17 | G.5a. twopx benchmark CSV exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/results/v10_twopx_benchmark_final.csv |
| 18 | G.5b. twopx benchmark has exactly 6 rows (2 targets x 3 method_class) | PASS | rows=6 |
| 19 | G.5c. twopx benchmark has all 3 method_classes | PASS | [] |
| 20 | G.5d. Putirka row is NaN (deferred) | PASS | n_putirka=2, rmse_nan=True |
| 21 | G.5e. Table S8.9 markdown non-trivial | PASS | 1353B |
| 22 | G.6a. figure audit CSV exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/results/v10_phase_g_figure_audit.csv |
| 23 | G.6b. figure audit has 6 rows (fig24-fig29) | PASS | rows=6 |
| 24 | G.6c. all figure audit rows PASS | PASS | n_fail=0 |
| 25 | G.6d. fig24_per_regime_rmse_opx_liq pdf+png+txt | PASS | pdf=True, png=True, txt=True |
| 26 | G.6d. fig25_per_regime_residual_violins_opx_liq pdf+png+txt | PASS | pdf=True, png=True, txt=True |
| 27 | G.6d. fig26_generalization_opx_liq pdf+png+txt | PASS | pdf=True, png=True, txt=True |
| 28 | G.6d. fig27_shap_summary_opx_liq pdf+png+txt | PASS | pdf=True, png=True, txt=True |
| 29 | G.6d. fig28_bias_correction_opx_liq pdf+png+txt | PASS | pdf=True, png=True, txt=True |
| 30 | G.6d. fig29_twopx_benchmark pdf+png+txt | PASS | pdf=True, png=True, txt=True |
