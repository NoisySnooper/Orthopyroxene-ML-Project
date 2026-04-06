**Notebooks 03–09 — Consolidated summary**

Date: 2026-04-05

**Notebook 03 — Model summary (opx_liq)**
- **T_C, RF:** RMSE_train = 35.551; RMSE_test = 124.018; R² = 0.442
- **T_C, ERT:** RMSE_train = 26.190; RMSE_test = 125.711; R² = 0.427
- **T_C, XGB:** RMSE_train = 6.323; RMSE_test = 123.512; R² = 0.447
- **T_C, GB:** RMSE_train = 35.197; RMSE_test = 124.399; R² = 0.439
- **P_kbar, RF:** RMSE_train = 1.864; RMSE_test = 6.145; R² = 0.621
- **P_kbar, ERT:** RMSE_train = 0.001; RMSE_test = 6.636; R² = 0.558
- **P_kbar, XGB:** RMSE_train = 0.003; RMSE_test = 6.427; R² = 0.585
- **P_kbar, GB:** RMSE_train = 1.726; RMSE_test = 5.566; R² = 0.689
- **Source:** [results/nb03_results_summary.csv](results/nb03_results_summary.csv)

**Notebook 04 — Putirka vs ML (head‑to‑head)**
- **T — Putirka iterative (28a+29a):** RMSE = 120.866; MAE = 61.141; R² = 0.472; n_test = 143
- **T — ML-RF opx-liq:** RMSE = 124.018; MAE = 67.925; R² = 0.442; n_test = 170
- **P — Putirka iterative (28a+29a):** RMSE = 5.402; MAE = 3.622; R² = 0.703; n_test = 143
- **P — ML-RF opx-liq:** RMSE = 6.145; MAE = 4.281; R² = 0.621; n_test = 170
- **Note:** Putirka rows were evaluated on 143 samples; ML rows in this file use 170 samples.
- **Source:** [results/nb04_putirka_comparison.csv](results/nb04_putirka_comparison.csv)

**Notebook 05 — LOSO statistics**
- **T (LOSO):** median RMSE = 58.427; mean RMSE = 79.906; IQR = 56.891
  - **Worst 5 studies (study — n_test — RMSE):**
    - Wasylenki et al. (2003) — 8 — 463.218
    - Wagner & Grove (1998) — 2 — 382.027
    - Xirouchakis et al. (2001) — 5 — 272.004
    - Agee & Draper (2004) — 4 — 230.200
    - Patiño Douce (1995) — 3 — 160.143
- **P (LOSO):** median RMSE = 3.053; mean RMSE = 4.276; IQR = 3.907
  - **Worst 5 studies (study — n_test — RMSE):**
    - Agee & Draper (2004) — 4 — 21.982
    - Walter (1998) — 6 — 13.660
    - Xirouchakis et al. (2001) — 5 — 12.422
    - Fonseca et al. (2014) — 3 — 10.487
    - Colson & Gust (1989) — 3 — 10.345
- **Source:** [results/nb05_loso_T.csv](results/nb05_loso_T.csv), [results/nb05_loso_P.csv](results/nb05_loso_P.csv)

**Notebook 06 — SHAP**
- **Top 8 features for P:** liq_SiO2, liq_MgO, Al_VI, liq_Al2O3, Al2O3, FeO_total, liq_K2O, liq_CaO
- **Top 8 features for T:** liq_MgO, liq_SiO2, liq_Al2O3, liq_CaO, liq_Mg_num, FeO_total, Al_VI, liq_K2O
- **Beeswarm figures:** [figures/fig07_shap_P_beeswarm.png](figures/fig07_shap_P_beeswarm.png), [figures/fig08_shap_T_beeswarm.png](figures/fig08_shap_T_beeswarm.png)
- **Source:** [results/table5_shap_importance.csv](results/table5_shap_importance.csv)

**Notebook 07 — Bias correction**
- **Numeric raw vs corrected RMSE:** not found in results CSVs (no explicit raw vs corrected rows located).
- **Residuals figure:** [figures/fig11_bias_correction_residuals.png](figures/fig11_bias_correction_residuals.png)
- **Corrected model artifacts:** [models/model_RF_P_kbar_opx_liq_corrected.joblib](models/model_RF_P_kbar_opx_liq_corrected.joblib), [models/model_RF_T_C_opx_liq_corrected.joblib](models/model_RF_T_C_opx_liq_corrected.joblib)

**Notebook 08 — Natural samples**
- **Predictions:** 50 samples; T_pred range ≈ 909.312–1562.143 °C; P_pred range ≈ 0.677–139.273 kbar
- **Predictions CSV:** [results/nb08_natural_predictions.csv](results/nb08_natural_predictions.csv)
- **Natural data source:** [data/natural/lin_2023_ne_china_peridotites.csv](data/natural/lin_2023_ne_china_peridotites.csv) (file present; earlier read returned as binary)
- **Geotherm figure:** [figures/fig12_natural_samples_geotherm.png](figures/fig12_natural_samples_geotherm.png)

**Notebook 09 — Figures for review**
- **First figure suggested for review:** [figures/fig01_pt_distribution.png](figures/fig01_pt_distribution.png)
- **Figures directory:** [figures](figures)

**Files referenced**
- [results/nb03_results_summary.csv](results/nb03_results_summary.csv)
- [results/nb03_results_all.csv](results/nb03_results_all.csv)
- [results/nb04_putirka_comparison.csv](results/nb04_putirka_comparison.csv)
- [results/comparison_P_predictions.csv](results/comparison_P_predictions.csv)
- [results/comparison_T_predictions.csv](results/comparison_T_predictions.csv)
- [results/nb05_loso_T.csv](results/nb05_loso_T.csv)
- [results/nb05_loso_P.csv](results/nb05_loso_P.csv)
- [results/table5_shap_importance.csv](results/table5_shap_importance.csv)
- [results/nb08_natural_predictions.csv](results/nb08_natural_predictions.csv)
- [figures/fig07_shap_P_beeswarm.png](figures/fig07_shap_P_beeswarm.png)
- [figures/fig08_shap_T_beeswarm.png](figures/fig08_shap_T_beeswarm.png)
- [figures/fig11_bias_correction_residuals.png](figures/fig11_bias_correction_residuals.png)
- [figures/fig12_natural_samples_geotherm.png](figures/fig12_natural_samples_geotherm.png)

**Next steps**
- Saved to [results/notebooks_03-09_summary.md](results/notebooks_03-09_summary.md).
- Options: send figures one-at-a-time for publication-quality review; export tables to CSV/Excel; or re-run specific notebooks to compute missing corrected RMSEs. Reply with your choice.
