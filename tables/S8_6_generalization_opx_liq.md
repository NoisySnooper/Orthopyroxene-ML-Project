# Table S8.6: opx-liq generalization (4 grouped CV strategies)

Pooled OOF RMSE and MAE with bootstrap 95% CI over out-of-fold predictions. Strategies are ordered from strictest to weakest generalization test. LOSO leaves one Citation out at a time (93 folds). ClusterKFold uses k=5 KMeans clusters on standardized composition. TargetBinKFold uses the pre-registered P-regime labels (0/5/15/30/100 kbar). LeaveOneRegionOut uses a petrologic study-type categorization inferred from Citation text.

| Model/feat | Target | Strategy | n | folds | RMSE [95% CI] | MAE [95% CI] |
|---|---|---|---|---|---|---|
| ElasticNet/raw | T_C | LOSO | 600 | 93 | 69.40 [62.51, 76.35] °C | 48.83 [45.17, 52.64] °C |
| ElasticNet/raw | T_C | ClusterKFold | 600 | 5 | 68.64 [62.03, 75.79] °C | 48.19 [44.56, 52.02] °C |
| ElasticNet/raw | T_C | TargetBinKFold | 600 | 4 | 70.95 [64.39, 77.82] °C | 50.13 [46.43, 54.22] °C |
| ElasticNet/raw | T_C | LeaveOneRegionOut | 600 | 7 | 69.90 [63.59, 76.65] °C | 48.33 [44.38, 52.26] °C |
| MLP/raw | P_kbar | LOSO | 600 | 93 | 4.59 [4.15, 5.06] kbar | 3.21 [2.97, 3.48] kbar |
| MLP/raw | P_kbar | ClusterKFold | 600 | 5 | 7.29 [6.65, 7.87] kbar | 5.46 [5.06, 5.83] kbar |
| MLP/raw | P_kbar | TargetBinKFold | 600 | 4 | 6.51 [6.11, 6.94] kbar | 5.21 [4.91, 5.53] kbar |
| MLP/raw | P_kbar | LeaveOneRegionOut | 600 | 7 | 5.88 [5.26, 6.50] kbar | 4.03 [3.70, 4.36] kbar |
| ElasticNet/raw | P_kbar | LOSO | 600 | 93 | 5.40 [4.90, 5.90] kbar | 3.84 [3.55, 4.14] kbar |
| ElasticNet/raw | P_kbar | ClusterKFold | 600 | 5 | 7.44 [6.88, 7.97] kbar | 5.84 [5.46, 6.18] kbar |
| ElasticNet/raw | P_kbar | TargetBinKFold | 600 | 4 | 7.11 [6.53, 7.64] kbar | 5.30 [4.94, 5.67] kbar |
| ElasticNet/raw | P_kbar | LeaveOneRegionOut | 600 | 7 | 6.17 [5.56, 6.80] kbar | 4.33 [3.96, 4.69] kbar |