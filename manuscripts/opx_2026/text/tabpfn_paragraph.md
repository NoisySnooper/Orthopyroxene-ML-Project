# TabPFN v2 supplementary baseline (drop-in for Section 5 Discussion)

*Auto-filled from `results/tabpfn_head_to_head.csv` and `results/tabpfn_multiseed_summary.csv`. Numbers are rendered to 2 dp for kbar and 1 dp for C. Regenerate with `python scripts/tabpfn/opx_tb_nb03_fill_tabpfn_paragraph.py` after any baseline rerun.*

## Paragraph 1 — framing

Foundation models for small tabular data have become a standard reference point following Hollmann et al. (2025, *Nature* 637, doi:10.1038/s41586-024-08328-6), whose TabPFN v2 was pretrained on millions of synthetic tabular priors and, on public small-data regression benchmarks, matched or exceeded tuned ensembles without any hyperparameter search. To keep our comparison faithful to that paper's evaluation, we ran TabPFN v2 against every (pipeline, track, target) cell in the pre-registered v10 roster. The baseline uses CPU inference (no GPU available), raw oxide features only (the model autoscales and handles missingness internally, so the ALR and PWLR feature sets are not meaningful axes), and 20 seeds (42-61), matching the seed protocol used by the tuned families so stability estimates are on an equal footing. No Optuna tuning. No stacking. No SHAP. We evaluate the original TabPFN v2 release (Hollmann et al. 2025), not the subsequent TabPFN-2.5 release (Grinsztajn et al. 2025, arXiv:2511.08667), to maintain consistency with the peer-reviewed Nature publication and to comply with the permissive license associated with the v2 weights.

## Paragraph 2 — head-to-head numbers

Table A summarizes the head-to-head. Reading across the eight cells:


| track | target | v10 best | v10 RMSE | TabPFN RMSE | external best | verdict |
|---|---|---|---|---|---|---|
| cpx_liq | P_kbar | LightGBM/pwlr | 6.55 +/- 0.05 kbar | 7.51 +/- 0.13 kbar | -- | v10_wins |
| cpx_liq | T_C | ERT/pwlr | 72.5 +/- 0.2 C | 70.1 +/- 1.0 C | -- | tabpfn_wins |
| cpx_only | P_kbar | MLP/alr | 13.66 +/- 0.28 kbar | 13.41 +/- 0.58 kbar | -- | competitive |
| cpx_only | T_C | ERT/pwlr | 127.0 +/- 0.3 C | 130.6 +/- 3.0 C | -- | v10_wins |
| opx_liq | P_kbar | MLP/raw | 4.40 +/- 0.46 kbar | 5.46 +/- 0.14 kbar | -- | v10_wins |
| opx_liq | T_C | ElasticNet/raw | 77.1 C | 84.5 +/- 0.8 C | -- | v10_wins |
| opx_only | P_kbar | RF/pwlr | 10.35 +/- 0.04 kbar | 12.52 +/- 0.16 kbar | -- | v10_wins |
| opx_only | T_C | LightGBM/alr | 146.6 +/- 0.6 C | 150.6 +/- 1.5 C | -- | v10_wins |

## Paragraph 3 — interpretation

Across the 8 cells, TabPFN wins 1, v10 wins 6, 1 are competitive within 0.5 standard errors. Domain-specific feature engineering and hyperparameter tuning still matter at n between 600 and 2400 for experimental petrology. TabPFN's synthetic-data priors cover generic tabular nonlinearities well but may not capture silicate melt compositional manifolds, where feature sparsity and citation-group clustering create structure that the pretraining distribution did not sample. This is consistent with the ordinary-pattern-versus-specialty-data distinction drawn in the TabPFN paper's Discussion.

## Caveats (for internal use; not for the paper body)

- TabPFN is deliberately outside `BASE_ORDER`; the pre-registered model roster was locked 2026-04-17 and adding TabPFN would break registration discipline.
- Regime-stratified results for TabPFN are in `results/tabpfn_regime_rmse.csv` but are not spliced into the main scorecard because that scorecard tests the v10 tuned pipeline specifically.
- The full sample-by-sample test predictions (one row per test sample per seed) are in `results/tabpfn_predictions.csv` for independent re-analysis, but are not referenced from any nb04/nbF/nb09 cell.
