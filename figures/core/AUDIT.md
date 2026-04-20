# Core figure audit (Phase 5, Tier 1)

Entry template:

```
Core_NN_<stem>
  Caption claim: "..."
  Data source:   results/XYZ.csv (or data/processed/*.parquet)
  Models shown:  [list]
  TabPFN:        yes | no | N/A (reason)
  Regression:    matches caption; RMSE within 0.01 of CSV; winner verdict matches scoreboard
  Visual check:  yes | no
  Notes:         (any caveats)
```

Status codes: `[new]` built in Tier 1, `[rebuild]` rebuild in Tier 2,
`[pending-audit]` figure exists but not yet audited.

---

## Core_01_fig_dataset_map  [new]

- Caption claim: "P-T distribution of all experiments in ExPetDB 2025-07-21 across 4 pyroxene tracks, colored by pre-registered P regime"
- Data source:   data/processed/{opx,cpx}_clean_{opx_liq,opx_only,cpx_liq,cpx_only}.parquet; data/splits/*_indices_*.npy
- Models shown:  N/A (pure data figure)
- TabPFN:        N/A
- Regression:    n per track matches parquet row counts (600 / 1035 / 2385 / 2897); regime edges [0,5,15,30,100] kbar match docs/preregistration/p_regime_preregistration.md
- Visual check:  pending
- Notes:         train/test marker distinction via alpha + edge; regime boundary lines at 5/15/30 kbar

## Core_02_fig_citation_split  [new]

- Caption claim: "Citation-grouped 10-fold split vs naive row-random split; illustrates why within-publication grouping is needed"
- Data source:   schematic (no CSV)
- Models shown:  N/A
- TabPFN:        N/A
- Regression:    schematic only; text claims StratifiedGroupKFold, 10 folds, min_train_fold=50 must match what nb01/nb03 actually use
- Visual check:  pending
- Notes:         uses 30x6 grid for visualization clarity; actual opx corpus has ~93 publications

## Core_03_fig_methods_flowchart  [new]

- Caption claim: "10-stage pipeline from ExPetDB raw to per-regime test evaluation; 9 model families; 8 tuned + TabPFN untuned"
- Data source:   schematic (no CSV)
- Models shown:  9 families (ElasticNet, RF, ERT, GB, XGB, LightGBM, CatBoost, MLP, TabPFN)
- TabPFN:        yes (called out in stage 4 and as stage-5 alt path)
- Regression:    stage numbering matches methods prose in CAVEATS / manuscript; Optuna 200 trials seed 42 + TabPFN default n_estimators=8 match Phase 1 actuals
- Visual check:  pending
- Notes:         arrows adjusted manually; inspect for overlap at stages 5-6

## Core_09_fig_opx_four_combos  [new]

- Caption claim: "4-panel per-regime RMSE across 5 candidates; only panel (d) opx_only P ships a post-correction win"
- Data source:   results/preregistered_scorecard_postcorrection.csv, results/bias_correction_shipped.csv
- Models shown:  tuned pre/post, Putirka, TabPFN pre/post
- TabPFN:        yes (2 of 5 bars per regime)
- Regression:    winner abbreviations (Tp/TP/PU/TFp/TFP) match scorecard 'winner' field; ship box matches 'ship_a' OR 'ship_b' in bias_correction_shipped.csv per (track,target,model)
- Visual check:  pending
- Notes:         opx_only T_C has no external benchmark (NaN); Putirka bars absent for (c)

## Core_10_fig_temperature_null  [new]

- Caption claim: "Temperature null: neither tuned nor TabPFN corrections ship on opx_liq T; TabPFN ships marginally on opx_only T deeper_mantle"
- Data source:   results/preregistered_scorecard_postcorrection.csv, results/bias_correction_shipped.csv
- Models shown:  tuned pre/post, Putirka (opx_liq only), TabPFN pre/post
- TabPFN:        yes (2 of 5 bars per regime)
- Regression:    verdict box reflects actual shipped.csv rows; opx_only T TabPFN Form A ships per Phase 1 outcome
- Visual check:  pending
- Notes:         honest complement to Core_08; both should be shown in paper

---

## Existing figures awaiting audit in Tier 2

These are already on disk at figures/ (flat). They will be re-audited
and, where needed, rebuilt to include TabPFN in the candidate pool.

- Core_04_fig_nb04_cross_pipeline_heatmap  [pending-audit, may rebuild]
- Core_05_fig30_bias_correction_per_regime_rmse  [pending-audit]
- Core_06_fig31_bias_correction_residuals  [pending-audit, may rebuild to add TabPFN overlay]
- Core_07_fig34_bias_correction_scorecard_delta  [pending-audit, may rebuild]
- Core_08_fig45_opx_only_P_headline  [pending-audit]
- Core_11_fig_nb08_twopx_1to1  [pending-audit]

---

## SI figures (not in this audit pass)

fig24, fig25, fig26, fig27, fig28, fig29, fig30, fig31, fig32, fig33, fig34, fig35, fig44, fig_aug01, fig_aug02, fig_aug03, fig_aug04 all remain in figures/ (flat) as SI set. Full SI audit deferred to Tier 2.
