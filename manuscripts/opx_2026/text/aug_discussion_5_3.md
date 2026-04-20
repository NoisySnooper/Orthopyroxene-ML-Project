### Section 5.3 Augmentation sensitivity (Discussion, autofilled)

The augmentation hypothesis is not supported. Form B fails to ship on 0/4 opx combinations under the Agreda-Lopez 15x Gaussian augmentation protocol, and augmentation additionally degrades aggregate RMSE on every combination. For small experimental petrology datasets with publication-level clustering, 15x Gaussian composition noise is counterproductive. The differentiator between our negative Form B result and Agreda-Lopez positive one must be mineral-specific (orthopyroxene vs clinopyroxene residual structure) or protocol-specific beyond augmentation alone.

Per-combination headline (best cell under each protocol):

- **opx_liq/T_C**: non-aug ElasticNet/raw RMSE 77.06+/-0.00 (canon winner: excluded); aug ElasticNet/raw RMSE 77.19+/-0.11; Form A ships 0/20 seeds, Form B ships 0/20 seeds.
- **opx_liq/P_kbar**: non-aug MLP/raw RMSE 4.40+/-0.46 (canon winner: excluded); aug XGB/raw RMSE 4.76+/-0.08; Form A ships 17/20 seeds, Form B ships 0/20 seeds.
- **opx_only/T_C**: non-aug LightGBM/alr RMSE 146.63+/-0.59 (canon winner: excluded); aug XGB/raw RMSE 153.78+/-0.78; Form A ships 20/20 seeds, Form B ships 4/20 seeds.
- **opx_only/P_kbar**: non-aug RF/pwlr RMSE 10.35+/-0.04 (canon winner: A); aug ERT/pwlr RMSE 11.12+/-0.11; Form A ships 20/20 seeds, Form B ships 0/20 seeds.

Across all 80 augmented (combo, seed) cells: Form A wins 57, Form B wins 0, neither ships 23.
See fig_aug01 for ship-verdict comparison, fig_aug02 for aggregate RMSE delta, fig_aug03 for residual structure per regime on opx-only P_kbar, and fig_aug04 for Form B breakpoint stability.