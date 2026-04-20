# Terminology cleanup report v2 (two-pass)

Generated 2026-04-20 by `scripts/audits/terminology_cleanup.py`.

## Pass 1 rules (compound, longest-first)

- `\bv10 tuned pre\b` -> `pre-correction tuned baseline`
- `\bv10 tuned post\b` -> `post-correction tuned baseline`
- `\bv10_post[-\s]correction\b` -> `post-correction`
- `\bv10 post[-\s]correction\b` -> `post-correction`
- `\bv10 pre[-\s]registered\b` -> `pre-registered`
- `\bv10 pre[-\s]correction\b` -> `pre-correction`
- `\bv10 best\b` -> `best tuned model`
- `\bv10 RMSE\b` -> `tuned RMSE`
- `\bv10_pre\b` -> `pre-correction`
- `\bv10_corrected\b` -> `post-correction`
- `\bv10_post\b` -> `post-correction`
- `\braw v10\b` -> `raw (uncorrected) tuned baseline`
- `\bv10 pre\b` -> `pre-correction`
- `\bv10 post\b` -> `post-correction`
- `\bv10 pipeline\b` -> `the tuned ML pipeline`
- `\bv10\b` -> `the tuned ML baseline`
- `\bv9 finding\b` -> `earlier finding`
- `\bv9 empty\b` -> `earlier empty`
- `\bv9\b` -> `the previous iteration`
- `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase`
- `\bPhase[- ]?G\b` -> `the final phase`
- `\bPhase[- ]?\d+\.?\d*[a-z]?\b` -> `the relevant phase`
- `\bphase g\.?\d+[a-z]?\b` -> `the final phase`
- `\btwo-axis honesty bar\b` -> `dual robustness check`
- `\bno-degradation rule\b` -> `no-regime-worsens requirement`
- `\b"?ships A/B"?\b` -> `accepts Form A or Form B`
- `\b"?ships A"?\b` -> `accepts Form A`
- `\b"?ships B"?\b` -> `accepts Form B`
- `\bopx-tb\b` -> `the opx ML thermobarometer`
- `\bopx_tb\b` -> `opx ML thermobarometer`

## Pass 2 rules (cosmetic fixups)

- `\bthe the tuned ML baseline\b` -> `the tuned ML baseline`
- `\bthe the tuned\b` -> `the tuned`
- `\bthe the final phase\b` -> `the final phase`
- `\bthe the relevant phase\b` -> `the relevant phase`
- `\bthe the previous iteration\b` -> `the previous iteration`
- `\bthe the\b` -> `the`
- `\bpost-correction-correction\b` -> `post-correction`
- `\bpre-correction-correction\b` -> `pre-correction`
- `\bpre-correction-registered\b` -> `pre-registered`
- `\btuned ML baseline tuned pre-correction\b` -> `pre-correction tuned baseline`
- `\btuned ML baseline tuned post-correction\b` -> `post-correction tuned baseline`
- `\btuned ML baseline tuned (pre|post)-correction\b` -> `\1-correction tuned baseline`

## Targets

- `manuscripts/**/*.md`
- `figures/**/*.txt`
- `deliverables\lee_package_20260420\figures/**/*.txt`

## Files modified

Total: 19 files, 46 replacements.

### `deliverables\lee_package_20260420\figures\fig28_bias_correction_opx_liq.txt`

- [pass 1] `\bv9 finding\b` -> `earlier finding` (x1)

### `deliverables\lee_package_20260420\figures\fig30_bias_correction_per_regime_rmse.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x2)
- [pass 1] `\b"?ships A/B"?\b` -> `accepts Form A or Form B` (x1)
- [pass 2] `\bthe the final phase\b` -> `the final phase` (x1)

### `deliverables\lee_package_20260420\figures\fig31_bias_correction_residuals.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x1)

### `deliverables\lee_package_20260420\figures\fig32_bias_correction_form_comparison.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x2)
- [pass 2] `\bthe the final phase\b` -> `the final phase` (x1)

### `deliverables\lee_package_20260420\figures\fig34_bias_correction_scorecard_delta.txt`

- [pass 2] `\bpost-correction-correction\b` -> `post-correction` (x1)

### `deliverables\lee_package_20260420\figures\fig35_tabpfn_vs_opx_tb.txt`

- [pass 2] `\bpre-correction-registered\b` -> `pre-registered` (x1)

### `deliverables\lee_package_20260420\figures\fig45_opx_only_P_headline.txt`

- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x1)
- [pass 2] `\btuned ML baseline tuned pre-correction\b` -> `pre-correction tuned baseline` (x1)
- [pass 2] `\btuned ML baseline tuned post-correction\b` -> `post-correction tuned baseline` (x1)

### `figures\fig25_per_regime_residual_violins_opx_liq.txt`

- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x1)

### `figures\fig28_bias_correction_opx_liq.txt`

- [pass 1] `\bv9 finding\b` -> `earlier finding` (x1)

### `figures\fig29_twopx_benchmark.txt`

- [pass 1] `\bv9 empty\b` -> `earlier empty` (x1)
- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x1)
- [pass 2] `\bthe the final phase\b` -> `the final phase` (x1)

### `figures\fig30_bias_correction_per_regime_rmse.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x2)
- [pass 1] `\b"?ships A/B"?\b` -> `accepts Form A or Form B` (x1)
- [pass 2] `\bthe the final phase\b` -> `the final phase` (x1)

### `figures\fig31_bias_correction_residuals.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x1)

### `figures\fig32_bias_correction_form_comparison.txt`

- [pass 1] `\bPhase[- ]?G\.?\d+[a-z]?\b` -> `the final phase` (x2)
- [pass 2] `\bthe the final phase\b` -> `the final phase` (x1)

### `figures\fig34_bias_correction_scorecard_delta.txt`

- [pass 2] `\bpost-correction-correction\b` -> `post-correction` (x1)

### `figures\fig35_tabpfn_vs_opx_tb.txt`

- [pass 2] `\bpre-correction-registered\b` -> `pre-registered` (x1)

### `figures\fig45_opx_only_P_headline.txt`

- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x1)
- [pass 2] `\btuned ML baseline tuned pre-correction\b` -> `pre-correction tuned baseline` (x1)
- [pass 2] `\btuned ML baseline tuned post-correction\b` -> `post-correction tuned baseline` (x1)

### `manuscripts\opx_2026\text\bias_correction_autofilled.md`

- [pass 1] `\bno-degradation rule\b` -> `no-regime-worsens requirement` (x2)

### `manuscripts\opx_2026\text\bias_correction_numbers.md`

- [pass 1] `\b"?ships A"?\b` -> `accepts Form A` (x2)

### `manuscripts\opx_2026\text\regime_results_autofilled.md`

- [pass 1] `\btwo-axis honesty bar\b` -> `dual robustness check` (x6)
- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x4)

## Intentionally NOT changed

- CSV column names (e.g., `v10_pre_rmse`, `v10_post_rmse`). Machine contracts.
- JSON keys in `form_a_params` / `form_b_params` fields.
- Log files under `logs/` (archival of what was run).
- Notebook code cells (identifiers follow CSV columns).
- Script filenames (`opx_tb_nb03_*.py`).
- The phrase `ship-if-better` itself (user preference: clearer).