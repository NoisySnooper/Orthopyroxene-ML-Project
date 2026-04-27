# Terminology cleanup report v3 (three-pass)

Generated 2026-04-20 by `scripts/audits/terminology_cleanup_v3.py`.

## Pass 0 rules (frankenword pre-clean, NEW in v3)

- `\bpre-registered the tuned ML baseline\b` -> `pre-registered ML baseline`
- `\buncorrected the tuned ML baseline predictions\b` -> `uncorrected predictions`
- `\buncorrected the tuned ML baseline prediction\b` -> `uncorrected prediction`
- `\bour the tuned ML baseline\b` -> `our tuned ML baseline`
- `\bthe the tuned ML pipeline\b` -> `the tuned ML pipeline`
- `\ba the tuned ML baseline\b` -> `a tuned ML baseline`
- `\ban the tuned ML baseline\b` -> `a tuned ML baseline`
- `\bthe tuned ML baseline baseline\b` -> `tuned ML baseline`

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
- `\bthe tuned ML baseline post-correction\b` -> `tuned ML baseline post-correction`
- `\bthe tuned ML baseline pre-correction\b` -> `tuned ML baseline pre-correction`
- `\bthe tuned ML baseline tuned\b` -> `tuned ML baseline`

## Targets

- `manuscripts/**/*.md`
- `figures/**/*.txt`
- `deliverables\lee_package_20260420\figures/**/*.txt`

## Files modified

Total: 9 files, 18 replacements.

### `deliverables\lee_package_20260420\figures\Core_05_fig30_bias_correction_per_regime_rmse.txt`

- [pass 1] `\b"?ships A"?\b` -> `accepts Form A` (x1)
- [pass 1] `\b"?ships B"?\b` -> `accepts Form B` (x1)

### `deliverables\lee_package_20260420\figures\Core_07_fig34_bias_correction_scorecard_delta.txt`

- [pass 1] `\bv10 post\b` -> `post-correction` (x1)
- [pass 1] `\bv10\b` -> `the tuned ML baseline` (x2)
- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x1)

### `figures\core\Core_05_fig30_bias_correction_per_regime_rmse.txt`

- [pass 1] `\b"?ships A"?\b` -> `accepts Form A` (x1)
- [pass 1] `\b"?ships B"?\b` -> `accepts Form B` (x1)

### `figures\core\Core_07_fig34_bias_correction_scorecard_delta.txt`

- [pass 1] `\bv10 post\b` -> `post-correction` (x1)
- [pass 1] `\bv10\b` -> `the tuned ML baseline` (x2)
- [pass 2] `\bthe the tuned ML baseline\b` -> `the tuned ML baseline` (x1)

### `figures\core\Core_08_fig45_opx_headline.txt`

- [pass 1] `\bv10 tuned pre\b` -> `pre-correction tuned baseline` (x1)
- [pass 1] `\bv10 tuned post\b` -> `post-correction tuned baseline` (x1)

### `figures\core\Core_12_fig_shap_winners.txt`

- [pass 1] `\bPhase[- ]?\d+\.?\d*[a-z]?\b` -> `the relevant phase` (x1)

### `manuscripts\opx_2026\text\aug_methods_3_9.md`

- [pass 0] `\bpre-registered the tuned ML baseline\b` -> `pre-registered ML baseline` (x1)

### `manuscripts\opx_2026\text\bias_correction_autofilled.md`

- [pass 0] `\buncorrected the tuned ML baseline predictions\b` -> `uncorrected predictions` (x1)

### `manuscripts\opx_2026\text\tabpfn_paragraph.md`

- [pass 0] `\bpre-registered the tuned ML baseline\b` -> `pre-registered ML baseline` (x1)
