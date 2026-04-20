# Terminology cleanup report

Generated 2026-04-20 by `scripts/audits/terminology_cleanup.py`.

## Replacement rules

- `\bv10 best\b` -> `best tuned model`
- `\bv10 RMSE\b` -> `tuned RMSE`
- `\bv10_pre\b` -> `pre-correction`
- `\bv10_corrected\b` -> `post-correction`
- `\bv10_post\b` -> `post-correction`
- `\braw v10\b` -> `raw (uncorrected) tuned baseline`
- `\bv10 pre\b` -> `pre-correction`
- `\bv10 post\b` -> `post-correction`
- `\bv10\b` -> `the tuned ML baseline`
- `\bopx-tb\b` -> `the opx ML thermobarometer`
- `\bopx_tb\b` -> `opx ML thermobarometer`

## Targets

- `manuscripts/**/*.md`
- `figures/**/*.txt`

## Files modified

### `figures\fig24_per_regime_rmse_opx_liq.txt`

- `\bv10 best\b` -> `best tuned model` (x1)

### `figures\fig25_per_regime_residual_violins_opx_liq.txt`

- `\bv10\b` -> `the tuned ML baseline` (x1)

### `figures\fig26_generalization_opx_liq.txt`

- `\bv10\b` -> `the tuned ML baseline` (x1)

### `figures\fig29_twopx_benchmark.txt`

- `\bv10\b` -> `the tuned ML baseline` (x2)

### `figures\fig34_bias_correction_scorecard_delta.txt`

- `\bv10_post\b` -> `post-correction` (x2)
- `\bv10 post\b` -> `post-correction` (x1)
- `\bv10\b` -> `the tuned ML baseline` (x1)

### `figures\fig35_tabpfn_vs_opx_tb.txt`

- `\bv10 pre\b` -> `pre-correction` (x1)
- `\bv10\b` -> `the tuned ML baseline` (x1)

### `figures\fig45_opx_only_P_headline.txt`

- `\bv10\b` -> `the tuned ML baseline` (x3)

### `manuscripts\opx_2026\text\aug_methods_3_9.md`

- `\bv10\b` -> `the tuned ML baseline` (x1)

### `manuscripts\opx_2026\text\bias_correction_autofilled.md`

- `\bv10\b` -> `the tuned ML baseline` (x4)

### `manuscripts\opx_2026\text\bias_correction_numbers.md`

- `\bv10_pre\b` -> `pre-correction` (x3)
- `\bv10_corrected\b` -> `post-correction` (x9)
- `\bv10_post\b` -> `post-correction` (x1)
- `\braw v10\b` -> `raw (uncorrected) tuned baseline` (x2)
- `\bv10\b` -> `the tuned ML baseline` (x2)

### `manuscripts\opx_2026\text\regime_results_autofilled.md`

- `\bv10 best\b` -> `best tuned model` (x2)
- `\bv10 RMSE\b` -> `tuned RMSE` (x3)
- `\bv10\b` -> `the tuned ML baseline` (x13)

### `manuscripts\opx_2026\text\tabpfn_paragraph.md`

- `\bv10 best\b` -> `best tuned model` (x1)
- `\bv10 RMSE\b` -> `tuned RMSE` (x1)
- `\bv10_pre\b` -> `pre-correction` (x1)
- `\bv10_corrected\b` -> `post-correction` (x1)
- `\bv10\b` -> `the tuned ML baseline` (x2)

## Intentionally NOT changed

- CSV column names (e.g., `v10_pre_rmse`, `v10_post_rmse` in `results/preregistered_scorecard_postcorrection.csv`). These are machine contracts consumed by code; renaming would break every downstream script.
- JSON keys in `results/bias_correction_shipped.csv` `form_a_params` and `form_b_params` fields.
- Log files under `logs/` (archival of what was run).
- Notebook code cells (code identifier names follow CSV columns).
- Script filenames (`opx_tb_nb03_*.py`) — these are stable path contracts; renaming would orphan log cross-references.