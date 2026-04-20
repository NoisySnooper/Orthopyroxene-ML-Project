# Data sources

All numeric assertions in `00_ADVISOR_REVIEW.ipynb` resolve to rows
or columns of the CSVs below. Every cell loads via `load_csv()` /
`load_json()` which raise `FileNotFoundError` rather than fabricating.

- `results/preregistered_scorecard_postcorrection.csv`
- `results/bias_correction_shipped.csv`
- `results/opx_multiseed_summary.csv`
- `results/tabpfn_bias_correction_summary.csv`
- `results/tabpfn_bias_correction_perseed.csv`
- `results/tabpfn_multiseed_summary.csv`
- `results/tabpfn_head_to_head.csv`
- `results/nb08_natural_predictions.csv`
- `results/nb08_cross_mineral_agreement.csv`
- `results/regime_allmodels_postcorrection.csv`