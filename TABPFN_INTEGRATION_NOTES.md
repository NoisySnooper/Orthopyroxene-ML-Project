# TabPFN Option B integration — working notes

Date started: 2026-04-19. Scope: promote TabPFN v2 from supplementary
baseline to the 9th first-class model family alongside the 8 Optuna-
tuned families. 11 commits planned (C1-C11).

Safety tag at pre-C1 HEAD: `pre-tabpfn-option-b-20260419` (at commit
`541fe9d`, Phase 1.5 C19 end).

## C1 deliverables

### Schema audit: TabPFN vs canonical

Compared `results/tabpfn_*.csv` against the canonical tuned-family
tables. Summary:

| TabPFN CSV | Canonical counterpart | Schema verdict |
|---|---|---|
| `tabpfn_multiseed_summary.csv` | `opx_multiseed_summary.csv`, `cpx_multiseed_summary.csv` | Exact match: `['pipeline','model','target','track','feature_set','mean','std','min','max','count']`. Direct concat safe. |
| `tabpfn_multiseed_results.csv` | `opx_multiseed_results.csv`, `cpx_multiseed_results.csv` | Superset: canonical has 7 cols `['pipeline','model','target','track','feature_set','seed','test_rmse']`; TabPFN has 12 (extras: `rmse`, `mae`, `r2`, `n_train`, `n_test`). C3 merge must project to the canonical 7-col shape OR canonical must absorb the 5 extras. Recommendation: project TabPFN to 7 cols on merge; write the extras to a separate `tabpfn_per_seed_metrics.csv` if we want them preserved. |
| `tabpfn_regime_rmse.csv` | `regime_allmodels.csv` | Exact match: 18 columns identical. `method_family='tabpfn'`, `method='TabPFN/raw'`, `method_label='TabPFN (foundation model)'` already set. Direct concat safe. Do NOT merge into `regime_allmodels_postcorrection.csv` (TabPFN is excluded from bias correction per C5). |

### Seed count change (carried over from Phase 1.5 C17)

User changed TabPFN protocol: 5 seeds (42-46) → 20 seeds (42-61),
matching the tuned families for an apples-to-apples stability
comparison.

**State on disk at Option B C1 start:**

- Script default (`scripts/tabpfn/opx_tb_nb03_tabpfn_baseline.py`): 20
  seeds. ✓
- Checkpoints (`results/tabpfn_checkpoints/`): 40 files covering
  s42-s46 across 8 combos. Missing: s47-s61.
- CSVs on disk: aggregated from 5 seeds. Shapes:
  - `tabpfn_multiseed_results.csv`: (40, 12) — target (160, 12) after 20-seed regen
  - `tabpfn_multiseed_summary.csv`: (8, 10) with `count=5` — target `count=20`
  - `tabpfn_regime_rmse.csv`: (40, 18) with 5-seed predictions mean — target: same shape, updated mean
  - `tabpfn_predictions.csv`: per-seed-per-sample; regenerates fully

### Regen invocation

```
PYTHONPATH=. .venv-tabpfn/Scripts/python.exe \
  scripts/tabpfn/opx_tb_nb03_tabpfn_baseline.py --pipeline all
```

Default seeds = 42-61. Existing checkpoints s42-s46 load from disk
(fast); 15 new seeds × 8 combos = 120 new fits.

CPU runtime envelope (rough): `n_estimators=8` for opx (n_train≈600-1000),
`n_estimators=4` for cpx (n_train≈2400). Expect several hours end-to-end.

**Log**: `logs/v10_nb03_tabpfn.log` (append-mode).

## C2 plan — register TabPFN in `BASE_ORDER`

- `src/models.py`: add `TabPFN` entry to `BASE_MODELS` and `MODEL_CLASSES`.
  Do NOT add to `PIPELINE_MODELS` (TabPFN doesn't go through the tuned
  Optuna pipeline).
- `PARAM_GRIDS`: empty dict for TabPFN (no tunable hyperparameters in
  the Hollmann protocol).
- Change `BASE_ORDER` from 8 to 9 entries: add `'TabPFN'` at the end.
- `STACKING_BASE_ORDER` stays 4 (do NOT add TabPFN; intentional subset).
- Test: `tests/test_tabpfn_registry.py` asserting the 9-tuple.
- Scan consumers for hardcoded 8-length checks; update to 9.
- Pytest gate: expect 55 → 56 tests after adding the new one.

## C3 plan — merge 20-seed TabPFN rows into canonical multiseed CSVs

Blocked until regen completes. Merge strategy:

1. Read `tabpfn_multiseed_summary.csv` (8 rows after regen).
2. Split by pipeline (`opx`/`cpx`) and append to the corresponding
   `{pipeline}_multiseed_summary.csv` (target shape: 96 + 4 = 100 rows
   per pipeline).
3. Read `tabpfn_multiseed_results.csv`, project to 7 canonical columns
   (drop `rmse, mae, r2, n_train, n_test`), split by pipeline, append
   to `{pipeline}_multiseed_results.csv` (1920 + 80 = 2000 rows each).

## C4 plan — merge TabPFN regime rows

Read `tabpfn_regime_rmse.csv` (40 rows), append to
`regime_allmodels.csv`. Skip `regime_allmodels_postcorrection.csv`.

## C5 plan — exclude TabPFN from bias correction

TabPFN has no OOF residuals (single in-context forward pass, no
training step that yields holdout residuals across folds). 8 rows to
`bias_correction_shipped.csv`:

```
pipeline=opx|cpx, track, target, model='TabPFN',
feature_set='raw', canonical_seed=NaN, winner='excluded',
reason='no OOF residuals available — in-context model'
```

## C6-C11 plan

See TabPFN Option B master prompt. To be detailed as each commit is
executed.

## Self-audit checklist

- [x] Safety tag set: `pre-tabpfn-option-b-20260419` at `541fe9d`
- [x] Schema audit complete; TabPFN CSVs compatible
- [ ] 20-seed regen in progress (background)
- [ ] Pytest 55/55 (Phase 1.5 baseline) → expect 56/56 after C2
- [x] TabPFN not in `STACKING_BASE_ORDER` (locked)
- [x] TabPFN excluded from bias correction post-correction tables
- [ ] Winner logic extended for TabPFN in C7
