# tabpfn_head_to_head.csv column legend

Rewritten 2026-04-20. Columns previously prefixed `v10_best_*` have been
renamed to `tuned_best_*` for manuscript terminology consistency. The
`verdict` column values (`v10_wins`, `tabpfn_wins`, `competitive`) are
kept as-is because they are enum codes, not prose.

## Columns

| column | meaning |
| --- | --- |
| `track` | pipeline+feature track: opx_liq, opx_only, cpx_liq, cpx_only |
| `target` | T_C (temperature) or P_kbar (pressure) |
| `tuned_best_model` | winning model family from 20-seed multiseed summary |
| `tuned_best_fs` | winning feature set (raw / alr / pwlr) |
| `tuned_best_rmse` | 20-seed mean test RMSE for the winning tuned family |
| `tuned_best_std` | 20-seed std of test RMSE for the winning tuned family |
| `tabpfn_rmse` | 20-seed mean test RMSE for TabPFN (n_estimators=8, raw features) |
| `tabpfn_std` | 20-seed std of TabPFN test RMSE |
| `external_best_rmse` | best external benchmark RMSE (Putirka/Jorgenson/Agreda) when available |
| `external_best_method` | method name of the winning external benchmark |
| `external_best_family` | family of the winning external benchmark |
| `verdict` | comparison outcome: `v10_wins` (tuned > TabPFN), `tabpfn_wins`, `competitive` |

## Verdict rule

`verdict` derives from per-track delta and joint std:

```
delta = tabpfn_rmse - tuned_best_rmse
tol   = 0.5 * max(tuned_best_std, tabpfn_std)
if abs(delta) <= tol:  'competitive'
elif delta > 0:        'v10_wins'     # tuned is lower RMSE
else:                  'tabpfn_wins'  # TabPFN is lower RMSE
```

Note the code enum `v10_wins` is retained for backward compat with two
downstream scripts (`scripts/tabpfn/opx_tb_nb03_*`). Display code maps
it to the string "Tuned wins" at render time.
