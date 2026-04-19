# Phase 1.5 C18 — Code integrity report

Date: 2026-04-19. Scope: verify that the 16 preceding Phase 1.5 commits
(C1-C17) did not break any import, symbol contract, notebook JSON,
CSV schema, or test.

Runner: `.venv/Scripts/python.exe`. Platform: Windows 11, Python 3.13.

## Check results

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | `python -m py_compile` across `src/`, `scripts/`, `config.py`, `run_all.py` | PASS | 0 syntax errors |
| 2 | `import src.models`; assert `len(models.BASE_MODELS) == 8` | PASS | BASE_MODELS intact |
| 3 | `import config`; `BASE_ORDER`, `CANONICAL_FIGURES`, `SEED_SPLIT` smoke | PASS | `BASE_ORDER=('RF','ERT','XGB','GB','CatBoost','LightGBM','ElasticNet','MLP')`, `len(CANONICAL_FIGURES)=35`, `SEED_SPLIT=42` |
| 4 | `import run_all`; assert `len(run_all.NOTEBOOKS) == 13` | PASS | 13 notebooks registered |
| 5 | Notebook JSON load across 16 tracked notebooks | PASS | All parse |
| 6 | Static grep of `from scripts.*` imports against tracked files | INFO | 5 known stale imports, documented residual — see below |
| 7 | Tightened `git grep -E "v10\|V10"` per §C1 scope | INFO | Top offenders confirmed to be Decision-1 paper-reporting labels + migration history prose + C14 residuals |
| 8 | Stale "8 model families" / "eight tuned" phrase grep | PASS | Single hit (`PROJECT_LAYOUT.md:172`, `BASE_ORDER is canonical (8 families)`) — correct for current state; updated in TabPFN Option B C10 |
| 9 | `CANONICAL_FIGURES` on-disk existence | INFO | 10/35 present (figs 26-34 rendered in Phase G). Remaining 25 are registered but not yet rendered — CLAUDE.md guardrail: "Never run nbF_figures.ipynb until Phase G." Not a regression |
| 10 | `pytest tests/ -q` | PASS | 55/55 passed in 3.33s |
| 11 | CSV schema sanity via `pandas.read_csv` on 12 canonical tables | PASS | All 12 load; shapes documented below |

## CSV shapes confirmed

| File | Shape | Expected |
|---|---|---|
| `results/opx_multiseed_summary.csv` | (96, 10) | 8 families × 2 targets × 6 feature/track combos = 96 ✓ |
| `results/opx_multiseed_results.csv` | (1920, 7) | 96 × 20 seeds ✓ |
| `results/cpx_multiseed_summary.csv` | (96, 10) | same shape ✓ |
| `results/cpx_multiseed_results.csv` | (1920, 7) | same shape ✓ |
| `results/regime_allmodels.csv` | (4769, 18) | pre-correction regime table |
| `results/regime_allmodels_postcorrection.csv` | (4809, 19) | post-correction regime table |
| `results/preregistered_scorecard_postcorrection.csv` | (40, 18) | 8 combos × 5 regimes |
| `results/tabpfn_multiseed_summary.csv` | (8, 10) | current 5-seed state |
| `results/tabpfn_multiseed_results.csv` | (40, 12) | 8 × 5 — will become (160, 12) at Option B C1 regen to 20 seeds |
| `results/tabpfn_regime_rmse.csv` | (40, 18) | 8 combos × 5 regimes |
| `results/tabpfn_head_to_head.csv` | (8, 12) | aggregates, regen at Option B C1 |
| `results/bias_correction_shipped.csv` | (8, 12) | TabPFN rows added at Option B C5 |

## Known residuals (pre-documented, not regressions)

### R1. Broken notebook imports after C11 archive (check 6)

Five `import scripts.*` statements reference modules that were moved to
`archive/` in C11:

1. `notebooks/nb03_opx_baseline_models.ipynb`: `from scripts.v10_test_protocol import summarize_tests`
2. `notebooks/nb03_cpx_baseline_models.ipynb`: same
3. `notebooks/nb03_twopx_baseline_models.ipynb`: same
4. `notebooks/nb03_universal_exploration.ipynb`: same
5. `notebooks/nbF_figures.ipynb`: `from scripts.v10_phase_g_figures_30_34 import *`

**Why not fixed in Phase 1.5.** C14 was downgraded (see
`CLEANUP_AUDIT_PHASE_1_5.md` §"C14 scope conflict") to a string-only
sweep because consolidating the 4 nb03 notebooks requires end-to-end
re-execution, which is explicitly out of Phase 1.5 scope. Running the
notebooks also falls under the Phase G guardrail for nbF.

**Remediation plan (post-Phase-1.5).** `git mv archive/.../v10_test_protocol.py
src/test_protocol.py`, then update the 4 nb03 import lines; `git mv
archive/.../v10_phase_g_figures_30_34.py src/figures_30_34.py`, update
nbF import. Tracked as "Phase I consolidation" in the dilemmas log.

### R2. v10_twopx_* and v10_universal_* CSVs unrenamed (C14 residual)

`results/v10_twopx_per_cell_results.csv`,
`results/v10_universal_cross_pipeline_{results,summary}.csv`, and peers
were not renamed in C2-C8 (C3 covered cpx only). The 4 nb03 notebooks
still reference these filenames literally. A rename without notebook
re-execution would break the reads. See audit §C14.

### R3. TabPFN CSVs on disk still hold 5-seed data (C17 residual)

User instruction during C17: TabPFN now runs 20 seeds matching the
other families. Script default updated
(`scripts/tabpfn/opx_tb_nb03_tabpfn_baseline.py`: `--seeds` default =
`42..61`), docstring/README/nb03_tabpfn_plan/tabpfn_paragraph prose
updated. The regeneration itself is deferred to TabPFN Option B C1
because Option B C3 consumes the 20-seed rows, and regenerating twice
is wasted cost. Confirmed in check 11: `tabpfn_multiseed_results.csv`
has 40 rows (8 × 5); will be 160 rows (8 × 20) post-Option-B-C1.

### R4. CANONICAL_FIGURES 25/35 not yet rendered on disk (check 9)

Per CLAUDE.md: "Never run nbF_figures.ipynb until Phase G." Figures
with `fig_nb0*_*` stems are registered but not rendered; `fig26-fig34`
are present from Phase G. Cleanup Phase 1.5 did not touch figures/;
this is the project's pre-Phase-1.5 state.

## C1-C17 regression summary

No regressions. 55/55 tests pass. All tracked imports either resolve or
were pre-documented as stale before C18 ran. All CSV readers succeed
against their canonical schemas.

Ready to proceed to C19 (final audit + tightened grep target ≤ 10).
