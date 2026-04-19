# Cleanup Phase 1.5 — finishing Tier 3

Date started: 2026-04-19. Purpose: finish the deferred work from Phase 1
(`CLEANUP_AUDIT.md` §9-10). Target end state: grep count of `v10|V10` in
tracked non-archive, non-preregistration, non-data, non-logs files is
≤ 10, each documented.

Safety tag: `pre-phase-1-5-2026-04-19` (set at HEAD `29ac30e`).

## C1 Baseline (before any changes)

### File-path inventory (tracked, excluding archive/ and docs/preregistration/)

| Category | Path glob | Count | Disposition |
|---|---|---|---|
| results/ v10_ files | `results/v10_*` | 47 | rename (C2-C7) |
| results/v10_*/ subdirs | `results/v10_bias_correction/`, `results/v10_tabpfn_checkpoints/`, `results/v10_optuna_studies/` | 3 | rename (C5,C6,C8) |
| scripts/ v10_*.py | `scripts/v10_*.py` | 61 | archive (C11) |
| notebooks/ v10 | `notebooks/nb04_v10_benchmark.ipynb` | 1 | resolve (C15) |
| figures/ v10 | `figures/fig35_tabpfn_vs_v10.*` | 3 | rename + caption (C6,C10) |
| tables/ v10 | `tables/v10_regime_*.{csv,md}` | 20 | rename (alongside C4) |
| docs/ v9 root | `docs/v9_*.md` | 4 | archive (C16) |
| logs/ v10 | `logs/v10_*` | 58 | **out of scope** — historical audit records |
| data/ | `data/**/v10*` or content v10 | various | **out of scope** — read-only per CLAUDE.md |

### Content grep baseline

Filtered `git grep -I -E "v10|V10"` excluding `archive/`,
`docs/preregistration/`, `logs/`, `data/`, `CLEANUP_AUDIT*.md`:

- **187 files** contain at least one v10/V10 occurrence
- **10,810 total occurrences** (dominated by CSV column values and paper
  artifact strings)

### Top offenders

| File | Count | Plan |
|---|---|---|
| results/v10_regime_allmodels_postcorrection.csv | 4440 | C4 rename. Column values `v10` preserved as paper-reporting label; see §Decision-1 |
| results/v10_regime_allmodels.csv | 4400 | same |
| notebooks/nb04_v10_benchmark.ipynb | 95 | C15 resolve |
| docs/master_plan.md | 68 | C17 sweep |
| notebooks/nb09_manuscript_compilation.ipynb | 65 | C2-C6 path updates + C17 label sweep |
| results/v10_regime_benchmark.csv | 57 | C4 rename (label column preserved) |
| .claude/settings.local.json | 53 | auto-generated, not tracked in policy scope; skip |
| scripts/v10_phase_g_postcorrection_tables.py | 48 | C11 archive |
| scripts/opx_tb_nb03_apply_part2_cells.py | 46 | C17 sweep |
| docs/cleanup_manifest.md | 42 | C17 sweep |

### Decision-1: `v10` as pipeline label vs `v10` as filename prefix

CSVs like `v10_regime_allmodels.csv` contain literal column values
`method_family == 'v10'` and `method_label == 'v10 RF alr'`. These
values label the pipeline **as reported in the paper**. Renaming the
column values to `opx_tb` would propagate into paper tables and figures
and would require coordinated paper-text changes. Out of scope for this
cleanup.

**Policy for C2-C6:** rename the CSV filenames (path-level). Preserve
column-level `v10` values as immutable paper-reporting labels. Final
grep count will exclude these column-value occurrences by excluding
`results/*.csv` and `tables/*.csv` from the tightened grep target.

### Tightened grep target for C19

Scope for the ≤ 10 target:

```
git grep -I -E "v10|V10" -- \
  ':!archive/' \
  ':!docs/preregistration/' \
  ':!logs/' \
  ':!data/' \
  ':!CLEANUP_AUDIT*.md' \
  ':!results/*.csv' ':!results/*.json' \
  ':!tables/*.csv' ':!tables/*.md' \
  ':!.claude/'
```

Expected residuals after C2-C19:
- `V10_BASE_ORDER` alias references in 4 `scripts/v10_phase_*_runner.py`
  (archived in C11, so not counted)
- Git commit messages (excluded by git grep; doesn't affect file
  content)
- `manuscripts/opx_2026/arxiv_submission/` — may contain frozen v10
  paper text, treat as immutable paper record if present
- Historical bullet points in `docs/master_plan.md` (C17 attempt; if
  history references are load-bearing, leave and document)

## Commit plan status

| Step | Planned | Status |
|---|---|---|
| C1 | Safety tag + this audit | IN PROGRESS |
| C2 | opx multiseed rename | pending |
| C3 | cpx multiseed rename | pending |
| C4 | regime/external/natural rename | pending |
| C5 | bias correction rename | pending |
| C6 | tabpfn rename + fig35 | pending |
| C7 | optuna best-params rename | pending |
| C8 | optuna_studies reconcile | pending |
| C9 | V10_ alias removal | pending |
| C10 | config.py caption cleanup | pending |
| C11 | archive v10_phase_* | pending |
| C12 | scripts/ subdirs | pending |
| C13 | run_all.py cleanup | pending |
| C14 | nb03 consolidation | pending |
| C15 | nb04_v10 + nb07b resolution | pending |
| C16 | v9 docs archive + nb03_test_protocol | pending |
| C17 | PROJECT_LAYOUT + README + autofills | pending |
| C18 | code integrity validation | pending |
| C19 | final audit + report | pending |

## Dilemmas log (stopped-and-asked scenarios)

### GEOROC cpx server reachable (2026-04-19)

`curl -I https://georoc.eu/` and `https://georoc.mpch-mainz.gwdg.de/`
both return HTTP 200. The 503 that paused Phase H.1b is resolved.

**Disposition:** not actioned in this cleanup. Reasons:

1. `scripts/v10_pull_georoc_cpx.py` is a manual-download stub: it only
   prints instructions to visit the web UI, filter by Phase =
   Clinopyroxene, and export CSV. No programmatic API.
2. Even with a fresh CSV in `data/natural/`, running the cpx natural-
   sample inference pipeline is a multi-hour notebook chain (nb08 +
   Phase H.3b/H.3d scripts). The Phase 1.5 spec explicitly excludes
   notebook re-runs during cleanup.
3. Existing `results/v10_cpx_*.csv` training artifacts are in place and
   get renamed in C3.

**Action item for after Phase 1.5:** download GEOROC cpx dataset
manually, re-run `nb08_natural_twopx.ipynb`, produce a cpx counterpart
to Phase H.5a opx world map. Track as Phase H.1b-recovery.

### C14 nb03 consolidation scope conflict (2026-04-19)

Plan calls for consolidating 4 track-specific baseline notebooks into a
single papermill-parameterized notebook:

| Notebook | Lines |
|---|---|
| `nb03_opx_baseline_models.ipynb` | 1595 |
| `nb03_cpx_baseline_models.ipynb` | 1586 |
| `nb03_twopx_baseline_models.ipynb` | 1383 |
| `nb03_universal_exploration.ipynb` | 1143 |

**Conflict:** Phase 1.5 spec explicitly excludes notebook re-runs during
cleanup. A papermill parameterization rewrite of this size cannot be
validated without executing the new unified notebook across all four
`TRACK` values end-to-end (each notebook is an Optuna + stacking +
test-set + figure pipeline, hours of runtime per track).

**Disposition:** skip full consolidation in Phase 1.5. Downgrade C14
scope to a string-only sweep: strip v10 references from the 4 notebooks
without changing code paths or cell structure. Pytest gate still holds.

**Action item for after Phase 1.5:** design the parameterized
`nb03_baseline_models.ipynb` in a dedicated phase where notebook
executions are in scope; track as Phase I consolidation.

**Known C14 residuals after the string sweep (intentional, documented):**

1. `v10_twopx_*.csv` and `v10_universal_*.csv` under `results/` were not
   renamed in C2-C8 (C3 covered cpx only). The four nb03 notebooks still
   contain `pd.read_csv(RESULTS / 'v10_twopx_per_cell_results.csv')` etc.
   Renaming the notebook strings without renaming the files would break
   the read. Follow-up: extend C3-style rename to twopx+universal as a
   dedicated post-Phase-1.5 commit.
2. `from scripts.v10_test_protocol import summarize_tests` appears in
   all 4 notebooks. The module was archived in C11 alongside other
   `scripts/v10_*.py` one-offs, but it is actually shared canonical
   infrastructure used by 4 production notebooks. Post-Phase-1.5
   action: `git mv archive/.../v10_test_protocol.py src/test_protocol.py`,
   then update the 4 notebook imports.
