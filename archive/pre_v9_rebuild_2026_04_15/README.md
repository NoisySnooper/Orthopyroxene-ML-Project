# Pre-v9 Rebuild Archive

**Archived:** 2026-04-15

Contents represent the v8 final state of the opx ML thermobarometer pipeline before the v9 clean-slate rebuild. See `docs/codebase_consistency_audit_optionB.md` and the v9 rebuild plan for context on why this archive exists.

## Subdirectory guide

| Subdir | What it contains |
|---|---|
| `notebooks_executed/` | All 12 papermill-executed notebooks from v8 (nb01-nb09, nb04b, nb07b, nbF) |
| `results/` | 79 CSV/JSON outputs from v8 pipeline runs |
| `figures/` | All figures at the top level of `figures/` from v8 (PDFs, PNGs, caption TXTs) |
| `figures_archive/` | Pre-existing `figures/archive/` content (fig_nb10b_two_pyroxene) |
| `models/` | All 37 canonical joblibs + isolation forests from v8 (~325 MB) |
| `logs/` | Pipeline run logs, FAILURE logs, codebase_audit.md, pipeline_health.txt |
| `logs_archive/` | Pre-existing `logs/archive/` content |
| `scripts_obsolete_patches/` | 44 one-shot builder/patch scripts from v5-v8, already applied to canonical notebooks |
| `root_orphans/` | Orphan files from repo root: stale runners (run_all.py, run_from.py), training logs, app inventory, extract_results.py |
| `examples_archive/` | `examples/test_sample_for_app_validation.csv` (app deployment fixture) |
