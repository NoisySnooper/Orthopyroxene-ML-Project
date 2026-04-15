# v9 Rebuild Deletion Plan

**Generated:** 2026-04-15
**Purpose:** Enumerate files and directories to permanently delete (not archived) during Phase 1. Each entry has a one-sentence justification and is confirmed not referenced by any KEEP-CANONICAL file.

---

## Files and directories to delete

### 1. `build_smoke/` (entire directory, 16 files, 96 KB)

**Justification:** Every file in this dir references notebooks that no longer exist.

| File | Imports nonexistent notebook |
|---|---|
| `nb04b_lepr_arcpl_validation_imports.py` | nb04b (absorbed into nb04) |
| `nb08_natural_samples_imports.py` | nb08_natural_samples (replaced by nb08_natural_twopx) |
| `nb10_extended_analyses_imports.py` | nb10 (absorbed into nb09) |
| `nb10b_two_pyroxene_benchmark_imports.py` | nb10b (absorbed into nb08) |
| `nb11_model_family_ceiling_imports.py` | nb11 (absorbed into nb03) |
| `nb10_cells_dump.txt` | nb10 cell source dump (reference only) |
| nb01..nb09, nbF `_imports.py` | files for live notebooks but smoke-test concept is obsolete in v9 |

No KEEP-CANONICAL file imports from `build_smoke/`. Confirmed via grep: zero references from `notebooks/`, `src/`, `config.py`, `run_all_v7.py`.

### 2. `wipe_old_files_v7.py` (root, 5 KB)

**Justification:** One-time v7 wipe script, already executed at v7 transition. No callers.

### 3. `__pycache__/` (root, 8 KB)

**Justification:** Python bytecode cache, regenerated on next import.

### 4. `src/__pycache__/` (93 KB)

**Justification:** Python bytecode cache, regenerated on next import.

### 5. `data/external/agreda_lopez_2024/repo/.git/` (23 MB)

**Justification:** Nested git repo from a git clone of the Agreda-Lopez ML_PT_Pyworkflow reference code. The `.git/` folder bloats repository size and creates submodule confusion if this project is ever pushed upstream. The source code (`ML_PT_Pyworkflow/`, `README.md`) inside `agreda_lopez_2024/repo/` stays.

### 6. `data/external/agreda_lopez_2024/repo/.DS_Store` (6 KB)

**Justification:** macOS filesystem metadata, transferred during clone. Platform noise.

### 7. ~~`data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv`~~ — USER DECISION: KEEP

User confirmed 2026-04-15: keep the raw 102 MB CSV for source provenance. Not deleted.

---

## Total bytes to be reclaimed

| Item | Size |
|---|---|
| `build_smoke/` | 96 KB |
| `wipe_old_files_v7.py` | 5 KB |
| `__pycache__/` (root) | 8 KB |
| `src/__pycache__/` | 93 KB |
| `data/external/.../agreda.../repo/.git/` | 23 MB |
| `data/external/.../agreda.../repo/.DS_Store` | 6 KB |
| **TOTAL** | **~23 MB** |

---

## Verification before deletion

All files/dirs above were checked for inbound references from KEEP-CANONICAL files via:
```
grep -r "<pattern>" notebooks/ src/ config.py run_all_v7.py requirements.txt
```

Results:
- `build_smoke/` — zero references from canonical files
- `wipe_old_files_v7.py` — zero references
- `__pycache__/` — no references (bytecode dir, never imported directly)
- `data/external/agreda_lopez_2024/repo/.git/` — no references (git internals)
- `data/external/agreda_lopez_2024/repo/ML_PT_Pyworkflow/` — KEPT, referenced by `src/external_models.py`
- `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` — NO references from canonical files (only from `natural_sample_prep_script.py` which is itself reproducibility-only)

---

## Deletion commands (for Phase 1 prompt)

```bash
# entire dirs
rm -rf build_smoke/
rm -rf __pycache__/
rm -rf src/__pycache__/
rm -rf data/external/agreda_lopez_2024/repo/.git/

# single files
rm wipe_old_files_v7.py
rm data/external/agreda_lopez_2024/repo/.DS_Store

# pending user confirmation (raw 102 MB CSV)
# rm data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv
```

---

## Safety checklist

- [ ] User has read and approved this deletion plan
- [ ] User has approved the raw natural CSV deletion specifically (or explicitly chosen to keep)
- [ ] Archive plan executed successfully before any deletions run (archive is the safety net)
- [ ] No file in the deletion list is referenced by any file in KEEP-CANONICAL list (verified above)

If any check fails, HALT Phase 1 before executing deletions.
