# Cleanup Phase 1 — Stage A audit

Date: 2026-04-19. Scope: read-only inventory + Stage B/C proposal. No
content edits or file moves performed. STOP here for user approval
before proceeding to Stage B.

Canonical naming to adopt project-wide:
- "ML orthopyroxene thermobarometer" / "opx thermobarometer" (prose).
- File prefix `opx_tb_` replaces `v10_` on artifact filenames. Drop the
  phase prefix entirely where the subdirectory already disambiguates
  (e.g. `results/v10_opx_multiseed_summary.csv` ->
  `results/opx_multiseed_summary.csv`).
- Symbol `V10_BASE_ORDER` -> `BASE_ORDER` (keep a module-level alias for
  one release so stale notebooks don't break).

Immutable zones (do not touch in any stage):
- `data/raw/`, `data/processed/`, `data/splits/`, `data/external/`,
  `data/natural/` (CLAUDE.md hard rule).
- `models/external/` (read-only per CLAUDE.md).
- `manuscripts/opx_2026/text/draft_v1.md` (not yet present; pre-sealed).
- `docs/preregistration/*` (to be created in B.1; treated as immutable
  once sealed).

---

## 1 Executive summary

| Category | Count | Comment |
|---|---|---|
| Notebooks | 17 | 6x `nb03_*` variants to consolidate |
| Scripts (scripts/*.py) | 67 | Many phase-legacy one-shots |
| src modules | 21 | 1 misnamed (`v10_phase_c_analysis.py`) |
| Tests | 3 | pytest: 55/55 pass (2026-04-19) |
| Docs (docs/*) | 27 | Includes 0-byte `docs/_nul_` anomaly |
| Figures (figures/*) | 51 files | 8 orphan `v10_regime_*.{pdf,png}` |
| Tables (tables/*) | 26 | S8_* + table_4_* + v10_regime_* mix |
| Results (results/*) | 70 items | Heavy `v10_` prefix load |
| Logs (logs/*) | 58 | All `v10_phase_*`, safe to archive |
| Models (models/*) | 387 files, 1.5 GB | canonical/, ablation/, external/ |
| Archives (archive/*) | 5 subdirs + loose `.ipynb` | 742 MB |
| Prompts (prompts/*) | 0 files | 4 empty subdirs |

Repository-wide `v10|V10` scan (excluding `data/**`): **11,262
occurrences across 250 files.** Bulk of the rename happens in B.2.

---

## 2 Notebook inventory (17)

| Notebook | Cells | Bytes | Status / decision |
|---|---|---|---|
| `nb01_data_cleaning.ipynb` | 17 | 21 KB | Keep. Rename-safe. |
| `nb02_eda_pca.ipynb` | 17 | 681 KB | Keep. |
| `nb03_baseline_models.ipynb` | 46 | 114 KB | **Consolidated target.** Superset; becomes canonical nb03 after B.3. |
| `nb03_opx_baseline_models.ipynb` | 18 | 127 KB | **Merge into nb03_baseline_models** via `TRACK="opx"` dispatch. |
| `nb03_cpx_baseline_models.ipynb` | 18 | 127 KB | **Merge** via `TRACK="cpx"`. |
| `nb03_twopx_baseline_models.ipynb` | 18 | 119 KB | **Merge** via `TRACK="twopx"`. |
| `nb03_universal_exploration.ipynb` | 17 | 105 KB | **Merge** via `TRACK="universal"` (side-project flag). |
| `nb03_tabpfn_baseline.ipynb` | 4 | 4.2 KB | Keep separate (different model class, driver-only). |
| `nb04_putirka_benchmark.ipynb` | 38 | 2.0 MB | Keep. Source of TabPFN head-to-head CSV. |
| `nb04_v10_benchmark.ipynb` | 25 | 243 KB | Decide: merge into nb04_putirka_benchmark or retain. Leaning **merge**. |
| `nb05_loso_validation.ipynb` | 8 | 167 KB | Keep. |
| `nb06_shap_analysis.ipynb` | 23 | 239 KB | Keep. |
| `nb07_bias_correction.ipynb` | 14 | 43 KB | Keep. |
| `nb07b_arcpl_bias_probe.ipynb` | 16 | 839 KB | Keep; retire after Phase H.6 if subsumed. |
| `nb08_natural_twopx.ipynb` | 7 | 245 KB | Keep. |
| `nb09_manuscript_compilation.ipynb` | 31 | 28.2 MB | Keep. Large due to embedded figures. |
| `nbF_figures.ipynb` | 24 | 226 KB | Keep. Canonical figure regen. |

**B.3 notebook consolidation target:**
A single `nb03_baseline_models.ipynb` dispatched by a `TRACK` parameter
({"opx","cpx","twopx","universal"}), with a small header cell that
resolves pipeline targets and feature sets from `config.py`. The 4
current variants merge as parametric executions; a notebook parameter
tag (`tags: ["parameters"]`) on the header cell keeps the file
papermill-executable.

---

## 3 Figure audit vs CANONICAL_FIGURES

Registry size: 30 entries. Figures on disk: 51 files.

### 3.1 Registered + present (OK)
Entries 1-25, 30-35. All have corresponding `.pdf/.png/.txt` triples.

### 3.2 Gap in registry (fig26-29)
`fig26_generalization_opx_liq`, `fig27_shap_summary_opx_liq`,
`fig28_bias_correction_opx_liq`, `fig29_twopx_benchmark` exist on disk
(with `.txt` sidecars) but are **absent from `CANONICAL_FIGURES`**.
Decision in B.2: add these four as registry entries so nbF can
regenerate them and nb09 can reference them consistently.

### 3.3 Orphan figures (no registry entry, no phase mapping)
| File | Bytes | Action |
|---|---|---|
| `figures/v10_regime_all.pdf` / .png | ~ | **Archive** (superseded by fig24). |
| `figures/v10_regime_combined.{pdf,png}` | ~ | **Archive**. |
| `figures/v10_regime_cpx.{pdf,png}` | ~ | **Archive**. |
| `figures/v10_regime_opx.{pdf,png}` | ~ | **Archive**. |
| `figures/fig_nb04_cross_pipeline_heatmap.png` | ~ | Keep (referenced by nb04). |
| `figures/fig_nb04_ensemble_lift.png` | ~ | Keep. |
| `figures/fig_nb04_per_regime_rmse_opx_liq.{pdf,png}` | ~ | **Duplicate** of fig24; archive. |
| `figures/fig_nb04_winning_base_histogram.png` | ~ | Keep. |
| `figures/fig_h5ac_opx_world_map.png` | ~ | Keep (Phase H.5a). |
| `figures/fig_h5b_opx_interactive.html` | ~ | Keep (Phase H.5b). |

**Archive destination (created in B.1):**
`archive/pre_consolidation_2026_04_18/figures/`.

---

## 4 Script audit (67)

Grouped by purpose:

### 4.1 Canonical drivers (keep, rename `v10_` -> `opx_tb_`)
- `v10_nb03_tabpfn_baseline.py` -> `opx_tb_nb03_tabpfn_baseline.py`
- `v10_nb03_fill_tabpfn_paragraph.py` -> `opx_tb_nb03_fill_tabpfn_paragraph.py`
- `v10_nb03_apply_part2_cells.py` -> `opx_tb_nb03_apply_part2_cells.py`
- `v10_nb03_test_tabpfn_smoke.py` -> `opx_tb_nb03_test_tabpfn_smoke.py`
- `v10_nb03_test_t15.py`, `v10_nb03_test_t16_t18.py` -> keep, rename.
- `v10_external_models_audit.py`
- `v10_test_protocol.py`
- `v10_audit.py`

### 4.2 Phase-legacy one-shots (archive after verifying outputs landed)
All scripts prefixed `v10_phase_{a,b,c,d,e,f,g,h}_*` (55 files). Each
represents a completed Phase execution; the produced artifacts live in
`results/`, `figures/`, `tables/`, and `logs/v10_phase_*_execution_log.md`.

Move destination: `archive/pre_consolidation_2026_04_18/scripts/phase_legacy/`.

Exceptions to keep live (still called by active drivers):
- `v10_phase_g_regime_allmodels.py` (nb04 imports)
- `v10_phase_g_postcorrection_tables.py` (nb09 imports)
- `v10_phase_h1a_opx_with_coords.py` (Phase H opx_liq still pending)
- `v10_phase_h2_locality_coverage.py`
- `v10_phase_h3a_opx_inference.py`
- `v10_phase_h3d_opx_rf_uncertainty.py`
- `v10_phase_h5a_opx_world_map.py`
- `v10_phase_h5b_opx_interactive_map.py`
- `v10_pull_georoc_cpx.py` (Phase H.1b pending)
- `v10_georoc_cpx_schema_diff.py` (diagnostic)

### 4.3 Pre-v10 audit scripts (archive)
- `audit_optionB_prediction.py`, `audit_putirka_paths.py`,
  `audit_structure.py`, `audit_thermobar_kd_api.py` -> archive.

### 4.4 Runners (archive)
- `run_phase2_optuna.py`, `run_phase35_to_312.py` -> archive.

### 4.5 Producer/consumer adjacency (top cells)
- `scripts/v10_nb03_tabpfn_baseline.py` -> produces
  `results/v10_tabpfn_{multiseed_results,multiseed_summary,predictions,
  regime_rmse}.csv` + `results/v10_tabpfn_checkpoints/*.pkl`.
- `notebooks/nb04_putirka_benchmark.ipynb` (TABPFN_PART2_BLOCK cell) ->
  produces `results/v10_tabpfn_head_to_head.csv`.
- `notebooks/nbF_figures.ipynb` (TABPFN_PART2_BLOCK) ->
  `figures/fig35_tabpfn_vs_v10.{pdf,png,txt}`.
- `notebooks/nb09_manuscript_compilation.ipynb` (TABPFN_PART2_BLOCK) ->
  `tables/S8_12_tabpfn_benchmark.{csv,md,tex}`.
- `scripts/v10_nb03_fill_tabpfn_paragraph.py` ->
  `manuscripts/opx_2026/text/tabpfn_paragraph.md`.

---

## 5 src/ audit

| File | Status |
|---|---|
| `src/v10_phase_c_analysis.py` | **Rename** to `src/opx_tb_analysis.py`. Only module still carrying `v10_` prefix. Update imports in scripts and notebooks. |
| `src/__init__.py` + 19 others | Keep, no rename needed. |

---

## 6 Anomalies

1. **`docs/_nul_`** — 0-byte file, Windows reserved-name escape. Delete
   in B.1.
2. **Empty root `Prior-Labs/`** — delete in B.1.
3. **Empty `prompts/*` subdirs** (bias_correction, cleanup/phase_1,
   cleanup/phase_2, tabpfn) — delete in B.1.
4. **`notebooks/nb03_baseline_models.ipynb`** (46 cells) coexists with
   the 4 track-specific nb03_*_baseline_models.ipynb variants; B.3
   unifies by parameterizing dispatch. The superset already contains
   the merged logic.
5. **Figures 26-29 are unregistered** (see 3.2). Add to
   `CANONICAL_FIGURES` in B.2.
6. **Untracked `results/v10_bias_correction/`** directory is on disk
   but not covered in this cleanup; deferred pending user direction.

---

## 7 Proposed directory structure (post-B)

```
config.py                         # BASE_ORDER (replaces V10_BASE_ORDER), keep alias
run_all.py                        # renamed from run_all_v10.py
README.md                         # rewritten in Stage C
PROJECT_LAYOUT.md                 # new in Stage C
PROJECT_OVERVIEW.md               # retained (status log)
CLAUDE.md                         # retained
CLEANUP_AUDIT.md                  # this file
requirements.txt, requirements-tabpfn.txt

data/                             # read-only, untouched
docs/
  preregistration/                # created; contains v10_p_regime_preregistration.md moved here
  master_plan.md                  # renamed from v10_master_plan.md
  <strategy docs>                 # unchanged
  archive_superseded/             # existing
figures/                          # canonical only; orphans -> archive
logs/                             # orphan phase logs -> archive
manuscripts/opx_2026/             # unchanged
models/                           # unchanged
notebooks/
  nb01_data_cleaning.ipynb
  nb02_eda_pca.ipynb
  nb03_baseline_models.ipynb      # parametric: TRACK in {opx,cpx,twopx,universal}
  nb03_tabpfn_baseline.ipynb      # retained (different model class)
  nb04_benchmark.ipynb            # merged putirka + v10_benchmark
  nb05_loso_validation.ipynb
  nb06_shap_analysis.ipynb
  nb07_bias_correction.ipynb
  nb07b_arcpl_bias_probe.ipynb
  nb08_natural_twopx.ipynb
  nb09_manuscript_compilation.ipynb
  nbF_figures.ipynb
results/                          # renamed v10_* -> (no prefix)
scripts/                          # active drivers only
src/
  opx_tb_analysis.py              # renamed from v10_phase_c_analysis.py
  <other modules unchanged>
tables/                           # canonical only; orphans -> archive
tests/                            # unchanged
archive/pre_consolidation_2026_04_18/
  figures/                        # orphaned figures moved here
  logs/                           # phase-legacy logs moved here
  scripts/phase_legacy/           # one-shot phase scripts moved here
  tables/                         # orphaned table variants moved here
  notebooks/nb03_track_variants/  # the 4 split nb03_*_baseline_models.ipynb
```

---

## 8 Stage B and C plan

### B.1 pure moves/renames (git mv only, no content edits)
1. `git mv src/v10_phase_c_analysis.py src/opx_tb_analysis.py`
2. `git mv run_all_v10.py run_all.py`
3. `git mv docs/v10_master_plan.md docs/master_plan.md`
4. `mkdir docs/preregistration &&
   git mv docs/v10_p_regime_preregistration.md
          docs/preregistration/p_regime_preregistration.md`
5. `git mv docs/v10_cleanup_manifest.md docs/cleanup_manifest.md`
   (and other `docs/v10_*` per manifest — full list in B.1 execution).
6. Archive moves (create target first):
   ```
   mkdir -p archive/pre_consolidation_2026_04_18/{figures,logs,scripts/phase_legacy,tables,notebooks/nb03_track_variants}
   git mv figures/v10_regime_*.{pdf,png} archive/pre_consolidation_2026_04_18/figures/
   git mv figures/fig_nb04_per_regime_rmse_opx_liq.{pdf,png}
          archive/pre_consolidation_2026_04_18/figures/
   git mv logs/v10_phase_{a,b,c,d,e,f,g}_*.{log,md,stdout}
          archive/pre_consolidation_2026_04_18/logs/
   git mv scripts/audit_*.py scripts/run_phase*.py
          archive/pre_consolidation_2026_04_18/scripts/phase_legacy/
   # phase_g and phase_h one-shots that are no longer imported, per 4.2 exceptions list
   ```
7. Delete anomalies: `git rm docs/_nul_`, `rmdir Prior-Labs`,
   `rmdir prompts/bias_correction prompts/cleanup/phase_1 prompts/cleanup/phase_2 prompts/tabpfn prompts/cleanup prompts`.
8. `git commit -m "[Stage B.1] Moves and renames only"`
9. Run pytest. Must stay green.

### B.2 content edits only (no moves)
1. `src/opx_tb_analysis.py` (renamed) — update module docstring.
2. `config.py` —
   - Introduce `BASE_ORDER = [...]`.
   - Add compatibility alias: `V10_BASE_ORDER = BASE_ORDER` with a
     deprecation comment.
   - Add figs 26, 27, 28, 29 to `CANONICAL_FIGURES`.
3. Update imports in every caller of `V10_BASE_ORDER`. (grep lists are
   small; see count above.)
4. Rewrite string literals that embed `v10_` file names to `opx_tb_`
   across: all `notebooks/*.ipynb`, all `scripts/*.py`, caption
   sidecars in `figures/*.txt`, manuscript autofills in
   `manuscripts/opx_2026/text/*.md` (except `draft_v1.md` if present).
5. Rename `results/v10_*` files/dirs to drop the prefix (`git mv`).
   Update all readers to the new names.
6. Run pytest. Spot-check `nbF_figures.ipynb` renders fig35.
7. `git commit -m "[Stage B.2] V10 -> opx_tb rename and registry updates"`

### B.3 notebook consolidation
1. Promote `nb03_baseline_models.ipynb` to the canonical multi-track
   notebook by adding a parameters cell (papermill tag).
2. Move the 4 track-specific variants to
   `archive/pre_consolidation_2026_04_18/notebooks/nb03_track_variants/`.
3. Consolidate `nb04_v10_benchmark.ipynb` into `nb04_putirka_benchmark`
   (rename to `nb04_benchmark.ipynb` at commit time) if the
   TABPFN_PART2_BLOCK cell provenance is preserved.
4. Run pytest. Smoke-run `nb03_baseline_models.ipynb` with
   `TRACK="opx"` end-to-end.
5. `git commit -m "[Stage B.3] Consolidate nb03 track variants"`

### C rewrite docs
1. Rewrite `README.md` around:
   - Canonical naming "opx thermobarometer".
   - Post-consolidation directory layout (cross-link to
     PROJECT_LAYOUT.md).
   - Running order: nb01 -> nb02 -> nb03 (parameterized) ->
     nb03_tabpfn -> nb04 -> nb05 -> ... -> nbF.
   - TabPFN supplementary baseline subsection (already drafted in
     current README).
2. Create `PROJECT_LAYOUT.md` — a single-page tree view + one-line
   purpose per top-level entry.
3. `git commit -m "[Stage C] Rewrite README and add PROJECT_LAYOUT"`.

### Post-phase self-audit checklist (14 items)
1. `git status` is clean.
2. Every rename shows up as R (rename) in `git log --stat`.
3. No orphan `v10_` occurrences except the `V10_BASE_ORDER` alias.
4. `CANONICAL_FIGURES` has 34 entries (1-35, no gaps after 26-29
   added).
5. `results/opx_tb_tabpfn_*.csv` files exist (renamed from
   `v10_tabpfn_*.csv`).
6. `figures/fig35_tabpfn_vs_v10.{pdf,png,txt}` exist and validate.
7. `tables/S8_12_tabpfn_benchmark.{csv,md,tex}` exist and validate.
8. Manuscript paragraph re-fills cleanly via
   `opx_tb_nb03_fill_tabpfn_paragraph.py`.
9. `pytest tests/` = 55/55 pass (or same count pre-phase).
10. `nbF_figures.ipynb` executes end-to-end with 0 errors.
11. `nb03_baseline_models.ipynb` executes with each TRACK value
    without errors.
12. No reads from `data/processed/` or `data/raw/` are broken.
13. `models/external/` untouched (diff empty).
14. `CLEANUP_AUDIT.md` updated with any deviations encountered.

---

## 9 Deferred items (not in Phase 1 scope)

- `results/v10_bias_correction/` directory triage.
- cpx Phase H.1b download (pending GRO 503 recovery).
- Opx Phase H.6 per-locality RMSE + curated correction panel.
- Compression/pruning of archive/ (742 MB) — deferred to a Phase 2.
- `models/` 1.5 GB retention review — deferred.

---

**Status: Stage A complete. Awaiting user approval to proceed with B.1.**
