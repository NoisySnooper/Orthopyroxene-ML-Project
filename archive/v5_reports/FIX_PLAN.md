# OPX ML Thermobarometry Pipeline — Master Fix Plan v5.0

**Executor:** Claude Code (VS Code IDE)
**Repo root:** `Final Project/`
**Mode:** Sequential, verification-gated
**Audience for Claude Code reports:** Use Caveman Mode (install in Phase 0). All user-facing progress messages, summaries, and status reports use Caveman Mode (full or ultra). Code, docstrings, markdown cells in notebooks, README, PROJECT_OVERVIEW, and all written deliverables use proper formal English.

---

## Setup: Install Caveman Mode

Run before starting Phase 0:

```bash
claude plugin marketplace add JuliusBrussee/caveman && claude plugin install caveman@caveman
```

Activate ultra mode for status reports. Keep formal English for files written to disk.

---

## Operating Rules

1. Phases run in order. Do not start phase N+1 until phase N verification passes.
2. After each phase, print Caveman-mode checklist showing what done, what left.
3. Before deleting files, list them and confirm count.
4. Existing `config.py` is canonical. Add to it. Do not replace.
5. This refactor is structural. Do NOT change scientific logic except where Phase 4 explicitly authorizes.
6. Every computation cell needs a markdown cell above it explaining purpose in 1-3 sentences.
7. Every figure self-contained: title, axis labels with units, legend, caption text element. Figure must be evaluable standalone without reading surrounding code.
8. Progress bars: use tqdm. Nested loops use `position=` and `leave=False`.
9. User writing style for all written deliverables: no em dashes, no AI cliches, active voice, first person where natural, direct, blunt. Cite sources with working links.
10. Commit after each phase: `git add -A && git commit -m "phase N complete: <summary>"`

---

## Phase 0: Backup and Baseline

**Goal:** Snapshot everything before changes.

### Steps

1. Verify clean git state. If uncommitted changes exist, commit them.
2. Create branch `refactor/v5-master-fix`.
3. Tag baseline `baseline-pre-refactor`.
4. Print disk usage and file count.
5. Verify environment matches `requirements.txt`.
6. Install tqdm if missing.

### Verification

- Git branch is `refactor/v5-master-fix`.
- Tag `baseline-pre-refactor` exists.
- All packages in requirements.txt importable.

---

## Phase 1: Aggressive Cleanup

### 1.1 Archive Pipeline V1

Move `notebooks/Pipeline V1` to `archive/pipeline_v1_legacy`. Add `archive/README.md`.

### 1.2 Delete Outdated Build Scripts

11 files: `_nbbuild.py`, `_build_nb01.py` through `_build_nb09.py` (including `_build_nb04b.py`).

### 1.3 Delete Outdated Roadmap Files

`execution_roadmap.md`, `revised_master_prompt_opx_thermobar.md`, `opx_ml_master_fix_v4.1.md`, `files.zip`.

### 1.4 Delete Stale Result Files

12 pre-fix CSVs in results/ (subsumed by 1.6 wipe).

### 1.5 Delete Standalone NB06b Notebook

Save content first to `/tmp/nb06b_content.json` for Phase 3 merge.

### 1.6 Wipe Generated Outputs

Delete contents of `data/processed/`, `data/splits/`, `models/`, `results/`, `figures/`, `reports/figures/`, `logs/`. Keep folders with `.gitkeep`. Do not touch `data/raw/` or `data/natural/`.

---

## Phase 2: Source Module Refactor (`src/`)

Create `src/` with these modules:

- `features.py`: raw/alr/pwlr feature engineering, `build_feature_matrix`.
- `models.py`: `predict_median`, `predict_iqr`, `parse_params`, `build_model`, `BASE_MODELS`.
- `data.py`: loaders for processed data and splits; path constants import from `config.py`.
- `evaluation.py`: `compute_metrics`, `residual_by_bin`, LOSO loop, cluster-KFold.
- `geotherm.py`: `hasterok_chapman_geotherm(q_s_mW, z_km)`.
- `io_utils.py`: `save_figure`, `save_table`, `with_progress`.

Canonical source for features: NB03c version. Eliminate 12 copies of feature functions and 5 copies of `predict_median`. All notebooks now import from `src`.

---

## Phase 3: Notebook Consolidation

### 3.1 Merge NB03b + NB03c → NB03

NB03c is canonical. Pull `n_aug` sensitivity cells from NB03b into NB03c. Set canonical `N_AUG` per representation from sensitivity test. Save `results/nb03_n_aug_sensitivity.csv`. Rename to `nb03_baseline_models.ipynb`.

### 3.2 Merge NB06b → NB06

Add sections: "SHAP Robustness Across Seeds", "SHAP Interaction Effects", "Feature Importance Stability".

### 3.3 Fix NB01/NB02 Write Collision

NB01 writes `data/processed/opx_clean_core.parquet`. NB02 adds cluster column and writes `data/processed/opx_clean_core_with_clusters.parquet`. Loaders: `load_opx_core()` vs `load_opx_core_clustered()`.

### 3.4 Resolve NB06 + NB09 Table5 Collision

NB06 writes `results/nb06_shap_feature_importance.csv` only. NB09 produces `table5_shap_importance.csv`.

### 3.5 Resolve NB02 + NBF Figure Collision

NB02 writes `figures/eda_pt_distribution.png`, `figures/eda_pca_biplot.png`. NBF writes the manuscript-name versions.

---

## Phase 4: Authorized Bug Fixes

### 4.1 NB09 Schema Drift

Replace `('seed', 'nunique')` with `('split_seed', 'nunique')`. Replace `('fold', 'nunique')` with `('group', 'nunique')`. Fix track/target join. After: `multi_win` defined, downstream cells execute.

### 4.2 NB11 H2O Case Mismatch

Replace `H2O_liq` / `liq_H2O` references with `H2O_Liq`. Populates `results/nb11_h2o_dependence.csv`.

### 4.3 NBF Geotherm Function Wiring

Replace linear approximation in NBF cell 10 with `hasterok_chapman_geotherm` called for q_s in [40, 60, 80] mW/m².

### 4.4 Rename "MC Dropout" → "Analytical Uncertainty Propagation"

Rename function, section, and output file (`nb11_mc_inference.csv` → `nb11_analytical_uncertainty.csv`). Document the distinction from epistemic uncertainty.

### 4.5 NB07 Conformal Calibration

Add TODO note only; not authorized as logic change.

### 4.6 Document Pressure-Range Bias

Add markdown caveat in NB07 and NB09: valid P range approximately 5-20 kbar.

---

## Phase 5: Centralize Configuration

Add to `config.py`: KD bounds, cation/oxide totals, feature set flag `WIN_FEAT`, default `N_AUG`, QRF quantiles, analytical noise levels, geotherm heat flows, MC iteration count, and output path constants. Replace hardcoded `random_state=42` in notebooks with `SEED_MODEL`.

---

## Phase 6: Notebook Polish

- Every code cell has a 1-3 sentence markdown cell above it.
- Every figure: descriptive title, axis labels with units, legend, caption, saved at 300 dpi.
- tqdm on any cell running more than 5 seconds. Nested loops use `position=` and `leave=False`.
- Inline display via `%matplotlib inline`.

---

## Phase 7: Run-Fresh Execution

Restart all kernels. Wipe generated folders if state drifted. Run in order: NB01, NB02, NB03, NB04, NB04b, NB05, NB06, NB07, NB08, NB11, NB09, NBF. Caveman status per notebook. Halt on error.

Optional: `run_all.py` using papermill.

---

## Phase 8: Documentation

### 8.1 README.md

Short GitHub landing with Quick Start, Project Structure, Pipeline Overview mermaid graph.

### 8.2 PROJECT_OVERVIEW.md

Detailed documentation: background, approach, pipeline architecture (mermaid flowchart), notebook reference, module reference, configuration reference, reproducibility steps, limitations, citations.

### 8.3 Pipeline PNG

`scripts/render_pipeline_diagram.py` produces `figures/pipeline_diagram.png` for manuscript.

### 8.4 Update requirements.txt

Add `tqdm`, `papermill` (optional), `graphviz`.

---

## Phase 9: Final Audit

Re-run static audit. Expect zero orphan outputs, zero write collisions, zero schema mismatches, zero duplicated functions. Print final file tree. Commit and tag. Update this file's STATUS.

---

END OF FIX PLAN v5.0
