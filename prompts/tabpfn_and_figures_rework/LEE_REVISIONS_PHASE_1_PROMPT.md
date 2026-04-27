# Claude Code Prompt: Dr. Lee Revision Phase 1

## Context

Professor Dr. Kanani K. M. Lee reviewed the current opx ML thermobarometer manuscript draft and raised specific methodological concerns that require remediation before manuscript resubmission. This prompt instructs Claude Code to execute the complete remediation plan autonomously.

Repository root: `C:\Users\NQTa\Documents\MLCourse\Final Project`
Target branch: `lee_revisions_phase_1` (create from current `main`)
Audit timestamp reference: `MANUSCRIPT_WRITING_AUDIT.md` at root (2026-04-20, SHA a9f102f8)
Manuscript draft location: `manuscripts/opx_2026/text/draft/sections/`

## Three deliverables

1. Fix the ElasticNet zero-std artifact (replace seed-variance CI with bootstrap CI on residuals across ALL 9 families for ALL 8 cells)
2. Expand the natural-sample pairing matrix (22 rows + Method Agreement Matrix heatmap, fixed-axis convention)
3. Complete automatic finalization (update manuscript Table 2, insert §4.7 natural-sample section, render all figures, run test suite)

Read-write scope: `results/`, `src/`, `scripts/`, `figures/`, `manuscripts/`, `tests/`. Do NOT edit `data/`, `docs/preregistration/`.

## Deliverable 1 — Fix the zero-std artifact

### Diagnostic step (first, before any code changes)

Open `python3 -i`, load the opx_multiseed_summary.csv, and for every (pipeline, model, feature_set, track, target) row print the 20-seed RMSE values to 8 decimal places (not rounded). Identify which cells have std < 0.001 AND verify whether this is a true zero-variance fit or a display precision artifact.

Write the diagnostic to `results/diagnostic_seed_variance_check.csv` with columns:
- pipeline, model, feature_set, track, target
- n_seeds
- rmse_std_8dp
- is_exactly_zero (bool)
- is_rounded_zero (bool, meaning < 1e-4 but > 1e-8)
- inferred_source (one of: deterministic_fit, seed_bug, rounding_artifact)

For every `inferred_source == seed_bug` row, run a minimal reproducibility script: fit the winning family on seed=42 split, then seed=43 split, then compare RMSE to 8dp. If the two RMSE values are identical at 8dp, escalate: the split-seed is not propagating. Patch `scripts/multiseed_refit.py` to fix the bug, re-run the affected cells, and regenerate `results/opx_multiseed_summary.csv` and `results/cpx_multiseed_summary.csv`. Log the patch in `results/seed_variance_patch_log.md`.

### Primary fix: bootstrap CI for every family × cell

Add a new evaluator in `src/evaluation.py`:

```python
import numpy as np
from sklearn.utils import check_random_state

def bootstrap_rmse_ci(y_true, y_pred, n_boot=500, alpha=0.05, random_state=42):
    """
    Bootstrap 95% CI on paired-residual RMSE.
    Works for deterministic models (ElasticNet) and stochastic models alike.
    Returns (rmse_point, ci_lower, ci_upper).
    """
    rng = check_random_state(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    boot_rmse = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rmse[b] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
    rmse_point = np.sqrt(np.mean((y_true - y_pred) ** 2))
    lo, hi = np.percentile(boot_rmse, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rmse_point, lo, hi
```

Add a companion function `bootstrap_rmse_ci_aggregated_over_seeds` that takes per-seed (y_true, y_pred) sample-level arrays from the 20-seed protocol and returns the median across seeds of the lower and upper CI bounds.

Write a new script `scripts/evaluation/compute_bootstrap_cis_all_cells.py` that:
1. Loads per-seed test-set predictions from `results/per_seed_test_predictions.parquet` (or equivalent; look in `results/` for files matching the pattern `*_test_predictions*` and determine the canonical source).
2. For every (pipeline, model, feature_set, track, target, seed) computes bootstrap CI using the function above with n_boot=500, seed=42.
3. Aggregates to (pipeline, model, feature_set, track, target) using the median-across-seeds rule.
4. Writes `results/bootstrap_rmse_cis_all_cells.csv` with columns:
   - pipeline, model, feature_set, track, target
   - rmse_point (20-seed mean)
   - rmse_seed_std (20-seed std, kept for backward compatibility)
   - rmse_ci_lo (median across seeds of bootstrap 2.5th percentile)
   - rmse_ci_hi (median across seeds of bootstrap 97.5th percentile)
   - ci_half_width (rmse_ci_hi - rmse_ci_lo) / 2
5. Include all 9 families (8 tuned + TabPFN) on every cell where predictions exist. Use `ci_source = 'bootstrap_n500'` column for provenance.

If `per_seed_test_predictions.parquet` does not exist, reconstruct it by running `scripts/multiseed_refit.py` in a prediction-save mode. Do not silently skip a cell — every family × cell needs a CI.

### Propagate to manuscript Table 2

Update `manuscripts/opx_2026/tables/T2_model_roster.csv` (and corresponding `.tex`) with columns:
- pipeline, track, target
- winning family, feature_set
- RMSE (20-seed mean)
- 95% CI [lo, hi]
- 2nd-place family, RMSE, CI
- TabPFN rank (1 to 9) and CI

If T2 does not yet exist, create it from `results/opx_multiseed_summary.csv`, `results/cpx_multiseed_summary.csv`, `results/tabpfn_multiseed_summary.csv`, and `results/bootstrap_rmse_cis_all_cells.csv` joined on the key columns.

Add a footnote to T2: "95% CI computed by bootstrap resampling (n_boot=500) of test-set residuals per seed, aggregated as median across 20 seeds. Bootstrap CI is non-trivial for all families including deterministic fits (ElasticNet), and replaces seed-variance std as the primary uncertainty estimate."

### Update Results Section 4.1

Edit `manuscripts/opx_2026/text/draft/sections/04_results.md`:
- Find paragraph 2 of §4.1 (starting "Five different model families win...") and replace every `± std` with `[95% CI lo, hi]` format.
- Example: `"ElasticNet (raw features) wins opx-liq T at 77.06 °C [CI 72.1, 82.3]"` (replace with actual CI values from the bootstrap CSV).
- Add a one-sentence methodological note at the end of §4.1 paragraph 2: "All uncertainty intervals are bootstrap 95% CIs computed on test-set residuals; see Methods §3.4 for protocol."
- Ensure Table 2 is referenced at the correct sentence.

### Update Methods Section 3.4

Edit `manuscripts/opx_2026/text/draft/sections/03_methods.md`:
- Find §3.4 and add a new subsection "**Uncertainty quantification.**" with the bootstrap CI description.
- Exact text to insert: "Uncertainty for every (family, feature set, track, target) cell is quantified by bootstrap resampling of the test-set residuals at each of the 20 SPLIT_SEEDS (n_boot = 500 per seed, random_state = 42), aggregated as the median across seeds of the 2.5th and 97.5th percentile bounds. This protocol produces non-trivial 95% confidence intervals for deterministic fits (ElasticNet at frozen hyperparameters) and stochastic fits alike, and replaces the 20-seed RMSE standard deviation as the primary uncertainty estimate. Seed-variance std is retained as a secondary stability diagnostic in Supplementary Table S1."

### Test assertions

Add test cases to `tests/test_evaluation.py`:
- `test_bootstrap_ci_nontrivial_for_deterministic_fit`: synthetic data with a constant model must produce a non-trivial CI (CI width > 0).
- `test_bootstrap_ci_deterministic_given_seed`: same inputs, same random_state must produce identical CIs.
- `test_bootstrap_rmse_cis_all_cells_exists_and_complete`: load `results/bootstrap_rmse_cis_all_cells.csv` and assert that every (pipeline, model, feature_set, track, target) combination from `opx_multiseed_summary.csv` and `cpx_multiseed_summary.csv` has a matching row with non-null CI columns.

Run `python -m pytest tests/test_evaluation.py -v` and confirm all tests pass before proceeding to Deliverable 2.

## Deliverable 2 — Natural-sample pairing matrix

### Scope

The current natural-sample validation uses a 327-pair LEPR two-pyroxene corpus (`results/nb08_natural_predictions.csv` and related files). Expand to a 22-row pairing list plus a 10-method Method Agreement Matrix. Per user decision: keep y-axes in native units with fixed ranges across panels; do NOT compute CR-RMSE or other normalized metrics.

### Final pairing list (22 rows)

Implement in `scripts/pairing/compute_pairing_matrix.py`. For each row, compute on the 327 LEPR natural samples: RMSE of disagreement (T) and RMSE of disagreement (P), with bootstrap CIs (n_boot=500). Also compute mean-absolute-difference and median-absolute-difference.

| Row | Method A | Method B |
|---|---|---|
| A | our ML opx-liq | our ML cpx-liq |
| B | our ML opx-liq | Ágreda-López 2024 cpx-liq |
| C | our ML opx-liq | Putirka 2008 opx-liq (28a + 29a) |
| D | our ML opx-liq | Jorgenson 2022 cpx-liq |
| E | our ML opx-liq | Wang 2021 cpx |
| F | our ML opx-liq | our ML opx-only |
| G | our ML opx-only | Putirka 2-px eq 36/37 + 39 |
| H | our ML opx-only | Brey & Köhler 1990 two-pyroxene |
| I | our ML opx-only | our ML cpx-only |
| J | Ágreda-López 2024 cpx-only | Jorgenson 2022 cpx-only |
| M | our ML opx-only | Ágreda-López 2024 cpx-only |
| N | our ML opx-only | Jorgenson 2022 cpx-only |
| O | our ML opx-only | Putirka 32a/32b/32c cpx-only |
| P | our ML opx-only | Wang 2021 cpx |
| Q | our ML cpx-liq | Ágreda-López 2024 cpx-liq |
| R | our ML cpx-liq | Jorgenson 2022 cpx-liq |
| S | our ML cpx-only | Ágreda-López 2024 cpx-only |
| T | our ML cpx-only | Jorgenson 2022 cpx-only |
| U | our ML opx-liq T | Putirka 28a opx-liq T |
| V | our ML opx-liq P | Putirka 29a opx-liq P |

(Rows K and L intentionally omitted per user decision.)

### Output CSV schema

`results/pairing_matrix_22rows.csv` with columns:
- row_label (A through V)
- method_a_name, method_b_name
- n_pairs (should be 327 for all rows unless a method cannot predict on a subset)
- T_rmse_disagreement, T_ci_lo, T_ci_hi, T_mean_abs_diff, T_median_abs_diff
- P_rmse_disagreement, P_ci_lo, P_ci_hi, P_mean_abs_diff, P_median_abs_diff
- method_a_rmse_vs_truth_T, method_a_rmse_vs_truth_P (if LEPR has reported values; mark NaN otherwise)
- method_b_rmse_vs_truth_T, method_b_rmse_vs_truth_P (same)

### Method Agreement Matrix

Build `scripts/pairing/make_fig_method_agreement_matrix.py`. Output:
- `figures/core/Core_13a_fig_agreement_matrix_T.pdf` and `.png`
- `figures/core/Core_13b_fig_agreement_matrix_P.pdf` and `.png`
- `figures/core/Core_13_txt_caption.txt`

Methods to include on both axes (10 × 10 matrix):
1. our ML opx-liq
2. our ML opx-only
3. our ML cpx-liq
4. our ML cpx-only
5. Putirka opx-liq (28a/29a)
6. Putirka 2-px (eq 36/37 + 39)
7. Ágreda-López 2024 cpx-liq
8. Ágreda-López 2024 cpx-only
9. Jorgenson 2022 cpx-liq
10. Jorgenson 2022 cpx-only

Heatmap cell value: RMSE of disagreement between method i and method j on the 327 pairs. Diagonal is zero. One matrix for T (colormap 0-200 °C), one for P (colormap 0-15 kbar). Use a sequential viridis-like colormap. Cluster methods by agreement (apply scipy.cluster.hierarchy on the average-disagreement matrix and reorder rows/columns). Annotate each cell with its RMSE value.

Caption text for Core_13:
"Method Agreement Matrix on 327 LEPR natural two-pyroxene pairs. Cell (i, j) shows the RMSE of disagreement between method i and method j on T (panel a) and P (panel b). Methods are hierarchically clustered by average pairwise disagreement. Tight clusters indicate methods in the same calibration community; outlier methods disagree most with the consensus. Our ML opx-only [row 2] is compared to 9 alternative methods simultaneously."

### Paired-comparison panel figures (fixed axes)

Build `scripts/pairing/make_fig_pairing_panels.py`. Output:
- `figures/core/Core_14a_fig_pairing_panels_T.pdf`
- `figures/core/Core_14b_fig_pairing_panels_P.pdf`
- `figures/core/Core_14_txt_caption.txt`

Layout: 22 subpanels in a 4 × 6 grid (one panel per row A-V). Each panel is a 1:1 scatter plot of Method A predicted vs Method B predicted on the 327 pairs.

**Fixed axis convention per user instruction:**
- T panels: both axes [750, 1800] °C with identity line and ±50 °C envelope as dashed grey lines
- P panels (opx + opx-liq + mixed): both axes [0, 60] kbar with identity line and ±5 kbar envelope
- P panels (cpx-only pure): both axes [0, 150] kbar with identity line and ±10 kbar envelope

Annotate each panel with: row label, n_pairs, RMSE of disagreement (with CI), and a color-coded point cloud if the sample has an equilibrium_pair_flag.

### Update Results Section 4.7 (restore natural-sample content)

Re-insert §4.7 in `manuscripts/opx_2026/text/draft/sections/04_results.md` after §4.6. Content:

- Paragraph 1: corpus description (327 LEPR two-pyroxene pairs, reference to Methods §3.10 for LEPR SHA256 provenance).
- Paragraph 2: Row A headline — our opx-liq vs our cpx-liq within-pipeline consistency. Report T_rmse_disagreement and P_rmse_disagreement with CIs.
- Paragraph 3: Rows F, I — our internal-consistency checks (opx-liq vs opx-only; opx-only vs cpx-only). Interpret the magnitude.
- Paragraph 4: Rows C, U, V — our opx-liq vs Putirka 2008 opx-liq.
- Paragraph 5: Rows G, H — our opx-only vs classical two-pyroxene (Putirka eq 36/37/39; Brey-Köhler 1990).
- Paragraph 6: Rows M, N, O, P — our opx-only vs published cpx methods.
- Paragraph 7: Rows Q, R, S, T — our cpx vs published cpx (replication sanity check).
- Paragraph 8: Method Agreement Matrix interpretation, with reference to Figure 13.
- Paragraph 9: Natural-sample bias finding — our opx-only reads systematically hotter and deeper than classical two-pyroxene and Jorgenson 2022 cpx-only references. State the bias numerically (T bias, P bias) and recommend against using our opx-only as primary T estimator on natural 2-px rocks when classical alternatives are available.

Use the actual numbers from `results/pairing_matrix_22rows.csv` — do not leave placeholders.

### Update Methods Section 3.10

Insert a new §3.10 "Natural-sample validation protocol" in `03_methods.md` before the current Software/Reproducibility section. Content:

- LEPR data source, curation date, SHA256 prefix `c96ffea4`
- Filter rules (two-pyroxene inner join on Experiment, 327 pairs after completeness filter)
- List of 10 methods compared on the corpus
- 22-row pairing specification
- Fixed-axis rules: [750, 1800] °C for T, [0, 60] kbar for P (opx-scope), [0, 150] kbar for P (cpx-only scope)
- Method Agreement Matrix construction (10×10, hierarchical clustering by average disagreement)
- Statement that no normalization (CR-RMSE, DoU) is applied; absolute RMSE in native units is the primary metric

Renumber the existing §3.10 Software/Reproducibility to §3.11.

## Deliverable 3 — Automatic finalization tasks

Execute these in order, after Deliverables 1 and 2 complete.

### 3.1 Regenerate all core figures

For every file matching `figures/core/Core_*.pdf`, run its producer script (listed in the Core_NN sidecar `.txt` file under "producer_script:"). If the producer script fails, log the failure and continue; do not block subsequent figures.

Confirm the final figure inventory matches:
- Core_01 through Core_12 (existing)
- Core_13a, Core_13b (new, method agreement matrices T and P)
- Core_14a, Core_14b (new, pairing panels T and P)

Update `figures/core/AUDIT.md` with the final inventory and producer-script mapping.

### 3.2 Terminology consistency sweep

Run case-sensitive grep across `manuscripts/opx_2026/text/draft/sections/` for these forbidden strings:
- `v10` (not in prose; CSV column names exempt)
- `Phase-G`, `phase-G`, `phase G`
- `ship-if-better`
- `two-axis honesty bar`
- `ships Form A`, `Form A ships`, `Form B ships`

For every match, rewrite using the canonical terminology:
- `v10` → `tuned ML baseline` or `tuned family`
- `Phase-G` → (remove; this is internal dev slang)
- `ship-if-better` → `tolerance-band acceptance rule`
- `two-axis honesty bar` → `dual robustness check`
- `ships Form A` → `accepts Form A`

Also grep for em dashes (`—` character U+2014). Replace every instance with a space-hyphen-space ` - ` per user style preference.

Log every replacement to `results/terminology_sweep_log.md` with file, line, before, after.

### 3.3 Bootstrap CI propagation to Results §4.1

Confirm that every instance of `± std` in `04_results.md` has been converted to `[CI lo, hi]` format. If any remain, replace them using values from `results/bootstrap_rmse_cis_all_cells.csv`.

### 3.4 Test suite run

Run the full preregistration test suite:

```bash
python -m pytest tests/test_preregistration.py tests/test_evaluation.py -v --tb=short
```

Expect T15 (headline claim pass), T19-T22 (bias-correction constants), and the three new bootstrap-CI tests to pass. If any fail, log to `results/test_suite_failure_log.md` and continue. Do not roll back completed work on a test failure — report and move on.

### 3.5 Word document regeneration

Run `python build_manuscript_docx.py` from project root to regenerate `manuscripts/opx_2026/opx_ml_thermobarometer_draft.docx` from the updated section markdown files. The build script is committed at the project root and uses Cambria font, all-black text, centered display equations, and plain-border tables. If the build script is not present, skip this step and log to `results/finalization_log.md`.

### 3.6 Audit refresh

Generate an updated manuscript audit to `MANUSCRIPT_WRITING_AUDIT_v2.md` (parallel to the existing v1). Same 15-section structure. Include:
- Section 1 pre-registration state (verify lock dates)
- Section 2 dataset inventory (unchanged)
- Section 3 model roster with bootstrap CIs (new)
- Section 4 external benchmarks with bootstrap CIs (new)
- Section 5 bias correction scorecard (unchanged; confirm cell counts)
- Section 6 regime results (unchanged)
- Section 7 TabPFN verdict (unchanged)
- Section 8 SHAP inventory (unchanged)
- Section 9 natural sample validation with 22-row pairing matrix (NEW, expanded)
- Section 10 augmentation ablation (unchanged)
- Section 11 figure traceability with Core_13 and Core_14 added (new)
- Section 12 manuscript autofill inventory with word counts (updated)
- Section 13 open issues — remove closed items (ElasticNet zero-std, natural-sample expansion), add any new issues raised during execution
- Section 14 compute provenance (unchanged)
- Section 15 gap checklist — mark completed items, flag remaining

### 3.7 Commit and branch management

After all steps complete:
1. Run `git status --short` and log dirty files to `results/finalization_log.md`.
2. Commit all changes in `results/`, `src/`, `scripts/`, `figures/`, `manuscripts/`, `tests/` with one of three commit messages depending on scope:
   - `"fix: bootstrap CI for ElasticNet zero-std, all 9 families"`
   - `"feat: 22-row pairing matrix + method agreement matrix Core_13/14"`
   - `"chore: manuscript finalization post-Lee-revision"`
3. Do not push. Leave the branch `lee_revisions_phase_1` local for review.
4. Produce a final delta report at `results/LEE_REVISIONS_PHASE_1_SUMMARY.md` with sections: Fixed Issues, New Artifacts, Known Limitations, Time Spent, Remaining TODO.

## Execution constraints

- Caveman tone in all generated prose: short sentences, no filler, no em dashes, no AI clichés.
- Manuscript prose stays in formal research style; only internal logs, TODOs, and commit messages use caveman.
- Every numeric claim in updated manuscript sections must cite a CSV + column. No fabrication.
- If a required CSV does not exist, do not invent values — flag in `results/finalization_log.md` with "NOT FOUND: expected at X, impact on manuscript Y."
- Run-time budget: ~6-8 hours of CPU. If a step would exceed 2 hours, checkpoint state and document the partial completion.
- If the TabPFN environment is not available in `.venv-tabpfn`, skip TabPFN-specific bootstrap-CI refits and use existing predictions from `results/tabpfn_multiseed_summary.csv`. Log the skip.
- If any LEPR-dependent computation fails because `data/LEPR.xlsx` is missing or has hash drift from `c96ffea4`, halt and report — do not fabricate natural-sample numbers.

## Acceptance criteria

Execution is complete when ALL of the following hold:

1. `results/bootstrap_rmse_cis_all_cells.csv` exists with ≥ 8 cells × ≥ 9 families = ≥ 72 rows, all with non-null CI columns.
2. `results/pairing_matrix_22rows.csv` exists with 22 rows (A-V excluding K, L).
3. `figures/core/Core_13a_fig_agreement_matrix_T.pdf` and `Core_13b_*.pdf` exist.
4. `figures/core/Core_14a_fig_pairing_panels_T.pdf` and `Core_14b_*.pdf` exist.
5. `manuscripts/opx_2026/text/draft/sections/03_methods.md` contains a new §3.4 paragraph on bootstrap CI and a new §3.10 on natural-sample protocol.
6. `manuscripts/opx_2026/text/draft/sections/04_results.md` contains §4.7 with 9 paragraphs using real numbers from the pairing CSV, and §4.1 uses `[CI lo, hi]` format instead of `± std`.
7. `manuscripts/opx_2026/tables/T2_model_roster.csv` exists with CI columns.
8. `MANUSCRIPT_WRITING_AUDIT_v2.md` exists at project root.
9. `results/LEE_REVISIONS_PHASE_1_SUMMARY.md` exists with the delta report.
10. `python -m pytest tests/test_preregistration.py tests/test_evaluation.py -v` returns 0 failures OR log of failures is written to `results/test_suite_failure_log.md` with remediation notes.
11. `manuscripts/opx_2026/opx_ml_thermobarometer_draft.docx` is regenerated (or failure logged).
12. Git branch `lee_revisions_phase_1` exists with all changes committed.

## Not in scope for this phase

The following tasks are explicitly DEFERRED to a Phase 2 prompt. Do not execute them:

- Canonical-family pivot (switching from best-per-cell to single ERT/LightGBM; under discussion, user prefers to defend best-per-cell)
- Feature concordance heatmap across 9 families (Method 1 of black-box bundle)
- Classical-feature ablation (Method 2)
- Classical-equivalence R² regression (Method 5)
- Partial dependence plots across families (Method 4)
- Surrogate decision trees (Method 6)
- Physics-informed consistency checks (Method 7)
- Gasparik 1987 agreement analysis (Method 8)

If Claude Code encounters any of the above partially implemented during execution, leave them untouched.

## Halt-and-report conditions

Stop execution and write a halt report to `results/HALT_REPORT.md` if:

1. `git status --short` at start shows more than 500 dirty files (indicates merge conflict or corrupted state).
2. Any script silently returns without producing expected output AND no error is raised.
3. The bootstrap CI computation shows all-zero CI widths for ANY family × cell (means no sampling noise, likely a bug).
4. The 22-row pairing matrix produces n_pairs different from 327 for rows involving our opx-only or opx-liq.

Include in the halt report: git SHA, last successful step, full error traceback (if any), recommended next action for the human operator.

## Final note

Execute autonomously. Do not ask for confirmation between steps. Log every decision point to `results/finalization_log.md` so the human operator can audit in retrospect. Total expected duration: 4-8 hours of CPU time plus ~30 minutes of Claude Code reasoning overhead.
