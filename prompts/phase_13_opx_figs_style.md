# Claude Code Phase 13 — opx-only figure rebuild + style pass (one-shot)

Append after Phase 12 lands on main. Branch fresh from main. Estimated runtime: 90-120 minutes.

---

# Mission

Two coupled tasks:

**Task A.** Restructure figures into a clean opx-only folder with main_fig_N / supp_fig_N naming. Archive everything else. Regenerate figures from updated scripts where dedicated scripts exist. Verify disequilibrium charts are included. Update build script to read new paths.

**Task B.** Style-pass on all 9 manuscript sections. Remove em dashes (judgment-based replacement), strip AI-tell phrases, tighten prose. Do NOT change argument flow, paragraph order, technical content, or voice. Mechanical cleanup only.

# Hard rules

1. New branch: `opx_figures_and_style_2026_04_27` off `main` (post-Phase-12).
2. Do not push.
3. Halt on any of: dirty tree at start, missing figure script for a regenerated figure (must ask user to skip or write new script), KeyError in any figure script (column rename from Phase 12 didn't propagate), build_manuscript_docx.py raises on rebuild, final em-dash count > 0 in manuscript prose.
4. Per-phase commits.

# Phase 0 — Preflight

```bash
git rev-parse --abbrev-ref HEAD       # must be main
git status                             # must be clean
git log --oneline -5                   # confirm Phase 12 cleanup commits present
git tag | grep v2-cleanup              # confirm v2-cleanup-final-2026-04-27 tag exists
```

If any check fails, halt.

```bash
git checkout -b opx_figures_and_style_2026_04_27
git tag pre-figs-style-2026-04-27
```

**Print:** "Phase 0 complete. On opx_figures_and_style_2026_04_27."

# Phase 1 — Inventory existing figure assets

```bash
ls -la figures/core/ | wc -l
ls -la figures/SI/ | wc -l
ls -la scripts/figures/ 2>/dev/null || echo "no scripts/figures dir"
find scripts/ -name "*.py" -path "*fig*" | head -30
find scripts/ -name "*.py" -path "*plot*" | head -30
```

Look for figure-generating scripts. Likely locations:
- `scripts/figures/`
- `scripts/figures_core/`
- `scripts/manuscript/figures/`
- Embedded in `notebooks/nbF_figures.ipynb`

For each main figure (Core_01 through Core_18) and each retained supplementary figure, classify as:
- **Has script** — regenerate-able
- **Orphan** — exists as PNG/PDF only, no source script (likely from a notebook)

Build inventory file `scripts/manuscript/figure_inventory.txt` with one line per figure:
```
Core_01,scripts/figures/make_dataset_map.py,has_script
Core_02,scripts/figures/make_citation_split.py,has_script
Core_03,UNKNOWN,orphan
...
```

**Halt and print:** "Figure inventory complete. N figures with scripts, M orphans. Proceed?"

Wait for user confirmation. If user says proceed, continue. If user says "make scripts for orphans," that's a separate task — halt and report what would be needed.

For orphan figures, default action is option (a): copy existing PNG from legacy folder to figures/opx_only/, rename to main_fig_N or supp_fig_N, write cleaned caption. Apply this default unless the orphan is data-driven and would benefit from regeneration.

# Phase 2 — Update figure scripts for column rename + opx-only

For each figure script identified in Phase 1:

## 2a. Column rename pass

Phase 12 renamed CSV columns from `v10*` to `our_*` and verdict cells from `v10_wins` to `our_method_better`. Figure scripts that read these CSVs need updating.

```bash
rg "v10|v10_best|v10_method|v10_rmse|v10_wins" scripts/figures/ scripts/ 2>/dev/null
```

For each match, replace with the corresponding new name. Reuse the renames dict from Phase 12:

```python
renames = {
    "v10": "our_method",
    "v10_best": "our_best_RMSE",
    "v10_best_model": "our_best_model",
    "v10_method": "our_method_label",
    "v10_rmse": "our_rmse",
    "v10_rmse_lo": "our_rmse_lo",
    "v10_rmse_hi": "our_rmse_hi",
}
```

Apply via str_replace per file. Halt if any script imports a column name that doesn't exist in the renames dict (means there's a column we missed in Phase 12).

## 2b. Drop cpx / twopx / universal pipelines

For each figure script that supports multiple pipelines:
- Find the pipeline-loop logic (often `for pipeline in ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only']`)
- Restrict to opx tracks only: `for pipeline in ['opx_liq', 'opx_only']`
- Find any panel-grid logic that hardcodes 4-row or 8-row layout (cpx + opx)
- Reduce to 2-row or 4-row (opx only)
- Find any `assert len(rows) == 8` or similar guards and update

Specific known scripts to update (verify against Phase 1 inventory):
- Figure 4 heatmap script: 4-row → 2-row
- Figure 9a per-regime by family: drop cpx panels
- Figure 9b aggregate by family: drop cpx
- Figure 12 SHAP: 8 panels → 4 panels
- Figure 15 feature concordance: 8 panels → 4 panels
- Figure 16 classical equivalence: 4 tracks → 2 tracks
- Figure 17 partial dependence: was opx-only already, verify

Re-run after edits:
```bash
rg "cpx_liq|cpx_only|twopx" scripts/figures/ scripts/manuscript/figures/ 2>/dev/null | head -20
```

Should be near-empty for any figures kept in main text. Some scripts may legitimately produce both pipelines and just suppress cpx output for this paper — that's fine, they just need a flag.

Commit:
```bash
git add scripts/figures/
git commit -m "phase-13(scripts): rename v10 columns, drop cpx/twopx panels"
```

# Phase 3 — Create new folders and archive existing

```bash
mkdir -p figures/opx_only
mkdir -p figures/legacy
```

Move all current core and SI figures to legacy:
```bash
git mv figures/core/* figures/legacy/ 2>/dev/null
git mv figures/SI/* figures/legacy/ 2>/dev/null
rmdir figures/core figures/SI 2>/dev/null
```

Verify nothing was missed:
```bash
ls figures/
# Should show: opx_only/, legacy/
ls figures/legacy/ | wc -l
# Should show 80+ files (everything from old core + SI)
```

Commit:
```bash
git add figures/
git commit -m "phase-13(structure): archive existing figures, create opx_only folder"
```

# Phase 4 — Regenerate figures into opx_only/

## 4a. Establish naming map

```
Old name                                    New name
Core_01_fig_dataset_map                  → main_fig_1
Core_03_fig_methods_flowchart            → main_fig_2  (Fig 2 was the deleted CV schematic)
Core_04_fig_nb04_cross_pipeline_heatmap  → main_fig_3
Core_05_fig30_bias_correction_per_regime → main_fig_4
Core_06_fig31_bias_correction_residuals  → main_fig_5
Core_07_fig34_bias_correction_scorecard  → main_fig_6
Core_08_fig45_opx_headline               → main_fig_7
Core_09a_fig_opx_regime_families         → main_fig_8a
Core_09b_fig_opx_overall_families        → main_fig_8b
Core_10_fig_best_vs_putirka              → main_fig_9
Core_12_fig_shap_winners                 → main_fig_10
Core_15_fig_feature_concordance          → main_fig_11
Core_16_fig_classical_equivalence        → main_fig_12
Core_17_fig_partial_dependence           → main_fig_13
Core_18_fig_surrogate_trees              → main_fig_14

DELETED: Core_02 (Figure 2, converted to text in Phase 12 if not yet done — verify)
DELETED: Core_11 (twopx 1:1)
DELETED: Core_13a/b (10x10 method matrix, cpx-laden)
DELETED: Core_14a (LEPR pairing)
DELETED: Core_01b, Core_10b/c/d (ArcPL, decided in Phase 12)
DELETED: TabPFN Figure 8 (handled in Phase 12)

Disequilibrium charts: Core_16 (classical equivalence) panels include
disequilibrium maps. Verify these are present in regenerated main_fig_12.
```

## 4b. Verify disequilibrium content

Phase 12 audit noted Core_16 caption: "Columns: (1) P 1:1 scatter of ML predicted vs classical-fit predicted in kbar, (2) T 1:1 scatter in Celsius, (3) ML-minus-classical-fit residual disequilibrium map."

Open the generation script for Core_16 (likely `scripts/figures/make_classical_equivalence.py` or similar). Verify it produces 3 columns per row including the disequilibrium residual map.

If the disequilibrium maps are missing from current Core_16, they need to be added back to the generation logic. Read the script to confirm.

If disequilibrium panels were lost in an earlier rebuild, regenerate by:
```python
# In the figure script, ensure third column produces:
# residual = y_ml - y_classical_fit
# scatter on (residual_T, residual_P) axes
# overlay tolerance box per-cell
```

Halt if the disequilibrium logic is unclear from the script alone.

## 4c. Regenerate main figures

For each main figure with a script:

```bash
python scripts/figures/make_dataset_map.py --opx-only --output figures/opx_only/main_fig_1
python scripts/figures/make_methods_flowchart.py --opx-only --output figures/opx_only/main_fig_2
python scripts/figures/make_family_heatmap.py --opx-only --output figures/opx_only/main_fig_3
# ... etc for all 14 main figures
```

If a script doesn't accept `--opx-only` or `--output` flags, run it as-is and rename the output. Check before each call:
```bash
python scripts/figures/<script>.py --help 2>/dev/null | head -10
```

For each regenerated figure, write a fresh `.txt` caption sidecar in `figures/opx_only/main_fig_N.txt`. The caption should be the existing caption from `figures/legacy/<old_name>.txt` BUT cleaned of:
- v10 references
- Amendment / Phase narrative
- cpx / twopx / universal mentions
- "v3 ship rule"
- File paths with `_v3.csv` or `_v2.csv`

Replace caption file in opx_only/, do NOT keep the old caption verbatim.

For orphan figures (no script): copy the PNG from `figures/legacy/` to `figures/opx_only/main_fig_N.png`, do the same for `.pdf` if present, and write a fresh caption.

```bash
cp figures/legacy/Core_03_fig_methods_flowchart.png figures/opx_only/main_fig_2.png
cp figures/legacy/Core_03_fig_methods_flowchart.pdf figures/opx_only/main_fig_2.pdf
# Write cleaned caption
echo "Methods flowchart. Three phases: data preprocessing (citation-grouped 80/20 split), nine-family model training (Optuna-tuned + TabPFN default), and evaluation (10-fold OOF for bias correction, held-out test for canonical scorecard)." > figures/opx_only/main_fig_2.txt
```

Halt if any expected file fails to materialize in `figures/opx_only/`.

## 4d. Regenerate supplementary figures

Supplementary figures retained per Phase 12 audit (after dropping cpx/twopx/ArcPL/TabPFN-specific):

```
Old name                                    New name
fig24_per_regime_rmse_opx_liq            → supp_fig_1
fig25_per_regime_residual_violins_opx_liq → supp_fig_2
fig26_generalization_opx_liq             → supp_fig_3
fig27_shap_summary_opx_liq               → supp_fig_4
fig28_bias_correction_opx_liq            → supp_fig_5
fig30_bias_correction_per_regime_rmse    → supp_fig_6  (was 8 cells, now 4 opx)
fig31_bias_correction_residuals          → supp_fig_7  (was 8 cells, now 4 opx)
fig32_bias_correction_form_comparison    → supp_fig_8  (opx-only after rebuild)
fig33_bias_correction_per_seed_stability → supp_fig_9  (opx-only after rebuild)
fig34_bias_correction_scorecard_delta    → supp_fig_10 (opx-only after rebuild)
fig_aug01_ship_verdict_comparison        → supp_fig_11
fig_aug02_aggregate_rmse_delta           → supp_fig_12
fig_aug03_residual_structure_per_regime  → supp_fig_13
fig_aug04_form_b_breakpoint_stability    → supp_fig_14

DELETED: fig29 (twopx benchmark)
DELETED: fig35 (TabPFN cpx vs opx)
DELETED: fig44 (TabPFN bias scoreboard)
DELETED: fig45 (already main_fig_7 as Core_08)
DELETED: fig_h5ac, fig_nb04*, fig_nb08* (broken stubs)
DELETED: ArcPL figures
```

For figures that previously contained 8 cells (4 opx + 4 cpx), regenerate with opx-only restriction. Verify the resulting layout is sensible (don't just hide cpx panels, drop them).

For each, regenerate via the appropriate script with `--opx-only` flag (or restricted pipeline list), output to `figures/opx_only/supp_fig_N.png` + `.pdf`, write cleaned caption sidecar.

## 4e. Verify file count

```bash
ls figures/opx_only/*.png | wc -l
# Expect: 14 main + 14 supp = 28 PNGs (or 29 if main_fig_8 has 8a + 8b separate files)
```

```bash
ls figures/opx_only/*.txt | wc -l
# Same count as PNGs
```

Halt if any figure is missing.

Commit:
```bash
git add figures/opx_only/
git commit -m "phase-13(figures): regenerate opx-only figures with main/supp naming"
```

# Phase 5 — Update build_manuscript_docx.py

`scripts/manuscript/build_manuscript_docx.py` currently references `figures/core/Core_01_*.png` etc. Update to read from new folder.

## 5a. Path constants

```python
# Old:
FIGURES_CORE = PROJECT_ROOT / "figures" / "core"
FIGURES_SI = PROJECT_ROOT / "figures" / "SI"

# New:
FIGURES_DIR = PROJECT_ROOT / "figures" / "opx_only"
```

## 5b. main_figures list

```python
main_figures = [
    ("Figure 1", "main_fig_1.png"),
    ("Figure 2", "main_fig_2.png"),
    ("Figure 3", "main_fig_3.png"),
    ("Figure 4", "main_fig_4.png"),
    ("Figure 5", "main_fig_5.png"),
    ("Figure 6", "main_fig_6.png"),
    ("Figure 7", "main_fig_7.png"),
    ("Figure 8a", "main_fig_8a.png"),
    ("Figure 8b", "main_fig_8b.png"),
    ("Figure 9", "main_fig_9.png"),
    ("Figure 10", "main_fig_10.png"),
    ("Figure 11", "main_fig_11.png"),
    ("Figure 12", "main_fig_12.png"),
    ("Figure 13", "main_fig_13.png"),
    ("Figure 14", "main_fig_14.png"),
]
```

## 5c. SI figures list

```python
supp_figs = sorted(FIGURES_DIR.glob("supp_fig_*.png"))
```

## 5d. Drop legacy fallbacks

Remove any code path that looks at `figures/core/` or `figures/SI/`. If the build script has retry logic for missing files, simplify it — if a file is missing after Phase 4, halt rather than silently skip.

Verify with a dry run if supported, otherwise just attempt build and check exit code.

Commit:
```bash
git add scripts/manuscript/build_manuscript_docx.py
git commit -m "phase-13(build): point build script at figures/opx_only"
```

# Phase 6 — Style pass

Mechanical cleanup of all 9 manuscript section files. Goal: remove AI-tell signatures without changing voice or argument.

## 6a. Em dash replacement

```bash
rg -c "—" manuscripts/opx_2026/text/draft/sections/
```

Should return non-zero in most files. Replace each em dash judgment-based:

**Replacement rules (general guidance, Claude Code uses judgment):**

1. **Parenthetical aside** → comma pair
2. **Strong break / pivot** → period
3. **Range / numeric span** → en dash (`–`), NOT em dash. Don't replace en dashes.
4. **List opener** → colon
5. **Definition / amplification** → colon or comma
6. **Sentence break for emphasis** → period

After replacement:
```bash
rg "—" manuscripts/opx_2026/text/draft/sections/
```
Must be empty. Halt if any em dash remains.

## 6b. AI-tell phrase removal

Search for and rewrite or delete:

```bash
rg -i "it is worth noting|notably|interestingly|of note|importantly|crucially|fundamentally|essentially,|indeed,|in fact,|that being said|with that said|having said that|moreover,|furthermore,|additionally," manuscripts/opx_2026/text/draft/sections/
```

For each match:
- "It is worth noting that X" → state X directly
- "Notably," / "Importantly," / "Interestingly," at sentence start → delete the word
- "Indeed," / "In fact," / "Moreover," / "Furthermore," / "Additionally," → delete or replace with "Also"
- "Fundamentally" / "Essentially" / "Crucially" → almost always droppable

```bash
rg -i "we read this|we treat this|we frame this|we position this|we interpret this" manuscripts/opx_2026/text/draft/sections/
```

For each match: rewrite as direct claim.

```bash
rg -i "by construction|by design|tends to|broadly speaking|in some sense|on balance|all things considered" manuscripts/opx_2026/text/draft/sections/
```

For each match: usually droppable.

```bash
rg -i "natural reviewer|natural question|reasonable definition|by any reasonable|the field|the literature|the community" manuscripts/opx_2026/text/draft/sections/
```

For each match:
- "The natural reviewer response" → state response directly
- "The natural question is" → just ask the question
- "By any reasonable definition" → delete
- "The field" / "The literature" / "The community" → name specific authors or drop

```bash
rg -i "we do not claim|we are not claiming|we do not assert|to be clear" manuscripts/opx_2026/text/draft/sections/
```

For each match: usually a hedge stack. Either delete or rewrite as direct negative claim.

## 6c. Tricolon reduction

```bash
rg "\b\w+, \w+, and \w+" manuscripts/opx_2026/text/draft/sections/ | head -20
```

Audit each:
- Legitimate enumeration (model names, etc.) → keep
- Rhetorical tricolon → cut to two
- Adjective stack → pick the strongest, drop the others

Use judgment. Don't auto-reduce.

## 6d. Specific known offenders

`01_introduction.md`:
- "Three gaps in the existing literature motivate this work." → "Three gaps motivate this work."
- "is a real gap" / "is a real explainability gap" / "is a real limitation" → drop "real"

`04_results.md`:
- "Each layer is its own falsifiable test with a pre-registered threshold; the bundle as a whole asks whether..." → trim "as a whole asks whether"

`05_discussion.md`:
- "We do not read the opx-only P result as evidence that ML thermobarometry has 'solved' orthopyroxene barometry." → "The opx-only P result does not solve orthopyroxene barometry."
- "Three caveats constrain the generality of the finding." → state caveats directly with numbers
- "We interpret this asymmetry as reflecting a real geologic difference rather than an artifact" → "The asymmetry reflects a real geologic difference, not an artifact"

`06_conclusions.md`:
- "Five contributions distinguish this work from prior ML thermobarometer literature." (now four after Phase 12) → "Four contributions follow."
- "We position these nulls as findings in their own right" → "These nulls are findings in their own right"

`09_cover_letter.md`:
- "Four features of this work make it a natural fit for JGR:MLC" → "Four features fit JGR:MLC's scope:"

## 6e. Verification pass

After all 6a-6d changes:

```bash
rg "—" manuscripts/opx_2026/text/draft/sections/ && echo "FAIL: em dashes remain"
rg -ic "it is worth noting|notably," manuscripts/opx_2026/text/draft/sections/
rg -ic "we read this|by construction" manuscripts/opx_2026/text/draft/sections/
```

First should be empty. Second and third should be near-zero (< 5 hits combined).

Commit:
```bash
git add manuscripts/opx_2026/text/draft/sections/
git commit -m "phase-13(style): em dash removal + AI-tell cleanup"
```

# Phase 7 — Update prose figure references

The figure renumbering from Phase 4 cascades into prose. References like "Figure 4 places TabPFN" or "Figure 9a displays" are now wrong.

```bash
rg "Figure \d+|Figure \d+[ab]" manuscripts/opx_2026/text/draft/sections/ | head -40
```

Renumbering map:

```
Old Figure 1   → Figure 1
Old Figure 2   → DELETED (now in §2.3 prose)
Old Figure 3   → Figure 2
Old Figure 4   → Figure 3
Old Figure 5   → Figure 4
Old Figure 6   → Figure 5
Old Figure 7   → Figure 6
Old Figure 8   → DELETED (TabPFN, removed in Phase 12)
Old Figure 9a  → Figure 8a
Old Figure 9b  → Figure 8b
Old Figure 10  → Figure 9
Old Figure 11  → DELETED (twopx)
Old Figure 12  → Figure 10
Old Figure 13a → DELETED
Old Figure 13b → DELETED
Old Figure 14a → DELETED
Old Figure 15  → Figure 11
Old Figure 16  → Figure 12
Old Figure 17  → Figure 13
Old Figure 18  → Figure 14
```

Use str_replace per substitution. Verify:
```bash
rg "Figure 1[5-8]|Figure Core_" manuscripts/opx_2026/text/draft/sections/
```
Should be empty.

Commit:
```bash
git add manuscripts/opx_2026/text/draft/sections/
git commit -m "phase-13(refs): renumber figure references in prose"
```

# Phase 8 — Rebuild docx

```bash
python scripts/manuscript/build_manuscript_docx.py
```

Verify:
```bash
ls -la manuscripts/opx_2026/arxiv_submission/manuscript.docx
```

Final inspection:
```python
python -c "
from docx import Document
import re
doc = Document('manuscripts/opx_2026/arxiv_submission/manuscript.docx')
n_para = len(doc.paragraphs)
n_tables = len(doc.tables)

em_dash_hits = sum(1 for p in doc.paragraphs if '—' in p.text)
v10_hits = sum(1 for p in doc.paragraphs if 'v10' in p.text or 'v9' in p.text)
ai_tells = sum(1 for p in doc.paragraphs if any(t in p.text.lower() for t in ['it is worth noting', 'notably,', 'interestingly,', 'we read this', 'by construction']))
fig_hits = sum(1 for p in doc.paragraphs if re.search(r'Figure (1[5-8]|2[0-9])', p.text))

print(f'Paragraphs: {n_para}')
print(f'Tables: {n_tables}')
print(f'Em dashes remaining: {em_dash_hits}')
print(f'v10/v9 mentions: {v10_hits}')
print(f'AI-tell phrases: {ai_tells}')
print(f'Figure refs out of valid range: {fig_hits}')
"
```

Halt if:
- em_dash_hits > 0
- v10_hits > 0
- ai_tells > 5
- fig_hits > 0

Commit:
```bash
git add manuscripts/opx_2026/arxiv_submission/manuscript.docx
git commit -m "phase-13(build): rebuild docx with new figures and style cleanup"
```

# Phase 9 — Final report

```
================================================================
PHASE 13 OPX-FIGURES + STYLE COMPLETE

Branch: opx_figures_and_style_2026_04_27
Pre-tag: pre-figs-style-2026-04-27
Output: manuscripts/opx_2026/arxiv_submission/manuscript.docx

Figure restructure:
  - figures/opx_only/         <N> main + <M> supp figures
  - figures/legacy/           ~80 archived files
  - figures/core/             REMOVED
  - figures/SI/               REMOVED

Main figures: <count>
Supp figures: <count>
Disequilibrium charts: present in main_fig_12

Style pass results:
  Em dashes: <before> → 0
  AI-tells: <before> → <after>
  v10 mentions: 0
  Figure references: all valid

Manuscript:
  Paragraphs: <count>
  Tables: <count>
  Size: <size> MB

Next steps:
  1. Visually scan manuscript.docx
  2. If satisfied:
       git checkout main
       git merge --no-ff opx_figures_and_style_2026_04_27
       git tag figs-style-final-2026-04-27
  3. Remaining pre-submission items:
     - Bibliography (99_references.md)
     - LaTeX equation rendering pass
     - Front-matter ORCIDs and email
     - TabPFN version pin tightening

Recovery:
  git reset --hard pre-figs-style-2026-04-27
  or git checkout main && git branch -D opx_figures_and_style_2026_04_27
```

# Halt conditions

- Phase 0: not on main, dirty tree, Phase 12 cleanup tag missing
- Phase 1: figure inventory fails to build
- Phase 2: KeyError in any figure script after column rename
- Phase 3: figure count mismatch after archive
- Phase 4: any expected file missing after regeneration; orphan figure handling unclear (default to option a unless specified)
- Phase 4b: disequilibrium logic missing from main_fig_12 script
- Phase 5: build script can't find files in opx_only/
- Phase 6: em dashes remain after replacement pass
- Phase 7: figure references in invalid range remain
- Phase 8: any of em_dash, v10, ai_tells, fig_hits checks fail
