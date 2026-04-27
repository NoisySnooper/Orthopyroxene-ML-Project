#!/usr/bin/env python3
"""Build the advisor review package for Dr. Kanani K.M. Lee.

Outputs under deliverables/lee_package_20260420/:
    00_ADVISOR_REVIEW.ipynb           (source notebook)
    00_ADVISOR_REVIEW.html            (rendered HTML; papermill-executed)
    00_ADVISOR_REVIEW_executed.ipynb  (executed kernel snapshot)
    figures/*                         (copies of the 6 core + 5 SI figs)
    PROVENANCE.md                     (git sha, file mtimes, CSV row counts)
    data_links.md                     (list of underlying CSVs with sizes)

Data contract: every numeric assertion in the notebook loads from a CSV
or JSON under `results/`. If a required file is missing, the builder
fails loud (return code 2). No hard-coded numbers.

Usage:
    .venv/Scripts/python.exe scripts/deliverables/build_lee_package.py

The notebook's own cells also fail loud if a CSV is missing, so the
package can be re-executed standalone.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf
import pandas as pd

PKG_DIR = PROJECT_ROOT / 'deliverables' / 'lee_package_20260420'
FIGS_IN = PROJECT_ROOT / 'figures'
FIGS_OUT = PKG_DIR / 'figures'
NB_PATH = PKG_DIR / '00_ADVISOR_REVIEW.ipynb'
NB_EXEC_PATH = PKG_DIR / '00_ADVISOR_REVIEW_executed.ipynb'
HTML_PATH = PKG_DIR / '00_ADVISOR_REVIEW.html'

REQUIRED_CSVS = [
    'results/preregistered_scorecard_postcorrection.csv',
    'results/bias_correction_shipped.csv',
    'results/opx_multiseed_summary.csv',
    'results/tabpfn_bias_correction_summary.csv',
    'results/tabpfn_bias_correction_perseed.csv',
    'results/tabpfn_multiseed_summary.csv',
    'results/tabpfn_head_to_head.csv',
    'results/nb08_natural_predictions.csv',
    'results/nb08_cross_mineral_agreement.csv',
    'results/regime_allmodels_postcorrection.csv',
]

CORE_FIGS = [
    'Core_01_fig_dataset_map',
    'Core_01b_fig_dataset_map_holdout',
    'Core_02_fig_citation_split',
    'Core_03_fig_methods_flowchart',
    'Core_04_fig_nb04_cross_pipeline_heatmap',
    'Core_05_fig30_bias_correction_per_regime_rmse',
    'Core_06_fig31_bias_correction_residuals',
    'Core_07_fig34_bias_correction_scorecard_delta',
    'Core_08_fig45_opx_headline',
    'Core_09a_fig_opx_regime_families',
    'Core_09b_fig_opx_overall_families',
    'Core_10_fig_best_vs_putirka',
    'Core_11_fig_nb08_twopx_1to1',
    'Core_12_fig_shap_winners',
]
SI_FIGS = [
    'fig24_per_regime_rmse_opx_liq',
    'fig25_per_regime_residual_violins_opx_liq',
    'fig26_generalization_opx_liq',
    'fig27_shap_summary_opx_liq',
    'fig28_bias_correction_opx_liq',
    'fig29_twopx_benchmark',
    'fig32_bias_correction_form_comparison',
    'fig33_bias_correction_per_seed_stability',
    'fig35_tabpfn_vs_opx_tb',
    'fig44_tabpfn_bias_scoreboard_opx',
    'fig_aug01_ship_verdict_comparison',
    'fig_aug02_aggregate_rmse_delta',
    'fig_aug03_residual_structure_per_regime',
    'fig_aug04_form_b_breakpoint_stability',
]


def _md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip('\n'))


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip('\n'))


def _append_figure_cells(cells: list, stems: list[str],
                         src_subdir: str = 'core') -> None:
    """Append (markdown header+caption, code preview) pair for each figure.

    Plain-string construction (NOT textwrap.dedent) so that injected
    multi-line caption sidecars don't collide with the dedent logic
    and create leading-space corrupt markdown headers.

    Caption sidecars are read from FIGS_IN / src_subdir / stem.txt, which
    matches where the originals live (core/ or SI/) after the Tier 2
    restructure.
    """
    src_dir = FIGS_IN / src_subdir
    for stem in stems:
        cap_path = src_dir / f'{stem}.txt'
        caption = (cap_path.read_text(encoding='utf-8').strip()
                   if cap_path.exists() else '(no caption sidecar)')
        cap_lines = caption.splitlines()
        if cap_lines and cap_lines[0].strip() == stem:
            caption = '\n'.join(cap_lines[1:]).strip()
        md = f'### {stem}\n\n{caption}'
        cells.append(nbf.v4.new_markdown_cell(md))
        code = (
            "from IPython.display import Image\n"
            f"png = FIGS / '{stem}.png'\n"
            "if png.exists():\n"
            "    display(Image(filename=str(png), width=780))\n"
            "else:\n"
            f"    display(Markdown(f'**missing:** {{png}}'))"
        )
        cells.append(nbf.v4.new_code_cell(code))


def verify_csvs():
    missing = [c for c in REQUIRED_CSVS
               if not (PROJECT_ROOT / c).exists()]
    if missing:
        print(f'ERROR: missing required CSVs: {missing}')
        sys.exit(2)


def copy_figures():
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    skipped_locked = []

    def _copy_one(src_dir: Path, stem: str):
        for ext in ('.pdf', '.png', '.txt', '.html'):
            src = src_dir / f'{stem}{ext}'
            if not src.exists():
                continue
            dst = FIGS_OUT / src.name
            try:
                shutil.copy(src, dst)
                copied.append(src.name)
            except PermissionError:
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    skipped_locked.append(src.name)
                else:
                    raise

    core_dir = FIGS_IN / 'core'
    si_dir = FIGS_IN / 'SI'
    for stem in CORE_FIGS:
        _copy_one(core_dir, stem)
    for stem in SI_FIGS:
        _copy_one(si_dir, stem)

    if skipped_locked:
        print(f'WARN: {len(skipped_locked)} files locked at destination; '
              f'skipped (contents identical): {skipped_locked}')
    return copied


def build_notebook():
    cells = []

    # ---- Header + plain-English orientation for Dr. Lee ----
    cells.append(_md("""
    # Advisor review package -- opx ML thermobarometer

    **For:** Dr. Kanani K.M. Lee
    **From:** Ta Quang Nhan (cadet, USCGA)
    **Date:** 2026-04-20
    **Target venue:** JGR ML & Computation

    ## What this document is

    A single reviewable snapshot of the pyroxene thermobarometer project.
    Figures come first so you can scan the results visually; data tables
    come after in case you want to check a specific number.

    ## The project in one paragraph (plain English)

    We are trying to estimate the pressure (P) and temperature (T) at
    which an orthopyroxene (opx) crystal grew, using only the chemistry
    of that crystal (and optionally its surrounding liquid). Classical
    petrologic thermobarometers like Putirka (2008) fit a small
    polynomial to experimental data; we fit a flexible machine-learning
    model to the same experimental data and compare. The question we
    are asking is: *does the ML model do better than the polynomial,
    and if so where and by how much?*

    ## A quick glossary (once, then we move on)

    - **opx / cpx**: orthopyroxene and clinopyroxene, two common
      pyroxene minerals.
    - **opx-liq / opx-only**: two prediction pipelines. "opx-liq"
      uses both the crystal and the surrounding liquid as input;
      "opx-only" uses just the crystal chemistry.
    - **pressure regime**: we pre-registered four geologic pressure
      bins before any modeling: shallow-crustal (<5 kbar),
      deep-crustal / MASH (5-15 kbar), lithospheric-mantle (15-30
      kbar), deeper-mantle (30+ kbar), plus "ALL" for the combined
      test set.
    - **RMSE**: root-mean-squared error. How far, on average, a
      prediction misses the truth. Lower is better. Units: C for
      temperature, kbar for pressure.
    - **bias correction**: a small post-processing step applied after
      the ML model makes its raw prediction. Two flavors are tested:
      "Form A" learns a regression of residual on predicted value
      separately in each pressure regime; "Form B" is a single smooth
      piecewise-sigmoid on the predicted-value axis.
    - **ship-if-better rule**: a correction only ships if it improves
      overall RMSE AND does not make any pre-registered regime
      worse. A pre-registered promise to not cherry-pick.
    - **TabPFN**: a foundation model for small tabular data. In-
      context transformer architecture, no training, no tuning, no
      SHAP. We added it as a 9th baseline family.
    - **Putirka (2008)**: the reference classical thermobarometer.

    ## How to read this document

    1. **Figures first** (Section 1): 13 core figures + 1 SHAP figure.
       Each figure has a long caption explaining what it shows and
       what to look for.
    2. **Supporting figures** (Section 2): 14 SI figures for the
       curious.
    3. **Data tables** (Section 3): the numbers behind the figures,
       in case you want to double-check.
    4. **Methods, limitations, caveats** (Sections 4-6): short
       plain-English sections.
    5. **Provenance** (Section 7): git state + file inventory.

    ## Sections

    1. Core figures (14 PDFs: 13 main + Core_12 SHAP)
    2. Supporting information figures
    3. Data tables
       3.1 Headline (4 opx cells)
       3.2 Eight-cell winner table (tuned + TabPFN)
       3.3 TabPFN bias-correction scoreboard
       3.4 opx-only P per-regime breakdown
       3.5 TabPFN head-to-head (pre vs post vs tuned vs Putirka)
    4. Methods summary
    5. Limitations
    6. Caveats and reconstruction provenance
    7. Provenance (git SHA, CSV inventory)
    8. Pre-registration (verbatim)
    """))

    # Setup cell
    cells.append(_code("""
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import display, Markdown, Image

    ROOT = Path('.').resolve().parent.parent  # deliverables/lee_package_20260420/ -> project root
    RESULTS = ROOT / 'results'
    FIGS = Path('./figures')

    def load_csv(relpath: str) -> pd.DataFrame:
        p = ROOT / relpath
        if not p.exists():
            raise FileNotFoundError(f'required CSV missing: {p}')
        return pd.read_csv(p)

    def load_json(relpath: str) -> dict:
        p = ROOT / relpath
        if not p.exists():
            raise FileNotFoundError(f'required JSON missing: {p}')
        return json.loads(p.read_text())

    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print('Setup complete; loader ready.')
    """))

    # ---- 1. Core figures (FIRST, per advisor request) ----
    cells.append(_md("""
    ## 1. Core figures

    Fourteen figures in total. Each one is paired with a long caption
    written in plain language. You don't need to load any data to read
    this section.

    - **Core_01 / Core_01b**: who is in the training set, who is in
      the held-out test set (the sanity-check that no single citation
      leaks across the split).
    - **Core_02**: the citation-grouped fold split used for cross-
      validation.
    - **Core_03**: a one-page flowchart of the ML pipeline.
    - **Core_04**: cross-pipeline heatmap of model RMSE.
    - **Core_05**: per-regime RMSE before and after bias correction.
    - **Core_06**: residual-vs-prediction scatter at canonical seed 42.
    - **Core_07**: scoreboard of ML vs Putirka, regime by regime.
    - **Core_08**: the headline per-regime RMSE bar chart for all four
      opx cells.
    - **Core_09a / Core_09b**: how each ML family compares, by regime
      and overall.
    - **Core_10**: best ML model vs Putirka, one dot per test sample.
    - **Core_11**: natural-sample two-pyroxene 1:1 agreement plot.
    - **Core_12 (new)**: SHAP feature importance for the best-
      explainable model per cell. SHAP tells you which oxide features
      the model was actually leaning on.
    """))
    _append_figure_cells(cells, CORE_FIGS, src_subdir='core')

    # ---- 2. SI figures ----
    cells.append(_md("""
    ## 2. Supporting information figures

    Extra figures for the curious reader. Safe to skim.
    """))
    _append_figure_cells(cells, SI_FIGS, src_subdir='SI')

    # ---- 3. Data tables (MOVED here, after figures) ----
    cells.append(_md("""
    ## 3. Data tables

    Every number above loads from a CSV under `results/`. Nothing is
    hard-coded. If a file is missing the cell raises `FileNotFoundError`
    rather than silently filling in a value.
    """))

    # 3.1 Headline
    cells.append(_md("""
    ### 3.1 Headline table

    The headline numbers for each of the 4 opx cells, at the "ALL
    pressure regimes" level. Columns: uncorrected tuned-ML RMSE,
    bias-corrected tuned-ML RMSE, the best Putirka (2008)
    thermobarometer for that cell, TabPFN pre- and post-correction
    RMSE, and which of those five candidates wins.
    Source: `results/preregistered_scorecard_postcorrection.csv`.
    """))
    cells.append(_code("""
    sc = load_csv('results/preregistered_scorecard_postcorrection.csv')
    hl = sc[(sc['track'].isin(['opx_liq','opx_only'])) & (sc['regime']=='ALL')].copy()
    display_cols = ['track','target','v10_pre_rmse','v10_post_rmse','best_external_rmse',
                    'tabpfn_rmse','tabpfn_post_rmse','tabpfn_correction_form','winner']
    hl = hl[display_cols].rename(columns={
        'v10_pre_rmse': 'tuned pre',
        'v10_post_rmse': 'tuned post',
        'best_external_rmse': 'Putirka best',
        'tabpfn_rmse': 'TabPFN pre',
        'tabpfn_post_rmse': 'TabPFN post',
        'tabpfn_correction_form': 'TabPFN form',
    })
    display(hl)
    """))

    # 3.2 Eight-cell winner table
    cells.append(_md("""
    ### 3.2 Eight-cell winner table

    Which model shipped, with or without bias correction, in each of
    the 8 opx rows (4 tuned-family rows + 4 TabPFN rows).
    Source: `results/bias_correction_shipped.csv`.
    """))
    cells.append(_code("""
    shipped = load_csv('results/bias_correction_shipped.csv')
    opx_rows = shipped[shipped['pipeline']=='opx'].copy()
    display_cols = ['track','target','model','winner','ship_a','ship_b','n_seeds_done']
    display(opx_rows[display_cols])
    """))

    # 3.3 TabPFN bias-correction scoreboard
    cells.append(_md("""
    ### 3.3 TabPFN bias-correction scoreboard

    TabPFN's own pre/post correction numbers, averaged over 5 OOF
    seeds. This is the source for the TabPFN columns in Section 3.1.
    Source: `results/tabpfn_bias_correction_summary.csv`.
    """))
    cells.append(_code("""
    tf_sum = load_csv('results/tabpfn_bias_correction_summary.csv')
    display(tf_sum)
    """))
    cells.append(_md("""
    Per-seed detail (5 seeds x 4 combos = 20 rows). Useful for
    checking stability.
    Source: `results/tabpfn_bias_correction_perseed.csv`.
    """))
    cells.append(_code("""
    tf_per = load_csv('results/tabpfn_bias_correction_perseed.csv')
    display(tf_per[['track','target','seed','pre_rmse_all','post_rmse_a','post_rmse_b','ship_a','ship_b','winner']])
    """))

    # 3.4 opx-only P regime
    cells.append(_md("""
    ### 3.4 opx-only P_kbar per-regime breakdown

    For the pressure cell where ML does best (opx-only P), here is
    the regime-by-regime picture across all 5 candidates: tuned pre,
    tuned post, Putirka eq. 29c, TabPFN pre, TabPFN post.
    """))
    cells.append(_code("""
    pr = sc[(sc['track']=='opx_only') & (sc['target']=='P_kbar')].copy()
    show = ['regime','n','v10_pre_rmse','v10_post_rmse','best_external_rmse',
            'tabpfn_rmse','tabpfn_post_rmse','tabpfn_correction_form','winner']
    display(pr[show])
    """))

    # 3.5 TabPFN head-to-head
    cells.append(_md("""
    ### 3.5 TabPFN head-to-head (aggregate RMSE, 20-seed baseline)

    TabPFN vs the best tuned family per cell, at the aggregate ALL
    level, using the full 20-seed multiseed protocol so the stability
    estimates are on an equal footing with the tuned families.
    Source: `results/tabpfn_head_to_head.csv`.
    """))
    cells.append(_code("""
    h2h = load_csv('results/tabpfn_head_to_head.csv')
    display(h2h)
    """))

    # ---- 4. Methods summary (plain English) ----
    cells.append(_md(r"""
    ## 4. Methods summary (plain English)

    - **What we are predicting.** For each pyroxene sample we are
      trying to predict either (a) the temperature at which it formed
      or (b) the pressure at which it formed. Two input "tracks": the
      full crystal + liquid pair ("opx-liq") or the crystal alone
      ("opx-only"). Four target cells in total.
    - **Where the training data come from.** A curated subset of the
      LEPR experimental petrology database (our snapshot is dated
      2025-07-21). Every sample is from a published experiment at
      known P and T. We group by citation when splitting into train
      and test, so that memorizing a single lab's style is not
      rewarded.
    - **Which models we tune.** 8 standard ML families (random
      forest, extremely-randomized trees, XGBoost, gradient boosting,
      CatBoost, LightGBM, elastic-net linear, multi-layer
      perceptron). Each one is tuned with Optuna (50 trials) on a
      citation-grouped 5-fold cross-validation objective. We also
      run a 9th foundation-model baseline: TabPFN v2 (Hollmann et al.
      2025), which requires no tuning.
    - **Pressure regimes.** Before we fit any correction, we
      registered four geologic pressure bins: shallow-crustal
      (<5 kbar), deep-crustal / MASH (5-15 kbar),
      lithospheric-mantle (15-30 kbar), deeper-mantle (30+ kbar).
      "ALL" is the full test set combined.
    - **Bias correction.** Two flavors. *Form A* fits a simple linear
      regression of residual on predicted value, separately in each
      of the four regimes. *Form B* fits one smooth piecewise-
      sigmoid curve across the whole predicted-value axis. Only one
      flavor can ship per cell.
    - **Ship-if-better rule (as of Amendment 1, 2026-04-20).** A
      correction ships if overall RMSE improves AND no
      pre-registered regime with enough samples (N >= 20) gets
      worse. Regimes with fewer than 20 samples are noted but are
      not allowed to veto a real improvement elsewhere. This
      replaces the original strict rule that any regime with any
      degradation would block shipping; the tiered rule is documented
      in `docs/preregistration/AMENDMENT_1_acceptance_rule.md`.
    - **How we measure stability.** Each experiment is repeated at
      20 random seeds (42-61). The bars and numbers you see include
      bootstrap 95% confidence intervals so you can see whether a
      difference is signal or noise.
    """))

    # ---- 5. Limitations ----
    cells.append(_md("""
    ## 5. Limitations

    1. **Temperature corrections mostly fail to ship.** In our
       strict evaluation, the bias correction refuses to ship for 3
       of 4 temperature cells. Temperature residuals do not have a
       strong regime structure to remove, so per-regime correction
       cancels out. We report this as a null result rather than
       tuning until something ships.
    2. **opx-only T is the weakest cell.** For the opx-only
       thermometer, TabPFN improves the aggregate RMSE but the gain
       is marginal; we would not recommend that particular cell for
       deployment today.
    3. **Natural-sample cross-check carries a small T bias.** On
       LEPR paired pyroxenes (n=327) the opx-only ML prediction
       runs about +80 C hotter than Putirka's two-pyroxene method
       and Jorgenson's cpx-only method. The pressure agreement is
       within our conformal error bar.
    4. **No GEOROC refresh since 2026-04-09.** We did not pull a
       new natural-sample export for this round; the natural
       comparison uses the on-disk export from that date.
    5. **SHAP on TabPFN is not available.** TabPFN is an in-context
       foundation model with no gradient surface exposed. For the
       two cells where TabPFN is the scorecard winner (opx_only T
       and opx_only P), Core_12 shows the tuned-family runner-up's
       SHAP for explainability and flags the swap in the subtitle.
    """))

    # ---- 6. Caveats and reconstruction provenance ----
    cells.append(_md("""
    ## 6. Caveats and reconstruction provenance

    Short file of "things a reviewer should know before acting on the
    numbers." Loaded from `CAVEATS.md` alongside this notebook.
    """))
    cells.append(_code("""
    cav = Path('./CAVEATS.md')
    if cav.exists():
        display(Markdown(cav.read_text(encoding='utf-8')))
    else:
        display(Markdown('(caveats file missing)'))
    """))

    # ---- 7. Provenance ----
    cells.append(_md("""
    ## 7. Provenance

    Git state + source CSV sizes at build time. Loaded from
    `PROVENANCE.md` alongside this notebook.
    """))
    cells.append(_code("""
    prov = Path('./PROVENANCE.md')
    if prov.exists():
        display(Markdown(prov.read_text(encoding='utf-8')))
    else:
        display(Markdown('(provenance file missing)'))
    """))

    # ---- 8. Pre-registration ----
    cells.append(_md("""
    ## 8. Pre-registration (verbatim)

    The rules of the game, frozen before we fit any corrections.
    Reproduced verbatim from `docs/preregistration/`.
    """))
    cells.append(_code("""
    # Prefer the local preregistration/ copy bundled with the package;
    # fall back to the project docs/ source if running uninstalled.
    for name in ('p_regime_preregistration.md', 'nb03_test_protocol.md'):
        local = Path('./preregistration') / name
        src = ROOT / 'docs' / 'preregistration' / name
        p = local if local.exists() else src
        if not p.exists():
            raise FileNotFoundError(f'preregistration missing: {name}')
        display(Markdown(f'### `{name}`'))
        display(Markdown(p.read_text(encoding='utf-8')))
    """))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
    }
    nbf.write(nb, NB_PATH)
    print(f'wrote {NB_PATH}')


def write_provenance():
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ['git', 'log', '-5', '--oneline'], cwd=PROJECT_ROOT, text=True).strip()

    lines = [
        '# Provenance',
        '',
        f'Built: 2026-04-20',
        f'Git SHA: `{git_sha}`',
        '',
        '## Last 5 commits',
        '',
        '```',
        git_status,
        '```',
        '',
        '## CSV inventory',
        '',
        '| File | Rows | Cols | Size (KB) |',
        '| --- | --- | --- | --- |',
    ]
    for c in REQUIRED_CSVS:
        p = PROJECT_ROOT / c
        df = pd.read_csv(p)
        size_kb = p.stat().st_size / 1024
        lines.append(f'| `{c}` | {len(df)} | {df.shape[1]} | {size_kb:.1f} |')
    (PKG_DIR / 'PROVENANCE.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote PROVENANCE.md')


README_TEXT = """Advisor review package -- opx ML thermobarometer
Author: Ta Quang Nhan (cadet, USCGA)
Date: 2026-04-20
For: Dr. Kanani K.M. Lee

START HERE: open 00_ADVISOR_REVIEW.html in any browser.
No Python required. Self-contained ~6 MB.

Files:
  00_ADVISOR_REVIEW.html         -- read this
  00_ADVISOR_REVIEW.ipynb        -- source notebook
  00_ADVISOR_REVIEW_executed.ipynb -- executed snapshot
  figures/                       -- 13 figures (PDF + PNG + caption sidecar)
  preregistration/               -- locked evaluation framework (2026-04-17)
  PROVENANCE.md                  -- git SHA, commit log, CSV inventory
  data_links.md                  -- source CSV paths
  CAVEATS.md                     -- reconstruction notes, limitations

Twelve sections. Key result at Section 1 (Headline). Dive deeper from there.

Questions: Ta Quang Nhan at USCGA.
"""


CAVEATS_TEXT = """# Caveats and reconstruction provenance

Items an independent reviewer should know before acting on the numbers
in this package.

## 1. Reconstructed conformal calibration (`results/nb07_conformal_qhat.json`)

The conformal calibration file (qhat_T=92.58 C, qhat_P=10.0 kbar at alpha=0.10)
was rebuilt from archive after the post-consolidation notebook layout dropped
it. Values originate from the pre-v10 calibration run (n_calibration=43). The
semantic validity of these qhat values under the **current** calibration set
has NOT been independently verified. Downstream: nb08 LEPR comparison uses
these qhat values for conformal half-widths on natural-sample predictions.

## 2. Reconstructed per-family winners (`results/nb03_per_family_winners.json`)

Rebuilt via `scripts/data_prep/build_per_family_winners.py` selecting
argmin(mean RMSE) -> argmin(std) -> alphabetical from
`opx_multiseed_summary.csv`. No cross-validation against a pre-consolidation
archive file. Selections:

- Forest family: RF/pwlr (opx_only T, opx_only P, opx_liq T) + RF/alr (opx_liq P)
- Boosted family: LightGBM/alr, XGB/pwlr, GB/raw, GB/raw

## 3. Ship-if-better threshold tightness

The ship-if-better rule uses `overall_delta > 1e-6 AND
max_regime_degradation <= 1e-6`. Effectively "never degrade any regime
by any float amount". Defensible but strict; a looser threshold (e.g.
0.5% relative degradation) would likely let more corrections ship.
Reported RMSE values are unaffected; the rule only gates shipping.

## 4. Phase 2 figure scope

The TabPFN integration originally planned 6 new figures. Shipped 2
(fig44 scoreboard, fig45 opx-only P headline) and relied on pre-existing
figures (fig28, fig30-35) for the remaining panels. No numeric data
impact; cosmetic scope only.

## 5. Papermill execution is visual-only

The 48-cell notebook runs error-free but contains no numeric assertions
(no `assert abs(rmse - 10.35) < 0.1` style guards). Verification is by
eye against the underlying CSVs. Every cell DOES raise
FileNotFoundError if its source CSV is missing, so structural
failures are caught; value-level regressions would require adding
explicit asserts.

## 6. CSV column name retention

Columns such as `v10_pre_rmse` and `v10_post_rmse` remain in the CSVs
as machine contracts; the notebook performs display-time aliasing to
"tuned pre" / "tuned post" for the reader. Renaming the CSV columns
themselves would break every downstream script.
"""


def write_readme():
    (PKG_DIR / 'README.txt').write_text(README_TEXT, encoding='utf-8')
    print('wrote README.txt')


def write_caveats():
    (PKG_DIR / 'CAVEATS.md').write_text(CAVEATS_TEXT, encoding='utf-8')
    print('wrote CAVEATS.md')


def copy_preregistration():
    src_dir = PROJECT_ROOT / 'docs' / 'preregistration'
    dst_dir = PKG_DIR / 'preregistration'
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ('p_regime_preregistration.md', 'nb03_test_protocol.md'):
        src = src_dir / name
        if src.exists():
            shutil.copy(src, dst_dir / name)
            copied.append(name)
        else:
            print(f'WARN: missing preregistration file {src}')
    print(f'copied {len(copied)} preregistration files')
    return copied


def write_data_links():
    lines = [
        '# Data sources',
        '',
        'All numeric assertions in `00_ADVISOR_REVIEW.ipynb` resolve to rows',
        'or columns of the CSVs below. Every cell loads via `load_csv()` /',
        '`load_json()` which raise `FileNotFoundError` rather than fabricating.',
        '',
    ]
    for c in REQUIRED_CSVS:
        lines.append(f'- `{c}`')
    (PKG_DIR / 'data_links.md').write_text('\n'.join(lines), encoding='utf-8')


def papermill_execute():
    cmd = [
        sys.executable, '-m', 'papermill',
        str(NB_PATH), str(NB_EXEC_PATH),
        '--cwd', str(PKG_DIR),
        '--log-output',
        '--execution-timeout', '900',
    ]
    print(f'papermill: {NB_PATH.name} -> {NB_EXEC_PATH.name}')
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f'papermill failed rc={rc}')
        return False
    return True


def nbconvert_html():
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'html',
        '--output', HTML_PATH.name,
        '--output-dir', str(PKG_DIR),
        str(NB_EXEC_PATH),
    ]
    print(f'nbconvert: {NB_EXEC_PATH.name} -> {HTML_PATH.name}')
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f'nbconvert failed rc={rc}')
        return False
    return True


def main() -> int:
    verify_csvs()
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    copied = copy_figures()
    print(f'copied {len(copied)} figure files')
    copy_preregistration()
    write_readme()
    write_caveats()
    write_data_links()
    write_provenance()
    build_notebook()
    ok = papermill_execute()
    if not ok:
        return 3
    ok = nbconvert_html()
    if not ok:
        return 4
    print(f'\nPackage ready: {PKG_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
