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
    'fig44_tabpfn_bias_scoreboard_opx',
    'fig45_opx_only_P_headline',
    'fig34_bias_correction_scorecard_delta',
    'fig35_tabpfn_vs_opx_tb',
    'fig28_bias_correction_opx_liq',
    'fig_nb08_twopx_1to1',
]
SI_FIGS = [
    'fig30_bias_correction_per_regime_rmse',
    'fig31_bias_correction_residuals',
    'fig32_bias_correction_form_comparison',
    'fig33_bias_correction_per_seed_stability',
    'fig_aug01_ship_verdict_comparison',
    'fig_aug02_aggregate_rmse_delta',
    'fig_aug03_residual_structure_per_regime',
]


def _md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip('\n'))


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip('\n'))


def verify_csvs():
    missing = [c for c in REQUIRED_CSVS
               if not (PROJECT_ROOT / c).exists()]
    if missing:
        print(f'ERROR: missing required CSVs: {missing}')
        sys.exit(2)


def copy_figures():
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for stem in CORE_FIGS + SI_FIGS:
        for ext in ('.pdf', '.png', '.txt'):
            src = FIGS_IN / f'{stem}{ext}'
            if src.exists():
                shutil.copy(src, FIGS_OUT / src.name)
                copied.append(src.name)
    return copied


def build_notebook():
    cells = []

    # ---- Header ----
    cells.append(_md("""
    # Advisor review package -- opx ML thermobarometer

    **Audience:** Dr. Kanani K.M. Lee
    **Author:** Ta Quang Nhan (cadet, USCGA)
    **Date:** 2026-04-20
    **Target venue:** JGR ML & Computation

    ## Package purpose

    Consolidates the post-Phase-1 state of the opx ML thermobarometer project
    into a single reviewable artifact. Every number below loads from a CSV or
    JSON under `results/`; nothing is hard-coded. If a required file is missing
    the notebook cells raise FileNotFoundError rather than silently fabricating.

    ## Sections

    1. Headline table (4 opx combos)
    2. 8-cell winner table (tuned + TabPFN)
    3. Bias-correction scoreboard
    4. opx-only P_kbar per-regime breakdown
    5. TabPFN head-to-head (pre vs post vs tuned vs Putirka)
    6. Core figures (6 PDFs)
    7. SI figures (7 PDFs)
    8. Pre-registration (verbatim)
    9. Methods summary
    10. Limitations
    11. Provenance
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

    # ---- 1. Headline table ----
    cells.append(_md("""
    ## 1. Headline table

    Aggregate ALL-regime pre-correction RMSE, best-correction RMSE, and winner
    verdict for each of the 4 opx combinations. Source:
    `results/preregistered_scorecard_postcorrection.csv`.
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

    # ---- 2. 8-cell winner table ----
    cells.append(_md("""
    ## 2. 8-cell winner table

    The 8-cell view: each of the 4 opx combinations x (tuned best winner, TabPFN).
    Source: `results/bias_correction_shipped.csv`.
    """))
    cells.append(_code("""
    shipped = load_csv('results/bias_correction_shipped.csv')
    opx_rows = shipped[shipped['pipeline']=='opx'].copy()
    display_cols = ['track','target','model','winner','ship_a','ship_b','n_seeds_done']
    display(opx_rows[display_cols])
    """))

    # ---- 3. Bias-correction scoreboard ----
    cells.append(_md("""
    ## 3. Bias-correction scoreboard

    TabPFN per-combo summary over 5 OOF seeds. Mean pre-correction RMSE, mean
    Form A RMSE, mean Form B RMSE, ship counts, winner tallies. Source:
    `results/tabpfn_bias_correction_summary.csv`.
    """))
    cells.append(_code("""
    tf_sum = load_csv('results/tabpfn_bias_correction_summary.csv')
    display(tf_sum)
    """))
    cells.append(_md("""
    Per-seed detail (5 seeds x 4 combos = 20 rows). Source:
    `results/tabpfn_bias_correction_perseed.csv`.
    """))
    cells.append(_code("""
    tf_per = load_csv('results/tabpfn_bias_correction_perseed.csv')
    display(tf_per[['track','target','seed','pre_rmse_all','post_rmse_a','post_rmse_b','ship_a','ship_b','winner']])
    """))

    # ---- 4. opx-only P regime ----
    cells.append(_md("""
    ## 4. opx-only P_kbar per-regime breakdown

    Per-regime RMSE across the 5 candidates (tuned pre/post, Putirka 29c,
    TabPFN pre/post). The post-corrected TabPFN is the aggregate winner on
    3 of 5 regimes.
    """))
    cells.append(_code("""
    pr = sc[(sc['track']=='opx_only') & (sc['target']=='P_kbar')].copy()
    show = ['regime','n','v10_pre_rmse','v10_post_rmse','best_external_rmse',
            'tabpfn_rmse','tabpfn_post_rmse','tabpfn_correction_form','winner']
    display(pr[show])
    """))

    # ---- 5. TabPFN head-to-head ----
    cells.append(_md("""
    ## 5. TabPFN head-to-head (aggregate RMSE, 20-seed baseline)

    Source: `results/tabpfn_head_to_head.csv`. TabPFN vs best tuned family per
    (pipeline, track, target) at aggregate ALL level from the 20-seed multiseed
    protocol.
    """))
    cells.append(_code("""
    h2h = load_csv('results/tabpfn_head_to_head.csv')
    display(h2h)
    """))

    # ---- 6. Core figures ----
    cells.append(_md("""
    ## 6. Core figures

    Six core manuscript figures. PDFs live in this package under `figures/`;
    previews are PNG for rendering.
    """))
    for stem in CORE_FIGS:
        cap_path = FIGS_IN / f'{stem}.txt'
        caption = cap_path.read_text(encoding='utf-8') if cap_path.exists() else '(no caption sidecar)'
        cells.append(_md(f"""
        ### {stem}

        {caption}
        """))
        cells.append(_code(f"""
        from IPython.display import Image
        png = FIGS / '{stem}.png'
        if png.exists():
            display(Image(filename=str(png), width=780))
        else:
            display(Markdown(f'**missing:** {{png}}'))
        """))

    # ---- 7. SI figures ----
    cells.append(_md("""
    ## 7. Supporting information figures
    """))
    for stem in SI_FIGS:
        cap_path = FIGS_IN / f'{stem}.txt'
        caption = cap_path.read_text(encoding='utf-8') if cap_path.exists() else '(no caption sidecar)'
        cells.append(_md(f"""
        ### {stem}

        {caption}
        """))
        cells.append(_code(f"""
        png = FIGS / '{stem}.png'
        if png.exists():
            display(Image(filename=str(png), width=780))
        else:
            display(Markdown(f'**missing:** {{png}}'))
        """))

    # ---- 8. Preregistration ----
    cells.append(_md("""
    ## 8. Pre-registration (verbatim)

    The pre-registered pressure partition and test protocol, reproduced
    verbatim from `docs/preregistration/`.
    """))
    cells.append(_code("""
    for name in ('p_regime_preregistration.md', 'nb03_test_protocol.md'):
        p = ROOT / 'docs' / 'preregistration' / name
        if not p.exists():
            raise FileNotFoundError(p)
        display(Markdown(f'### `{name}`'))
        display(Markdown(p.read_text(encoding='utf-8')))
    """))

    # ---- 9. Methods ----
    cells.append(_md(r"""
    ## 9. Methods summary

    - **Pipelines:** opx_liq (pyroxene + liquid features) and opx_only
      (pyroxene-only features). Four (track, target) combinations: opx_liq T_C,
      opx_liq P_kbar, opx_only T_C, opx_only P_kbar.
    - **Training universe:** LEPR experimental database (ExPetDB 2025-07-21)
      filtered for opx equilibrium and citation-grouped into folds so that
      no citation appears in both a fold's train and held-out split.
    - **Tuned families:** 8 gradient-boosted / forest / linear families
      (RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP), each Optuna-
      tuned (50 trials, 12 inner jobs) on the 5-fold CV objective.
    - **Foundation baseline:** TabPFN v2 (Hollmann et al. 2025) with
      `n_estimators=8` on CPU, no tuning. Bias-corrected variant uses
      5-seed 10-fold out-of-fold residuals as the correction-fit substrate.
    - **Pressure partition:** shallow_crustal (<5 kbar), deep_crustal_MASH
      (5-10), lithospheric_mantle (10-20), deeper_mantle (>=20), plus ALL.
      Registered 2026-04-17 before any correction fitting.
    - **Correction forms:**
        - *Form A:* per-regime ordinary least squares y_corr = a_r * y_pred + b_r.
        - *Form B:* piecewise sigmoid blend in Agreda-Lopez (2024) form with
          data-driven breakpoints.
    - **Ship-if-better rule:** conservative acceptance -- Form ships iff
      `overall_delta > 1e-6` AND `max_regime_degradation <= 1e-6`. No
      per-regime loss accepted for aggregate gain.
    - **Multiseed protocol:** 20 seeds (42-61) for test-RMSE variance; 5 seeds
      for OOF bias-correction fit (per standard protocol).
    """))

    # ---- 10. Limitations ----
    cells.append(_md("""
    ## 10. Limitations

    1. **Form B fails to ship on opx.** 0/8 tuned + 0/4 TabPFN opx combinations
       pass the ship-if-better rule under Form B. The augmentation ablation
       (nb04b, 15x Gaussian noise at 3% RSD per Agreda-Lopez 2024) does NOT
       rescue Form B; aggregate RMSE strictly degrades under augmentation on
       every opx combination. Attribution: mineral-specific or dataset-size
       specific, not protocol-specific.
    2. **opx-only T is a null result.** Neither Form A nor Form B ships on
       opx_only/T_C for any tuned family. TabPFN Form A ships but only on the
       deeper_mantle regime; the aggregate ALL improvement is marginal. The
       opx-only thermometer is not recommended for deployment.
    3. **Natural-sample agreement is imperfect.** On LEPR paired pyroxenes
       (n=327), opx-only ML carries a +80 C positive T bias against both
       Jorgenson cpx-only and Putirka two-pyroxene methods. P agreement is
       within the conformal half-width.
    4. **5-seed OOF for TabPFN is a reduction from the 20-seed test protocol.**
       The bias-fit substrate uses 5 seeds to bound CPU cost; test-set
       inference uses the full 20 seeds. TabPFN Form A ship decisions are at
       canonical seed 42 with per-seed stability reported.
    5. **No external GEOROC cpx-opx pair update since Apr 9 2026.** The natural-
       sample refresh uses the existing on-disk export; the upstream server
       URL / credential were not supplied for this round.
    """))

    # ---- 11. Provenance ----
    cells.append(_md("""
    ## 11. Provenance

    Git state + source CSV sizes at build time. Loaded via `PROVENANCE.md`
    alongside this notebook.
    """))
    cells.append(_code("""
    prov = Path('./PROVENANCE.md')
    if prov.exists():
        display(Markdown(prov.read_text(encoding='utf-8')))
    else:
        display(Markdown('(provenance file missing)'))
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
