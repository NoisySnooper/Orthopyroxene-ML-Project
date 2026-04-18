#!/usr/bin/env python3
"""Phase G.1 kickoff: build and execute nb04_v10_benchmark.ipynb.

This notebook is the headline cross-pipeline comparison. It loads
per-cell and ensemble results from all four pipelines (opx, cpx,
twopx, universal) and renders:

  1. A unified best-RMSE table across all pipelines x targets x tracks.
  2. Cross-pipeline best-RMSE heatmap (primary figure).
  3. Model family contribution: which base model wins most often?
  4. Ensemble-vs-best-base delta per pipeline.

External model integration (Agreda, Jorgenson, Petrelli, Putirka,
Wang) is scoped as Phase G.1b and happens separately.

Run from project root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [text]}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': [text],
            'outputs': [], 'execution_count': None}


def build_cells():
    cells = []
    cells.append(md(
        "# nb04_v10_benchmark (headline cross-pipeline comparison)\n"
        "\n"
        "Consolidates Phase C/D/E/F results into a unified view. Loads per-cell and "
        "ensemble CSVs from `results/` for all four pipelines (opx, cpx, twopx, "
        "universal) and produces:\n"
        "\n"
        "1. Unified best-RMSE table\n"
        "2. Cross-pipeline heatmap (headline figure)\n"
        "3. Winning-base histogram (which family wins most often?)\n"
        "4. Ensemble-vs-best-base delta per pipeline\n"
        "\n"
        "**Scope:** internal ML comparison only. External models (Agreda, Jorgenson, "
        "Petrelli, Wang, Putirka) are integrated in the follow-up benchmark "
        "(Phase G.1b).\n"
    ))
    cells.append(code(
        "import os, sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd()\n"
        "if ROOT.name == 'notebooks':\n"
        "    ROOT = ROOT.parent\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from config import RESULTS\n"
        "pd.set_option('display.width', 220)\n"
        "pd.set_option('display.max_columns', 50)\n"
    ))

    cells.append(md(
        "## Load per-pipeline results\n"
        "\n"
        "Per-cell CSVs carry one row per (model, target, track, feature_set). "
        "Ensemble CSVs carry one row per (method, target, track, feature_set).\n"
    ))
    cells.append(code(
        "pipelines = {\n"
        "    'opx':       RESULTS / 'v10_opx_per_cell_results.csv',\n"
        "    'cpx':       RESULTS / 'v10_cpx_per_cell_results.csv',\n"
        "    'twopx':     RESULTS / 'v10_twopx_per_cell_results.csv',\n"
        "    'universal': RESULTS / 'universal' / 'v10_universal_per_cell_results.csv',\n"
        "}\n"
        "ens_paths = {\n"
        "    'opx':       RESULTS / 'v10_opx_ensemble_results.csv',\n"
        "    'cpx':       RESULTS / 'v10_cpx_ensemble_results.csv',\n"
        "    'twopx':     RESULTS / 'v10_twopx_ensemble_results.csv',\n"
        "    'universal': RESULTS / 'universal' / 'v10_universal_ensemble_results.csv',\n"
        "}\n"
        "\n"
        "frames_base, frames_ens = [], []\n"
        "for p, path in pipelines.items():\n"
        "    df = pd.read_csv(path)\n"
        "    df['pipeline'] = p\n"
        "    frames_base.append(df)\n"
        "for p, path in ens_paths.items():\n"
        "    df = pd.read_csv(path)\n"
        "    df['pipeline'] = p\n"
        "    frames_ens.append(df)\n"
        "\n"
        "all_bases = pd.concat(frames_base, ignore_index=True)\n"
        "all_ens   = pd.concat(frames_ens, ignore_index=True)\n"
        "print(f'base rows: {len(all_bases)}  ensemble rows: {len(all_ens)}')\n"
    ))

    cells.append(md("## Unified best-RMSE table\n"
                    "Per (pipeline, target, track): winning base model + "
                    "winning ensemble method.\n"))
    cells.append(code(
        "idx_b = all_bases.groupby(['pipeline','target','track'])['test_rmse'].idxmin()\n"
        "best_b = all_bases.loc[idx_b, ['pipeline','target','track','model','feature_set','test_rmse']]\n"
        "best_b = best_b.rename(columns={'model':'best_base','test_rmse':'base_rmse','feature_set':'base_fs'})\n"
        "\n"
        "idx_e = all_ens.groupby(['pipeline','target','track'])['test_rmse'].idxmin()\n"
        "best_e = all_ens.loc[idx_e, ['pipeline','target','track','method','feature_set','test_rmse']]\n"
        "best_e = best_e.rename(columns={'method':'best_ens','test_rmse':'ens_rmse','feature_set':'ens_fs'})\n"
        "\n"
        "unified = best_b.merge(best_e, on=['pipeline','target','track'], how='outer')\n"
        "unified['ensemble_gain'] = unified['base_rmse'] - unified['ens_rmse']\n"
        "unified = unified.sort_values(['target', 'pipeline', 'track']).reset_index(drop=True)\n"
        "unified.round(3)\n"
    ))

    cells.append(md(
        "## Figure 1: cross-pipeline headline heatmap\n"
        "\n"
        "Rows = (pipeline, track), columns = target. Color = best base RMSE. "
        "Lower (darker) is better.\n"
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = unified[unified['target']==tgt].copy()\n"
        "    sub['row'] = sub['pipeline'] + ' / ' + sub['track']\n"
        "    rows = sub['row'].tolist()\n"
        "    vals = sub['base_rmse'].values.reshape(-1, 1)\n"
        "    im = ax.imshow(vals, cmap='viridis_r', aspect='auto')\n"
        "    ax.set_yticks(range(len(rows)))\n"
        "    ax.set_yticklabels(rows)\n"
        "    ax.set_xticks([0])\n"
        "    ax.set_xticklabels([f'best base RMSE'])\n"
        "    for i, v in enumerate(sub['base_rmse'].values):\n"
        "        ax.text(0, i, f'{v:.2f}', ha='center', va='center',\n"
        "                color='white' if v > sub['base_rmse'].median() else 'black',\n"
        "                fontsize=10)\n"
        "    ax.set_title(f'{tgt}  (units: {\"C\" if tgt==\"T_C\" else \"kbar\"})')\n"
        "    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
        "plt.suptitle('Cross-pipeline best-RMSE heatmap (v10 Phase G.1)', fontsize=12)\n"
        "plt.tight_layout()\n"
        "out = ROOT / 'figures' / 'fig_nb04_cross_pipeline_heatmap.png'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "plt.savefig(out, dpi=300, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'saved {out}')\n"
    ))

    cells.append(md(
        "## Figure 2: winning base family across all 10 cells\n"
        "\n"
        "Counts, across the 10 (pipeline, target, track) cells, how many times "
        "each base-model family wins. Signals which family to treat as the "
        "primary recommendation.\n"
    ))
    cells.append(code(
        "win_counts = unified['best_base'].value_counts()\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.bar(win_counts.index, win_counts.values, color='steelblue')\n"
        "ax.set_ylabel('cells won'); ax.set_xlabel('base model')\n"
        "ax.set_title('Winning base model across 10 cells (v10 Phase G.1)')\n"
        "for i, v in enumerate(win_counts.values):\n"
        "    ax.text(i, v + 0.05, str(v), ha='center', fontsize=10)\n"
        "plt.tight_layout()\n"
        "out = ROOT / 'figures' / 'fig_nb04_winning_base_histogram.png'\n"
        "plt.savefig(out, dpi=300, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'saved {out}')\n"
    ))

    cells.append(md(
        "## Figure 3: ensemble-vs-base delta per cell\n"
        "\n"
        "Positive `ensemble_gain` means the ensemble beat the best single base. "
        "Near-zero or negative means the best single base was already hard to "
        "beat. Bars colored by target.\n"
    ))
    cells.append(code(
        "tmp = unified.copy()\n"
        "tmp['cell'] = tmp['pipeline'] + ' / ' + tmp['track'] + ' / ' + tmp['target']\n"
        "tmp = tmp.sort_values('ensemble_gain')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "colors = ['#0072B2' if t == 'T_C' else '#D55E00' for t in tmp['target']]\n"
        "ax.barh(tmp['cell'], tmp['ensemble_gain'], color=colors)\n"
        "ax.axvline(0, color='k', lw=0.8)\n"
        "ax.set_xlabel('base_rmse - ens_rmse (positive = ensemble wins)')\n"
        "ax.set_title('Ensemble lift over best single base (v10 Phase G.1)')\n"
        "from matplotlib.patches import Patch\n"
        "ax.legend(handles=[Patch(color='#0072B2', label='T_C'),\n"
        "                   Patch(color='#D55E00', label='P_kbar')],\n"
        "          fontsize=9, loc='lower right')\n"
        "plt.tight_layout()\n"
        "out = ROOT / 'figures' / 'fig_nb04_ensemble_lift.png'\n"
        "plt.savefig(out, dpi=300, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'saved {out}')\n"
    ))

    cells.append(md(
        "## Summary table: written to disk\n"
        "\n"
        "Saves the unified best-RMSE table to `results/v10_phase_g1_unified_best.csv` "
        "for downstream use by manuscript + other notebooks.\n"
    ))
    cells.append(code(
        "out_csv = RESULTS / 'v10_phase_g1_unified_best.csv'\n"
        "unified.to_csv(out_csv, index=False)\n"
        "print(f'saved {out_csv}')\n"
        "print(f'\\nShape: {unified.shape}')\n"
    ))

    cells.append(md(
        "## Next: Phase G.1b\n"
        "\n"
        "Integrate external models (Agreda 2024 cpx-liq, Jorgenson 2022 "
        "cpx-only/cpx-liq, Petrelli 2020 cpx-only/cpx-liq, Wang 2021 cpx-liq, "
        "Putirka 2008 cpx-only/cpx-liq) on the shared ArcPL Kd-eq subset. "
        "Produces `fig_nb04_{pipeline}_method_benchmark_arcpl_kdEq.png` and the "
        "cross-method comparison headline.\n"
    ))
    return cells


def write_nb(path: Path, cells):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.13'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'wrote {path}')


def execute_nb(path: Path):
    t0 = time.time()
    cmd = [sys.executable, '-m', 'jupyter', 'nbconvert',
           '--to', 'notebook', '--execute', '--inplace',
           '--ExecutePreprocessor.timeout=600',
           str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f'FAIL  {path}  elapsed={dt:.1f}s')
        print(r.stdout[-1500:])
        print('STDERR:'); print(r.stderr[-1500:])
        return False
    print(f'executed {path}  elapsed={dt:.1f}s')
    return True


def main():
    path = Path('notebooks/nb04_v10_benchmark.ipynb')
    write_nb(path, build_cells())
    ok = execute_nb(path)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
