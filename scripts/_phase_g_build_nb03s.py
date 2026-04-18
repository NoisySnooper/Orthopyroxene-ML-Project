#!/usr/bin/env python3
"""Rebuild the four v10 nb03_* thin-orchestrator notebooks so they
run inline: load CSV results, display tables, render figures. No
subprocess calls (training already complete by Phase C/D/E/F
scripts). Then execute each notebook in-place via nbconvert so the
executed cell outputs are embedded in the .ipynb file.

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


def nb_opx():
    cells = []
    cells.append(md(
        "# nb03_opx_baseline_models (v10 Phase C)\n"
        "\n"
        "Loads and visualizes the Phase C opx training results produced by "
        "`scripts/v10_phase_c_opx_driver.py` + `scripts/v10_phase_c_opx_runner.py`. "
        "96 Optuna studies (8 models x 2 targets x 2 tracks x 3 feature sets), "
        "zero failures.\n"
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
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.max_columns', 50)\n"
    ))

    cells.append(md("## Base model results (96 rows)\n"))
    cells.append(code(
        "df_bases = pd.read_csv(RESULTS / 'v10_opx_per_cell_results.csv')\n"
        "print(f'n_rows={len(df_bases)}')\n"
        "df_bases.round(3)\n"
    ))

    cells.append(md("## Ensemble results (36 rows: ridge / two_level / greedy)\n"))
    cells.append(code(
        "df_ens = pd.read_csv(RESULTS / 'v10_opx_ensemble_results.csv')\n"
        "df_ens.round(3)\n"
    ))

    cells.append(md("## Best base per (target, track)\n"))
    cells.append(code(
        "idx = df_bases.groupby(['target', 'track'])['test_rmse'].idxmin()\n"
        "best_per_cell = df_bases.loc[idx].reset_index(drop=True)\n"
        "best_per_cell[['target','track','model','feature_set','test_rmse','test_r2']].round(3)\n"
    ))

    cells.append(md("## Best ensemble per (target, track)\n"))
    cells.append(code(
        "idx = df_ens.groupby(['target', 'track'])['test_rmse'].idxmin()\n"
        "best_ens = df_ens.loc[idx].reset_index(drop=True)\n"
        "best_ens.round(3)\n"
    ))

    cells.append(md("## Figure: per-model test RMSE heatmap\n"
                    "Rows = 8 models, columns = (target, track, feature_set). "
                    "Color = RMSE, lower is better (darker).\n"))
    cells.append(code(
        "def rmse_heatmap(df, title):\n"
        "    pv = df.pivot_table(index='model',\n"
        "                        columns=['target','track','feature_set'],\n"
        "                        values='test_rmse')\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)\n"
        "    for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "        sub = pv[tgt]\n"
        "        im = ax.imshow(sub.values, aspect='auto', cmap='viridis_r')\n"
        "        ax.set_xticks(range(sub.shape[1]))\n"
        "        ax.set_xticklabels([f'{t}/{f}' for t, f in sub.columns],\n"
        "                           rotation=45, ha='right', fontsize=8)\n"
        "        ax.set_yticks(range(sub.shape[0]))\n"
        "        ax.set_yticklabels(sub.index)\n"
        "        ax.set_title(f'{tgt}  RMSE  ({title})')\n"
        "        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
        "    plt.tight_layout()\n"
        "    return fig\n"
        "\n"
        "fig = rmse_heatmap(df_bases, 'opx bases')\n"
        "plt.show()\n"
    ))

    cells.append(md("## Figure: base vs ensemble RMSE per cell\n"
                    "Shows how much each ensemble beats the best base model.\n"))
    cells.append(code(
        "best_base = df_bases.loc[df_bases.groupby(['target','track'])['test_rmse']\n"
        "                                  .idxmin(), ['target','track','test_rmse']]\n"
        "best_base = best_base.rename(columns={'test_rmse': 'best_base_rmse'})\n"
        "merged = df_ens.merge(best_base, on=['target', 'track'])\n"
        "merged['delta'] = merged['test_rmse'] - merged['best_base_rmse']\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = merged[merged['target'] == tgt]\n"
        "    methods = sub['method'].unique()\n"
        "    tracks  = sorted(sub['track'].unique())\n"
        "    x = np.arange(len(tracks))\n"
        "    w = 0.25\n"
        "    for i, m in enumerate(methods):\n"
        "        vals = [sub[(sub['method']==m) & (sub['track']==t)]['delta'].values[0]\n"
        "                for t in tracks]\n"
        "        ax.bar(x + i*w - w, vals, w, label=m)\n"
        "    ax.axhline(0, color='k', lw=0.8)\n"
        "    ax.set_xticks(x); ax.set_xticklabels(tracks)\n"
        "    ax.set_title(f'{tgt}: ensemble RMSE - best base RMSE')\n"
        "    ax.set_ylabel('delta RMSE (negative = ensemble wins)')\n"
        "    ax.legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Test protocol results (T01/T02/T11/T12)\n"))
    cells.append(code(
        "from scripts.v10_test_protocol import summarize_tests\n"
        "tests_df = summarize_tests(pipeline='opx')\n"
        "print(f'n_test_rows={len(tests_df)}')\n"
        "tests_df[['test_id','target','track','passed','value','threshold']].head(60)\n"
    ))
    cells.append(code(
        "if not tests_df.empty:\n"
        "    rate = tests_df.groupby('test_id')['passed'].agg(['sum', 'count'])\n"
        "    rate['pass_rate'] = rate['sum'] / rate['count']\n"
        "    print('Pass rate per test (across cells):')\n"
        "    print(rate)\n"
    ))

    cells.append(md(
        "## Summary\n"
        "\n"
        "Phase C opx training is complete. Canonical models under "
        "`models/canonical/opx/`. Next phase: Phase G figures + manuscript "
        "compilation.\n"
    ))
    return cells


def nb_cpx():
    cells = []
    cells.append(md(
        "# nb03_cpx_baseline_models (v10 Phase D)\n"
        "\n"
        "Loads and visualizes Phase D cpx training results. 96 Optuna "
        "studies (same 8x2x2x3 shape as Phase C), zero failures.\n"
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
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.max_columns', 50)\n"
    ))

    cells.append(md("## Base model results (96 rows)\n"))
    cells.append(code(
        "df_bases = pd.read_csv(RESULTS / 'v10_cpx_per_cell_results.csv')\n"
        "print(f'n_rows={len(df_bases)}')\n"
        "df_bases.round(3)\n"
    ))

    cells.append(md("## Ensemble results (36 rows)\n"))
    cells.append(code(
        "df_ens = pd.read_csv(RESULTS / 'v10_cpx_ensemble_results.csv')\n"
        "df_ens.round(3)\n"
    ))

    cells.append(md("## Best base per (target, track)\n"))
    cells.append(code(
        "idx = df_bases.groupby(['target', 'track'])['test_rmse'].idxmin()\n"
        "best_per_cell = df_bases.loc[idx].reset_index(drop=True)\n"
        "best_per_cell[['target','track','model','feature_set','test_rmse','test_r2']].round(3)\n"
    ))

    cells.append(md("## Best ensemble per (target, track)\n"))
    cells.append(code(
        "idx = df_ens.groupby(['target', 'track'])['test_rmse'].idxmin()\n"
        "best_ens = df_ens.loc[idx].reset_index(drop=True)\n"
        "best_ens.round(3)\n"
    ))

    cells.append(md("## Figure: per-model test RMSE heatmap\n"))
    cells.append(code(
        "pv = df_bases.pivot_table(index='model',\n"
        "                          columns=['target','track','feature_set'],\n"
        "                          values='test_rmse')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = pv[tgt]\n"
        "    im = ax.imshow(sub.values, aspect='auto', cmap='viridis_r')\n"
        "    ax.set_xticks(range(sub.shape[1]))\n"
        "    ax.set_xticklabels([f'{t}/{f}' for t, f in sub.columns],\n"
        "                       rotation=45, ha='right', fontsize=8)\n"
        "    ax.set_yticks(range(sub.shape[0]))\n"
        "    ax.set_yticklabels(sub.index)\n"
        "    ax.set_title(f'{tgt}  RMSE  (cpx bases)')\n"
        "    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Figure: base vs ensemble RMSE per cell\n"))
    cells.append(code(
        "best_base = df_bases.loc[df_bases.groupby(['target','track'])['test_rmse']\n"
        "                                  .idxmin(), ['target','track','test_rmse']]\n"
        "best_base = best_base.rename(columns={'test_rmse': 'best_base_rmse'})\n"
        "merged = df_ens.merge(best_base, on=['target', 'track'])\n"
        "merged['delta'] = merged['test_rmse'] - merged['best_base_rmse']\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = merged[merged['target'] == tgt]\n"
        "    methods = sub['method'].unique()\n"
        "    tracks  = sorted(sub['track'].unique())\n"
        "    x = np.arange(len(tracks))\n"
        "    w = 0.25\n"
        "    for i, m in enumerate(methods):\n"
        "        vals = [sub[(sub['method']==m) & (sub['track']==t)]['delta'].values[0]\n"
        "                for t in tracks]\n"
        "        ax.bar(x + i*w - w, vals, w, label=m)\n"
        "    ax.axhline(0, color='k', lw=0.8)\n"
        "    ax.set_xticks(x); ax.set_xticklabels(tracks)\n"
        "    ax.set_title(f'{tgt}: ensemble RMSE - best base RMSE')\n"
        "    ax.set_ylabel('delta RMSE'); ax.legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Test protocol results\n"))
    cells.append(code(
        "from scripts.v10_test_protocol import summarize_tests\n"
        "tests_df = summarize_tests(pipeline='cpx')\n"
        "print(f'n_test_rows={len(tests_df)}')\n"
        "tests_df[['test_id','target','track','passed','value','threshold']].head(60)\n"
    ))
    cells.append(code(
        "if not tests_df.empty:\n"
        "    rate = tests_df.groupby('test_id')['passed'].agg(['sum', 'count'])\n"
        "    rate['pass_rate'] = rate['sum'] / rate['count']\n"
        "    print(rate)\n"
    ))

    cells.append(md(
        "## Summary\n"
        "\n"
        "Phase D cpx training complete. Canonical models under `models/canonical/cpx/`.\n"
    ))
    return cells


def nb_twopx():
    cells = []
    cells.append(md(
        "# nb03_twopx_baseline_models (v10 Phase E)\n"
        "\n"
        "Loads and visualizes Phase E twopx (paired opx+cpx) training "
        "results. 64 Optuna studies (8 models x 2 targets x 1 track x 4 "
        "feature sets), zero failures. Twopx has a dedicated feature set "
        "`twopx_components` on top of raw/alr/pwlr.\n"
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
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.max_columns', 50)\n"
    ))

    cells.append(md("## Base model results (64 rows)\n"))
    cells.append(code(
        "df_bases = pd.read_csv(RESULTS / 'v10_twopx_per_cell_results.csv')\n"
        "print(f'n_rows={len(df_bases)}')\n"
        "df_bases.round(3)\n"
    ))

    cells.append(md("## Ensemble results\n"))
    cells.append(code(
        "df_ens = pd.read_csv(RESULTS / 'v10_twopx_ensemble_results.csv')\n"
        "df_ens.round(3)\n"
    ))

    cells.append(md("## Best base per (target, feature_set)\n"))
    cells.append(code(
        "idx = df_bases.groupby(['target','feature_set'])['test_rmse'].idxmin()\n"
        "best_per_cell = df_bases.loc[idx].reset_index(drop=True)\n"
        "best_per_cell[['target','track','model','feature_set','test_rmse','test_r2']].round(3)\n"
    ))

    cells.append(md("## Best ensemble per (target, feature_set)\n"))
    cells.append(code(
        "idx = df_ens.groupby(['target','feature_set'])['test_rmse'].idxmin()\n"
        "best_ens = df_ens.loc[idx].reset_index(drop=True)\n"
        "best_ens.round(3)\n"
    ))

    cells.append(md("## Figure: per-model RMSE across feature sets\n"))
    cells.append(code(
        "pv = df_bases.pivot_table(index='model', columns=['target','feature_set'],\n"
        "                          values='test_rmse')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = pv[tgt]\n"
        "    im = ax.imshow(sub.values, aspect='auto', cmap='viridis_r')\n"
        "    ax.set_xticks(range(sub.shape[1])); ax.set_xticklabels(sub.columns, rotation=30)\n"
        "    ax.set_yticks(range(sub.shape[0])); ax.set_yticklabels(sub.index)\n"
        "    ax.set_title(f'{tgt} RMSE (twopx)')\n"
        "    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Figure: test RMSE by model and feature set (bars)\n"))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = df_bases[df_bases['target']==tgt]\n"
        "    models = sorted(sub['model'].unique())\n"
        "    fsets = sorted(sub['feature_set'].unique())\n"
        "    x = np.arange(len(models))\n"
        "    w = 0.18\n"
        "    for i, fs in enumerate(fsets):\n"
        "        vals = [sub[(sub['model']==m) & (sub['feature_set']==fs)]['test_rmse'].values[0]\n"
        "                for m in models]\n"
        "        ax.bar(x + (i - len(fsets)/2)*w + w/2, vals, w, label=fs)\n"
        "    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right')\n"
        "    ax.set_ylabel('test RMSE'); ax.set_title(f'twopx {tgt}')\n"
        "    ax.legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Test protocol results\n"))
    cells.append(code(
        "from scripts.v10_test_protocol import summarize_tests\n"
        "tests_df = summarize_tests(pipeline='twopx')\n"
        "print(f'n_test_rows={len(tests_df)}')\n"
        "tests_df[['test_id','target','track','passed','value','threshold']].head(60)\n"
    ))
    cells.append(code(
        "if not tests_df.empty:\n"
        "    rate = tests_df.groupby('test_id')['passed'].agg(['sum', 'count'])\n"
        "    rate['pass_rate'] = rate['sum'] / rate['count']\n"
        "    print(rate)\n"
    ))

    cells.append(md(
        "## Summary\n"
        "\n"
        "Phase E twopx training complete. P_kbar twopx RMSE ~4.26 kbar is "
        "the best of any opx/cpx track, consistent with the information "
        "gain of co-equilibrated phases.\n"
    ))
    return cells


def nb_universal():
    cells = []
    cells.append(md(
        "# nb03_universal_exploration (v10 Phase F)\n"
        "\n"
        "Loads and visualizes Phase F universal (mask-aware, single-model) "
        "training results. 16 Optuna studies (8 models x 2 targets x 1 "
        "track x 1 feature set `universal_raw`). Includes T13/T14 "
        "phase-scope evaluation.\n"
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
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.max_columns', 50)\n"
        "UNIV = RESULTS / 'universal'\n"
    ))

    cells.append(md("## Base model results\n"))
    cells.append(code(
        "df_bases = pd.read_csv(UNIV / 'v10_universal_per_cell_results.csv')\n"
        "print(f'n_rows={len(df_bases)}')\n"
        "df_bases.round(3)\n"
    ))

    cells.append(md("## Ensemble results\n"))
    cells.append(code(
        "df_ens = pd.read_csv(UNIV / 'v10_universal_ensemble_results.csv')\n"
        "df_ens.round(3)\n"
    ))

    cells.append(md("## Best base / ensemble per target\n"))
    cells.append(code(
        "best_base = df_bases.loc[df_bases.groupby('target')['test_rmse'].idxmin()]\n"
        "best_ens  = df_ens.loc[df_ens.groupby('target')['test_rmse'].idxmin()]\n"
        "print('Best base per target:'); print(best_base[['target','model','test_rmse']].round(3).to_string(index=False))\n"
        "print('\\nBest ensemble per target:'); print(best_ens[['target','method','test_rmse']].round(3).to_string(index=False))\n"
    ))

    cells.append(md("## Figure: RMSE by base model (universal)\n"))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = df_bases[df_bases['target']==tgt].sort_values('test_rmse')\n"
        "    ax.barh(sub['model'], sub['test_rmse'], color='steelblue')\n"
        "    ax.set_xlabel('test RMSE')\n"
        "    ax.set_title(f'universal {tgt}')\n"
        "    ax.invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "## Phase-scope stratified RMSE (T13/T14)\n"
        "\n"
        "Universal model RMSE stratified by phase scope on the test set. "
        "T14 requires `twopx_liq` (3 phases) to have the lowest RMSE. "
        "T13 requires median RMSE to decrease monotonically from "
        "1-phase -> 2-phase -> 3-phase.\n"
    ))
    cells.append(code(
        "df_scope = pd.read_csv(UNIV / 'v10_universal_scope_rmse.csv')\n"
        "df_scope.round(3)\n"
    ))
    cells.append(code(
        "scope_tiers = {'opx_only':1,'cpx_only':1,\n"
        "               'opx_liq':2,'cpx_liq':2,'twopx':2,\n"
        "               'twopx_liq':3}\n"
        "df_scope['tier'] = df_scope['phase_scope'].map(scope_tiers)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for ax, tgt in zip(axes, ['T_C', 'P_kbar']):\n"
        "    sub = df_scope[df_scope['target']==tgt].sort_values('tier')\n"
        "    colors = {1:'#D55E00', 2:'#E69F00', 3:'#0072B2'}\n"
        "    ax.bar(sub['phase_scope'], sub['rmse'],\n"
        "           color=[colors[t] for t in sub['tier']])\n"
        "    ax.set_ylabel('test RMSE'); ax.set_title(f'universal {tgt} by phase scope')\n"
        "    ax.set_xticklabels(sub['phase_scope'], rotation=30, ha='right')\n"
        "    for xi, (ps, n, r) in enumerate(zip(sub['phase_scope'], sub['n'], sub['rmse'])):\n"
        "        ax.text(xi, r, f'n={n}', ha='center', va='bottom', fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(md("## Test protocol (T01/T02/T11/T12/T13/T14)\n"))
    cells.append(code(
        "from scripts.v10_test_protocol import summarize_tests\n"
        "tests_df = summarize_tests(pipeline='universal')\n"
        "print(f'n_test_rows={len(tests_df)}')\n"
        "tests_df[['test_id','target','track','passed','value','threshold']].head(60)\n"
    ))
    cells.append(code(
        "if not tests_df.empty:\n"
        "    rate = tests_df.groupby('test_id')['passed'].agg(['sum', 'count'])\n"
        "    rate['pass_rate'] = rate['sum'] / rate['count']\n"
        "    print(rate)\n"
    ))

    cells.append(md(
        "## Summary\n"
        "\n"
        "Phase F universal training complete. Best P_kbar RMSE on the "
        "`twopx_liq` subset is ~4.07 kbar, validating T14 (all-phase "
        "subset is best). Canonical models under `models/canonical/universal/`.\n"
    ))
    return cells


def write_nb(path: Path, cells):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.13',
            },
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
    targets = [
        (Path('notebooks/nb03_opx_baseline_models.ipynb'), nb_opx()),
        (Path('notebooks/nb03_cpx_baseline_models.ipynb'), nb_cpx()),
        (Path('notebooks/nb03_twopx_baseline_models.ipynb'), nb_twopx()),
        (Path('notebooks/nb03_universal_exploration.ipynb'), nb_universal()),
    ]
    for path, cells in targets:
        write_nb(path, cells)

    ok = sum(execute_nb(p) for p, _ in targets)
    print(f'\nDONE  executed={ok}/{len(targets)}')
    return 0 if ok == len(targets) else 1


if __name__ == '__main__':
    sys.exit(main())
