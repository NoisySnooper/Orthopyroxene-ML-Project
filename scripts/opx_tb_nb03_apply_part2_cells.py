#!/usr/bin/env python3
"""One-shot helper to append the TabPFN Part 2 cells into nb04, nbF, and nb09.

Idempotent: detects existing tagged cells by a sentinel comment and skips if
already present. Run with the main venv (any python with nbformat).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

SENTINEL = '# --- TABPFN_PART2_BLOCK ---'


def _code_cell(source_lines: list[str]) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines,
    }


def _md_cell(source_lines: list[str]) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines,
    }


def append_cells(nb_path: Path, cells: list[dict]) -> bool:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb.get('cells', []):
        src = c.get('source', [])
        joined = ''.join(src) if isinstance(src, list) else src
        if SENTINEL in joined:
            print(f'[skip] {nb_path.name}: sentinel present')
            return False
    nb.setdefault('cells', []).extend(cells)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'[ok] {nb_path.name}: appended {len(cells)} cell(s)')
    return True


# ---------------------------------------------------------------------------
# nb04 cell: TabPFN head-to-head against v10 aggregate-best + external best.
# ---------------------------------------------------------------------------
NB04_MD = _md_cell([
    '## TabPFN v2 supplementary baseline (appended 2026-04-19)\n',
    '\n',
    'Joins `results/v10_tabpfn_multiseed_summary.csv` against the v10 '
    'aggregate-best tuned model per (pipeline, track, target) and the best '
    'classical/ML external reference from `results/v10_regime_allmodels.csv`. '
    'Writes `results/v10_tabpfn_head_to_head.csv`. Wrapped in try/except so '
    'nb04 still runs if TabPFN outputs are absent. See '
    '`docs/nb03_tabpfn_plan.md`.\n',
])

NB04_CODE = _code_cell([
    '# --- TABPFN_PART2_BLOCK --- head-to-head vs v10 best + external best\n',
    'try:\n',
    '    import pandas as pd\n',
    '    from pathlib import Path\n',
    '    from config import RESULTS\n',
    '\n',
    '    tab = pd.read_csv(RESULTS / "v10_tabpfn_multiseed_summary.csv")\n',
    '    opx = pd.read_csv(RESULTS / "opx_multiseed_summary.csv")\n',
    '    cpx = pd.read_csv(RESULTS / "v10_cpx_multiseed_summary.csv")\n',
    '    v10 = pd.concat([opx, cpx], ignore_index=True)\n',
    '\n',
    '    best_v10 = (v10.sort_values("mean")\n',
    '                   .groupby(["track", "target"])\n',
    '                   .first()\n',
    '                   .reset_index()[["track", "target", "model", '
    '"feature_set", "mean", "std"]]\n',
    '                   .rename(columns={"mean": "v10_best_rmse",\n',
    '                                    "std":  "v10_best_std",\n',
    '                                    "model": "v10_best_model",\n',
    '                                    "feature_set": "v10_best_fs"}))\n',
    '\n',
    '    tab_small = tab[["track", "target", "mean", "std"]].rename(\n',
    '        columns={"mean": "tabpfn_rmse", "std": "tabpfn_std"})\n',
    '\n',
    '    # External best: smallest "rmse" at regime=ALL among '
    'method_family in (putirka, agreda_lopez, jorgenson) for each cell.\n',
    '    reg = pd.read_csv(RESULTS / "v10_regime_allmodels.csv")\n',
    '    ext = reg[(reg.regime == "ALL") & \n',
    '              (reg.method_family.isin(["putirka", "agreda_lopez", '
    '"jorgenson"]))].copy()\n',
    '    ext_best = (ext.sort_values("rmse")\n',
    '                   .groupby(["track", "target"]).first()\n',
    '                   .reset_index()[["track", "target", "rmse", '
    '"method", "method_family"]]\n',
    '                   .rename(columns={"rmse": "external_best_rmse",\n',
    '                                    "method": "external_best_method",\n',
    '                                    "method_family": "external_best_family"}))\n',
    '\n',
    '    h2h = best_v10.merge(tab_small, on=["track", "target"], how="left")\n',
    '    h2h = h2h.merge(ext_best, on=["track", "target"], how="left")\n',
    '\n',
    '    def _verdict(r):\n',
    '        if pd.isna(r["tabpfn_rmse"]) or pd.isna(r["v10_best_rmse"]):\n',
    '            return "incomplete"\n',
    '        d = r["tabpfn_rmse"] - r["v10_best_rmse"]\n',
    '        tol = 0.5 * max(r.get("v10_best_std", 0) or 0, \n',
    '                       r.get("tabpfn_std", 0) or 0)\n',
    '        if d < -tol: return "tabpfn_wins"\n',
    '        if d >  tol: return "v10_wins"\n',
    '        return "competitive"\n',
    '    h2h["verdict"] = h2h.apply(_verdict, axis=1)\n',
    '\n',
    '    h2h.to_csv(RESULTS / "v10_tabpfn_head_to_head.csv", index=False)\n',
    '    print(f"wrote {RESULTS}/v10_tabpfn_head_to_head.csv ({len(h2h)} rows)")\n',
    '    print(h2h.to_string(index=False))\n',
    'except FileNotFoundError as e:\n',
    '    print(f"[skip] TabPFN head-to-head: {e}")\n',
    'except Exception as e:\n',
    '    print(f"[error] TabPFN head-to-head: {e}")\n',
])

# ---------------------------------------------------------------------------
# nbF cell: fig35_tabpfn_vs_v10 — 2x4 grouped-bar grid with verdict annotation.
# ---------------------------------------------------------------------------
NBF_MD = _md_cell([
    '## Fig 35: TabPFN v2 vs v10 vs external best\n',
    '\n',
    'Grouped-bar comparison of the TabPFN v2 supplementary baseline against '
    'the pre-registered v10 tuned pipeline and the best external reference '
    'per (pipeline, track, target). See `docs/nb03_tabpfn_plan.md` for '
    'the construction and limitations of this baseline.\n',
])

NBF_CODE = _code_cell([
    '# --- TABPFN_PART2_BLOCK --- fig35 tabpfn vs v10 vs external\n',
    'try:\n',
    '    import numpy as np\n',
    '    import pandas as pd\n',
    '    import matplotlib.pyplot as plt\n',
    '    from pathlib import Path\n',
    '    from config import RESULTS, FIGURES\n',
    '    from src.plot_style import FAMILY_COLORS, save_both\n',
    '\n',
    '    h2h = pd.read_csv(RESULTS / "v10_tabpfn_head_to_head.csv")\n',
    '    tracks = ["opx_liq", "opx_only", "cpx_liq", "cpx_only"]\n',
    '    targets = ["T_C", "P_kbar"]\n',
    '    UNITS = {"T_C": "C", "P_kbar": "kbar"}\n',
    '\n',
    '    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)\n',
    '    fig.suptitle("Fig 35. TabPFN v2 vs v10 tuned vs external best", \n',
    '                 fontsize=11)\n',
    '    COLOR_V10 = FAMILY_COLORS.get("boosted", "#0072B2")\n',
    '    COLOR_TAB = "#009E73"\n',
    '    COLOR_EXT = FAMILY_COLORS.get("putirka", "#56B4E9")\n',
    '\n',
    '    for ti, target in enumerate(targets):\n',
    '        for tj, track in enumerate(tracks):\n',
    '            ax = axes[ti, tj]\n',
    '            row = h2h[(h2h.track == track) & (h2h.target == target)]\n',
    '            if row.empty:\n',
    '                ax.set_visible(False); continue\n',
    '            r = row.iloc[0]\n',
    '            means = [r.get("v10_best_rmse"), r.get("tabpfn_rmse"),\n',
    '                     r.get("external_best_rmse")]\n',
    '            errs  = [r.get("v10_best_std", 0), r.get("tabpfn_std", 0),\n',
    '                     0]\n',
    '            labels = ["v10 best", "TabPFN", "external"]\n',
    '            colors = [COLOR_V10, COLOR_TAB, COLOR_EXT]\n',
    '            means_clean = [m if pd.notna(m) else 0.0 for m in means]\n',
    '            errs_clean  = [e if pd.notna(e) else 0.0 for e in errs]\n',
    '            bars = ax.bar(labels, means_clean, yerr=errs_clean,\n',
    '                          color=colors, edgecolor="black", linewidth=0.6,\n',
    '                          capsize=3)\n',
    '            ax.set_title(f"{track} / {target}", fontsize=9)\n',
    '            ax.set_ylabel(f"RMSE ({UNITS[target]})", fontsize=9)\n',
    '            ax.tick_params(labelsize=8)\n',
    '            verdict = r.get("verdict", "")\n',
    '            ax.text(0.02, 0.98, verdict, transform=ax.transAxes,\n',
    '                    va="top", ha="left", fontsize=8,\n',
    '                    bbox=dict(boxstyle="round,pad=0.2", \n',
    '                              fc="white", ec="black", alpha=0.7))\n',
    '\n',
    '    caption = ("Fig. 35. TabPFN v2 (Hollmann et al. 2025) versus the v10 "\n',
    '               "pre-registered tuned baseline and the best classical/ML "\n',
    '               "external reference per (pipeline, track, target) "\n',
    '               "combination. TabPFN error bars show 5-seed ensemble "\n',
    '               "stability; v10 error bars show 20-seed model-fit "\n',
    '               "variance. TabPFN receives raw oxide features only (no "\n',
    '               "ALR/PWLR) and is fit with default hyperparameters "\n',
    '               "(n_estimators=8 opx / 4 cpx, device=cpu). Sources: "\n',
    '               "results/v10_tabpfn_multiseed_summary.csv, "\n',
    '               "results/v10_tabpfn_head_to_head.csv, "\n',
    '               "results/v10_{opx,cpx}_multiseed_summary.csv.")\n',
    '    save_both(fig, FIGURES / "fig35_tabpfn_vs_v10", caption=caption,\n',
    '              dpi=300)\n',
    '    plt.show()\n',
    '    print("wrote fig35_tabpfn_vs_v10.{pdf,png,txt}")\n',
    'except FileNotFoundError as e:\n',
    '    print(f"[skip] fig35: {e}")\n',
    'except Exception as e:\n',
    '    print(f"[error] fig35: {e}")\n',
])

# ---------------------------------------------------------------------------
# nb09 cell: Table S12 tabpfn_benchmark.
# ---------------------------------------------------------------------------
NB09_MD = _md_cell([
    '## Table S12: TabPFN v2 supplementary baseline\n',
    '\n',
    'Head-to-head comparison of the TabPFN v2 baseline against the v10 '
    'tuned pipeline and the best classical/ML external reference per '
    '(pipeline, track, target). Wraps in try/except so nb09 still runs '
    'if TabPFN outputs are absent.\n',
])

NB09_CODE = _code_cell([
    '# --- TABPFN_PART2_BLOCK --- Table S12 tabpfn benchmark\n',
    'try:\n',
    '    import pandas as pd\n',
    '    from pathlib import Path\n',
    '    from config import RESULTS\n',
    '\n',
    '    h2h = pd.read_csv(RESULTS / "v10_tabpfn_head_to_head.csv")\n',
    '    def _fmt(m, s, prec=2):\n',
    '        if pd.isna(m): return "--"\n',
    '        if pd.isna(s) or s == 0: return f"{m:.{prec}f}"\n',
    '        return f"{m:.{prec}f} +/- {s:.{prec}f}"\n',
    '    disp = pd.DataFrame({\n',
    '        "pipeline": ["opx" if "opx" in t else "cpx" for t in h2h["track"]],\n',
    '        "track":    h2h["track"],\n',
    '        "target":   h2h["target"],\n',
    '        "v10_best": [_fmt(m, s, 2 if tgt == "P_kbar" else 1)\n',
    '                     for m, s, tgt in zip(h2h["v10_best_rmse"],\n',
    '                                           h2h["v10_best_std"],\n',
    '                                           h2h["target"])],\n',
    '        "v10_best_model": h2h["v10_best_model"].fillna(""),\n',
    '        "tabpfn":   [_fmt(m, s, 2 if tgt == "P_kbar" else 1)\n',
    '                     for m, s, tgt in zip(h2h["tabpfn_rmse"],\n',
    '                                           h2h["tabpfn_std"],\n',
    '                                           h2h["target"])],\n',
    '        "external_best": [_fmt(m, 0, 2 if tgt == "P_kbar" else 1)\n',
    '                          for m, tgt in zip(h2h["external_best_rmse"],\n',
    '                                             h2h["target"])],\n',
    '        "external_method": h2h["external_best_method"].fillna(""),\n',
    '        "verdict":  h2h["verdict"],\n',
    '    })\n',
    '    disp.to_csv("tables/S8_12_tabpfn_benchmark.csv", index=False)\n',
    '    with open("tables/S8_12_tabpfn_benchmark.md", "w", encoding="utf-8") as f:\n',
    '        f.write("# Table S12. TabPFN v2 baseline comparison\\n\\n")\n',
    '        f.write(disp.to_markdown(index=False))\n',
    '        f.write("\\n\\n*TabPFN fit with default hyperparameters '
    '(n_estimators=8 opx / 4 cpx, device=cpu, 5 seeds). Reference: '
    'Hollmann et al. 2025, Nature 637.*\\n")\n',
    '    tex = disp.to_latex(index=False, escape=True, longtable=False)\n',
    '    with open("tables/S8_12_tabpfn_benchmark.tex", "w", encoding="utf-8") as f:\n',
    '        f.write("% Table S12. TabPFN v2 baseline comparison\\n")\n',
    '        f.write(tex)\n',
    '    print(f"wrote S8_12_tabpfn_benchmark.{{csv,md,tex}} ({len(disp)} rows)")\n',
    '    print(disp.to_string(index=False))\n',
    'except FileNotFoundError as e:\n',
    '    print(f"[skip] S8_12: {e}")\n',
    'except Exception as e:\n',
    '    print(f"[error] S8_12: {e}")\n',
])


def main():
    nb04 = PROJECT_ROOT / 'notebooks' / 'nb04_putirka_benchmark.ipynb'
    nbF  = PROJECT_ROOT / 'notebooks' / 'nbF_figures.ipynb'
    nb09 = PROJECT_ROOT / 'notebooks' / 'nb09_manuscript_compilation.ipynb'
    append_cells(nb04, [NB04_MD, NB04_CODE])
    append_cells(nbF,  [NBF_MD,  NBF_CODE])
    append_cells(nb09, [NB09_MD, NB09_CODE])
    return 0


if __name__ == '__main__':
    sys.exit(main())
