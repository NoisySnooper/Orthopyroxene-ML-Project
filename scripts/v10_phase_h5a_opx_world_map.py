#!/usr/bin/env python3
"""Phase H.5a + H.5c: static world map, opx-only natural predictions.

Scope: opx track only. Cpx and twopx layers deferred until H.1b/c/d
complete. This produces the opx panels of the eventual 3-mineral map
stack.

Three panels (Robinson projection, global extent):
  Panel A  tectonic setting (categorical colormap)
  Panel B  predicted T_C    (continuous colormap; OOD-flagged samples
                             rendered with reduced alpha)
  Panel C  predicted P_kbar  (H.5c) regime-stratified discrete colormap
                              with pre-registered bin boundaries as ticks

Per Phase G collision 3, Panel B/C do NOT claim per-regime RMSE. The
pre-registered P-regime bins stratify the Panel C colorbar visually
only; regime assignment is descriptive context, not quantitative error.

Samples with missing lat/lon are dropped (~27/53050).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import (RESULTS, LOGS, FIGURES,
                    P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS)

PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
OUT_PNG = FIGURES / 'fig_h5ac_opx_world_map.png'
LOG_PATH = LOGS / 'v10_phase_h5ac_opx_world_map.log'


TECTONIC_COLOR = {
    'CONVERGENT MARGIN':                             '#D55E00',
    'INTRAPLATE VOLCANICS':                          '#0072B2',
    'RIFT VOLCANICS':                                '#009E73',
    'CONTINENTAL FLOOD BASALT':                      '#CC79A7',
    'ARCHEAN CRATON (INCLUDING GREENSTONE BELTS)':   '#56B4E9',
    'OCEAN ISLAND':                                  '#E69F00',
    'SEAMOUNT':                                      '#F0E442',
    'OCEANIC PLATEAU':                               '#999999',
    'COMPLEX VOLCANIC SETTINGS':                     '#660066',
    'SUBMARINE RIDGE':                               '#33FFCC',
    'UNKNOWN':                                       '#BBBBBB',
}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _draw_basemap(ax):
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#f5f2e9', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='#e7f0f7', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#444', zorder=1)
    ax.gridlines(linewidth=0.2, color='#888', alpha=0.5, zorder=2)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.5a opx world map', fh)
        df = pd.read_csv(PRED_CSV, low_memory=False)
        _log(f'loaded {len(df)} opx predictions', fh)

        df = df.dropna(subset=['lat', 'lon'])
        _log(f'after lat/lon dropna: {len(df)}', fh)

        df['tec_label'] = df['TECTONIC SETTING'].fillna('UNKNOWN')
        df.loc[~df['tec_label'].isin(TECTONIC_COLOR), 'tec_label'] = 'UNKNOWN'
        _log(f'tectonic label unique: {df.tec_label.nunique()}', fh)

        fig = plt.figure(figsize=(22, 9))
        proj = ccrs.Robinson()

        # ---- Panel A: tectonic setting ----
        axA = fig.add_subplot(1, 3, 1, projection=proj)
        _draw_basemap(axA)
        for label, color in TECTONIC_COLOR.items():
            sub = df[df.tec_label == label]
            if sub.empty:
                continue
            axA.scatter(sub.lon.values, sub.lat.values,
                        transform=ccrs.PlateCarree(),
                        s=3, c=color, alpha=0.55,
                        edgecolors='none', label=f'{label} (n={len(sub)})')
        axA.set_title('Panel A  Natural opx sample localities by tectonic setting',
                      fontsize=11)
        leg = axA.legend(loc='lower left', fontsize=7, markerscale=3,
                         framealpha=0.9, ncol=1)
        for t in leg.get_texts():
            t.set_fontsize(7)

        # ---- Panel B: predicted T_C (alpha down on OOD) ----
        axB = fig.add_subplot(1, 3, 2, projection=proj)
        _draw_basemap(axB)
        order = df['pred_T_C_opx_only'].argsort().values
        dfo = df.iloc[order]
        alpha = np.where(dfo['ood_flag'].values, 0.15, 0.55)
        scat = axB.scatter(dfo.lon.values, dfo.lat.values,
                           transform=ccrs.PlateCarree(),
                           c=dfo.pred_T_C_opx_only.values, cmap='magma',
                           vmin=900, vmax=1400,
                           s=4, alpha=alpha, edgecolors='none')
        cb = plt.colorbar(scat, ax=axB, orientation='horizontal',
                          fraction=0.04, pad=0.05, shrink=0.8)
        cb.set_label('Predicted T_C (opx-only, LightGBM / alr)  [C]', fontsize=9)
        axB.set_title(
            'Panel B  Predicted T; dim = OOD-flagged vs training.',
            fontsize=10)

        # ---- Panel C (H.5c): predicted P with pre-registered regime bins ----
        axC = fig.add_subplot(1, 3, 3, projection=proj)
        _draw_basemap(axC)
        regime_edges = list(P_REGIME_BIN_EDGES_KBAR)  # [0, 5, 15, 30, 100]
        # Discrete colormap aligned to 4 regimes: shallow, deep_crustal, lith_mantle, deeper_mantle
        regime_colors = ['#fde725', '#5ec962', '#21918c', '#3b528b']
        regime_cmap = mcolors.ListedColormap(regime_colors)
        regime_norm = mcolors.BoundaryNorm(regime_edges, regime_cmap.N)
        order_p = df['pred_P_kbar_opx_only'].argsort().values
        dfp = df.iloc[order_p]
        alpha_p = np.where(dfp['ood_flag'].values, 0.15, 0.60)
        # Clip predicted P into [0, P_ceiling) so BoundaryNorm covers every point
        p_clip = np.clip(dfp.pred_P_kbar_opx_only.values,
                         regime_edges[0], regime_edges[-1] - 1e-6)
        scatC = axC.scatter(dfp.lon.values, dfp.lat.values,
                            transform=ccrs.PlateCarree(),
                            c=p_clip, cmap=regime_cmap, norm=regime_norm,
                            s=4, alpha=alpha_p, edgecolors='none')
        cbC = plt.colorbar(scatC, ax=axC, orientation='horizontal',
                           fraction=0.04, pad=0.05, shrink=0.8,
                           ticks=regime_edges, spacing='proportional')
        cbC.set_label('Predicted P_kbar (opx-only, RF / pwlr) binned by '
                      'pre-registered regime', fontsize=9)
        cbC.ax.set_xticklabels([f'{int(e)}' for e in regime_edges])
        # Regime labels as secondary annotation under colorbar
        for i, lab in enumerate(P_REGIME_LABELS):
            midpoint = (regime_edges[i] + regime_edges[i + 1]) / 2
            cbC.ax.text(midpoint, -1.8, lab.replace('_', '\n'),
                        ha='center', va='top', fontsize=6,
                        transform=cbC.ax.transData)
        axC.set_title(
            'Panel C (H.5c)  Predicted P stratified by pre-registered '
            'regime bins\n(0/5/15/30/100 kbar). Regime is descriptive, '
            'not a per-bin RMSE claim.',
            fontsize=9)

        fig.suptitle(
            f'H.5a+c  Worldwide opx-only predictions  '
            f'(n={len(df):,}, GEOROC 2024-12)',
            fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote {OUT_PNG}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
