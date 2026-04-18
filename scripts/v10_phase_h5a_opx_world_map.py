#!/usr/bin/env python3
"""Phase H.5a (partial): static world map, opx-only natural predictions.

Scope: opx track only. Cpx and twopx layers deferred until H.1b/c/d
complete. This produces the opx panels of the eventual 3-mineral map
stack.

Two panels (side-by-side, Robinson projection, global extent):
  Panel A  tectonic setting (categorical colormap)
  Panel B  predicted T_C    (continuous colormap; OOD-flagged samples
                             rendered with reduced alpha)

Per Phase G collision 3, Panel B does NOT claim per-regime RMSE. The
pre-registered P-regime bins instead stratify the colorbar ticks
visually (panel subtitle notes that regime assignment is descriptive).

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
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import (RESULTS, LOGS, FIGURES,
                    P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS)

PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
OUT_PNG = FIGURES / 'fig_h5a_opx_world_map.png'
LOG_PATH = LOGS / 'v10_phase_h5a_opx_world_map.log'


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

        fig = plt.figure(figsize=(18, 9))
        proj = ccrs.Robinson()

        # ---- Panel A: tectonic setting ----
        axA = fig.add_subplot(1, 2, 1, projection=proj)
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
        axB = fig.add_subplot(1, 2, 2, projection=proj)
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
            'Panel B  Predicted T; dim = OOD-flagged vs training.\n'
            'Pre-registered P-regime bins (0/5/15/30/100 kbar) are '
            'descriptive only.',
            fontsize=10)

        fig.suptitle(
            f'H.5a  Worldwide opx-only predictions  '
            f'(n={len(df):,}, GEOROC 2024-12)',
            fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote {OUT_PNG}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
