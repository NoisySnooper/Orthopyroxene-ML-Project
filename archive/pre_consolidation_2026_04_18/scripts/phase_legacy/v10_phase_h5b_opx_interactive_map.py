#!/usr/bin/env python3
"""Phase H.5b (partial): interactive folium map for opx_only natural
predictions.

Complements the static H.5a+c cartopy figure. Renders as a single HTML
file with CartoDB Positron tiles, clustered markers (via folium's
FastMarkerCluster for 53k point scale), and per-sample tooltip showing
GEOROC ID, tectonic setting, predicted T and P, RF tree std, OOD flag.

Scope: opx only. Cpx + twopx panels deferred until H.1b/c/d.

Inputs:
  results/v10_natural_opx_opx_only_predictions.csv  (H.3a + H.3d augment)

Outputs:
  figures/fig_h5b_opx_interactive.html
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster

from config import RESULTS, LOGS, FIGURES, P_REGIME_BIN_EDGES_KBAR

PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
OUT_HTML = FIGURES / 'fig_h5b_opx_interactive.html'
LOG_PATH = LOGS / 'v10_phase_h5b_opx_interactive.log'

REGIME_COLORS = {
    0: '#fde725',  # shallow_crustal
    1: '#5ec962',  # deep_crustal_MASH
    2: '#21918c',  # lithospheric_mantle
    3: '#3b528b',  # deeper_mantle
}
REGIME_NAMES = {
    0: 'shallow_crustal',
    1: 'deep_crustal_MASH',
    2: 'lithospheric_mantle',
    3: 'deeper_mantle',
}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def assign_regime(p_kbar):
    edges = list(P_REGIME_BIN_EDGES_KBAR)
    idx = np.digitize(p_kbar, edges, right=False) - 1
    return np.clip(idx, 0, len(REGIME_COLORS) - 1)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.5b opx interactive map', fh)
        df = pd.read_csv(PRED_CSV, low_memory=False)
        _log(f'loaded {len(df)} rows', fh)
        df = df.dropna(subset=['lat', 'lon'])
        _log(f'after lat/lon dropna: {len(df)}', fh)

        df['regime_idx'] = assign_regime(df['pred_P_kbar_opx_only'])
        df['regime_name'] = df['regime_idx'].map(REGIME_NAMES)

        fmap = folium.Map(
            location=[20.0, 0.0], zoom_start=2, tiles='CartoDB positron',
            control_scale=True,
        )

        # One cluster per regime so users can toggle visibility.
        # FeatureGroups wrap clusters so LayerControl can toggle them.
        groups = {}
        for idx, name in REGIME_NAMES.items():
            fg = folium.FeatureGroup(name=f'{name} (P bin {idx})',
                                     show=True)
            fg.add_to(fmap)
            groups[idx] = MarkerCluster(name=name,
                                        disableClusteringAtZoom=7)
            groups[idx].add_to(fg)

        ood_group = folium.FeatureGroup(name='OOD flagged (all regimes)',
                                        show=False)
        ood_group.add_to(fmap)
        ood_cluster = MarkerCluster(name='OOD')
        ood_cluster.add_to(ood_group)

        has_sigma = 'pred_P_kbar_opx_only_tree_std' in df.columns
        sample_cap = 20000  # folium chokes beyond this
        if len(df) > sample_cap:
            df = df.sample(sample_cap, random_state=42)
            _log(f'subsampled to {sample_cap} markers for browser perf', fh)

        n_added = 0
        for _, r in df.iterrows():
            tooltip_parts = [
                f"<b>{r.get('UNIQUE_ID', '?')}</b>",
                f"Tectonic: {r.get('TECTONIC SETTING', '?')}",
                f"T = {r['pred_T_C_opx_only']:.0f} C",
                f"P = {r['pred_P_kbar_opx_only']:.1f} kbar ({r['regime_name']})",
            ]
            if has_sigma:
                tooltip_parts.append(
                    f"sigma(P) = {r['pred_P_kbar_opx_only_tree_std']:.1f} kbar")
            tooltip_parts.append(
                f"OOD: {'YES' if r['ood_flag'] else 'no'}")
            tooltip = '<br>'.join(tooltip_parts)
            color = REGIME_COLORS[int(r['regime_idx'])]
            marker = folium.CircleMarker(
                location=[r['lat'], r['lon']],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.55 if not r['ood_flag'] else 0.15,
                weight=0.4,
                tooltip=tooltip,
            )
            if r['ood_flag']:
                marker.add_to(ood_cluster)
            else:
                marker.add_to(groups[int(r['regime_idx'])])
            n_added += 1
        _log(f'added {n_added} markers', fh)

        folium.LayerControl(collapsed=False).add_to(fmap)

        # Simple custom legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background: white; padding: 10px; border: 1px solid #666;
                    font-family: sans-serif; font-size: 12px;">
          <b>H.5b &mdash; opx-only predicted P regime</b><br>
          <i style="background:#fde725;width:12px;height:12px;display:inline-block;"></i>
          shallow_crustal (0-5 kbar)<br>
          <i style="background:#5ec962;width:12px;height:12px;display:inline-block;"></i>
          deep_crustal_MASH (5-15 kbar)<br>
          <i style="background:#21918c;width:12px;height:12px;display:inline-block;"></i>
          lithospheric_mantle (15-30 kbar)<br>
          <i style="background:#3b528b;width:12px;height:12px;display:inline-block;"></i>
          deeper_mantle (30-100 kbar)<br>
          <span style="opacity:0.4;">lighter = OOD-flagged</span>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

        fmap.save(str(OUT_HTML))
        _log(f'wrote {OUT_HTML}', fh)
        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
