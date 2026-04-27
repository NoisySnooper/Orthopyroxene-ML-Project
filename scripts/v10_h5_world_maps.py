#!/usr/bin/env python3
"""H.5: World maps for natural-sample inference (static + interactive).

Static maps (publication, Robinson projection, cartopy):
  Core_19a_fig_world_map_opx.{pdf,png}    2 panels (tectonic + predicted T)
  Core_19b_fig_world_map_cpx.{pdf,png}    2 panels
  Core_19c_fig_world_map_twopx.{pdf,png}  2 panels

Interactive maps (folium):
  figures/interactive/world_map_opx.html
  figures/interactive/world_map_cpx.html  (skipped if 93k markers blow up)
  figures/interactive/world_map_twopx.html

Visualization stratification only — predicted-T color does not imply
a quantitative claim per the natural-sample pre-registration boundary
(no ground truth P/T on natural samples).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
INTERACTIVE_DIR = PROJECT_ROOT / 'figures' / 'interactive'
OUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

OKABE_TECTONIC = {
    'CONVERGENT MARGIN':                          '#D55E00',
    'INTRAPLATE VOLCANICS':                       '#CC79A7',
    'RIFT VOLCANICS':                             '#E69F00',
    'CONTINENTAL FLOOD BASALT':                   '#F0E442',
    'ARCHEAN CRATON (INCLUDING GREENSTONE BELTS)': '#999933',
    'OCEAN ISLAND':                               '#56B4E9',
    'SEAMOUNT':                                   '#44AA99',
    'OCEANIC PLATEAU':                            '#0072B2',
    'COMPLEX VOLCANIC SETTINGS':                  '#888888',
    'SUBMARINE RIDGE':                            '#5B5B5B',
    'OTHER':                                      '#AAAAAA',
}

OCEAN_C = '#eef5fc'
LAND_C = '#f0efe8'
COAST_C = '#444444'


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.5) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _setup_robinson_axis(ax):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_C, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_C, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COAST_C, linewidth=0.4)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='gray',
                      alpha=0.4)
    gl.xlocator = plt.matplotlib.ticker.FixedLocator(
        np.arange(-180, 181, 30))
    gl.ylocator = plt.matplotlib.ticker.FixedLocator(
        np.arange(-90, 91, 30))


def _color_by_tectonic(setting):
    s = (setting or '').strip().upper()
    return OKABE_TECTONIC.get(s, OKABE_TECTONIC['OTHER'])


def make_static_map(df: pd.DataFrame, mineral: str, t_pred_col: str,
                    out_stem: Path, title: str) -> None:
    valid = df.dropna(subset=['lat', 'lon']).copy()
    n = len(valid)
    _log(f'caveman: {mineral} static map: {n} samples with lat/lon')

    fig = plt.figure(figsize=(20, 8))
    proj = ccrs.Robinson()

    ax_a = fig.add_subplot(1, 2, 1, projection=proj)
    _setup_robinson_axis(ax_a)
    colors = [_color_by_tectonic(s) for s in valid['tectonic_setting']]
    ax_a.scatter(valid['lon'], valid['lat'], c=colors, s=4, alpha=0.4,
                 edgecolor='none', transform=ccrs.PlateCarree(), zorder=2)
    ax_a.set_title(f'A. {mineral.upper()} natural samples by tectonic '
                   f'setting (n={n})',
                   loc='left', fontsize=12, fontweight='bold', pad=10)

    legend_handles = []
    for s, color in OKABE_TECTONIC.items():
        if s == 'OTHER':
            continue
        n_in = int((valid['tectonic_setting'].astype(str).str.upper() == s).sum())
        if n_in == 0:
            continue
        legend_handles.append(plt.Line2D(
            [0], [0], marker='o', linestyle='', color=color,
            markersize=6, label=f'{s.title()} (n={n_in})'))
    ax_a.legend(handles=legend_handles, loc='lower left',
                fontsize=7, framealpha=0.85, ncol=1)

    ax_b = fig.add_subplot(1, 2, 2, projection=proj)
    _setup_robinson_axis(ax_b)
    valid_t = valid.dropna(subset=[t_pred_col]).copy()
    sc = ax_b.scatter(valid_t['lon'], valid_t['lat'], c=valid_t[t_pred_col],
                      cmap='viridis', vmin=600, vmax=1500, s=4, alpha=0.6,
                      edgecolor='none', transform=ccrs.PlateCarree(),
                      zorder=2)
    cb = plt.colorbar(sc, ax=ax_b, orientation='vertical', shrink=0.7,
                      pad=0.04)
    cb.set_label('Predicted T (°C)', fontsize=10)
    ax_b.set_title(f'B. {mineral.upper()} predicted T (canonical model, '
                   f'n={len(valid_t)})',
                   loc='left', fontsize=12, fontweight='bold', pad=10)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.02,
             'Visualization only. Predicted-T colors do not constitute a '
             'per-regime numeric claim on natural samples '
             '(no ground truth P/T).',
             ha='center', fontsize=8, style='italic', color='#444444')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    _log(f'caveman: wrote {out_stem.name}.pdf and .png')


def make_interactive_map(df: pd.DataFrame, mineral: str, t_pred_col: str,
                          out_path: Path, sample_cap: int = 8000) -> None:
    import folium
    from folium.plugins import MarkerCluster

    valid = df.dropna(subset=['lat', 'lon']).copy()
    if len(valid) > sample_cap:
        valid = valid.sample(sample_cap, random_state=42)
        _log(f'caveman: {mineral} interactive map subsampled to '
             f'{sample_cap} markers')

    m = folium.Map(location=[10, 0], zoom_start=2,
                    tiles='OpenStreetMap', prefer_canvas=True)
    cluster = MarkerCluster(name='samples').add_to(m)

    for _, row in valid.iterrows():
        color = _color_by_tectonic(row['tectonic_setting'])
        t_val = row.get(t_pred_col)
        popup = (f"<b>{row.get('sample_name', '?')}</b><br>"
                 f"Citation: {str(row.get('citation', '?'))[:80]}<br>"
                 f"Tectonic: {row.get('tectonic_setting', '?')}<br>"
                 f"Predicted T: {t_val:.0f} °C<br>"
                 if pd.notna(t_val) else
                 f"<b>{row.get('sample_name', '?')}</b><br>"
                 f"Citation: {str(row.get('citation', '?'))[:80]}<br>")
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color=color, fill=True, fill_color=color, fill_opacity=0.6,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.save(str(out_path))
    _log(f'caveman: wrote {out_path.name}')


def main() -> int:
    _log('caveman: H.5 world maps start')

    opx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_opx.csv',
        low_memory=False)
    cpx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_cpx.csv',
        low_memory=False)
    twopx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_twopx.csv',
        low_memory=False)
    _log(f'caveman: loaded opx={len(opx)} cpx={len(cpx)} twopx={len(twopx)}')

    make_static_map(
        opx, 'opx', 'T_canonical_opx_only_T',
        OUT_DIR / 'Core_19a_fig_world_map_opx',
        'Core_19a. Worldwide orthopyroxene natural samples (GEOROC 2024-12)')
    make_static_map(
        cpx, 'cpx', 'T_canonical_cpx_only_T',
        OUT_DIR / 'Core_19b_fig_world_map_cpx',
        'Core_19b. Worldwide clinopyroxene natural samples (GEOROC 2024-12)')
    make_static_map(
        twopx, 'twopx', 'T_canonical_twopx_T',
        OUT_DIR / 'Core_19c_fig_world_map_twopx',
        'Core_19c. Two-pyroxene natural sample pairs (GEOROC 2024-12)')

    # Interactive maps
    make_interactive_map(opx, 'opx', 'T_canonical_opx_only_T',
                          INTERACTIVE_DIR / 'world_map_opx.html')
    make_interactive_map(cpx, 'cpx', 'T_canonical_cpx_only_T',
                          INTERACTIVE_DIR / 'world_map_cpx.html')
    make_interactive_map(twopx, 'twopx', 'T_canonical_twopx_T',
                          INTERACTIVE_DIR / 'world_map_twopx.html',
                          sample_cap=12000)

    _log('caveman: H.5 world maps done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
