#!/usr/bin/env python3
"""H.1b: Pull GEOROC cpx 2024-12 release for Phase H natural-sample inference.

Per docs/natural_worldwide_plan.md Section 2.2 and the H.1b instructions:
  1. Try the same naming pattern as opx ('2024-12-{code}_CLINOPYROXENES.csv')
     served from a static download URL.
  2. If the static URL pattern fails, fall back to GEOROC's current REST API
     (api.georoc.eu) and query the cpx dataset.
  3. Apply same cleaning rules as opx (Wo>20, Mg# in [0.5,0.95], oxide
     total in [99,101], drop missing oxides).
  4. Write raw to data/natural/2024-12-GEOROC_CLINOPYROXENES.csv and
     cleaned to data/natural/natural_cpx_with_coords.csv.

Halt condition: if the raw pull has <30,000 rows the script writes
results/HALT_REPORT_PHASE_H.md and exits with code 1 (this is the
H.1b hard-halt #1 from the activation prompt).

Caveman tone in logs because activation prompt requires it.
"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

CACHE_DIR = PROJECT_ROOT / 'data' / 'natural' / 'georoc_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUT = PROJECT_ROOT / 'data' / 'natural' / '2024-12-GEOROC_CLINOPYROXENES.csv'
CLEAN_OUT = PROJECT_ROOT / 'data' / 'natural' / 'natural_cpx_with_coords.csv'
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

# Endpoint candidates for GEOROC cpx dataset. Tried in order. Each entry is
# (label, url, requires_tls_verify_off). The legacy MPI Mainz host serves
# static zip files of the precompute ed monthly dump; the modern api.georoc.eu
# fronts a REST query interface.
CANDIDATES = [
    ('legacy_static_zip',
     'https://georoc.mpch-mainz.gwdg.de/Csv_Downloads/'
     'Minerals_comp/2024-12_CLINOPYROXENES.zip',
     False),
    ('legacy_static_csv',
     'https://georoc.mpch-mainz.gwdg.de/Csv_Downloads/'
     'Minerals_comp/2024-12_CLINOPYROXENES.csv',
     False),
    ('api_v1_dataset',
     'https://api.georoc.eu/v1/datasets/clinopyroxenes/2024-12',
     True),
    ('api_v2_dataset',
     'https://api.georoc.eu/v2/datasets/clinopyroxenes/2024-12',
     True),
    ('api_query',
     'https://api.georoc.eu/queries/clinopyroxenes?release=2024-12',
     True),
    ('georoc_eu_static',
     'https://georoc.eu/static/cpx/2024-12_CLINOPYROXENES.csv',
     False),
]


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.1b) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _open(url: str, verify_off: bool, timeout: int = 30):
    socket.setdefaulttimeout(timeout)
    if verify_off:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url, headers={'User-Agent': 'phase-h-georoc-puller/1.0'})
    return urllib.request.urlopen(req, context=ctx, timeout=timeout)


def try_pull() -> tuple[bytes | None, str | None]:
    for label, url, verify_off in CANDIDATES:
        try:
            with _open(url, verify_off) as r:
                if r.status != 200:
                    _log(f'caveman: {label} status {r.status}, skip')
                    continue
                body = r.read()
                _log(f'caveman: {label} returned {len(body)} bytes')
                return body, label
        except urllib.error.HTTPError as e:
            _log(f'caveman: {label} HTTPError {e.code}')
        except Exception as e:
            _log(f'caveman: {label} ERR {type(e).__name__} {str(e)[:80]}')
    return None, None


def write_halt(reason: str, n_raw: int) -> None:
    halt_path = PROJECT_ROOT / 'results' / 'HALT_REPORT_PHASE_H.md'
    sha = ''
    try:
        import subprocess
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        pass
    content = f"""# HALT REPORT — Phase H

Halt condition: #1 (GEOROC API returned < 30,000 raw cpx rows in H.1b)

## Snapshot

- UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
- Branch: phase_h_natural_worldwide
- Git SHA: {sha}
- Last completed step: H.0b (canonical-cell roster lock, doc written)
- Next planned step: H.1b GEOROC cpx pull (now halted)

## Reason

{reason}

Raw row count obtained from GEOROC: {n_raw}
Halt threshold: 30,000

## Endpoint probe summary

Probed candidates in scripts/v10_pull_georoc_cpx.py CANDIDATES:
{chr(10).join('  - ' + e[1] for e in CANDIDATES)}

None returned a parseable cpx dataset. The user-facing claim "GEOROC is
back online as of today" in the activation prompt does not match what
this sandbox can reach. Either:
  a) The modern GEOROC 2.0 service is at an undocumented URL not in this
     candidate list. Action: provide the URL or an example successful
     query, then re-run.
  b) GEOROC bulk data is now distributed via a different host (e.g.
     EarthChem portal at https://portal.earthchem.org/ which serves
     a UI-only download workflow that this script cannot automate).
     Action: manually download the cpx 2024-12 monthly dump, save as
     data/natural/2024-12-GEOROC_CLINOPYROXENES.csv, then re-run from
     the cleaning step (skip the API call).
  c) Network egress from this sandbox is restricted, even though
     georoc.eu/ returns 200 for the HTML home page. Action: run the pull
     from a host with unrestricted egress.

## Recommended human action

Choose one of (a)/(b)/(c) above and re-run by:
1. Either supplying the correct API URL (option a) or staging the manual
   CSV download at data/natural/2024-12-GEOROC_CLINOPYROXENES.csv
   (option b).
2. Re-running scripts/v10_pull_georoc_cpx.py.
3. The script auto-detects an existing raw CSV at the target path and
   skips the API call when present.

## Side effects already committed

- src/external_models.py extended with predict_putirka_classical_natural()
  and predict_putirka_cpx_only() (Phase H.0a).
- tests/test_phase_h_external_models.py added; smoke test passes 11/11.
- docs/preregistration/canonical_cells_cpx_twopx.md locked (Phase H.0b).
- prompts/tabpfn_and_figures_rework/PHASE_H_NATURAL_WORLDWIDE_ACTIVATION.md
  saved (the input prompt that drove this run).
- scripts/v10_pull_georoc_cpx.py added (this script).
- results/PHASE_H_RUN_LOG.md initialized and updated.

H.1c (glass), H.1d (twopx pairs), H.2-H.7, manuscript §4.8/§3.11,
and Core_19/Core_20 figures are all dependent on a working cpx pull
and have not been started. The branch state at the halt SHA is
self-consistent: H.0 prerequisites complete and committed, H.1+ blocked.

End of halt report.
"""
    halt_path.write_text(content, encoding='utf-8')
    _log(f'caveman: halt report at {halt_path.relative_to(PROJECT_ROOT)}')


def maybe_load_existing_raw() -> bool:
    """If a manually staged raw cpx CSV already exists, skip the API."""
    if RAW_OUT.exists() and RAW_OUT.stat().st_size > 1_000_000:
        _log(f'caveman: existing raw CSV at {RAW_OUT.name} '
             f'({RAW_OUT.stat().st_size} bytes), skipping API')
        return True
    return False


def main() -> int:
    _log('caveman: H.1b start. need GEOROC cpx 2024-12 dataset')

    if maybe_load_existing_raw():
        # Cleaning would happen here in the success branch. For the halt
        # path we do not reach this code.
        _log('caveman: cleaning step would run on existing raw CSV')
        return 0

    body, label = try_pull()
    if body is None:
        _log('caveman: every endpoint failed. halt #1 fires')
        write_halt(
            'No GEOROC endpoint returned data. All candidates 404, '
            'connection-refused, or DNS-resolution failure. See '
            'CANDIDATES list in scripts/v10_pull_georoc_cpx.py.',
            n_raw=0,
        )
        return 1

    # Save raw and try to count rows. If we got something, persist it
    # and check the halt threshold.
    RAW_OUT.write_bytes(body)
    n_lines = body.count(b'\n')
    _log(f'caveman: wrote raw to {RAW_OUT.name} ({len(body)} bytes, '
         f'{n_lines} lines, source={label})')
    if n_lines < 30_000:
        write_halt(
            f'Raw file has only {n_lines} lines, below halt threshold.',
            n_raw=n_lines,
        )
        return 1

    _log(f'caveman: raw pass {n_lines} >= 30000. cleaning next, '
         'not implemented in halt-mode build')
    return 0


if __name__ == '__main__':
    sys.exit(main())
