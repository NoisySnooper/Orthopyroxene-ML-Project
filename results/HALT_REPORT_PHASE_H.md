# HALT REPORT — Phase H

Halt condition: #1 (GEOROC API returned < 30,000 raw cpx rows in H.1b)

## Snapshot

- UTC: 2026-04-26T23:53:57Z
- Branch: phase_h_natural_worldwide
- Git SHA: 1f39749e1a15f394c29d4fd96fdb02bca892af6d
- Last completed step: H.0b (canonical-cell roster lock, doc written)
- Next planned step: H.1b GEOROC cpx pull (now halted)

## Reason

No GEOROC endpoint returned data. All candidates 404, connection-refused, or DNS-resolution failure. See CANDIDATES list in scripts/v10_pull_georoc_cpx.py.

Raw row count obtained from GEOROC: 0
Halt threshold: 30,000

## Endpoint probe summary

Probed candidates in scripts/v10_pull_georoc_cpx.py CANDIDATES:
  - https://georoc.mpch-mainz.gwdg.de/Csv_Downloads/Minerals_comp/2024-12_CLINOPYROXENES.zip
  - https://georoc.mpch-mainz.gwdg.de/Csv_Downloads/Minerals_comp/2024-12_CLINOPYROXENES.csv
  - https://api.georoc.eu/v1/datasets/clinopyroxenes/2024-12
  - https://api.georoc.eu/v2/datasets/clinopyroxenes/2024-12
  - https://api.georoc.eu/queries/clinopyroxenes?release=2024-12
  - https://georoc.eu/static/cpx/2024-12_CLINOPYROXENES.csv

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
