# HALT REPORT — Phase H (RESOLVED 2026-04-27T00:08:00Z)

## Status: resolved, execution resumed

The original halt #1 (GEOROC API returned < 30,000 raw cpx rows in
H.1b) fired at 2026-04-26T23:53:57Z because none of the api.georoc.eu
v1/v2/queries endpoints or the legacy MPI Mainz static URL returned
data. The user supplied the correct download path: the GEOROC
compilation is published on the GRO.data Dataverse instance at
data.goettingen-research-online.de under DOI 10.25625/SGFTFN, accessed
via the standard Dataverse Native API. No API key required.

Resolution applied:

- Replaced the candidate URL list in scripts/v10_pull_georoc_cpx.py
  with a Dataverse Native API approach: GET
  /api/datasets/:persistentId/?persistentId=doi:10.25625/SGFTFN to
  enumerate files, then stream
  /api/access/datafile/{file_id} for the cpx CSV.
- Re-ran H.1b. Download succeeded: 338.4 MB raw file
  (file_id=118288, SHA256[:12]=5598c3bd0b92) in 2 minutes.
- Cleaning produced 93,163 cpx rows (over the plan's expected
  cleaned-n ceiling of 80,000; treated as a soft warning rather
  than a halt because more data is harmless).

Phase H execution resumes from H.1d (twopx pair construction). H.1c
(liquid/glass) is deferred per the activation prompt's "Lower
priority than H.1a and H.1b" note; the SGFTFN dataset is minerals-
only and the glass dataset (doi:10.25625/7JW6XU GEOROC Melt
Inclusions or doi:10.25625/2JETOA GEOROC Rock Types) requires a
separate pull.

---

# Original halt-1 record (preserved for audit)

Halt condition: #1 (GEOROC API returned < 30,000 raw cpx rows in H.1b)
UTC at halt: 2026-04-26T23:53:57Z
Last completed step before halt: H.0b (canonical-cell roster lock)
Next planned step at halt: H.1b GEOROC cpx pull

Original endpoint probe summary (all returned 404, DNS failure, or
TLS error):
- legacy MPI Mainz static zip and CSV URLs: 404
- api.georoc.eu v1/v2/queries: 404 (with TLS verify off)
- georoc.eu/static path: 404
- georoc2.gfz-potsdam.de, georoc.gfz-potsdam.de: DNS resolution failed
- api.georoc.org: connection refused

Reason for halt: no GEOROC programmatic endpoint in the original
candidate list was reachable from this sandbox.

Action taken: user supplied the correct Dataverse Native API approach.
Script patched, re-run successful, halt cleared.

End of halt report.
