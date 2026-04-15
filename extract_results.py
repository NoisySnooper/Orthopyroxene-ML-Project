"""Extract v7 results into a single markdown summary for Claude review.

Reads every canonical output from results/, models/, logs/, and figures/, and
produces reports/v7_review_summary.md. This file is what the user uploads to
Claude for analysis. Includes no figure bytes; figures are referenced by path
and size.

Usage:
    .venv\\Scripts\\python.exe extract_results.py [--out PATH]

Default output: reports/v7_review_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"


def read_csv_as_rows(path: Path, max_rows: int = 500) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(r)
            return rows
    except Exception as e:
        return [{"_READ_ERROR": str(e)}]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_READ_ERROR": str(e)}


def markdown_table(rows: list[dict], keys: list[str] | None = None, max_cols: int = 10) -> str:
    if not rows:
        return "*(empty)*"
    if keys is None:
        keys = list(rows[0].keys())
    keys = keys[:max_cols]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body = []
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def file_size_kb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / 1024.0


def latest_log(pattern: str) -> Path | None:
    matches = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def tail_text(path: Path, n_lines: int = 40) -> str:
    if not path.exists():
        return "*(not found)*"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as e:
        return f"*(read error: {e})*"


def section_header() -> str:
    return f"""# v7 Run Review Summary

Generated: {datetime.now().isoformat()}
Repo root: `{ROOT}`

This file is produced by `extract_results.py`. It is intended to be uploaded
to Claude for review. No figure bytes are included; figures are referenced
by filename, size, and existence check.
"""


def section_run_metadata() -> str:
    parts = ["## 1. Run metadata\n"]
    latest_run_log = latest_log("pipeline_run_v7_*.log")
    latest_resume_log = latest_log("pipeline_resume_*.log")
    if latest_run_log:
        parts.append(f"Latest v7 run log: `{latest_run_log.name}` "
                     f"({file_size_kb(latest_run_log):.1f} KB)\n")
        parts.append("```\n" + tail_text(latest_run_log, 30) + "```\n")
    else:
        parts.append("*(no pipeline_run_v7_*.log found)*\n")
    if (latest_resume_log and
            (not latest_run_log or latest_resume_log.stat().st_mtime > latest_run_log.stat().st_mtime)):
        parts.append(f"\nLatest resume log: `{latest_resume_log.name}` (more recent than run log)\n")
        parts.append("```\n" + tail_text(latest_resume_log, 20) + "```\n")

    failure_logs = sorted(LOGS.glob("FAILURE_v7_*.log"))
    halt_logs = sorted(LOGS.glob("HALT_v7_*.log"))
    missing_logs = sorted(LOGS.glob("MISSING_*.log"))

    if failure_logs:
        parts.append(f"\n**Failures detected ({len(failure_logs)}):**\n")
        for fl in failure_logs:
            parts.append(f"- `{fl.name}`:\n```\n{tail_text(fl, 15)}```\n")
    if halt_logs:
        parts.append(f"\n**Operator decisions hit ({len(halt_logs)}):**\n")
        for hl in halt_logs:
            parts.append(f"- `{hl.name}`:\n```\n{tail_text(hl, 15)}```\n")
    if missing_logs:
        parts.append(f"\n**Missing files flagged ({len(missing_logs)}):**\n")
        for ml in missing_logs:
            parts.append(f"- `{ml.name}`:\n```\n{tail_text(ml, 10)}```\n")

    if latest_run_log:
        txt = latest_run_log.read_text(encoding="utf-8", errors="replace")
        expected = ["NB01", "NB02", "NB03", "NB04", "NB04b", "NB05",
                    "NB06", "NB07", "NB08", "NB09", "NBF"]
        seen = {nb: bool(re.search(rf"=== {re.escape(nb)} complete ===", txt)) for nb in expected}
        parts.append("\n**Notebook completion markers detected:**\n")
        for nb, ok in seen.items():
            parts.append(f"- {nb}: {'ok' if ok else 'missing'}")

    return "\n".join(parts)


def section_per_family_winners() -> str:
    parts = ["## 2. Per-family canonical model winners (nb03 output)\n"]
    winners = read_json(RESULTS / "nb03_per_family_winners.json")
    if winners is None:
        parts.append("**`nb03_per_family_winners.json` not found.** v7 Part B did not complete.\n")
        parts.append("Falling back to legacy `nb03_winning_configurations.json`:\n")
        legacy = read_json(RESULTS / "nb03_winning_configurations.json")
        if legacy:
            parts.append("```json\n" + json.dumps(legacy, indent=2)[:2000] + "\n```\n")
        return "\n".join(parts)

    parts.append(f"Schema version: {winners.get('schema_version', 'unknown')}\n")
    protocol = winners.get("selection_protocol", {})
    seeds = protocol.get("seeds", [0])
    parts.append(f"Seeds used: {protocol.get('n_seeds', '?')} "
                 f"(range {min(seeds)}-{max(seeds)})\n")
    parts.append(f"Tiebreaker rule: {protocol.get('tiebreaker_rule', '?')}\n\n")

    for family_key in ("forest_family", "boosted_family"):
        fam = winners.get(family_key, {})
        if not fam:
            parts.append(f"### {family_key}: empty\n")
            continue
        parts.append(f"### {family_key.replace('_', ' ').title()}\n")
        rows = []
        for task, spec in fam.items():
            rows.append({
                "task": task,
                "model": spec.get("model_name"),
                "features": spec.get("feature_set"),
                "rmse_mean": spec.get("rmse_mean"),
                "rmse_std": spec.get("rmse_std"),
                "runner_up": f"{spec.get('runner_up_model')}+{spec.get('runner_up_feature_set')}",
                "runner_up_rmse": spec.get("runner_up_rmse_mean"),
                "tiebreaker": "yes" if spec.get("tiebreaker_applied") else "no",
            })
        parts.append(markdown_table(rows))
        parts.append("")

    return "\n".join(parts)


def section_multi_seed_summary() -> str:
    parts = ["## 3. Multi-seed training summary (nb03)\n"]
    path = RESULTS / "nb03_multi_seed_summary.csv"
    if not path.exists():
        path = RESULTS / "nb03_multi_seed_results.csv"
        if not path.exists():
            parts.append("*(nb03_multi_seed_summary.csv / _results.csv not found)*\n")
            return "\n".join(parts)
    rows = read_csv_as_rows(path, max_rows=200)
    parts.append(f"Total rows: {len(rows)} "
                 f"(expected 48 = 4 models x 3 feature sets x 2 tracks x 2 targets)\n\n")
    if rows:
        keys = ["track", "target", "model_name", "feature_set",
                "rmse_test_mean", "rmse_test_std", "r2_test_mean"]
        parts.append(markdown_table(rows, keys=keys, max_cols=7))
    return "\n".join(parts)


def section_benchmarks() -> str:
    parts = ["## 4. Benchmarks\n"]
    parts.append("### 4.1 NB04 unified benchmark (ExPetDB internal test)\n")
    path = RESULTS / "nb04_unified_benchmark.csv"
    if path.exists():
        rows = read_csv_as_rows(path, max_rows=50)
        parts.append(markdown_table(rows, max_cols=8))
    else:
        path_alt = RESULTS / "nb04_putirka_vs_ml.csv"
        if path_alt.exists():
            parts.append(f"*(unified table missing; legacy `{path_alt.name}` present with "
                         f"{len(read_csv_as_rows(path_alt, 10000))} rows)*\n")
        else:
            parts.append("*(not found)*\n")

    parts.append("\n### 4.2 Three-way ML benchmark on ArcPL\n")
    path = RESULTS / "nb04_three_way_ml_benchmark.csv"
    if path.exists():
        rows = read_csv_as_rows(path)
        parts.append(markdown_table(rows))
    else:
        parts.append("*(not found)*\n")

    parts.append("\n### 4.3 Putirka failure mode stratification\n")
    path = RESULTS / "nb04_failure_mode_stratification.csv"
    if path.exists():
        rows = read_csv_as_rows(path)
        parts.append(markdown_table(rows))
    else:
        parts.append("*(not found)*\n")

    parts.append("\n### 4.4 ArcPL opx-liq validation (NB04 Part 3, ex-NB04b)\n")
    # v7: NB04b was consolidated into NB04 Part 3. Prefer the new
    # nb04_arcpl_opx_liq_* filenames; fall back to legacy nb04b_arcpl_* aliases
    # that Part 3 still writes for backward compatibility.
    _arcpl_files = [
        "nb04_arcpl_opx_liq_metrics.csv",
        "nb04_arcpl_opx_liq_bootstrap.csv",
        "nb04_arcpl_opx_liq_predictions_forest.csv",
        "nb04_arcpl_opx_liq_predictions_boosted.csv",
        "nb04b_arcpl_unified_benchmark.csv",
        "nb04b_arcpl_predictions_forest.csv",
        "nb04b_arcpl_predictions_boosted.csv",
        "nb04b_arcpl_predictions.csv",
    ]
    _found_any = False
    for fname in _arcpl_files:
        p = RESULTS / fname
        if p.exists():
            _found_any = True
            rows = read_csv_as_rows(p, max_rows=10)
            total = len(read_csv_as_rows(p, 100000))
            parts.append(f"**{fname}** ({total} rows total, first 10):\n")
            parts.append(markdown_table(rows, max_cols=10))
            parts.append("")
    if not _found_any:
        parts.append("*(no ArcPL files found — run NB04 Part 3)*\n")

    return "\n".join(parts)


def section_generalization() -> str:
    parts = ["## 5. Generalization (NB05)\n"]
    path = RESULTS / "nb05_loso_pooled.csv"
    if path.exists():
        rows = read_csv_as_rows(path)
        parts.append(markdown_table(rows))
    else:
        parts.append("*(nb05_loso_pooled.csv not found)*\n")
    return "\n".join(parts)


def section_uncertainty() -> str:
    parts = ["## 6. Uncertainty calibration (NB07)\n"]
    for fname in ["nb07_conformal_coverage_forest.csv",
                  "nb07_conformal_coverage_boosted.csv",
                  "nb07_conformal_coverage_table.csv",
                  "nb07_bias_correction_null_result.csv"]:
        p = RESULTS / fname
        if p.exists():
            parts.append(f"### {fname}\n")
            parts.append(markdown_table(read_csv_as_rows(p)))
            parts.append("")
    return "\n".join(parts)


def section_cross_mineral() -> str:
    parts = ["## 7. Cross-mineral validation (NB08)\n"]
    for fname in ["nb08_disagreement_metrics.csv",
                  "nb08_natural_predictions.csv",
                  "nb08_natural_predictions_filtered.csv",
                  "nb08_natural_predictions_all.csv"]:
        p = RESULTS / fname
        if p.exists():
            all_rows = read_csv_as_rows(p, max_rows=100000)
            parts.append(f"### {fname} ({len(all_rows)} rows)\n")
            if "disagreement" in fname or len(all_rows) <= 20:
                parts.append(markdown_table(all_rows[:20]))
            else:
                parts.append(markdown_table(all_rows[:5]))
                parts.append(f"*(showing first 5 of {len(all_rows)})*")
            parts.append("")
    return "\n".join(parts)


def section_model_ceiling() -> str:
    parts = ["## 8. Model family ceiling (NB03 Section 3.6 or legacy NB11)\n"]
    for fname in ["nb03_model_family_ceiling.csv",
                  "nb11_model_family_ceiling.csv"]:
        p = RESULTS / fname
        if p.exists():
            rows = read_csv_as_rows(p)
            parts.append(f"### {fname}\n")
            parts.append(markdown_table(rows))
            parts.append("")
            break
    else:
        parts.append("*(not found)*\n")
    return "\n".join(parts)


def section_final_metrics() -> str:
    parts = ["## 9. Consolidated final metrics (NB09)\n"]
    data = read_json(RESULTS / "nb09_final_metrics.json")
    if data is None:
        parts.append("*(nb09_final_metrics.json not found)*\n")
        return "\n".join(parts)
    parts.append("```json\n" + json.dumps(data, indent=2)[:5000] + "\n```\n")
    return "\n".join(parts)


def section_figures() -> str:
    parts = ["## 10. Figure inventory\n"]
    if not FIGURES.exists():
        parts.append("*(figures/ directory not found)*\n")
        return "\n".join(parts)
    pngs = sorted(FIGURES.glob("*.png"))
    pdfs = sorted(FIGURES.glob("*.pdf"))
    parts.append(f"PNG files: {len(pngs)}\n")
    parts.append(f"PDF files: {len(pdfs)}\n\n")

    parts.append("### Figure file sizes (KB)\n")
    rows = []
    stems = sorted({p.stem for p in pngs} | {p.stem for p in pdfs})
    for stem in stems:
        png = FIGURES / f"{stem}.png"
        pdf = FIGURES / f"{stem}.pdf"
        rows.append({
            "figure": stem,
            "png_kb": f"{file_size_kb(png):.1f}" if png.exists() else "MISSING",
            "pdf_kb": f"{file_size_kb(pdf):.1f}" if pdf.exists() else "MISSING",
            "broken_render_risk": "yes" if (png.exists() and file_size_kb(png) < 10) else "no",
        })
    parts.append(markdown_table(rows, max_cols=4))
    return "\n".join(parts)


def section_anomaly_flags() -> str:
    parts = ["## 11. Anomaly flags\n"]
    flags = []

    if FIGURES.exists():
        for p in FIGURES.glob("*.png"):
            if file_size_kb(p) < 10:
                flags.append(f"Small PNG (likely broken render): `{p.name}` at {file_size_kb(p):.1f} KB")

    if RESULTS.exists():
        for p in RESULTS.glob("*.csv"):
            rows = read_csv_as_rows(p, max_rows=5)
            if len(rows) == 0:
                flags.append(f"Empty CSV: `{p.name}`")
            elif all(all(v in ("", "nan", "NaN", "None") for v in r.values()) for r in rows):
                flags.append(f"All-NaN CSV: `{p.name}`")

    if not (RESULTS / "nb03_per_family_winners.json").exists():
        flags.append("MISSING: `nb03_per_family_winners.json` (v7 Part B incomplete)")

    winners = read_json(RESULTS / "nb03_per_family_winners.json")
    if winners:
        expected_keys = ["opx_only_T_C", "opx_only_P_kbar",
                         "opx_liq_T_C", "opx_liq_P_kbar"]
        for family in ("forest_family", "boosted_family"):
            fam = winners.get(family, {})
            missing_tasks = [k for k in expected_keys if k not in fam]
            if missing_tasks:
                flags.append(f"{family} missing tasks: {missing_tasks}")
            for task, spec in fam.items():
                filename = spec.get("filename", "")
                if (filename
                        and not (RESULTS / filename).exists()
                        and not (MODELS / "canonical" / filename).exists()
                        and not (MODELS / filename).exists()):
                    flags.append(f"Winner filename not on disk: {family}/{task} -> {filename}")

    for summary_fname in ("nb03_multi_seed_summary.csv", "nb03_multi_seed_results.csv"):
        path = RESULTS / summary_fname
        if path.exists():
            rows = read_csv_as_rows(path, max_rows=200)
            for r in rows:
                target = r.get("target", "")
                try:
                    rmse = float(r.get("rmse_test_mean", "nan"))
                    if target == "T_C" and rmse > 250:
                        flags.append(f"Suspicious T RMSE in {r.get('track')}/{r.get('model_name')}/"
                                     f"{r.get('feature_set')}: {rmse:.1f} C")
                    if target == "P_kbar" and rmse > 20:
                        flags.append(f"Suspicious P RMSE in {r.get('track')}/{r.get('model_name')}/"
                                     f"{r.get('feature_set')}: {rmse:.2f} kbar")
                except (ValueError, TypeError):
                    pass
            break

    if not flags:
        parts.append("*(no anomalies detected)*\n")
    else:
        for f in flags:
            parts.append(f"- {f}")
    return "\n".join(parts)


def section_operator_decisions() -> str:
    parts = ["## 12. Operator decisions log\n"]
    path = LOGS / "operator_decisions.txt"
    if path.exists():
        parts.append("```\n" + tail_text(path, 50) + "```\n")
    else:
        parts.append("*(logs/operator_decisions.txt not found)*\n")
    return "\n".join(parts)


def section_file_audit() -> str:
    parts = ["## 13. File audit (what exists where)\n"]
    for d in [RESULTS, MODELS, FIGURES, LOGS]:
        if not d.exists():
            parts.append(f"- `{d.relative_to(ROOT)}/`: does not exist")
            continue
        count_total = sum(1 for _ in d.rglob("*") if _.is_file())
        parts.append(f"- `{d.relative_to(ROOT)}/`: {count_total} files")
    return "\n".join(parts)


def build_summary() -> str:
    sections = [
        section_header(),
        section_run_metadata(),
        section_per_family_winners(),
        section_multi_seed_summary(),
        section_benchmarks(),
        section_generalization(),
        section_uncertainty(),
        section_cross_mineral(),
        section_model_ceiling(),
        section_final_metrics(),
        section_figures(),
        section_anomaly_flags(),
        section_operator_decisions(),
        section_file_audit(),
    ]
    return "\n\n---\n\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=REPORTS / "v7_review_summary.md")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = build_summary()
    args.out.write_text(content, encoding="utf-8")
    sys.stdout.write(content)
    print(f"\n\n[extract_results] wrote {args.out}", file=sys.stderr)
    print(f"[extract_results] size: {file_size_kb(args.out):.1f} KB", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
