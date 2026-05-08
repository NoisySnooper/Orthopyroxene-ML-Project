"""Execute the full opx ML thermobarometer pipeline end-to-end via papermill.

Exit codes:
    0 - full pipeline complete
    1 - a notebook failed with non-zero return code
    2 - a notebook hit an OPERATOR DECISION REQUIRED block (halt for review)
    3 - a required notebook file is missing from notebooks/

Usage:
    .venv\\Scripts\\python.exe run_all.py [--resume-from NOTEBOOK_NAME]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = NB_DIR / "executed"
LOG_DIR = ROOT / "logs"

NOTEBOOKS = [
    "nb01_data_cleaning",
    "nb02_eda_pca",
    "nb03_opx_baseline_models",
    # nb03_tabpfn_baseline runs the TabPFN v2 foundation-model baseline
    # (9th family post-Option-B). The notebook kernel is the main .venv;
    # the notebook itself shells out to .venv-tabpfn via subprocess, so
    # papermill here does not need tabpfn installed. Must run before
    # nb04_putirka_benchmark (which consumes tabpfn_multiseed_summary for
    # the head-to-head merge) and before nb07_bias_correction (TabPFN is
    # excluded from bias correction so nb07 processes only the 8 tuned
    # families).
    "nb03_tabpfn_baseline",
    "nb04_putirka_benchmark",
    "nb04_regime_benchmark",
    # nb04b_aug_test is an optional augmentation ablation on 4 opx
    # combinations (Section 5.3 sensitivity probe; added 2026-04-19).
    # Not on the critical path to the pre-registered verdict. Heavy
    # compute (~3.3h for 1920 fits + ~30 min for OOF/bias) is delegated
    # to scripts/ablations/run_augmentation_ablation_opx.py and
    # scripts/ablations/run_augmentation_oof_bias.py, which MUST be run
    # before papermilling this notebook so the CSVs exist.
    "nb04b_aug_test",
    "nb05_loso_validation",
    "nb06_shap_analysis",
    "nb07_bias_correction",
    # nb07b_arcpl_bias_probe is a development-time diagnostic that probed
    # the +50 C / +1.5 kbar ArcPL residual structure. It is not on the
    # critical path: it produces no manuscript-bound output, no figure
    # appears in figures/opx_only/, and nothing downstream consumes its
    # CSVs. Run it manually if you want to inspect the probe; it requires
    # nb04_putirka_benchmark Part 3 to have produced
    # results/nb04_arcpl_opx_liq_predictions_forest.csv first.
    "nb09_manuscript_compilation",
    # nbF_figures is the figure-build orchestrator. It shells out to the
    # standalone scripts in scripts/figures/, scripts/pairing/,
    # scripts/shap/, and scripts/manuscript/ to produce the publication
    # figures and the manuscript .docx.
    "nbF_figures",
]

CELL_TIMEOUT_SECONDS = 10800
DECISION_MARKER = "OPERATOR DECISION REQUIRED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Skip all notebooks before this one (name without .ipynb).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_run_{run_id}.log"

    start_idx = 0
    if args.resume_from:
        try:
            start_idx = NOTEBOOKS.index(args.resume_from)
        except ValueError:
            print(f"ERROR: --resume-from '{args.resume_from}' not found in NOTEBOOKS list.")
            print(f"Available: {NOTEBOOKS}")
            return 3

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Pipeline run started at {datetime.now().isoformat()}\n")
        log.write(f"Run ID: {run_id}\n")
        log.write(f"Notebooks (in order): {NOTEBOOKS}\n")
        log.write(f"Starting from: {NOTEBOOKS[start_idx]} (index {start_idx})\n")
        log.write(f"Per-cell timeout: {CELL_TIMEOUT_SECONDS}s\n")
        log.write(f"Decision halt marker: {DECISION_MARKER!r}\n\n")
        log.flush()

        for nb in NOTEBOOKS[start_idx:]:
            ts = datetime.now().strftime("%H:%M:%S")
            banner = f"\n[{ts}] starting {nb}\n"
            print(banner.strip())
            log.write(banner)
            log.flush()

            nb_in = NB_DIR / f"{nb}.ipynb"
            if not nb_in.exists():
                msg = f"[{ts}] MISSING {nb_in} -- halting\n"
                print(msg.strip())
                log.write(msg)
                log.flush()
                miss_log = LOG_DIR / f"MISSING_{nb}.log"
                miss_log.write_text(
                    f"{nb} not found at {nb_in}\n"
                    f"Check pipeline execution order in run_all.py.\n",
                    encoding="utf-8",
                )
                return 3

            cmd = [
                sys.executable, "-m", "papermill",
                str(nb_in),
                str(OUT_DIR / f"{nb}_executed.ipynb"),
                "--log-output",
                "--progress-bar",
                "--execution-timeout", str(CELL_TIMEOUT_SECONDS),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None

            decision_triggered = False
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                if DECISION_MARKER in line:
                    decision_triggered = True
            proc.wait()

            ts = datetime.now().strftime("%H:%M:%S")

            if proc.returncode != 0:
                fail_log = LOG_DIR / f"FAILURE_{nb}.log"
                fail_log.write_text(
                    f"{nb} failed with returncode {proc.returncode} "
                    f"at {datetime.now().isoformat()}\n"
                    f"Full run log: {log_path}\n"
                    f"Resume command after fix: "
                    f"python run_all.py --resume-from {nb}\n",
                    encoding="utf-8",
                )
                msg = f"[{ts}] FAILED {nb} rc={proc.returncode}\n"
                print(msg.strip())
                log.write(msg)
                return 1

            if decision_triggered:
                halt_log = LOG_DIR / f"HALT_{nb}.log"
                next_idx = NOTEBOOKS.index(nb) + 1
                next_nb = NOTEBOOKS[next_idx] if next_idx < len(NOTEBOOKS) else "(no more)"
                halt_log.write_text(
                    f"{nb} completed successfully but hit {DECISION_MARKER}\n"
                    f"at {datetime.now().isoformat()}\n"
                    f"Review the notebook output for the decision context.\n"
                    f"Full run log: {log_path}\n"
                    f"Resume command after decision: "
                    f"python run_all.py --resume-from {next_nb}\n",
                    encoding="utf-8",
                )
                msg = (
                    f"[{ts}] HALT: operator decision required in {nb}\n"
                    f"Review {halt_log} and resume with --resume-from <next_nb>\n"
                )
                print(msg.strip())
                log.write(msg)
                return 2

            ok = f"[{ts}] OK {nb}\n"
            print(ok.strip())
            log.write(ok)
            log.flush()

        done = f"\n[{datetime.now():%H:%M:%S}] full pipeline complete\n"
        print(done.strip())
        log.write(done)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
