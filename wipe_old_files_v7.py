"""Clean stale artifacts before the v7 20-seed rerun.

Archives (does not delete) old FAILURE logs, orphan figures, and pre-v7
canonical files. Intended to be run once after applying v7 code edits
but before executing run_all_v7.py.

Usage:
    .venv\\Scripts\\python.exe wipe_old_files_v7.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_BASE = ROOT / "archive" / f"v7_preparation_{TS}"


def archive_file(src: Path, category: str, dry_run: bool) -> bool:
    if not src.exists():
        return False
    rel = src.relative_to(ROOT) if src.is_absolute() else src
    dst = ARCHIVE_BASE / category / rel
    if dry_run:
        print(f"  [dry-run] would move: {rel}  ->  {dst.relative_to(ROOT)}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  moved: {rel}  ->  {dst.relative_to(ROOT)}")
    return True


def archive_glob(pattern: str, category: str, dry_run: bool) -> int:
    count = 0
    for src in ROOT.glob(pattern):
        if src.is_file() and archive_file(src, category, dry_run):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be moved without moving anything.")
    args = parser.parse_args()

    print(f"v7 wipe script -- {'DRY RUN' if args.dry_run else 'LIVE RUN'}")
    if args.dry_run:
        print(f"Archive destination (would be): {ARCHIVE_BASE.relative_to(ROOT)}")
    else:
        print(f"Archive destination: {ARCHIVE_BASE.relative_to(ROOT)}")
    print()

    total_moved = 0

    print("[1/7] Stale FAILURE logs...")
    total_moved += archive_glob("logs/FAILURE_*.log", "failure_logs", args.dry_run)

    print("[2/7] Pipeline run logs (older than 7 days only)...")
    cutoff = datetime.now().timestamp() - (7 * 86400)
    pipeline_logs = (list((ROOT / "logs").glob("pipeline_run_*.log"))
                     + list((ROOT / "logs").glob("pipeline_resume_*.log")))
    old_pipeline_count = 0
    for log in pipeline_logs:
        if log.stat().st_mtime < cutoff:
            if archive_file(log, "pipeline_logs_old", args.dry_run):
                old_pipeline_count += 1
    total_moved += old_pipeline_count
    if old_pipeline_count == 0:
        print("  (none older than 7 days)")

    print("[3/7] Orphan figures from archived notebooks...")
    total_moved += archive_glob("figures/fig_nb10b_*.png", "orphan_figures", args.dry_run)
    total_moved += archive_glob("figures/fig_nb10b_*.pdf", "orphan_figures", args.dry_run)

    print("[4/7] Old per-combo winners JSON...")
    old_json = ROOT / "results" / "nb03_winning_configurations.json"
    if old_json.exists():
        print(f"  keeping: {old_json.relative_to(ROOT)} (deprecated but retained for migration)")
    else:
        print("  (not present)")

    print("[5/7] Root-level stragglers...")
    stragglers = [
        ("nb03c_frozen_params.json", "results"),
        ("nb03c_training.log", "logs"),
    ]
    for fname, target_dir in stragglers:
        src = ROOT / fname
        if not src.exists():
            continue
        dst_dir = ROOT / target_dir
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / fname
        if args.dry_run:
            print(f"  [dry-run] would relocate: {fname}  ->  {target_dir}/{fname}")
        else:
            shutil.move(str(src), str(dst))
            print(f"  relocated: {fname}  ->  {target_dir}/{fname}")
        total_moved += 1

    print("[6/7] Old executed notebooks...")
    executed_dir = ROOT / "notebooks" / "executed"
    if executed_dir.exists():
        moved_exec = 0
        for nb_out in executed_dir.glob("*.ipynb"):
            if archive_file(nb_out, "executed_notebooks_pre_v7", args.dry_run):
                moved_exec += 1
        total_moved += moved_exec
        if moved_exec == 0:
            print("  (none present)")
    else:
        print("  (executed dir does not exist; will be created on next run)")

    print("[7/7] models/canonical/ sanity check...")
    canonical_dir = ROOT / "models" / "canonical"
    if canonical_dir.exists():
        contents = list(canonical_dir.iterdir())
        if not contents:
            print(f"  {canonical_dir.relative_to(ROOT)} is empty; "
                  f"will be populated by v7 nb03 Phase 3.6")
        else:
            print(f"  {canonical_dir.relative_to(ROOT)} contains {len(contents)} files "
                  f"(will be overwritten by v7 nb03)")
    else:
        print("  (does not exist; v7 nb03 will create it)")

    print()
    print(f"Summary: {total_moved} items {'would be' if args.dry_run else 'were'} archived/relocated.")
    if not args.dry_run and total_moved > 0:
        print(f"Archive location: {ARCHIVE_BASE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
