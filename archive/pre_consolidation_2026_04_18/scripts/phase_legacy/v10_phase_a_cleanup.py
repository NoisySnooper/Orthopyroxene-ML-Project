#!/usr/bin/env python3
"""Phase A cleanup: archive v9 artifacts, delete per user review, scaffold new dirs."""

import os
import shutil
import json
import subprocess
import hashlib
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)

ARCHIVE_DATE = "2026_04_16"
ARCHIVE_DIR = PROJECT_ROOT / f"archive/pre_v10_rebuild_{ARCHIVE_DATE}"
TAG_REQUIRED = "pre_v10_cleanup_2026_04_16"
FREE_SPACE_MB = 1000

ARCHIVE_THEN_DELETE = [
    "results/manuscript_key_results.csv",
    "results/nb03_*.csv",
    "results/nb03_*.json",
    "results/nb03_*.npz",
    "results/nb04_*.csv",
    "results/nb04b_*.csv",
    "results/nb05_*.csv",
    "results/nb06_*.csv",
    "results/nb07*.csv",
    "results/nb07*.json",
    "results/nb08_*.csv",
    "results/nb10_*.csv",
    "results/nbF_*.csv",
    "results/table*.csv",
    "results/figure_inventory.csv",
    "results/optuna_studies/",
    "figures/fig*.pdf",
    "figures/fig*.png",
    "figures/fig*.txt",
    "models/*.joblib",
    "notebooks/executed/",
    "logs/*.txt",
    "logs/*.log",
    "nb03_optuna_best_params.json",
    "nb03c_training.log",
    "nb03_search.log",
]

DELETE_ITEMS = [
    "build_smoke/",
    "src/__pycache__/",
    "__pycache__/",
]

ARCHIVE_USER_DECISIONS = [
    "extract_results.py",
    "app_extract_inventory.txt",
]

STALE_RUNNERS = {
    "run_all.py": "DELETE",
    "run_from.py": "DELETE",
    "wipe_old_files_v7.py": "DELETE",
    "run_all_v7.py": "RENAME",
}

NEW_DIRS = [
    "manuscripts/opx_2026/figures",
    "manuscripts/opx_2026/tables",
    "manuscripts/opx_2026/text",
    "manuscripts/opx_2026/arxiv_submission",
    "manuscripts/cpx_2026/figures",
    "manuscripts/cpx_2026/tables",
    "manuscripts/cpx_2026/text",
    "manuscripts/cpx_2026/arxiv_submission",
    "models/canonical/opx",
    "models/canonical/cpx",
    "models/canonical/twopx",
    "models/canonical/universal",
    "models/ablation/resampled",
    "src/ablations",
]

KEEP_DIRS = [
    "src",
    "data/raw",
    "data/processed",
    "data/splits",
    "data/external",
    "data/natural",
    "models/external",
    "notebooks",
    "docs",
]

LOG_FILE = PROJECT_ROOT / "logs" / f"v10_phase_a_cleanup_{ARCHIVE_DATE}.txt"

# ============================================================================
def log_msg(msg):
    """Log to both stdout and file (ascii-safe)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(msg + "\n")

def sha256_file(filepath):
    """Compute SHA256 of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_free_space_mb(path):
    """Get free disk space in MB."""
    try:
        result = subprocess.run(
            f'powershell -Command "(Get-Volume -DriveLetter {path[0]}).SizeRemaining / 1MB"',
            capture_output=True, text=True, shell=True
        )
        return float(result.stdout.strip())
    except:
        return float("inf")

def glob_files(pattern):
    """Glob files matching pattern."""
    return list(PROJECT_ROOT.glob(pattern))

def archive_and_delete(src_pattern, archive_dir):
    """Archive files matching glob pattern, then delete."""
    files = glob_files(src_pattern)
    results = []
    for src in files:
        if src.is_file():
            rel_path = src.relative_to(PROJECT_ROOT)
            archive_path = archive_dir / rel_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, archive_path)
            src.unlink()
            results.append((str(rel_path), "OK"))
    return results

# ============================================================================
def main():
    log_msg("=" * 70)
    log_msg(f"v10 PHASE A CLEANUP [{datetime.now().isoformat()}]")
    log_msg("=" * 70)

    # [1] Check git tag
    log_msg("\n[1] Checking pre-cleanup git tag...")
    result = subprocess.run(
        f"git tag -l {TAG_REQUIRED}",
        capture_output=True, text=True, shell=True, cwd=PROJECT_ROOT
    )
    if TAG_REQUIRED not in result.stdout:
        log_msg(f"FATAL: Tag {TAG_REQUIRED} not found. Aborting.")
        return False
    log_msg(f"[OK] Tag {TAG_REQUIRED} found")

    # [2] Check git status
    log_msg("\n[2] Checking git status...")
    result = subprocess.run(
        "git status --porcelain",
        capture_output=True, text=True, shell=True, cwd=PROJECT_ROOT
    )
    if result.stdout.strip():
        log_msg("WARN: Git tree has uncommitted changes. Consider committing first.")
    else:
        log_msg("[OK] Git tree clean")

    # [3] Check free space
    log_msg(f"\n[3] Checking free disk space (need {FREE_SPACE_MB} MB)...")
    free_mb = get_free_space_mb(str(PROJECT_ROOT))
    if free_mb < FREE_SPACE_MB:
        log_msg(f"FATAL: Only {free_mb:.0f} MB free. Aborting.")
        return False
    log_msg(f"[OK] {free_mb:.0f} MB free")

    # [4] Create archive directory
    log_msg(f"\n[4] Creating archive directory...")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    log_msg(f"[OK] Archive dir: {ARCHIVE_DIR}")

    # [5] Archive then delete (manifest list)
    log_msg(f"\n[5] Archiving results, figures, models, logs...")
    archive_count = 0
    for pattern in ARCHIVE_THEN_DELETE:
        results = archive_and_delete(pattern, ARCHIVE_DIR)
        for rel_path, status in results:
            archive_count += 1
    log_msg(f"[OK] Archived {archive_count} items")

    # [6] Delete per user REVIEW decisions
    log_msg(f"\n[6] Deleting per user review decisions...")
    delete_count = 0
    for item in DELETE_ITEMS:
        paths = glob_files(item)
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            log_msg(f"    [OK] Deleted {p.relative_to(PROJECT_ROOT)}")
            delete_count += 1
    log_msg(f"[OK] Deleted {delete_count} items")

    # [7] Archive per user REVIEW decisions
    log_msg(f"\n[7] Archiving per user review decisions...")
    archive_user_count = 0
    for item in ARCHIVE_USER_DECISIONS:
        p = PROJECT_ROOT / item
        if p.exists():
            rel_path = p.relative_to(PROJECT_ROOT)
            archive_path = ARCHIVE_DIR / rel_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, archive_path)
            p.unlink()
            log_msg(f"    [OK] Archived {rel_path}")
            archive_user_count += 1
    log_msg(f"[OK] Archived {archive_user_count} items")

    # [8] Delete stale runners
    log_msg(f"\n[8] Handling stale runners...")
    runner_count = 0
    for runner, action in STALE_RUNNERS.items():
        p = PROJECT_ROOT / runner
        if p.exists():
            if action == "DELETE":
                p.unlink()
                log_msg(f"    [OK] Deleted {runner}")
            elif action == "RENAME":
                new_name = p.parent / "run_all_v10.py"
                p.rename(new_name)
                log_msg(f"    [OK] Renamed {runner} -> run_all_v10.py")
            runner_count += 1
    log_msg(f"[OK] Handled {runner_count} runners")

    # [9] Create new directory scaffolds
    log_msg(f"\n[9] Creating new directory scaffolds...")
    for d in NEW_DIRS:
        new_dir = PROJECT_ROOT / d
        new_dir.mkdir(parents=True, exist_ok=True)
    log_msg(f"[OK] Created {len(NEW_DIRS)} new directories")

    # [10] Generate data/hashes.json
    log_msg(f"\n[10] Generating data/hashes.json...")
    hashes = {}
    data_dir = PROJECT_ROOT / "data"
    for fpath in sorted(data_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "hashes.json":
            rel_path = str(fpath.relative_to(PROJECT_ROOT))
            sha = sha256_file(fpath)
            hashes[rel_path] = {
                "sha256": sha,
                "retrieved_date": "2026-04-16",
                "notes": "v10 Phase A baseline hash"
            }
    hashes_file = data_dir / "hashes.json"
    with open(hashes_file, "w") as f:
        json.dump(hashes, f, indent=2)
    log_msg(f"[OK] Generated hashes for {len(hashes)} data files")

    # [11] Sanity check: KEEP dirs intact
    log_msg(f"\n[11] Sanity check: verifying KEEP directories intact...")
    all_intact = True
    for keep_dir in KEEP_DIRS:
        p = PROJECT_ROOT / keep_dir
        if not p.exists():
            log_msg(f"    [X] MISSING: {keep_dir}")
            all_intact = False
    if not all_intact:
        log_msg("FATAL: Some KEEP directories missing!")
        return False
    log_msg("[OK] All KEEP directories intact")

    # [12] Final summary
    log_msg(f"\n[12] PHASE A CLEANUP COMPLETE")
    log_msg(f"    Archive location: {ARCHIVE_DIR}")
    log_msg(f"    Items archived: {archive_count + archive_user_count}")
    log_msg(f"    Items deleted: {delete_count}")
    log_msg(f"    New dirs created: {len(NEW_DIRS)}")
    log_msg(f"    Hashes generated: {len(hashes)}")
    log_msg(f"    Log: {LOG_FILE}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
