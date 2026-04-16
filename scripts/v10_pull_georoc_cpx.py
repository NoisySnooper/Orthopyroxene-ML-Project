#!/usr/bin/env python3
"""
Phase A sub-step: Pull GEOROC cpx global dataset (parallel to opx SGFTFN).

Author: NQTa (with Claude)
Date: 2026-04-16
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)

DATA_NATURAL = PROJECT_ROOT / "data" / "natural"

def main():
    print("=" * 70)
    print("v10 PULL GEOROC CPX GLOBAL DATASET")
    print("=" * 70)

    print("\n[1] GEOROC cpx global dataset location...")
    print("    GEOROC public database: https://georoc.eu/")
    print("    Query: Phase = Clinopyroxene, download as CSV")
    print("")
    print("    Status: Manual download required (GEOROC web interface)")
    print("    Expected file: GEOROC_CLINOPYROXENES_*.csv (~50-100 MB)")
    print("    Target location: data/natural/2024-XX-GEOROC_CLINOPYROXENES.csv")
    print("")
    print("[2] Download steps:")
    print("    1. Visit https://georoc.eu/")
    print("    2. Filter: Phase = 'Clinopyroxene' only")
    print("    3. Download CSV (all columns)")
    print("    4. Save to data/natural/2024-XX-GEOROC_CLINOPYROXENES.csv")
    print("    5. Verify: wc -l data/natural/2024-XX-GEOROC_CLINOPYROXENES.csv")
    print("")

    # Check if GEOROC cpx file exists
    cpx_files = list(DATA_NATURAL.glob("*GEOROC*CLINOPYROXENES*"))
    if cpx_files:
        print(f"[OK] Found existing GEOROC cpx file: {cpx_files[0].name}")
        print(f"    Size: {cpx_files[0].stat().st_size / 1e6:.1f} MB")
        return True
    else:
        print("[WARN] No GEOROC cpx file found in data/natural/")
        print("       User must download manually from https://georoc.eu/")
        print("       Once downloaded, Phase A will detect it.")
        return False

if __name__ == "__main__":
    success = main()
    print("\n[Note] This pull is manual; automated download requires GEOROC API key")
    sys.exit(0 if success else 1)
