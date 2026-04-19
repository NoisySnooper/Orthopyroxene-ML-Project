#!/usr/bin/env python3
"""
Phase B notebook compatibility audit: scan for hardcoded paths, feature sets, indices.

Companion to: v10_notebooks_compatibility_audit.md Section 3
Author: NQTa (with Claude)
Date: 2026-04-16
"""

import os
import json
import re
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"

AUDIT_CSV = RESULTS_DIR / "v10_phase_b_audit.csv"

def scan_notebook(nb_path):
    """Scan a notebook for hardcoded paths, feature sets, indices."""
    issues = []

    with open(nb_path, 'r', encoding='utf-8', errors='ignore') as f:
        nb = json.load(f)

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # Check for hardcoded .joblib filenames (outside of loader context)
        joblib_pattern = r'["\'](?!.*load_canonical).*\.joblib["\']'
        if re.search(joblib_pattern, source) and 'load_canonical' not in source:
            issues.append({
                'notebook': nb_path.name,
                'cell_index': cell_idx,
                'issue_type': 'hardcoded_joblib',
                'severity': 'WARN',
                'excerpt': source[:100],
                'suggested_fix': 'Use canonical_model_filename() or load_canonical_model()'
            })

        # Check for hardcoded feature_set strings outside loaders
        for feature_set in ['alr', 'pwlr', 'raw']:
            pattern = rf"['\"]({feature_set})['\"]"
            if re.search(pattern, source):
                # Try to detect if it's in a loader context (whitelist)
                if 'load_canonical' not in source and 'feature_set' not in source:
                    issues.append({
                        'notebook': nb_path.name,
                        'cell_index': cell_idx,
                        'issue_type': f'hardcoded_feature_set_{feature_set}',
                        'severity': 'WARN',
                        'excerpt': source[:100],
                        'suggested_fix': f"Reference 'per_family_winners' manifest for feature_set per pipeline"
                    })
                    break

        # Check for hardcoded test indices (numeric arrays)
        # Only flag if it's actually loading indices, not pandas DataFrame operations
        if re.search(r'test_indices\s*=\s*np\.array\(\[', source):
            if 'load_split' not in source and 'data.splits' not in source:
                # Exclude pandas operations like reset_index
                if 'reset_index' not in source and '.index' not in source:
                    issues.append({
                        'notebook': nb_path.name,
                        'cell_index': cell_idx,
                        'issue_type': 'hardcoded_test_indices',
                        'severity': 'BLOCKING',
                        'excerpt': source[:100],
                        'suggested_fix': "Use data/splits/*.npy files via src.data loaders, not hardcoded indices"
                    })

        # Check imports
        imports = re.findall(r'from src import (.*)', source)
        for imp in imports:
            modules = [m.strip() for m in imp.split(',')]
            for mod in modules:
                mod_path = PROJECT_ROOT / f"src/{mod.split('.')[0]}.py"
                if not mod_path.exists():
                    issues.append({
                        'notebook': nb_path.name,
                        'cell_index': cell_idx,
                        'issue_type': 'missing_module',
                        'severity': 'BLOCKING',
                        'excerpt': f"from src import {imp}",
                        'suggested_fix': f"Module src/{mod} does not exist"
                    })

    return issues

def main():
    print("=" * 70)
    print("v10 PHASE B NOTEBOOK COMPATIBILITY AUDIT")
    print("=" * 70)

    all_issues = []
    nb_files = sorted(NOTEBOOKS_DIR.glob("nb*.ipynb"))

    print(f"\n[1] Scanning {len(nb_files)} notebooks...")
    for nb_path in nb_files:
        if 'executed' in str(nb_path):
            continue
        print(f"    Scanning {nb_path.name}...")
        issues = scan_notebook(nb_path)
        all_issues.extend(issues)

    print(f"\n[2] Found {len(all_issues)} issues")

    # Write CSV
    if all_issues:
        df = pd.DataFrame(all_issues)
        df.to_csv(AUDIT_CSV, index=False)
        print(f"    [OK] Audit CSV: {AUDIT_CSV}")

        # Summary by severity
        blocking = len(df[df['severity'] == 'BLOCKING'])
        warn = len(df[df['severity'] == 'WARN'])

        print(f"\n[3] SEVERITY SUMMARY")
        print(f"    BLOCKING: {blocking}")
        print(f"    WARN: {warn}")

        if blocking > 0:
            print(f"\n[ALERT] {blocking} BLOCKING issues found. Phase B cannot proceed until fixed.")
            print(f"        See {AUDIT_CSV} for details.")
            return False
        else:
            print(f"\n[OK] No BLOCKING issues. {warn} WARNings logged for Phase G.")
            return True
    else:
        print(f"[OK] No issues found. All notebooks compatible.")
        df = pd.DataFrame(columns=['notebook', 'cell_index', 'issue_type', 'severity', 'excerpt', 'suggested_fix'])
        df.to_csv(AUDIT_CSV, index=False)
        return True

if __name__ == "__main__":
    success = main()
    print("\n[PHASE B AUDIT COMPLETE]")
    sys.exit(0 if success else 1)
