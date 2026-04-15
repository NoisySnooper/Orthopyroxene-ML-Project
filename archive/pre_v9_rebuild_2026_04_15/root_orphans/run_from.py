"""Resume pipeline execution from a given notebook name onward.

Usage:
    .venv\\Scripts\\python.exe run_from.py nb04_putirka_benchmark
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / 'notebooks'
OUT_DIR = NB_DIR / 'executed'
LOG_DIR = ROOT / 'logs'

NOTEBOOKS = [
    'nb01_data_cleaning',
    'nb02_eda_pca',
    'nb03_baseline_models',
    'nb04_putirka_benchmark',
    'nb04b_lepr_arcpl_validation',
    'nb05_loso_validation',
    'nb06_shap_analysis',
    'nb07_bias_correction',
    'nb08_natural_twopx',
    'nbF_figures',
    'nb09_manuscript_compilation',
]


def main() -> int:
    start_nb = sys.argv[1] if len(sys.argv) > 1 else NOTEBOOKS[0]
    if start_nb not in NOTEBOOKS:
        print(f'Unknown notebook: {start_nb}')
        return 2
    start_idx = NOTEBOOKS.index(start_nb)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'pipeline_resume_{run_id}.log'

    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'Resume run started at {datetime.now().isoformat()}\n')
        log.write(f'Starting from: {start_nb} (index {start_idx})\n')
        log.flush()

        for nb in NOTEBOOKS[start_idx:]:
            ts = datetime.now().strftime('%H:%M:%S')
            banner = f'\n[{ts}] starting {nb}\n'
            print(banner.strip())
            log.write(banner)
            log.flush()

            cmd = [
                sys.executable, '-m', 'papermill',
                str(NB_DIR / f'{nb}.ipynb'),
                str(OUT_DIR / f'{nb}_executed.ipynb'),
                '--log-output',
                '--progress-bar',
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()

            ts = datetime.now().strftime('%H:%M:%S')
            if proc.returncode != 0:
                fail_log = LOG_DIR / f'FAILURE_{nb}.log'
                fail_log.write_text(
                    f'{nb} failed with returncode {proc.returncode} '
                    f'at {datetime.now().isoformat()}\n'
                    f'Full run log: {log_path}\n',
                    encoding='utf-8',
                )
                msg = f'[{ts}] FAILED {nb} rc={proc.returncode}\n'
                print(msg.strip())
                log.write(msg)
                return 1

            ok = f'[{ts}] OK {nb}\n'
            print(ok.strip())
            log.write(ok)
            log.flush()

        done = f'\n[{datetime.now():%H:%M:%S}] resume run complete\n'
        print(done.strip())
        log.write(done)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
