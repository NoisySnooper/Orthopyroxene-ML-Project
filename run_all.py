"""Execute the full opx ML thermobarometer pipeline end-to-end via papermill.

Notebooks run in dependency order. NB10 (extended analyses) runs before NB09
(manuscript compilation) because NB09 reads NB10 outputs.

Usage:
    .venv\\Scripts\\python.exe run_all.py

Per-notebook stdout/stderr is streamed live and also appended to a timestamped
log file under logs/. On any notebook failure, execution stops and the error
tail is printed.
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
    'nb08_natural_samples',
    'nb10_extended_analyses',
    'nb10b_two_pyroxene_benchmark',
    'nb11_model_family_ceiling',
    'nbF_figures',
    'nb09_manuscript_compilation',
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'pipeline_run_{run_id}.log'

    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'Pipeline run started at {datetime.now().isoformat()}\n')
        log.flush()

        for nb in NOTEBOOKS:
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

        done = f'\n[{datetime.now():%H:%M:%S}] full pipeline complete\n'
        print(done.strip())
        log.write(done)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
