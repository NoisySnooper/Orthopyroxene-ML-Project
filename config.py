from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROC = ROOT / 'data' / 'processed'
DATA_SPLITS = ROOT / 'data' / 'splits'
DATA_NATURAL = ROOT / 'data' / 'natural'
MODELS = ROOT / 'models'
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'
LOGS = ROOT / 'logs'

for d in [DATA_RAW, DATA_PROC, DATA_SPLITS, DATA_NATURAL, MODELS, FIGURES, RESULTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

EXPETDB = DATA_RAW / 'ExPetDB_download_ExPetDB-2025-07-21.xlsx'
