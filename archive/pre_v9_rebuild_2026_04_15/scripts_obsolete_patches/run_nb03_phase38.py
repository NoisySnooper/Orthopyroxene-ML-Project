"""Execute nb03 Phase 3.8 (ceiling analysis) standalone.

Phase 3.4 (20-seed training) and Phase 3.5 (per-family winners) already
wrote their artifacts at 22:34 on 2026-04-14. Only the Phase 3.8 cells
(26-31 in the source notebook, code cells 27-30) still needed to run,
and they depend solely on on-disk artifacts + the patched canonical
resolver. This runner execs the four code cells in order under a fresh
namespace primed with the same preamble as the notebook.

Also injects the fresh outputs back into the executed notebook copy
under notebooks/executed/ so the manuscript-ready artifact shows Phase
3.8 complete.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NB_SRC = ROOT / "notebooks" / "nb03_baseline_models.ipynb"
NB_EXEC = ROOT / "notebooks" / "executed" / "nb03_baseline_models_executed.ipynb"

PHASE38_CELL_IDX = [27, 28, 29, 30]


def _preamble_namespace() -> dict:
    import ast as _ast
    import json as _json
    import numpy as _np
    import pandas as _pd
    import xgboost as _xgb
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    import config as _config
    from src.features import build_feature_matrix as _build_feat

    DATA_PROC = ROOT / "data" / "processed"
    DATA_SPLITS = ROOT / "data" / "splits"
    RESULTS = ROOT / "results"
    MODELS = ROOT / "models"
    FIGURES = ROOT / "figures"
    FIGS = FIGURES

    ns = dict(
        ast=_ast, json=_json, np=_np, pd=_pd, xgb=_xgb,
        RandomForestRegressor=RandomForestRegressor,
        ExtraTreesRegressor=ExtraTreesRegressor,
        Ridge=Ridge, MLPRegressor=MLPRegressor,
        Pipeline=Pipeline, StandardScaler=StandardScaler,
        mean_squared_error=mean_squared_error,
        mean_absolute_error=mean_absolute_error,
        r2_score=r2_score,
        plt=_plt,
        config=_config,
        build_feature_matrix=_build_feat,
        DATA_PROC=DATA_PROC, DATA_SPLITS=DATA_SPLITS,
        RESULTS=RESULTS, MODELS=MODELS,
        FIGURES=FIGURES, FIGS=FIGS,
        N_SPLIT_REPS=_config.N_SPLIT_REPS,
    )
    return ns


def _exec_with_capture(src: str, ns: dict) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src, "<phase38>", "exec"), ns)
    return buf.getvalue()


def main() -> int:
    nb = nbformat.read(str(NB_SRC), as_version=4)
    ns = _preamble_namespace()

    captured: dict[int, str] = {}
    for idx in PHASE38_CELL_IDX:
        cell = nb.cells[idx]
        assert cell.cell_type == "code", f"cell {idx} not code"
        print(f"--- running cell {idx} ---")
        out = _exec_with_capture(cell.source, ns)
        captured[idx] = out
        if out:
            print(out)

    print("\nPhase 3.8 done. Artifacts:")
    for p in [
        ROOT / "results" / "nb11_model_family_ceiling.csv",
        ROOT / "figures" / "fig_nb11_model_family_ceiling.png",
        ROOT / "figures" / "fig_nb11_model_family_ceiling.pdf",
    ]:
        print(f"  {'OK' if p.exists() else 'MISSING'}: {p}")

    if NB_EXEC.exists():
        nb_exec = nbformat.read(str(NB_EXEC), as_version=4)
        for idx in PHASE38_CELL_IDX:
            if idx >= len(nb_exec.cells):
                continue
            cell = nb_exec.cells[idx]
            if cell.cell_type != "code":
                continue
            cell.outputs = [
                new_output(output_type="stream", name="stdout",
                           text=captured.get(idx, ""))
            ] if captured.get(idx) else []
            cell.execution_count = cell.get("execution_count") or (idx + 1)
        nbformat.write(nb_exec, str(NB_EXEC))
        print(f"\nInjected Phase 3.8 outputs into {NB_EXEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
