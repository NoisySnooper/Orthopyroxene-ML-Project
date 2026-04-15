"""Wrappers for third-party cpx thermobarometers used for the v6 three-way ML
benchmark and NB08 cross-mineral validation.

Agreda-Lopez 2024  : joblib scaler + onnx model + JSON piecewise bias terms.
Jorgenson 2022     : via Thermobar.calculate_cpx_only_press_temp
Wang 2021          : via Thermobar.calculate_cpx_liq_press_temp
Putirka 2008       : via Thermobar.calculate_cpx_liq_press_temp

v7 Part E hardens two things:
- Vendor JSON loading uses `ast.literal_eval`, never `eval`.
- K->C conversion is explicit (pinned against Thermobar 1.0.70) rather than
  a >400 K heuristic that silently misbehaves on natural magmatic ranges.
"""
from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

try:
    import Thermobar as _pt  # noqa: F401 (imported for version check only)
    _TB_VERSION = getattr(_pt, '__version__', 'unknown')
except Exception:
    _TB_VERSION = 'unimportable'


def _check_thermobar_version() -> None:
    """Called lazily by predict_* wrappers so the import-time cost stays nil."""
    from config import THERMOBAR_PINNED_VERSION
    if _TB_VERSION in (THERMOBAR_PINNED_VERSION, 'unknown', 'unimportable'):
        return
    warnings.warn(
        f"Thermobar version {_TB_VERSION} has not been validated against the K->C "
        f"contract. Verify T return type or pin Thermobar=={THERMOBAR_PINNED_VERSION}.",
        RuntimeWarning,
        stacklevel=3,
    )


AGREDA_CPX_COLS = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx',
                   'MgO_Cpx',  'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx']
AGREDA_LIQ_COLS = ['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq',
                   'MgO_Liq',  'MnO_Liq', 'CaO_Liq',  'Na2O_Liq', 'K2O_Liq']
# Relative errors (Agreda-Lopez Parameters.oxide_rel_err)
AGREDA_CPX_STD = np.array([0.03, 0.08, 0.03, 0.03, 0.03, 0.08, 0.03, 0.08, 0.08])
AGREDA_K_STD   = np.array([0.08])


def _agreda_std(phase: Literal['cpx_only', 'cpx_liq']) -> np.ndarray:
    if phase == 'cpx_only':
        return AGREDA_CPX_STD
    # cpx_liq: cpx oxides + liq oxides (excluding K) + K rel err at the end
    return np.concatenate([AGREDA_CPX_STD, AGREDA_CPX_STD[:-1], AGREDA_K_STD])


def _bias_correct(y: np.ndarray, bias: dict) -> np.ndarray:
    """Piecewise linear bias correction (Agreda-Lopez eq S1):

        bias(x) = slope_left  * (x - ang_left)   if x < ang_left
                = slope_right * (x - ang_right)  if x > ang_right
                = 0                              otherwise

    Corrected prediction = y - bias(y).
    """
    sL = float(bias['slope']['left'][0])
    sR = float(bias['slope']['right'][0])
    aL = float(bias['angle']['left'])
    aR = float(bias['angle']['right'])
    b = np.zeros_like(y, dtype=float)
    left = y < aL
    right = y > aR
    b[left]  = sL * (y[left]  - aL)
    b[right] = sR * (y[right] - aR)
    return y - b


def _perturb(X: np.ndarray, rel_std: np.ndarray,
             n_perturb: int = 15, seed: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_perturb augmented samples per row of X using relative errors."""
    X_rep = np.repeat(X, repeats=n_perturb, axis=0)
    std = np.repeat([rel_std], repeats=len(X_rep), axis=0) * X_rep
    rng = np.random.default_rng(seed)
    X_perturb = rng.normal(X_rep, std)
    groups = np.repeat(np.arange(len(X)), repeats=n_perturb)
    return X_perturb, groups


def _load_vendor_bias_json(path: Path) -> dict:
    """Parse an Agreda-Lopez vendor bias file. The upstream release stores a
    Python dict literal inside a JSON string. Try JSON first (future-proof),
    fall back to ast.literal_eval (never eval)."""
    with open(path) as f:
        content = f.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = ast.literal_eval(content)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = ast.literal_eval(data)
    return data


def load_agreda(models_dir: Path, phase: Literal['cpx_only', 'cpx_liq'],
                target: Literal['T', 'P']) -> tuple:
    """Return (scaler, onnx_session, bias_dict) for one Agreda-Lopez model."""
    import onnxruntime as rt
    tag = 'cpx_liq' if phase == 'cpx_liq' else 'cpx'
    stem = Path(models_dir) / f'agreda_{tag}_{target}'
    scaler = joblib.load(str(stem) + '.joblib')
    sess = rt.InferenceSession(str(stem) + '.onnx', providers=['CPUExecutionProvider'])
    bias = _load_vendor_bias_json(Path(str(stem) + '.json'))
    return scaler, sess, bias


def predict_agreda(X_raw: np.ndarray, scaler, sess, bias: dict,
                   rel_std: np.ndarray, n_perturb: int = 15) -> dict:
    """Run Agreda-Lopez pipeline: perturb -> scale -> onnx -> bias correct -> aggregate.

    Returns median, 16th percentile, 84th percentile per input sample.
    """
    X_perturb, groups = _perturb(np.asarray(X_raw, dtype=float), rel_std, n_perturb)
    X_perturb_s = scaler.transform(X_perturb)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    y = sess.run([label_name], {input_name: X_perturb_s.astype(np.float32)})[0].ravel()
    # Reshape back to (n_samples, n_perturb) and aggregate
    y_grouped = y.reshape(-1, n_perturb)
    median = np.median(y_grouped, axis=1)
    p16 = np.percentile(y_grouped, 16, axis=1)
    p84 = np.percentile(y_grouped, 84, axis=1)
    # Bias-correct the point estimate
    median_bc = _bias_correct(median, bias)
    return {'median': median_bc, 'p16': p16, 'p84': p84, 'median_raw': median}


def predict_agreda_from_df(df: pd.DataFrame, models_dir: Path,
                           phase: Literal['cpx_only', 'cpx_liq'],
                           target: Literal['T', 'P'],
                           n_perturb: int = 15) -> dict:
    """Convenience: pull required columns from df and run the full pipeline."""
    cols = AGREDA_CPX_COLS if phase == 'cpx_only' else AGREDA_CPX_COLS + AGREDA_LIQ_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f'Agreda phase={phase} missing columns: {missing}')
    X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    scaler, sess, bias = load_agreda(models_dir, phase, target)
    return predict_agreda(X, scaler, sess, bias, _agreda_std(phase), n_perturb)


def predict_jorgenson(df: pd.DataFrame, target: Literal['T', 'P'],
                      phase: Literal['cpx_only', 'cpx_liq'] = 'cpx_only',
                      P_kbar: np.ndarray | None = None,
                      T_K: np.ndarray | None = None) -> np.ndarray:
    """Jorgenson 2022 via Thermobar. Returns predictions in C (T) or kbar (P)."""
    import Thermobar as pt
    _check_thermobar_version()
    cpx = df[[c for c in df.columns if c.endswith('_Cpx')]].copy()
    if phase == 'cpx_only':
        if target == 'T':
            out = pt.calculate_cpx_only_temp(
                equationT='T_Jorgenson2022_Cpx_only', cpx_comps=cpx, P=P_kbar)
            return _extract(out, celsius=True)
        out = pt.calculate_cpx_only_press(
            equationP='P_Jorgenson2022_Cpx_only', cpx_comps=cpx, T=T_K)
        return _extract(out, celsius=False)
    liq = df[[c for c in df.columns if c.endswith('_Liq')]].copy()
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Jorgenson2022_Cpx_Liq', cpx_comps=cpx, liq_comps=liq, P=P_kbar)
        return _extract(out, celsius=True)
    out = pt.calculate_cpx_liq_press(
        equationP='P_Jorgenson2022_Cpx_Liq', cpx_comps=cpx, liq_comps=liq, T=T_K)
    return _extract(out, celsius=False)


def predict_wang(df: pd.DataFrame, target: Literal['T', 'P'],
                 P_kbar: np.ndarray | None = None,
                 T_K: np.ndarray | None = None) -> np.ndarray:
    """Wang 2021 cpx-liq via Thermobar."""
    import Thermobar as pt
    _check_thermobar_version()
    cpx = df[[c for c in df.columns if c.endswith('_Cpx')]].copy()
    liq = df[[c for c in df.columns if c.endswith('_Liq')]].copy()
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Wang2021_eq2', cpx_comps=cpx, liq_comps=liq, P=P_kbar)
        return _extract(out, celsius=True)
    out = pt.calculate_cpx_liq_press(
        equationP='P_Wang2021_eq1', cpx_comps=cpx, liq_comps=liq, T=T_K)
    return _extract(out, celsius=False)


def predict_putirka_cpx_liq(df: pd.DataFrame, target: Literal['T', 'P'],
                            P_kbar: np.ndarray | None = None,
                            T_K: np.ndarray | None = None) -> np.ndarray:
    """Putirka 2008 cpx-liq eq33 (T) / eq30 (P)."""
    import Thermobar as pt
    _check_thermobar_version()
    cpx = df[[c for c in df.columns if c.endswith('_Cpx')]].copy()
    liq = df[[c for c in df.columns if c.endswith('_Liq')]].copy()
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Put2008_eq33', cpx_comps=cpx, liq_comps=liq, P=P_kbar)
        return _extract(out, celsius=True)
    out = pt.calculate_cpx_liq_press(
        equationP='P_Put2008_eq30', cpx_comps=cpx, liq_comps=liq, T=T_K)
    return _extract(out, celsius=False)


def _extract(x, celsius: bool) -> np.ndarray:
    """Pull a 1-D float array out of a Thermobar return (may be DataFrame,
    Series, or numpy array). Apply K->C conversion based on an explicit,
    pinned contract (Thermobar 1.0.70 returns Kelvin for T functions).

    When the return is a DataFrame with a column name we trust the column
    name. When the return is a bare Series/ndarray (no column label) we
    apply the pinned contract: if `celsius=True` and the caller told us
    Thermobar returns Kelvin, subtract 273.15 unconditionally. No
    distributional heuristic, because magmatic T ranges around ~950 C
    overlap the ~1200 K band and make mean>400 non-discriminative.
    """
    from config import THERMOBAR_T_RETURNS_KELVIN
    if isinstance(x, pd.DataFrame):
        for col in ['T_C_calc', 'T_C', 'T_K_calc', 'T_K',
                    'P_kbar_calc', 'P_kbar']:
            if col in x.columns:
                arr = np.array(x[col].values, dtype=float, copy=True)
                if celsius and col.startswith('T_K'):
                    arr = arr - 273.15
                return arr
        return np.array(x.iloc[:, 0].values, dtype=float, copy=True)
    arr = np.array(x, dtype=float, copy=True)
    if celsius and THERMOBAR_T_RETURNS_KELVIN:
        arr = arr - 273.15
    return arr
