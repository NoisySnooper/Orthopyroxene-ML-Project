"""Preregistration tests T15-T22 covering the bias-correction ship rule.

These tests verify behavioral invariants registered in:
  - docs/preregistration/p_regime_preregistration.md (Sections 4.1, 6.1-6.6)
  - Pre-existing preregistration tests T1-T14 (covered elsewhere).

T15. Backward compatibility: `ship_decision` with `n_min_for_veto=0`
     reproduces the pre-tolerance baseline on synthetic fixtures (no
     silent behavior change for historical callers).

T16. Tiered gate semantics: a regime with n < N_MIN_FOR_VETO that
     degrades is logged to `low_n_degradations` but does NOT veto a
     global-improving correction.

T17. N_MIN_FOR_VETO threshold boundary: the registered constant is
     fixed at 20. Regimes with exactly 20 samples DO veto (threshold
     is inclusive on the veto side, `n >= n_min_for_veto`), while
     regimes with 19 samples do not.

T18. TabPFN bias-correction exclusion: the shipped CSV preserves
     TabPFN's pre-tolerance verdicts verbatim under the canonical
     rescore (TabPFN has no OOF residuals; the rule cannot change a
     decision that was never fit).

T19. Tolerance backward compatibility: `ship_decision` with
     `degradation_tol_abs=0` and `degradation_tol_rel=0` reproduces
     the pre-tolerance baseline byte-for-byte on synthetic fixtures.
     The absolute/relative tolerance parameters default to zero so
     historical callers are unchanged.

T20. Tolerance semantics: a high-n regime whose degradation is at or
     below `max(SHIP_TOL, T_ABS, T_REL * pre_r)` does NOT veto a
     global-improving correction. The same degradation with tolerance
     parameters = 0 would veto.

T21. Max rule: when `T_REL * pre_r > T_ABS`, the relative tolerance
     is what binds. The rescore script uses the per-target absolute
     tolerance (10 degC for T, 1 kbar for P) alongside T_REL=0.10 and
     N_MIN_FOR_VETO=20, matching the registered constants in
     Section 6.1.

T22. Canonical shipped CSV integrity: every opx non-TabPFN row in
     bias_correction_shipped.csv carries a decoded `ship_a_v3` and
     `ship_b_v3` JSON blob, the recorded `winner_v3` is consistent with
     those two decisions under `choose_winner`, and the tolerance
     parameters (n_min_for_veto, degradation_tol_abs,
     degradation_tol_rel) match the registered constants in
     Section 6.1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.bias_correction import (  # noqa: E402
    SHIP_TOL, ShipDecision, choose_winner, ship_decision,
)


# -------------------------------------------------------------------------
# T15: backward compatibility
# -------------------------------------------------------------------------

def test_T15_ship_decision_default_reproduces_v1_on_shipping_case():
    """v1 behavior: ship if overall improves and no regime degrades."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 11.9},
    )
    assert r.ships is True
    assert r.form == 'A'
    assert r.overall_delta > SHIP_TOL


def test_T15_ship_decision_default_reproduces_v1_on_regime_veto():
    """v1 behavior: degrading regime vetoes even with global improvement."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},
    )
    assert r.ships is False
    assert r.max_regime_degradation > SHIP_TOL


def test_T15_ship_decision_default_reproduces_v1_on_overall_veto():
    """v1 behavior: non-improving overall vetoes even if regimes improve."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=10.0 + 1e-9,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.0, 'b': 11.0},
    )
    assert r.ships is False
    assert r.overall_delta <= SHIP_TOL


# -------------------------------------------------------------------------
# T16: tiered gate semantics
# -------------------------------------------------------------------------

def test_T16_low_n_degradation_does_not_veto_under_v2():
    """With n_min_for_veto=20, a small-n degrading regime is logged
    but does not veto; the correction ships."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},   # b degrades by +1
        per_regime_n={'a': 100, 'b': 5},          # b is low-n
        n_min_for_veto=20,
    )
    assert r.ships is True
    assert 'b' in r.low_n_degradations
    assert r.low_n_degradations['b'] == pytest.approx(1.0, abs=1e-9)


def test_T16_high_n_degradation_still_vetoes_under_v2():
    """With n_min_for_veto=20, a high-n degrading regime still vetoes
    (v2 is strictly weaker, never stronger, than v1)."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},
        per_regime_n={'a': 100, 'b': 50},
        n_min_for_veto=20,
    )
    assert r.ships is False


# -------------------------------------------------------------------------
# T17: N_MIN threshold boundary
# -------------------------------------------------------------------------

def test_T17_n_equal_threshold_still_vetoes():
    """Threshold is inclusive on the veto side: n >= N_MIN vetoes."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},
        per_regime_n={'a': 100, 'b': 20},
        n_min_for_veto=20,
    )
    assert r.ships is False


def test_T17_n_just_below_threshold_does_not_veto():
    """n = 19 is strictly below N_MIN=20, so this regime is non-vetoing."""
    r = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},
        per_regime_n={'a': 100, 'b': 19},
        n_min_for_veto=20,
    )
    assert r.ships is True
    assert 'b' in r.low_n_degradations


def test_T17_n_min_value_matches_amendment():
    """Section 6.1 of the prereg doc fixes N_MIN_FOR_VETO=20. The
    canonical rescore script must use this value; if someone edits
    it silently, this test flags it."""
    script = (PROJECT_ROOT
              / 'scripts/bias_correction/rescore_under_v3_rule.py')
    assert script.exists(), 'rescore script missing'
    text = script.read_text(encoding='utf-8')
    assert 'N_MIN = 20' in text, (
        'Section 6.1 of the prereg doc fixes N_MIN_FOR_VETO=20 in the '
        'rescore script; an edit has drifted from the registered '
        'threshold.')


# -------------------------------------------------------------------------
# T18: TabPFN bias-correction exclusion is preserved under v2
# -------------------------------------------------------------------------

@pytest.mark.skipif(
    not (PROJECT_ROOT / 'results/bias_correction_shipped_v2.csv').exists(),
    reason='v2 shipped CSV not yet generated')
def test_T18_tabpfn_v1_verdicts_preserved_in_v2():
    """TabPFN rows in bias_correction_shipped_v2.csv must match v1
    verdicts byte-for-byte (TabPFN has no OOF residuals; v2 never refits)."""
    ship_v1 = pd.read_csv('results/bias_correction_shipped.csv')
    ship_v2 = pd.read_csv('results/bias_correction_shipped_v2.csv')
    tab_v1 = ship_v1[ship_v1.model == 'TabPFN'].copy().reset_index(drop=True)
    tab_v2 = ship_v2[ship_v2.model == 'TabPFN'].copy().reset_index(drop=True)
    assert len(tab_v1) == len(tab_v2), (
        f'tabpfn row count differs: v1={len(tab_v1)} v2={len(tab_v2)}')
    for col in ('pipeline', 'track', 'target', 'feature_set',
                'canonical_seed', 'winner'):
        if col in tab_v1.columns and col in tab_v2.columns:
            assert (tab_v1[col].tolist() == tab_v2[col].tolist()), (
                f'TabPFN column {col} drifted between v1 and v2')


# -------------------------------------------------------------------------
# T19: v3 backward compatibility with v2 at zero tolerance
# -------------------------------------------------------------------------

def test_T19_ship_decision_v3_default_tolerances_match_v2():
    """With degradation_tol_abs=0 and degradation_tol_rel=0 the v3 rule
    must collapse to v2 verdicts byte-for-byte. The `max(tol, t_abs,
    t_rel * pre)` reduces to `tol` and the tiered-n gate is unchanged."""
    common = dict(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 12.0},
        per_regime_post={'a': 4.9, 'b': 13.0},
        per_regime_n={'a': 100, 'b': 50},
        n_min_for_veto=20,
    )
    r_v2 = ship_decision(**common)
    r_v3 = ship_decision(**common,
                         degradation_tol_abs=0.0,
                         degradation_tol_rel=0.0)
    assert r_v2.ships == r_v3.ships
    assert r_v2.overall_delta == pytest.approx(r_v3.overall_delta)
    assert r_v2.max_regime_degradation == pytest.approx(
        r_v3.max_regime_degradation)


# -------------------------------------------------------------------------
# T20: v3 tolerance semantics
# -------------------------------------------------------------------------

def test_T20_v3_tolerance_relaxes_regime_veto():
    """A high-n regime degrading by +1 kbar on a pre=11 kbar RMSE must
    veto under v2 but NOT under v3 with T_ABS=1, T_REL=0.10 since
    max(SHIP_TOL, 1, 1.1) = 1.1 >= 1.0."""
    common = dict(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 11.0},
        per_regime_post={'a': 4.9, 'b': 12.0},   # b degrades by +1.0
        per_regime_n={'a': 100, 'b': 50},
        n_min_for_veto=20,
    )
    r_v2 = ship_decision(**common)
    assert r_v2.ships is False, 'v2 must still veto a +1 kbar degradation'

    r_v3 = ship_decision(**common,
                         degradation_tol_abs=1.0,
                         degradation_tol_rel=0.10)
    assert r_v3.ships is True, (
        'v3 must accept +1 kbar degradation since max(1e-6, 1.0, 0.1*11)=1.1 '
        '>= 1.0')


def test_T20_v3_tolerance_preserves_veto_when_out_of_band():
    """Same setup as above but the degradation now exceeds the tolerance
    band; v3 must still veto (v3 is strictly weaker, never stronger,
    than v2 on the same inputs)."""
    r_v3 = ship_decision(
        form='A',
        pre_overall=10.0, post_overall=9.0,
        per_regime_pre={'a': 5.0, 'b': 11.0},
        per_regime_post={'a': 4.9, 'b': 13.0},   # +2 kbar > 1.1
        per_regime_n={'a': 100, 'b': 50},
        n_min_for_veto=20,
        degradation_tol_abs=1.0,
        degradation_tol_rel=0.10,
    )
    assert r_v3.ships is False


# -------------------------------------------------------------------------
# T21: v3 max rule + per-target tolerance constants
# -------------------------------------------------------------------------

def test_T21_v3_relative_binds_when_larger_than_absolute():
    """When T_REL * pre_r > T_ABS the relative bound governs. With
    T_ABS=10 degC and T_REL=0.10, on a pre=200 degC regime the band is
    max(tol, 10, 20)=20; a degradation of +15 must NOT veto."""
    r = ship_decision(
        form='A',
        pre_overall=100.0, post_overall=80.0,
        per_regime_pre={'a': 80.0, 'b': 200.0},
        per_regime_post={'a': 78.0, 'b': 215.0},
        per_regime_n={'a': 100, 'b': 50},
        n_min_for_veto=20,
        degradation_tol_abs=10.0,
        degradation_tol_rel=0.10,
    )
    assert r.ships is True


def test_T21_v3_constants_match_amendment():
    """Amendment 2 fixes T_ABS_T=10 degC, T_ABS_P=1 kbar, T_REL=0.10,
    N_MIN=20 in the rescore script."""
    script = (PROJECT_ROOT
              / 'scripts/bias_correction/rescore_under_v3_rule.py')
    assert script.exists(), 'v3 rescore script missing'
    text = script.read_text(encoding='utf-8')
    for tok in ('N_MIN = 20', 'T_ABS_T = 10.0', 'T_ABS_P = 1.0',
                'T_REL = 0.10'):
        assert tok in text, (
            f'Amendment 2 constant "{tok}" drifted from v3 rescore script.')


# -------------------------------------------------------------------------
# T22: v3 shipped CSV integrity
# -------------------------------------------------------------------------

@pytest.mark.skipif(
    not (PROJECT_ROOT / 'results/bias_correction_shipped.csv').exists(),
    reason='v3 shipped CSV not yet generated')
def test_T22_v3_ship_csv_integrity():
    """Every opx non-TabPFN row in bias_correction_shipped.csv must
    carry decodable ship_a_v3 / ship_b_v3 JSON, a winner_v3 consistent
    with `choose_winner`, and tolerance parameters that match
    Amendment 2 (N_MIN=20; T_ABS=10 for T_C, 1 for P_kbar; T_REL=0.10)."""
    ship_v3 = pd.read_csv('results/bias_correction_shipped.csv')
    opx = ship_v3[(ship_v3.pipeline == 'opx')
                  & (ship_v3.model != 'TabPFN')].copy()
    assert len(opx) > 0, 'no opx non-TabPFN rows in v3 shipped CSV'

    for _, row in opx.iterrows():
        cell = f'{row.track}/{row.target}/{row.model}/{row.feature_set}'
        dec_a = json.loads(row['ship_a_v3'])
        dec_b = json.loads(row['ship_b_v3'])
        assert 'ships' in dec_a and 'ships' in dec_b, (
            f'{cell}: ship_a_v3 / ship_b_v3 JSON missing "ships"')
        sd_a = ShipDecision(
            form='A', ships=bool(dec_a['ships']),
            overall_delta=float(dec_a.get('overall_delta', float('nan'))),
            max_regime_degradation=float(
                dec_a.get('max_regime_degradation', float('nan'))),
            reason=str(dec_a.get('reason', '')),
        )
        sd_b = ShipDecision(
            form='B', ships=bool(dec_b['ships']),
            overall_delta=float(dec_b.get('overall_delta', float('nan'))),
            max_regime_degradation=float(
                dec_b.get('max_regime_degradation', float('nan'))),
            reason=str(dec_b.get('reason', '')),
        )
        winner_expected = choose_winner(sd_a, sd_b)
        winner_actual = str(row['winner_v3'])
        assert winner_expected == winner_actual, (
            f'{cell}: winner_v3={winner_actual} but choose_winner returned '
            f'{winner_expected}')

        assert int(row['n_min_for_veto']) == 20, (
            f'{cell}: n_min_for_veto != 20 in v3 shipped CSV')
        t_abs_expected = 10.0 if row['target'] == 'T_C' else 1.0
        assert float(row['degradation_tol_abs']) == pytest.approx(
            t_abs_expected), (
            f'{cell}: degradation_tol_abs '
            f'{row["degradation_tol_abs"]} != {t_abs_expected}')
        assert float(row['degradation_tol_rel']) == pytest.approx(0.10), (
            f'{cell}: degradation_tol_rel != 0.10 in v3 shipped CSV')
