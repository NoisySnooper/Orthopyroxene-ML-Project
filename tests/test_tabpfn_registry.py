"""Registry-level contract tests for TabPFN as the 9th BASE_ORDER family.

These tests exercise the symbol surface (BASE_ORDER, TUNED_BASES,
BASE_MODELS, MODEL_CLASSES, PARAM_GRIDS, PIPELINE_MODELS,
STACKING_BASE_ORDER) without instantiating any TabPFN estimator, so they
run green in the main .venv that does not have tabpfn installed.
"""
from __future__ import annotations


def test_base_order_is_9_with_tabpfn_last():
    from src.opx_tb_analysis import BASE_ORDER
    assert len(BASE_ORDER) == 9
    assert BASE_ORDER[-1] == 'TabPFN'


def test_tuned_bases_excludes_tabpfn_and_has_length_8():
    from src.opx_tb_analysis import BASE_ORDER, TUNED_BASES
    assert len(TUNED_BASES) == 8
    assert 'TabPFN' not in TUNED_BASES
    assert tuple(TUNED_BASES) == BASE_ORDER[:-1]


def test_stacking_base_order_stays_4_and_excludes_tabpfn():
    from config import STACKING_BASE_ORDER
    assert len(STACKING_BASE_ORDER) == 4
    assert 'TabPFN' not in STACKING_BASE_ORDER


def test_tabpfn_not_in_pipeline_models():
    from src.models import PIPELINE_MODELS
    assert 'TabPFN' not in PIPELINE_MODELS


def test_tabpfn_param_grid_is_empty():
    from src.models import PARAM_GRIDS
    assert PARAM_GRIDS.get('TabPFN', None) == {}


def test_tabpfn_present_in_base_models_and_model_classes():
    from src.models import BASE_MODELS, MODEL_CLASSES
    assert 'TabPFN' in BASE_MODELS
    assert 'TabPFN' in MODEL_CLASSES


def test_base_models_factories_are_not_instantiated_at_import():
    # The lazy _make_tabpfn import-guard keeps the main-venv import path
    # green; we do NOT invoke BASE_MODELS['TabPFN']() here because tabpfn
    # is not installed in this venv.
    from src.models import BASE_MODELS
    factory = BASE_MODELS['TabPFN']
    assert callable(factory)
