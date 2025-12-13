"""Validation tests for SubsystemOptions configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.subsystems.affinity import AffinityWeights
from codeintel.analytics.subsystems.materialize import SubsystemOptions
from codeintel.config.primitives import SnapshotRef

EXPECTED_MIN_MODULES = 5
EXPECTED_MAX_SUBSYSTEMS = 10
EXPECTED_IMPORT_WEIGHT = 2.5
EXPECTED_SYMBOL_WEIGHT = 1.5
EXPECTED_CONFIG_WEIGHT = 0.75


def test_options_are_applied_and_typed() -> None:
    """Options should populate config with validated numeric values."""
    _ = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path().resolve())

    weights = AffinityWeights(
        import_weight=EXPECTED_IMPORT_WEIGHT,
        symbol_weight=EXPECTED_SYMBOL_WEIGHT,
        config_weight=EXPECTED_CONFIG_WEIGHT,
    )
    options = SubsystemOptions(
        min_modules=EXPECTED_MIN_MODULES,
        max_subsystems=EXPECTED_MAX_SUBSYSTEMS,
        weights=weights,
    )

    if options.min_modules != EXPECTED_MIN_MODULES:
        pytest.fail(f"min_modules not applied: {options.min_modules}")
    if options.max_subsystems != EXPECTED_MAX_SUBSYSTEMS:
        pytest.fail(f"max_subsystems not applied: {options.max_subsystems}")
    if options.weights is None:
        pytest.fail("weights not applied")
    if options.weights.import_weight != EXPECTED_IMPORT_WEIGHT:
        pytest.fail(f"import_weight not applied: {options.weights.import_weight}")
    if options.weights.symbol_weight != EXPECTED_SYMBOL_WEIGHT:
        pytest.fail(f"symbol_weight not applied: {options.weights.symbol_weight}")
    if options.weights.config_weight != EXPECTED_CONFIG_WEIGHT:
        pytest.fail(f"config_weight not applied: {options.weights.config_weight}")
