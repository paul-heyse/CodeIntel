"""Validation tests for SubsystemsStepConfig overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from codeintel.config import ConfigBuilder

EXPECTED_MIN_MODULES = 5
EXPECTED_MAX_SUBSYSTEMS = 10
EXPECTED_IMPORT_WEIGHT = 2.5
EXPECTED_SYMBOL_WEIGHT = 1.5
EXPECTED_CONFIG_WEIGHT = 0.75


def test_overrides_are_applied_and_typed() -> None:
    """Overrides should populate config with validated numeric values."""
    builder = ConfigBuilder.from_snapshot(
        repo="demo/repo", commit="abc123", repo_root=Path().resolve()
    )
    cfg = builder.subsystems(
        min_modules=EXPECTED_MIN_MODULES,
        max_subsystems=EXPECTED_MAX_SUBSYSTEMS,
        import_weight=EXPECTED_IMPORT_WEIGHT,
        symbol_weight=EXPECTED_SYMBOL_WEIGHT,
        config_weight=EXPECTED_CONFIG_WEIGHT,
    )

    if cfg.min_modules != EXPECTED_MIN_MODULES:
        pytest.fail(f"min_modules not applied: {cfg.min_modules}")
    if cfg.max_subsystems != EXPECTED_MAX_SUBSYSTEMS:
        pytest.fail(f"max_subsystems not applied: {cfg.max_subsystems}")
    if cfg.import_weight != EXPECTED_IMPORT_WEIGHT:
        pytest.fail(f"import_weight not applied: {cfg.import_weight}")
    if cfg.symbol_weight != EXPECTED_SYMBOL_WEIGHT:
        pytest.fail(f"symbol_weight not applied: {cfg.symbol_weight}")
    if cfg.config_weight != EXPECTED_CONFIG_WEIGHT:
        pytest.fail(f"config_weight not applied: {cfg.config_weight}")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_modules": cast("int | None", "invalid")}, "integer"),
        ({"max_subsystems": cast("int | None", 1.5)}, "integer"),
        ({"import_weight": cast("float | None", "heavy")}, "numeric"),
    ],
)
def test_invalid_overrides_raise(overrides: dict[str, object], message: str) -> None:
    """Invalid override types should raise a TypeError."""
    builder = ConfigBuilder.from_snapshot(
        repo="demo/repo", commit="abc123", repo_root=Path().resolve()
    )
    with pytest.raises(TypeError) as excinfo:
        kwargs = cast("dict[str, Any]", overrides)
        builder.subsystems(**kwargs)
    if message not in str(excinfo.value):
        pytest.fail(f"Expected '{message}' in error: {excinfo.value}")
