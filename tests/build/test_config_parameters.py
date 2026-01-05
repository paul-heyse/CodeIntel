"""Tests for build configuration loading and parameter access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import load_build_config
from codeintel.build.parameters import ParameterError, TargetParameters
from tests._helpers import make_build_config, write_build_config
from tests._helpers.assertions import expect_equal, expect_in, expect_true

if TYPE_CHECKING:
    from pathlib import Path


def test_load_build_config_merges_module_and_target(tmp_path: Path) -> None:
    """Module-level values merge with target-level overrides."""
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    write_build_config(
        project_root,
        {
            "analytics": {"threshold": 1, "shared": "module"},
            "analytics.function_types": {"threshold": 3, "enabled": True},
            "graphs": {"sampling_rate": 0.2},
        },
    )

    config = load_build_config(project_root)
    params = config.parameters_for("function_types")

    expect_equal(params.get_typed("threshold", int), 3)
    expect_true(params.get_typed("enabled", bool) is True)

    expect_equal(params.get_typed("sampling_rate", float), 0.2)
    expect_equal(params.get_typed("shared", str), "module")


def test_load_build_config_missing_or_invalid_returns_empty(tmp_path: Path) -> None:
    """Missing or invalid config files fall back to empty config."""
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)

    empty_config = load_build_config(project_root)
    expect_equal(empty_config.sections, {})
    expect_true(empty_config.parameters_for("anything").has("missing") is False)

    bad_path = project_root / "config/codeintel.build.toml"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not = { valid = ", encoding="utf-8")
    recovered = load_build_config(project_root)
    expect_equal(recovered.sections, {})


def test_build_config_get_nested_with_defaults() -> None:
    """Nested lookup handles defaults and None values."""
    cfg = make_build_config(
        {
            "analytics": {
                "function_types": {"max_commits": 5},
                "enabled": True,
            },
            "value": None,
            "nested": 4,
        }
    )

    expect_equal(cfg.get("analytics.function_types.max_commits"), 5)
    expect_equal(cfg.get("analytics.function_types.missing", default="fallback"), "fallback")

    expect_equal(cfg.get("value", default="default"), "default")

    expect_equal(cfg.get("nested.deeper", default=0), 0)


def test_target_parameters_success_and_merge() -> None:
    """TargetParameters returns typed values and merges overrides."""
    params = TargetParameters({"count": 10, "enabled": True})
    expect_equal(params.get_typed("count", int), 10)
    expect_true(params.get_typed("enabled", bool) is True)
    expect_true(params.get_optional("missing", str) is None)

    merged = params.merge(TargetParameters({"count": 20, "name": "demo"}))
    expect_equal(merged.get_typed("count", int), 20)
    expect_equal(merged.get_typed("name", str), "demo")
    expect_true(merged.has("enabled") is True)


def test_target_parameters_errors() -> None:
    """TargetParameters raises on missing or mismatched values."""
    params = TargetParameters({"count": 10, "flag": "yes"})

    with pytest.raises(ParameterError):
        params.get_typed("missing", int)

    with pytest.raises(ParameterError) as exc_info:
        params.get_typed("flag", bool)
    expect_in("bool", str(exc_info.value))
    expect_in("str", str(exc_info.value))

    with pytest.raises(ParameterError):
        params.get_optional("flag", int)
