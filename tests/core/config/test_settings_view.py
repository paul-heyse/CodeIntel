"""Tests for SettingsView access helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.core.config.settings import (
    BuildSettings,
    CliSettings,
    HamiltonExecutionSettings,
    ObservabilitySettings,
    ServingSettings,
)
from codeintel.core.config.view import SettingsView
from codeintel.core.runtime import RuntimeSettings, VariantConfig


def test_settings_view_from_runtime_round_trip(tmp_path: Path) -> None:
    """Round-trip SettingsView creation from runtime settings."""
    runtime = RuntimeSettings(
        build=BuildSettings(engine_version="test"),
        cli=CliSettings(),
        execution=HamiltonExecutionSettings(),
        serving=ServingSettings(serve_dir=tmp_path),
        observability=ObservabilitySettings(),
        variants=VariantConfig(),
    )
    view = SettingsView.from_runtime(runtime)
    assert view.build == runtime.build
    assert view.require_serving() == runtime.serving
    assert view.require_observability() == runtime.observability


def test_settings_view_require_raises_when_missing() -> None:
    """Raise when required settings are missing from the view."""
    view = SettingsView(
        build=BuildSettings(engine_version="test"),
        execution=HamiltonExecutionSettings(),
    )
    with pytest.raises(ValueError, match="Serving settings"):
        view.require_serving()
    with pytest.raises(ValueError, match="Observability settings"):
        view.require_observability()
