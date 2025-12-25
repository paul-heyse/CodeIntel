"""OpenTelemetry configuration parsing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.otel import observability_config_from_settings


def test_observability_config_from_settings_uses_default_service_name() -> None:
    """Ensure the default service name is applied."""
    settings = ObservabilitySettings(enabled=True)
    config = observability_config_from_settings(settings, default_service_name="default")
    assert config.service_name == "default"


def test_runtime_settings_parses_sampler_and_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure sampler and OTLP settings parse from environment variables."""
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.5")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    settings = load_runtime_settings().observability

    assert settings.traces_sampler == "parentbased_traceidratio"
    assert settings.traces_sampler_arg == 0.5
    assert settings.otlp.endpoint == "http://collector:4318"
    assert settings.otlp.protocol == "http/protobuf"


def test_runtime_settings_reads_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure runtime settings read the experimental config file."""
    config_path = tmp_path / "otel.yaml"
    config_path.write_text("receivers: {}", encoding="utf-8")
    monkeypatch.setenv("OTEL_EXPERIMENTAL_CONFIG_FILE", str(config_path))

    settings = load_runtime_settings().observability

    assert settings.config_file == config_path
