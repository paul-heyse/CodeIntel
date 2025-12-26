"""OpenTelemetry configuration parsing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability import otel as otel_module
from codeintel.observability.otel import (
    ObservabilityConfig,
    bootstrap_observability,
    observability_config_from_settings,
    shutdown_observability,
)

SAMPLER_ARG = 0.5


def test_observability_config_from_settings_uses_default_service_name() -> None:
    """Ensure the default service name is applied."""
    settings = ObservabilitySettings(enabled=True)
    config = observability_config_from_settings(settings, default_service_name="default")
    assert config.service_name == "default"


def test_runtime_settings_parses_sampler_and_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure sampler and OTLP settings parse from environment variables."""
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", str(SAMPLER_ARG))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    settings = load_runtime_settings().observability

    assert settings.traces_sampler == "parentbased_traceidratio"
    assert settings.traces_sampler_arg == SAMPLER_ARG
    assert settings.otlp.endpoint == "http://collector:4318"
    assert settings.otlp.protocol == "http/protobuf"


def test_runtime_settings_reads_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure runtime settings read the experimental config file."""
    config_path = tmp_path / "otel.yaml"
    config_path.write_text("receivers: {}", encoding="utf-8")
    monkeypatch.setenv("OTEL_EXPERIMENTAL_CONFIG_FILE", str(config_path))

    settings = load_runtime_settings().observability

    assert settings.config_file == config_path


def test_config_file_overrides_sdk_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config file should override OTEL_SDK_DISABLED."""
    config_path = tmp_path / "otel.yaml"
    config_path.write_text("receivers: {}", encoding="utf-8")
    monkeypatch.setenv("OTEL_EXPERIMENTAL_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    settings = load_runtime_settings().observability

    assert settings.enabled is True


def test_config_file_bootstrap_attaches_log_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Log handler should attach when config file bootstraps observability."""
    pytest.importorskip("opentelemetry.sdk._logs")
    config_path = tmp_path / "otel.yaml"
    config_path.write_text("receivers: {}", encoding="utf-8")

    monkeypatch.setattr(otel_module, "_apply_config_file", lambda _path: True)
    shutdown_observability()

    runtime = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            export_traces=False,
            export_metrics=False,
            export_logs=False,
            config_file=config_path,
        )
    )

    assert runtime.log_handler is not None
    shutdown_observability()
