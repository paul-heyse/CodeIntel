"""Config resolver provenance snapshot tests."""

from __future__ import annotations

import json
from pathlib import Path

from codeintel.core.config.settings import ObservabilitySettings, OtlpExporterSettings
from codeintel.observability.runtime import ConfigResolver


def test_config_resolver_snapshot_matches_golden() -> None:
    """Resolved config snapshots should match the golden fixture."""
    settings = ObservabilitySettings(
        enabled=True,
        service_name=None,
        service_version="1.2.3",
        deployment_environment="dev",
        resource_attributes=(("region", "us-east-1"),),
        propagators=("tracecontext",),
        config_file=Path("otel.yaml"),
        export_traces=True,
        export_metrics=True,
        export_logs=True,
        prometheus_enabled=False,
        logs_auto_instrument=True,
        log_correlation=True,
        logs_trace_filter=True,
        otlp=OtlpExporterSettings(endpoint="http://collector:4317"),
    )
    resolver = ConfigResolver(default_service_name="codeintel-default")
    resolved = resolver.resolve(
        settings,
        overrides={"metrics.prometheus_enabled": True},
    )
    payload = resolved.to_payload()
    fixture_path = Path(__file__).parent / "fixtures" / "config_resolver_snapshot.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload == expected
