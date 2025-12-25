"""CLI telemetry tests."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pytest
from cyclopts import App, Parameter

from codeintel.cli.errors import OutputFormat
from codeintel.core.config.settings import (
    BuildSettings,
    CliSettings,
    HamiltonExecutionSettings,
    ObservabilitySettings,
    ServingSettings,
)
from codeintel.core.runtime import RuntimeSettings
from codeintel.observability import cli as cli_observability


def test_run_cli_with_telemetry_calls_shutdown_on_parse_error() -> None:
    """Ensure shutdown is invoked when CLI parsing fails."""
    app = App(name="demo")

    @app.command
    def demo(value: Annotated[int, Parameter()]) -> None:
        _ = value

    calls = {"shutdown": 0}

    def _shutdown() -> None:
        calls["shutdown"] += 1

    def _bootstrap_cli(_verbosity: int) -> None:
        return None

    runtime_settings = RuntimeSettings(
        build=BuildSettings(engine_version="test"),
        cli=CliSettings(),
        execution=HamiltonExecutionSettings(),
        serving=ServingSettings(serve_dir=Path("serve")),
        observability=ObservabilitySettings(cli_enabled=False),
    )
    deps = cli_observability.RunCliTelemetryDeps(
        load_settings=lambda: runtime_settings,
        bootstrap=_bootstrap_cli,
        shutdown=_shutdown,
        get_observability=cli_observability.get_observability,
    )

    with pytest.raises(SystemExit):
        cli_observability.run_cli_with_telemetry(
            app,
            output_format=OutputFormat.TEXT,
            argv=("demo", "--value", "not-an-int"),
            deps=deps,
        )

    assert calls["shutdown"] == 1
