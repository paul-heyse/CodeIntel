"""CLI telemetry tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pytest
from cyclopts import App, Parameter

from codeintel.cli.errors import OutputFormat
from codeintel.core.config.settings import ObservabilitySettings
from codeintel.observability import cli as cli_observability


@dataclass(frozen=True)
class _RuntimeSettingsStub:
    observability: ObservabilitySettings


def test_run_cli_with_telemetry_calls_shutdown_on_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(cli_observability, "shutdown_observability", _shutdown)
    monkeypatch.setattr(cli_observability, "bootstrap_cli", _bootstrap_cli)
    monkeypatch.setattr(
        cli_observability,
        "load_runtime_settings",
        lambda: _RuntimeSettingsStub(observability=ObservabilitySettings(cli_enabled=False)),
    )

    with pytest.raises(SystemExit):
        cli_observability.run_cli_with_telemetry(
            app,
            output_format=OutputFormat.TEXT,
            argv=("demo", "--value", "not-an-int"),
        )

    assert calls["shutdown"] == 1
