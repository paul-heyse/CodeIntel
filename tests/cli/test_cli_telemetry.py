"""CLI telemetry tests."""

from __future__ import annotations

from dataclasses import dataclass
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
from tests._helpers.env import temporary_env


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

    with temporary_env(CODEINTEL_TEST_TELEMETRY_MODE="inherit"), pytest.raises(SystemExit):
        cli_observability.run_cli_with_telemetry(
            app,
            output_format=OutputFormat.TEXT,
            argv=("demo", "--value", "not-an-int"),
            deps=deps,
        )

    assert calls["shutdown"] == 1


@dataclass(frozen=True, slots=True)
class _DummyFlags:
    verbose: int = 1
    json: bool = False
    run_context: cli_observability.RunContext | None = None


def test_flatten_arg_names_includes_shared_flags() -> None:
    """Ensure shared flags are flattened into argument names."""
    arguments = {"flags": _DummyFlags(), "target": "modules"}
    names = cli_observability.flatten_arg_names(arguments, ignored_names=set())
    assert "flags.verbose" in names
    assert "flags.json" in names
    assert "flags.run_context" not in names
    assert "target" in names


def test_normalize_allowlist_expands_flags_prefix() -> None:
    """Ensure allowlist expansion includes shared flag prefixes."""
    allowlist = cli_observability.normalize_allowlist(("verbose", "target"))
    assert "flags.verbose" in allowlist
    assert "verbose" in allowlist
