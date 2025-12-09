"""Unit-level validation of the scoped help patch helper."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cyclopts_help import build_patched_app
from tests._helpers.assertions.expectation_assertions import expect_in, expect_not_in


@dataclass
class Options:
    """Simple options used to exercise help rendering."""

    flag: Annotated[bool, Parameter(name="--flag", help="Example flag.")] = False
    mode: Annotated[str | None, Parameter(name="--mode", help="Optional mode.")] = None


def _make_app() -> App:
    local_app = App()

    @local_app.command
    def cmd(
        options: Annotated[Options | None, Parameter(name="*")] = None,
    ) -> None:  # pragma: no cover - help only
        _ = options or Options()

    return local_app


def test_patched_app_help_with_missing_metadata() -> None:
    """Patched app should render help without crashing and with readable defaults."""
    app = build_patched_app(_make_app)

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["cmd", "--help"], result_action="return_value", exit_on_error=False, print_error=False)

    output = stdout.getvalue().lower()
    expect_in("usage", output)
    expect_in("--flag", output)
    expect_in("false", output)
    expect_in("--mode", output)
    expect_in("(none)", output)
    expect_not_in("simplenamespace", output)
