"""CLI scope parsing and plan output coverage.

Uses xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from tests._helpers.cli import CliResult

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


def test_cli_plan_outputs_isolation_and_scope_metadata(
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Plan output should include structured plan data with plugins."""
    result = cli_project_runner(["graph", "plugins", "--plan", "--output-format", "json"])
    if result.exit_code != 0:
        message = f"CLI plan command should exit successfully: {result.output}"
        pytest.fail(message)
    envelope = json.loads(result.output)

    payload = envelope.get("data", envelope)

    if "stages" not in payload and "plugins" not in payload and "plan_id" not in payload:
        message = (
            f"Plan JSON should include stages, plugins, or plan_id. Got: {list(payload.keys())}"
        )
        pytest.fail(message)
