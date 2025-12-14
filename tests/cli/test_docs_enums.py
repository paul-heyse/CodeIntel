"""CLI enum validation coverage for docs export.

Uses xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.errors import CLI_EXIT_VALIDATION
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from tests._helpers.cli import CLIContext

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


def test_docs_export_invalid_validation_mode(cli_ctx: CLIContext) -> None:
    """Invalid validation-mode yields exit 1 with friendly message."""
    result = run_cli(
        ["docs", "export", "--validation-mode", "invalid"],
        env=cli_ctx.env,
        cwd=cli_ctx.repo_root,
    )

    expect_equal(result.exit_code, CLI_EXIT_VALIDATION)
    expect_in('Invalid value for "--validation-mode"', result.stderr)
