"""CLI env var stability checks for help output."""

from __future__ import annotations

from tests._helpers.cli import assert_success, run_cli


def test_cli_env_vars_include_command_path() -> None:
    """Ensure help output uses command-scoped env vars for shared flags.

    Raises
    ------
    AssertionError
        If expected env vars are missing from help output.
    """
    result = run_cli(["build", "run", "--help"])
    assert_success(result)

    expected = [
        "CODEINTEL_BUILD_RUN_TARGETS",
        "CODEINTEL_BUILD_RUN_ROOT",
        "CODEINTEL_BUILD_RUN_OUTPUT_FORMAT",
        "CODEINTEL_BUILD_RUN_JSON",
        "CODEINTEL_BUILD_RUN_VERBOSE",
    ]
    missing = [name for name in expected if name not in result.stdout]
    if missing:
        message = "Missing env vars in CLI help: " + ", ".join(missing)
        raise AssertionError(message)
