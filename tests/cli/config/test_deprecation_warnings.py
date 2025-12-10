"""Tests for deprecation warnings in CLI functions."""

from __future__ import annotations

import contextlib
import warnings

from codeintel.cli.cyclopts_common import (
    RuntimeCLI,
    RuntimeCliError,
    build_runtime_from_cli,
    runtime_cli_to_options,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_true,
)


def test_build_runtime_from_cli_warning() -> None:
    """Verify build_runtime_from_cli emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # We expect this to fail since there's no project, so suppress the error
        with contextlib.suppress(RuntimeCliError):
            build_runtime_from_cli(RuntimeCLI())

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        expect_true(len(deprecation_warnings) >= 1)
        expect_in("RuntimeResolver", str(deprecation_warnings[0].message))


def test_runtime_cli_to_options_warning() -> None:
    """Verify runtime_cli_to_options emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        runtime_cli_to_options(RuntimeCLI())

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        expect_true(len(deprecation_warnings) >= 1)
        expect_in("RuntimeParams", str(deprecation_warnings[0].message))


def test_deprecation_warning_stacklevel() -> None:
    """Verify deprecation warnings point to caller, not internal function."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        runtime_cli_to_options(RuntimeCLI())

        # The warning should reference this test file, not cyclopts_common.py
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        expect_true(len(deprecation_warnings) >= 1)
        # Filename should be this test file
        expect_in("test_deprecation_warnings", deprecation_warnings[0].filename)
