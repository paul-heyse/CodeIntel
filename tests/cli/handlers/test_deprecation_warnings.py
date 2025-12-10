"""Tests for deprecation warnings in legacy handler modules."""

from __future__ import annotations

import importlib
import sys
import warnings

from tests._helpers.assertions.expectation_assertions import (
    expect_true,
)


def _test_deprecation_warning(module_name: str, expected_replacement: str) -> None:
    """Test that a module emits deprecation warning on import.

    Parameters
    ----------
    module_name
        Full module name to test.
    expected_replacement
        Expected text in deprecation message about replacement module.
    """
    # Remove module if already imported to force reimport
    if module_name in sys.modules:
        del sys.modules[module_name]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        importlib.import_module(module_name)

        module_short_name = module_name.rsplit(".", maxsplit=1)[-1]
        deprecation_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, DeprecationWarning)
            and module_short_name in str(warning.message)
        ]

        expect_true(
            len(deprecation_warnings) >= 1,
            message=f"Expected deprecation warning for {module_name}",
        )
        expect_true(
            expected_replacement in str(deprecation_warnings[0].message),
            message=f"Expected '{expected_replacement}' in warning message",
        )


def test_datasets_handlers_deprecation_warning() -> None:
    """Verify datasets_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.datasets_handlers",
        "handlers.datasets",
    )


def test_docs_handlers_deprecation_warning() -> None:
    """Verify docs_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.docs_handlers",
        "handlers.docs",
    )


def test_graphs_handlers_deprecation_warning() -> None:
    """Verify graphs_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.graphs_handlers",
        "handlers.graphs",
    )


def test_storage_handlers_deprecation_warning() -> None:
    """Verify storage_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.storage_handlers",
        "handlers.storage",
    )


def test_ops_handlers_deprecation_warning() -> None:
    """Verify ops_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.ops_handlers",
        "handlers.ops",
    )


def test_build_handlers_deprecation_warning() -> None:
    """Verify build_handlers.py emits deprecation warning on import."""
    _test_deprecation_warning(
        "codeintel.cli.build_handlers",
        "handlers.build",
    )
