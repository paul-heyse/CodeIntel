"""Centralized pytest option helpers for the test suite."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def register_pytest_options(parser: pytest.Parser) -> None:
    """Register custom pytest CLI options."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden files instead of comparing.",
    )


def apply_pytest_options(config: pytest.Config) -> None:
    """Apply pytest options to process-level configuration."""
    if config.getoption("--update-golden"):
        os.environ["UPDATE_GOLDEN"] = "1"


__all__ = ["apply_pytest_options", "register_pytest_options"]
