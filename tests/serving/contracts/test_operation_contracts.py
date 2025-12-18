"""Serving operation-contract checks."""

from __future__ import annotations

import pytest

from codeintel.serving.contracts.check_operation_contracts import main


def test_operation_contracts_check_passes() -> None:
    """Serving public surfaces remain coherent."""
    exit_code = main()
    if exit_code != 0:
        pytest.fail(f"Expected operation-contract check to pass, got exit_code={exit_code}")
