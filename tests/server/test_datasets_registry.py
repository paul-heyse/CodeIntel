"""Dataset registry should expose function_validation."""

from __future__ import annotations

import pytest

from codeintel.serving.http.datasets import build_dataset_registry


def test_function_validation_dataset_present() -> None:
    """Ensure function_validation is exposed via the dataset registry."""
    registry = build_dataset_registry(include_docs_views="exclude")
    if "function_validation" not in registry:
        pytest.fail("function_validation missing from dataset registry")
    if registry["function_validation"] != "analytics.function_validation":
        pytest.fail(f"unexpected mapping: {registry['function_validation']}")
