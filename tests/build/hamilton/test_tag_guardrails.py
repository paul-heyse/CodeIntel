"""Tests for Hamilton node-tag invariants.

This module ensures the build driver graph satisfies the invariants enforced by
``codeintel.build.hamilton.validate`` (tags, produced outputs, and compute purity).
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.graph_validation import (
    validate_graph,
    validation_result_to_json,
)
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


def test_validate_graph_has_no_errors(hamilton_runtime: HamiltonRuntimeBundle) -> None:
    """Ensure runtime bundle produces a tag-clean graph."""
    result = validate_graph(runtime=hamilton_runtime)
    if result.has_errors:
        payload = validation_result_to_json(result)
        pytest.fail(f"Graph validation errors:\n{payload}")
