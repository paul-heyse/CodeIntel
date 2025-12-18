"""Tests for Hamilton node-tag invariants.

This module ensures the build driver graph satisfies the invariants enforced by
``codeintel.build.hamilton.validate`` (tags, produced outputs, and compute purity).
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.validate import validate_graph, validation_result_to_json


def test_validate_graph_has_no_errors() -> None:
    """Ensure build_driver() produces a tag-clean graph."""
    result = validate_graph()
    if result.has_errors:
        payload = validation_result_to_json(result)
        pytest.fail(f"Graph validation errors:\n{payload}")
