"""Tests for tag taxonomy guardrail output payloads."""

from __future__ import annotations

import json

from codeintel.build.hamilton.validate import (
    GraphValidationIssue,
    GraphValidationResult,
    validation_result_to_json,
)


def test_validation_result_includes_node_provenance() -> None:
    """Include node provenance metadata in serialized validation results."""
    issue = GraphValidationIssue(
        severity="error",
        code="missing_tag",
        message="Materialize node missing target tag",
        node="t__missing",
    )
    result = GraphValidationResult(errors=(issue,), warnings=())
    payload = validation_result_to_json(
        result,
        node_provenance={"t__missing": {"module": "codeintel_targets.example"}},
    )
    decoded = json.loads(payload)
    assert decoded["errors"][0]["node_provenance"]["module"] == "codeintel_targets.example"
