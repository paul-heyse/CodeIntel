"""Evidence collection helpers for analytics compute."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import cast

import pytest

from codeintel.analytics.compute.evidence.collection import (
    EvidenceCollector,
    EvidenceSample,
    validate_evidence_samples,
)
from tests._helpers.assertions import assert_mapping_value


def test_evidence_sample_from_ast_and_to_dict() -> None:
    """Create evidence samples from AST nodes with snippet capture."""
    source = "def sample() -> None:\n    return None\n"
    node = ast.parse(source).body[0]
    sample = EvidenceSample.from_ast(
        path="module.py",
        lines=source.splitlines(),
        node=node,
        details={"kind": "function"},
        tags=("example",),
    )
    serialized = sample.to_dict()
    assert serialized["path"] == "module.py"
    assert serialized["lineno"] == 1
    snippet = assert_mapping_value(serialized, "snippet", str)
    assert "def sample" in snippet
    assert assert_mapping_value(serialized, "details", dict) == {"kind": "function"}
    assert assert_mapping_value(serialized, "tags", list) == ["example"]


def test_evidence_collector_deduplicates_and_caps() -> None:
    """Ensure collector enforces max samples and deduplicates by key."""
    collector = EvidenceCollector(max_samples=1)
    first = EvidenceSample(path="a.py", lineno=1, end_lineno=1, snippet="first")
    duplicate = EvidenceSample(path="a.py", lineno=1, end_lineno=1, snippet="first")
    second = EvidenceSample(path="a.py", lineno=2, end_lineno=2, snippet="second")

    collector.add(first)
    collector.add(duplicate)
    collector.add(second)

    assert collector.samples == [first]


def test_evidence_collector_add_from_ast_and_extend() -> None:
    """Add samples from AST and extend from existing samples."""
    source = "value = 1\nvalue = value + 1\n"
    tree = ast.parse(source)
    collector = EvidenceCollector(max_samples=3)
    collector.add_from_ast(
        path="math.py",
        lines=source.splitlines(),
        node=tree.body[0],
        details={"assign": True},
        tags=("assign",),
    )
    collector.extend(
        [
            EvidenceSample(path="math.py", lineno=2, end_lineno=2, snippet="value = value + 1"),
            EvidenceSample(path="math.py", lineno=3, end_lineno=3, snippet="value = value + 2"),
        ]
    )
    dicts = collector.to_dicts()
    assert len(dicts) == 3
    assert dicts[0]["details"] == {"assign": True}
    assert dicts[1]["snippet"] == "value = value + 1"


def test_validate_evidence_samples_errors() -> None:
    """Validate payloads raise on bad inputs."""
    with pytest.raises(TypeError):
        validate_evidence_samples([{"path": 1, "lineno": 1, "end_lineno": 1, "snippet": "x"}])
    with pytest.raises(ValueError, match="missing required field"):
        validate_evidence_samples([{"path": "p", "lineno": 1, "end_lineno": 1}])
    invalid_sample = cast("Mapping[str, object]", "not a mapping")
    with pytest.raises(TypeError):
        validate_evidence_samples([invalid_sample])
