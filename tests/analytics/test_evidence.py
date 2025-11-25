"""Unit tests for evidence helpers."""

from __future__ import annotations

import pytest

from codeintel.analytics.evidence import (
    EvidenceCollector,
    EvidenceSample,
    validate_evidence_samples,
)


def test_evidence_collector_deduplicates_and_truncates() -> None:
    """EvidenceCollector should de-dup and enforce max_samples."""
    max_samples = 2
    collector = EvidenceCollector(max_samples=max_samples)
    sample = EvidenceSample(
        path="file.py",
        lineno=1,
        end_lineno=1,
        snippet="line",
        details={"kind": "example"},
    )
    collector.add(sample)
    collector.add(sample)
    collector.add(
        EvidenceSample(
            path="file.py",
            lineno=2,
            end_lineno=2,
            snippet="other",
            details={},
        )
    )
    collector.add(
        EvidenceSample(
            path="file.py",
            lineno=3,
            end_lineno=3,
            snippet="extra",
            details={},
        )
    )
    evidence = collector.to_dicts()
    if len(evidence) != max_samples:
        pytest.fail("Collector should cap evidence at max_samples")
    if evidence[0]["details"] != {"kind": "example"}:
        pytest.fail("First evidence entry should retain details")


def test_validate_evidence_samples_rejects_invalid_payloads() -> None:
    """validate_evidence_samples should enforce required keys and types."""
    valid = [
        {
            "path": "file.py",
            "lineno": 1,
            "end_lineno": 1,
            "snippet": "line",
            "details": {},
            "tags": ["decorator"],
        }
    ]
    validate_evidence_samples(valid)
    with pytest.raises(ValueError, match="missing required field"):
        validate_evidence_samples([{"path": "file.py"}])
    with pytest.raises(TypeError, match="must be a string"):
        validate_evidence_samples(
            [
                {
                    "path": 1,
                    "lineno": 1,
                    "end_lineno": 1,
                    "snippet": "line",
                }
            ]
        )
