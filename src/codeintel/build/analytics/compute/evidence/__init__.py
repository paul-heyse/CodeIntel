"""Evidence computation module.

This package provides data structures and utilities for collecting
and representing evidence from analytics outputs.
"""

from __future__ import annotations

from codeintel.build.analytics.compute.evidence.collection import (
    EvidenceBundle,
    EvidenceCollector,
    EvidenceSample,
    validate_evidence_samples,
)

__all__ = [
    "EvidenceBundle",
    "EvidenceCollector",
    "EvidenceSample",
    "validate_evidence_samples",
]
