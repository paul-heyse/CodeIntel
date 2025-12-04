"""Data models computation module.

This package provides pure computation functions for analyzing data model usage.
"""

from __future__ import annotations

from codeintel.analytics.compute.data_models.usage import (
    ModelIndex,
    ModelInfo,
    ModelUsageArtifacts,
    ModelUsageResult,
    compute_data_model_usage,
)

__all__ = [
    "ModelIndex",
    "ModelInfo",
    "ModelUsageArtifacts",
    "ModelUsageResult",
    "compute_data_model_usage",
]
