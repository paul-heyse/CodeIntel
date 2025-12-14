"""Data models computation module.

This package provides pure computation functions for analyzing data model usage.

For Hamilton native execution, use `build_data_model_usage_rows` to get row
tuples, then materialize with `materialize_rows`.
"""

from __future__ import annotations

from codeintel.analytics.compute.data_models.usage import (
    DATA_MODEL_USAGE_COLS,
    ModelIndex,
    ModelInfo,
    ModelUsageArtifacts,
    ModelUsageResult,
    build_data_model_usage_rows,
    compute_data_model_usage,
)

__all__ = [
    "DATA_MODEL_USAGE_COLS",
    "ModelIndex",
    "ModelInfo",
    "ModelUsageArtifacts",
    "ModelUsageResult",
    "build_data_model_usage_rows",
    "compute_data_model_usage",
]
