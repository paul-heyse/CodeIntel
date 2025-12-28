"""Data models computation module.

This package provides pure computation functions for analyzing data model usage.

For Hamilton native execution, use ``build_data_model_usage_rows`` to get a
LazyFrame, then materialize with the columnar materializers.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.data_models``
"""

from __future__ import annotations

from codeintel.analytics.compute.data_models.usage import (
    DATA_MODEL_USAGE_COLS,
    DATA_MODEL_USAGE_TABLE_KEY,
    ModelIndex,
    ModelInfo,
    ModelUsageArtifacts,
    ModelUsageResult,
    build_data_model_usage_rows,
)

__all__ = [
    "DATA_MODEL_USAGE_COLS",
    "DATA_MODEL_USAGE_TABLE_KEY",
    "ModelIndex",
    "ModelInfo",
    "ModelUsageArtifacts",
    "ModelUsageResult",
    "build_data_model_usage_rows",
]
