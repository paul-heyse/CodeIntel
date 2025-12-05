"""Pipeline orchestration utilities.

This module re-exports pipeline execution utilities from their canonical locations.

For pipeline execution, use::

    from codeintel.pipeline import run_pipeline, FULL_PIPELINE

For history and export operations::

    from codeintel.pipeline.execution.step_runner import (
        run_history_timeseries,
        run_export_docs,
        HistoryTimeseriesParams,
        ExportHooks,
    )

For gateway caching::

    from codeintel.storage.gateway_cache import (
        get_gateway,
        close_gateways,
        gateway_cache_stats,
    )
"""

from __future__ import annotations

# Re-export from canonical locations for backward compatibility
from codeintel.pipeline.execution.runner import run_pipeline
from codeintel.pipeline.execution.step_runner import (
    DEFAULT_BUILD_SUBDIR,
    ExportHooks,
    HistoryTimeseriesParams,
    run_export_docs,
    run_history_timeseries,
)
from codeintel.storage.gateway_cache import (
    close_gateways,
    gateway_cache_stats,
    get_gateway,
)

__all__ = [
    "DEFAULT_BUILD_SUBDIR",
    "ExportHooks",
    "HistoryTimeseriesParams",
    "close_gateways",
    "gateway_cache_stats",
    "get_gateway",
    "run_export_docs",
    "run_history_timeseries",
    "run_pipeline",
]
