"""Shared function analytics nodes for typing and validation outputs."""

from __future__ import annotations

from hamilton.function_modifiers import cache

from codeintel.build.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_analytics_result_from_tabular,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)


@cache()
def function_analytics_result(
    env: BuildEnv, q__core__goids: InferableTabularInput
) -> FunctionAnalyticsResult:
    """Compute function typing rows from core.goids.

    Returns
    -------
    FunctionAnalyticsResult
        Types rows plus validation reporter.
    """
    return compute_function_analytics_result_from_tabular(q__core__goids, env.snapshot)


__all__ = ["function_analytics_result"]
