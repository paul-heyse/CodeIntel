"""Compatibility shim for result_types module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.core.result_types`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.result_types import OperationListResult

    # New (preferred):
    from codeintel.cli.core.result_types import OperationListResult
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.result_types' is deprecated. "
    "Use 'codeintel.cli.core.result_types' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.core.result_types import (
    BuildExecutionResult,
    BuildHistoryResult,
    BuildRunResult,
    BuildStatusResult,
    BuildTargetInfo,
    ConfigShowResult,
    DatasetDescribeResult,
    DatasetListResult,
    DatasetVerifyResult,
    DocsGenerateResult,
    DocsStatusResult,
    DryRunResult,
    DryRunStep,
    GraphPlanResult,
    GraphPluginInfo,
    GraphPluginsResult,
    GraphQueryResult,
    GraphStatsResult,
    HistoryDetailResult,
    HistoryListResult,
    IdeConfigResult,
    IdeStatusResult,
    OperationCallResult,
    OperationListResult,
    StorageQueryResult,
    StorageStatusResult,
    SubsystemDetailResult,
    SubsystemListResult,
)

__all__ = [
    "BuildExecutionResult",
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "BuildTargetInfo",
    "ConfigShowResult",
    "DatasetDescribeResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "DocsGenerateResult",
    "DocsStatusResult",
    "DryRunResult",
    "DryRunStep",
    "GraphPlanResult",
    "GraphPluginInfo",
    "GraphPluginsResult",
    "GraphQueryResult",
    "GraphStatsResult",
    "HistoryDetailResult",
    "HistoryListResult",
    "IdeConfigResult",
    "IdeStatusResult",
    "OperationCallResult",
    "OperationListResult",
    "StorageQueryResult",
    "StorageStatusResult",
    "SubsystemDetailResult",
    "SubsystemListResult",
]
