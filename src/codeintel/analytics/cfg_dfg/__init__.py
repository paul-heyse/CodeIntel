"""
CFG/DFG analytics per function.

The logic from `analytics.cfg_dfg_metrics` is now organized into submodules,
with `compute_cfg_metrics` and `compute_dfg_metrics` exported here for callers.

For Hamilton native execution, use the pure compute functions:
- `compute_cfg_metrics_pure` returns `CfgMetricsResult` without writing
- `compute_dfg_metrics_pure` returns `DfgMetricsResult` without writing
"""

from __future__ import annotations

from codeintel.analytics.cfg_dfg.compute import (
    CfgMetricsResult,
    DfgMetricsResult,
    compute_cfg_metrics_pure,
    compute_dfg_metrics_pure,
)
from codeintel.analytics.cfg_dfg.helpers import (
    degree_dict,
    load_function_metadata,
    parse_block_idx,
)
from codeintel.analytics.cfg_dfg.materialize import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.cfg_dfg.types import FnContextProtocol

__all__ = [
    "CfgMetricsResult",
    "DfgMetricsResult",
    "FnContextProtocol",
    "compute_cfg_metrics",
    "compute_cfg_metrics_pure",
    "compute_dfg_metrics",
    "compute_dfg_metrics_pure",
    "degree_dict",
    "load_function_metadata",
    "parse_block_idx",
]
