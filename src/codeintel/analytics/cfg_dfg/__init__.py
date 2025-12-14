"""CFG/DFG analytics per function.

For Hamilton native execution, use the pure compute functions:
- ``compute_cfg_metrics_pure`` returns ``CfgMetricsResult`` without writing
- ``compute_dfg_metrics_pure`` returns ``DfgMetricsResult`` without writing

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.cfg_dfg``
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
from codeintel.analytics.cfg_dfg.types import FnContextProtocol

__all__ = [
    "CfgMetricsResult",
    "DfgMetricsResult",
    "FnContextProtocol",
    "compute_cfg_metrics_pure",
    "compute_dfg_metrics_pure",
    "degree_dict",
    "load_function_metadata",
    "parse_block_idx",
]
