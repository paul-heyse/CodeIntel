"""CFG/DFG metrics plugins.

This package provides plugins for computing control-flow graph and
data-flow graph metrics per function.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.cfg_dfg.metrics import CfgDfgMetricsPlugin

__all__ = ["CfgDfgMetricsPlugin"]
