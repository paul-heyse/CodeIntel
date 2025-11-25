"""
CFG/DFG analytics per function.

The logic from `analytics.cfg_dfg_metrics` is now organized into submodules,
with `compute_cfg_metrics` and `compute_dfg_metrics` exported here for callers.
"""

from __future__ import annotations

from codeintel.analytics.cfg_dfg.materialize import compute_cfg_metrics, compute_dfg_metrics

__all__ = ["compute_cfg_metrics", "compute_dfg_metrics"]
