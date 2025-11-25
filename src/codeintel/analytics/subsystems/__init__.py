"""
Subsystem analytics orchestrator and helpers.

The legacy `analytics.subsystems` module has been split into focused submodules:
- affinity: module graphs and clustering
- edge_stats: fan-in/out and edge counts per subsystem
- risk: risk aggregation utilities
- materialize: end-to-end build pipeline
"""

from __future__ import annotations

from codeintel.analytics.subsystems.materialize import build_subsystems

__all__ = ["build_subsystems"]
