"""
Subsystem analytics orchestrator and helpers.

Split across focused submodules:
- affinity: module graphs and clustering
- edge_stats: fan-in/out and edge counts per subsystem
- risk: risk aggregation utilities
- materialize: end-to-end build pipeline
"""

from __future__ import annotations

from codeintel.analytics.subsystems.materialize import (
    SubsystemCacheBenchmark,
    benchmark_subsystem_cache_reads,
    build_subsystems,
    refresh_subsystem_caches,
    refresh_subsystem_coverage_cache,
    refresh_subsystem_profile_cache,
)

__all__ = [
    "SubsystemCacheBenchmark",
    "benchmark_subsystem_cache_reads",
    "build_subsystems",
    "refresh_subsystem_caches",
    "refresh_subsystem_coverage_cache",
    "refresh_subsystem_profile_cache",
]
