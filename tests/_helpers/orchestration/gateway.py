"""Re-export gateway helpers for orchestration module consistency.

This module re-exports symbols from tests._helpers.gateway for use within
the orchestration package and for external imports via orchestration.
"""

from __future__ import annotations

from tests._helpers.gateway import (
    DuckDBConnection,
    GatewayFactory,
    ScopeRecordingQuery,
    build_duckdb_backend,
    build_duckdb_query_service,
    build_scope_parsing_service,
    memory_con_with_macros,
    seed_tables,
)

__all__ = [
    "DuckDBConnection",
    "GatewayFactory",
    "ScopeRecordingQuery",
    "build_duckdb_backend",
    "build_duckdb_query_service",
    "build_scope_parsing_service",
    "memory_con_with_macros",
    "seed_tables",
]
