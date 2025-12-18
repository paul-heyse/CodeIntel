"""Backend session primitives for storage.

This package is a small, backend-focused layer that owns DuckDB session
lifecycle concerns (connection creation, tuning, attach/export, etc.).

It intentionally avoids importing higher-level storage APIs to keep the
dependency direction clean.
"""

from __future__ import annotations

from codeintel.storage.backend.duckdb_session import DuckDBConnectConfig, DuckDBSession

__all__ = ["DuckDBConnectConfig", "DuckDBSession"]
