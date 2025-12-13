"""Shared constants for the storage layer.

This module centralizes constants used across multiple storage modules
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

__all__ = ["DUCKDB_DIALECT", "SCHEMAS"]

DUCKDB_DIALECT = "duckdb"
"""SQLGlot dialect identifier for DuckDB SQL generation."""

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
"""Database schema names used in the CodeIntel data warehouse."""
