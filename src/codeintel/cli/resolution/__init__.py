"""Centralized project and runtime resolution for CLI operations.

This package provides:

- ``RuntimeResolver``: Resolve project runtime from CLI parameters
- ``RuntimeParams``: Canonical runtime parameters type
- ``BackendFlags``: Graph backend configuration
- ``GatewayManager``: Manage storage gateway lifecycle
- ``ResolvedRuntime``: Immutable result of runtime resolution
- ``ResolutionError``: Exception for resolution failures

Examples
--------
>>> from codeintel.cli.resolution import resolve_runtime
>>> runtime = resolve_runtime(ctx)  # doctest: +SKIP
>>> runtime.db_path  # doctest: +SKIP
PosixPath('build/db/codeintel.duckdb')
"""

from __future__ import annotations

from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.gateway import (
    GatewayManager,
    open_gateway_for_context,
)
from codeintel.cli.resolution.params import BackendFlags, RuntimeParams
from codeintel.cli.resolution.runtime import RuntimeResolver, resolve_runtime
from codeintel.cli.resolution.types import ResolvedRuntime

__all__ = [
    "BackendFlags",
    "GatewayManager",
    "ResolutionError",
    "ResolvedRuntime",
    "RuntimeParams",
    "RuntimeResolver",
    "open_gateway_for_context",
    "resolve_runtime",
]
