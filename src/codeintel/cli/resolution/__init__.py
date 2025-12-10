"""Centralized project and runtime resolution for CLI operations.

This package provides:

- ``resolve_from_params``: Primary API for runtime resolution from params dict
- ``RuntimeParams``: Canonical runtime parameters type
- ``BackendFlags``: Graph backend configuration
- ``ResolvedRuntime``: Immutable result of runtime resolution
- ``ResolutionError``: Exception for resolution failures

Examples
--------
>>> from codeintel.cli.resolution import resolve_from_params
>>> runtime = resolve_from_params({"project_root": "."})  # doctest: +SKIP
>>> runtime.db_path  # doctest: +SKIP
PosixPath('build/db/codeintel.duckdb')
"""

from __future__ import annotations

from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.params import BackendFlags, RuntimeParams
from codeintel.cli.resolution.runtime import resolve_from_params
from codeintel.cli.resolution.types import ResolvedRuntime

__all__ = [
    "BackendFlags",
    "ResolutionError",
    "ResolvedRuntime",
    "RuntimeParams",
    "resolve_from_params",
]
