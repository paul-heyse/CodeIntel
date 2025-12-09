"""Shared option objects for test environments.

Provides lightweight dataclasses for gateway and environment creation so helpers
can accept a single options object instead of numerous keyword parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO


@dataclass(frozen=True)
class GatewayOptions:
    """Configuration for building a test StorageGateway.

    Attributes
    ----------
    file_backed
        Whether to use a file-backed database instead of in-memory.
    db_path
        Path to the database file (required if file_backed is True).
    repo
        Repository identifier for the gateway.
    commit
        Commit hash for the gateway.
    apply_schema
        Whether to apply database schema on creation.
    ensure_views
        Whether to ensure views are created.
    validate_schema
        Whether to validate the schema after creation.
    strict_schema
        Whether to enforce strict schema mode (enables views/validation).
    """

    file_backed: bool = False
    db_path: Path | None = None
    repo: str | None = None
    commit: str | None = None
    apply_schema: bool = True
    ensure_views: bool = True
    validate_schema: bool = True
    strict_schema: bool = True


@dataclass(frozen=True)
class EnvOptions:
    """Configuration for building a full TestContext."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    file_backed: bool = False
    repo_root: Path | None = None
    build_dir: Path | None = None
    db_path: Path | None = None


__all__ = [
    "EnvOptions",
    "GatewayOptions",
]
