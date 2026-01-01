"""Data structures for AST-derived function features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class IoFlags:
    """Flags describing IO usage within a span or function."""

    uses_network: bool = False
    uses_db: bool = False
    uses_filesystem: bool = False
    uses_subprocess: bool = False

    @property
    def io_bound(self) -> bool:
        """Return True when any IO flag is set."""
        return self.uses_network or self.uses_db or self.uses_filesystem or self.uses_subprocess


@dataclass(frozen=True)
class FunctionAstFeatures:
    """
    Semantic feature vector derived from a single function AST.

    These are inputs to higher-level classifiers (roles, entrypoints, etc.), not classifications
    themselves.
    """

    goid: int
    rel_path: str
    qualname: str
    is_async: bool

    decorators: tuple[str, ...]
    imports: Mapping[str, str]
    libraries_used: frozenset[str]

    io_flags: IoFlags
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool

    http_client_libs: frozenset[str]
    http_server_libs: frozenset[str]
    db_libs: frozenset[str]
    message_libs: frozenset[str]

    config_read_count: int
    feature_flag_count: int

    extra: Mapping[str, object] = field(default_factory=dict)


__all__ = [
    "FunctionAstFeatures",
    "IoFlags",
]
