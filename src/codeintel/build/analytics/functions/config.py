"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.parsing import ParsedModule, SourceSpan
    from codeintel.core.validation.reporters import FunctionValidationReporter


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    snapshot: SnapshotRef
    now: datetime


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Runtime options for function analytics computation.

    These options are provided at execution time to reuse parsed AST data
    and collect validation findings.
    """

    validation_reporter: FunctionValidationReporter | None = None
    function_ast_map: dict[int, FunctionAst] | None = None
    missing_function_goids: set[int] = field(default_factory=set)

    def get_ast_map(self) -> dict[int, FunctionAst]:
        """Return the function AST map.

        Returns
        -------
        dict[int, FunctionAst]
            The AST map, empty if not provided.
        """
        if self.function_ast_map is not None:
            return self.function_ast_map
        return {}

    def get_missing_goids(self) -> set[int]:
        """Return the set of missing GOIDs.

        Returns
        -------
        set[int]
            The missing GOIDs set.
        """
        return self.missing_function_goids

    def has_ast_data(self) -> bool:
        """Check if AST data is available.

        Returns
        -------
        bool
            True if AST data is available.
        """
        return self.function_ast_map is not None


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    snapshot: SnapshotRef
    cache: dict[str, ParsedModule | None]
    span_index: dict[int, SourceSpan]
    reporter: FunctionValidationReporter
    ctx: ProcessContext
