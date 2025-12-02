"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.function_ast_cache import FunctionAst
    from codeintel.analytics.parsing.models import ParsedModule, SourceSpan
    from codeintel.analytics.parsing.validation import FunctionValidationReporter
    from codeintel.config import FunctionAnalyticsStepConfig


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    cfg: FunctionAnalyticsStepConfig
    now: datetime


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Optional hooks and cached context for function analytics.

    Attributes
    ----------
    validation_reporter
        Optional reporter for validation issues.
    function_ast_map
        Mapping of GOID to FunctionAst (from AstProvider).
    missing_function_goids
        Set of GOIDs that could not be parsed (from AstProvider).
    """

    validation_reporter: FunctionValidationReporter | None = None
    function_ast_map: dict[int, FunctionAst] | None = None
    missing_function_goids: set[int] = field(default_factory=set)

    def get_ast_map(self) -> dict[int, FunctionAst]:
        """Return the function AST map from either source.

        Returns
        -------
        dict[int, FunctionAst]
            The AST map, empty if neither source is provided.
        """
        if self.context is not None:
            return self.context.function_ast_map
        if self.function_ast_map is not None:
            return self.function_ast_map
        return {}

    def get_missing_goids(self) -> set[int]:
        """Return the set of missing GOIDs from either source.

        Returns
        -------
        set[int]
            The missing GOIDs set.
        """
        if self.context is not None:
            return self.context.missing_function_goids
        return self.missing_function_goids

    def has_ast_data(self) -> bool:
        """Check if AST data is available from any source.

        Returns
        -------
        bool
            True if AST data is available.
        """
        if self.context is not None:
            return bool(self.context.function_ast_map)
        return self.function_ast_map is not None


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsStepConfig
    cache: dict[str, ParsedModule | None]
    span_index: dict[int, SourceSpan]
    reporter: FunctionValidationReporter
    ctx: ProcessContext
