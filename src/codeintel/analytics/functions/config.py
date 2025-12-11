"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
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
    """Configuration options for function analytics computation.

    This dataclass serves as the typed options model for the function_metrics
    plugin. It contains both:
    - Config-driven fields (settable via profiles/config files)
    - Dynamic fields (set at runtime, e.g., AST caches)

    Config-Driven Fields
    --------------------
    These can be set via configuration files or profile overrides:

    include_graph_metrics : bool
        Whether to compute graph-derived metrics (PageRank, centrality).
    include_coverage_metrics : bool
        Whether to join coverage data for functions.
    complexity_threshold : int
        Maximum cyclomatic complexity before flagging as too complex.
    type_strictness : str
        Type checking strictness level ("strict", "standard", "lenient").
    scope_paths : list[str] | None
        If set, only process functions in these paths.

    Dynamic Fields
    --------------
    These are set at execution time, not from configuration:

    validation_reporter : FunctionValidationReporter | None
        Optional reporter for validation issues.
    function_ast_map : dict[int, FunctionAst] | None
        Pre-built AST map from AstProvider.
    missing_function_goids : set[int]
        GOIDs that could not be parsed.
    """

    include_graph_metrics: bool = True
    include_coverage_metrics: bool = True
    complexity_threshold: int = 10
    type_strictness: str = "standard"
    scope_paths: list[str] | None = None

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

    def should_compute_graph_metrics(self) -> bool:
        """Check if graph metrics should be computed.

        Returns
        -------
        bool
            True if graph metrics are enabled.
        """
        return self.include_graph_metrics

    def should_compute_coverage_metrics(self) -> bool:
        """Check if coverage metrics should be computed.

        Returns
        -------
        bool
            True if coverage metrics are enabled.
        """
        return self.include_coverage_metrics


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsStepConfig
    cache: dict[str, ParsedModule | None]
    span_index: dict[int, SourceSpan]
    reporter: FunctionValidationReporter
    ctx: ProcessContext
