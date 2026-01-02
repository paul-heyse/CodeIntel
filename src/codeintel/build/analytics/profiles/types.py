"""Shared value objects used by analytics profile recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl


@dataclass(frozen=True)
class FunctionProfileInputs:
    """Snapshot handle for function profile computations."""

    repo: str
    commit: str
    created_at: datetime
    slow_test_threshold_ms: float
    function_metrics: pl.DataFrame
    function_types: pl.DataFrame
    modules: pl.DataFrame
    typedness: pl.DataFrame
    diagnostics: pl.DataFrame
    goid_risk_factors: pl.DataFrame
    graph_metrics_functions: pl.DataFrame
    function_effects: pl.DataFrame
    function_contracts: pl.DataFrame
    semantic_roles_functions: pl.DataFrame
    docstrings: pl.DataFrame
    hotspots: pl.DataFrame
    call_graph_edges: pl.DataFrame
    call_graph_nodes: pl.DataFrame


@dataclass(frozen=True)
class FunctionProfileFrames:
    """Frame bundle required to build function profile inputs."""

    function_metrics: pl.DataFrame
    function_types: pl.DataFrame
    modules: pl.DataFrame
    typedness: pl.DataFrame
    diagnostics: pl.DataFrame
    goid_risk_factors: pl.DataFrame
    graph_metrics_functions: pl.DataFrame
    function_effects: pl.DataFrame
    function_contracts: pl.DataFrame
    semantic_roles_functions: pl.DataFrame
    docstrings: pl.DataFrame
    hotspots: pl.DataFrame
    call_graph_edges: pl.DataFrame
    call_graph_nodes: pl.DataFrame


@dataclass(frozen=True)
class FileProfileInputs:
    """Snapshot handle for file profile computations."""

    repo: str
    commit: str
    created_at: datetime
    function_profile: pl.DataFrame
    ast_metrics: pl.DataFrame
    hotspots: pl.DataFrame
    typedness: pl.DataFrame
    static_diagnostics: pl.DataFrame
    modules: pl.DataFrame


@dataclass(frozen=True)
class FileProfileFrames:
    """Frame bundle required to build file profile inputs."""

    function_profile: pl.DataFrame
    ast_metrics: pl.DataFrame
    hotspots: pl.DataFrame
    typedness: pl.DataFrame
    static_diagnostics: pl.DataFrame
    modules: pl.DataFrame


@dataclass(frozen=True)
class ModuleProfileInputs:
    """Snapshot handle for module profile computations."""

    repo: str
    commit: str
    created_at: datetime
    modules: pl.DataFrame
    function_profile: pl.DataFrame
    file_profile: pl.DataFrame
    import_graph_edges: pl.DataFrame
    semantic_roles_modules: pl.DataFrame


@dataclass(frozen=True)
class ModuleProfileFrames:
    """Frame bundle required to build module profile inputs."""

    modules: pl.DataFrame
    function_profile: pl.DataFrame
    file_profile: pl.DataFrame
    import_graph_edges: pl.DataFrame
    semantic_roles_modules: pl.DataFrame


@dataclass(frozen=True)
class FunctionBaseInfo:
    """Static function metadata pulled from symbol tables and metrics."""

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    module: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    complexity_bucket: str | None
    param_count: int
    positional_params: int
    keyword_params: int
    vararg: bool
    kwarg: bool
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    total_params: int
    annotated_params: int
    return_type: str | None
    param_types: object
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    file_typed_ratio: float | None
    static_error_count: int
    has_static_errors: bool


@dataclass(frozen=True)
class FunctionRiskView:
    """Risk-level attributes for a function."""

    function_goid_h128: int
    risk_score: float
    risk_level: str | None
    hotspot_score: float | None
    tags: object
    owners: object


@dataclass(frozen=True)
class FunctionGraphFeatures:
    """Call-graph degree metrics used by profiles."""

    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_edge_in_count: int
    call_edge_out_count: int
    call_is_leaf: bool
    call_is_entrypoint: bool
    call_is_public: bool


@dataclass(frozen=True)
class FunctionEffectsView:
    """Effect summaries inferred from static analysis."""

    function_goid_h128: int
    is_pure: bool
    uses_io: bool
    touches_db: bool
    uses_time: bool
    uses_randomness: bool
    modifies_globals: bool
    modifies_closure: bool
    spawns_threads_or_tasks: bool
    has_transitive_effects: bool
    purity_confidence: float | None


@dataclass(frozen=True)
class FunctionContractView:
    """Contract metadata derived from function_contracts."""

    function_goid_h128: int
    param_nullability_json: object
    return_nullability: str | None
    has_preconditions: bool
    has_postconditions: bool
    has_raises: bool
    contract_confidence: float | None


@dataclass(frozen=True)
class FunctionRoleView:
    """Semantic role metadata."""

    function_goid_h128: int
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: object


@dataclass(frozen=True)
class FunctionDocView:
    """Docstring-derived views."""

    function_goid_h128: int
    doc_short: str | None
    doc_long: str | None
    doc_params: object
    doc_returns: object
