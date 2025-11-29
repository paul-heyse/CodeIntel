"""Shared value objects used by analytics profile recipes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from codeintel.storage.gateway import DuckDBConnection


@dataclass(frozen=True)
class ProfileInputs:
    """Snapshot handle for profile computations."""

    con: DuckDBConnection
    repo: str
    commit: str
    created_at: datetime
    slow_test_threshold_ms: float


FunctionProfileInputs = ProfileInputs
FileProfileInputs = ProfileInputs
ModuleProfileInputs = ProfileInputs


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
class CoverageSummary:
    """Coverage and test-derived metrics for a function."""

    function_goid_h128: int
    executable_lines: int
    covered_lines: int
    coverage_ratio: float | None
    tested: bool
    untested_reason: str | None
    tests_touching: int
    failing_tests: int
    slow_tests: int
    flaky_tests: int
    last_test_status: str | None
    dominant_test_status: str | None


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


@dataclass(frozen=True)
class FunctionHistoryView:
    """History and churn metrics."""

    function_goid_h128: int
    created_in_commit: str | None
    created_at_history: datetime | None
    last_modified_commit: str | None
    last_modified_at: datetime | None
    age_days: int | None
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    churn_score: float | None
    stability_bucket: str | None
