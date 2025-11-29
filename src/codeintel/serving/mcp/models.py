"""Typed MCP request/response models and error payloads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.mcp.view_utils import normalize_entrypoints_payload

_TIME_WINDOW_LEN = 2


class MappingModel(BaseModel):
    """Base model that exposes mapping-style access for compatibility."""

    model_config = ConfigDict(extra="ignore")

    def __getitem__(self, key: str) -> object:
        """
        Return value for key using model_dump mapping semantics.

        Returns
        -------
        object
            Value associated with the provided key.
        """
        return self.model_dump()[key]

    def get(self, key: str, default: object | None = None) -> object | None:
        """
        Dict-like get helper with a default fallback.

        Returns
        -------
        object | None
            Retrieved value when present, otherwise the provided default.
        """
        return self.model_dump().get(key, default)


class ViewRow(BaseModel):
    """Generic row wrapper for DuckDB view/table results."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> object:
        """
        Allow dict-style access for compatibility with callers expecting mapping semantics.

        Parameters
        ----------
        key : str
            Field name to retrieve.

        Returns
        -------
        object
            Value for the requested field.
        """
        return self.model_dump()[key]

    def get(self, key: str, default: object | None = None) -> object | None:
        """
        Provide a dict-like getter for backward compatibility.

        Parameters
        ----------
        key:
            Field name to retrieve.
        default:
            Value to return when the field is missing.

        Returns
        -------
        object | None
            Value for the requested field or the provided default.
        """
        try:
            return self[key]
        except KeyError:
            return default


class ProblemDetail(BaseModel):
    """Problem Details payload for MCP error responses."""

    type: str = Field(default="about:blank")
    title: str
    detail: str | None = None
    status: int | None = None
    instance: str | None = None
    data: dict[str, object] | None = None


class Message(BaseModel):
    """Structured message attached to responses."""

    code: str
    severity: Literal["info", "warning", "error"]
    detail: str
    context: dict[str, object] | None = None


class ResponseMeta(BaseModel):
    """Response metadata including clamping and messaging."""

    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = Field(default_factory=list)


class GraphPluginDescriptor(BaseModel):
    """Descriptor for available graph metric plugins."""

    name: str
    stage: str
    description: str
    enabled_by_default: bool
    scope_aware: bool | None = None
    supported_scopes: tuple[str, ...] = ()
    requires_isolation: bool | None = None
    isolation_kind: str | None = None
    scope: object | None = None


class GraphPlanSkipped(BaseModel):
    """Skipped plugin entry returned by plan endpoint."""

    name: str
    reason: Literal["disabled"]


class GraphPlanResponse(BaseModel):
    """Resolved graph metric plan including ordering and dependency graph."""

    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[GraphPlanSkipped, ...]
    dep_graph: dict[str, tuple[str, ...]]


class GraphScopePayload(BaseModel):
    """Client-provided scope payload parsed into GraphRunScope."""

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_time_window(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        window = values.get("time_window")
        if window is None:
            return values
        if isinstance(window, (list, tuple)) and len(window) == _TIME_WINDOW_LEN:
            try:
                start = (
                    datetime.fromisoformat(window[0]) if isinstance(window[0], str) else window[0]
                )
                end = datetime.fromisoformat(window[1]) if isinstance(window[1], str) else window[1]
                values["time_window"] = (start, end)
            except (TypeError, ValueError):
                values["time_window"] = None
        else:
            values["time_window"] = None
        return values


def parse_graph_scope(scope: GraphScopePayload | None) -> GraphRunScope | None:
    """
    Convert a GraphScopePayload into a GraphRunScope.

    Parameters
    ----------
    scope:
        Optional scope payload from MCP requests.

    Returns
    -------
    GraphRunScope | None
        Parsed scope or ``None`` when not provided.
    """
    if scope is None:
        return None
    return GraphRunScope(
        paths=tuple(scope.paths),
        modules=tuple(scope.modules),
        time_window=scope.time_window,
    )


class FunctionSummaryRow(MappingModel):
    """
    Typed row for ``docs.v_function_summary`` used by MCP consumers.

    Fields capture the most commonly consumed attributes; extra columns emitted
    by the view are ignored to allow forward-compatible extensions.
    """

    repo: str
    commit: str
    rel_path: str
    function_goid_h128: int
    urn: str | None = None
    language: str | None = None
    kind: str | None = None
    qualname: str | None = None
    loc: int | None = None
    logical_loc: int | None = None
    cyclomatic_complexity: int | None = None
    complexity_bucket: str | None = None
    param_count: int | None = None
    positional_params: int | None = None
    keyword_only_params: int | None = None
    has_varargs: bool | None = None
    has_varkw: bool | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    coverage_ratio: float | None = None
    tested: bool | None = None
    test_count: int | None = None
    failing_test_count: int | None = None
    last_test_status: str | None = None


class CallGraphEdgeRow(MappingModel):
    """Edge row emitted by ``docs.v_call_graph_enriched``."""

    caller_goid_h128: int
    caller_repo: str
    caller_commit: str
    caller_urn: str | None = None
    caller_rel_path: str | None = None
    caller_qualname: str | None = None
    caller_risk_level: str | None = None
    caller_risk_score: float | None = None
    callee_goid_h128: int
    callee_repo: str
    callee_commit: str
    callee_urn: str | None = None
    callee_rel_path: str | None = None
    callee_qualname: str | None = None
    callee_risk_level: str | None = None
    callee_risk_score: float | None = None
    callsite_path: str | None = None
    callsite_line: int | None = None
    callsite_col: int | None = None
    language: str | None = None
    kind: str | None = None
    resolved_via: str | None = None
    confidence: float | None = None
    evidence_json: str | None = None


class FileSummaryRow(MappingModel):
    """
    Typed row for ``docs.v_file_summary`` with nested function summaries.

    Additional columns (tags, owners, AST counts) are tolerated via extra
    fields to keep the contract resilient to view extensions.
    """

    repo: str
    commit: str
    rel_path: str
    module: str | None = None
    language: str | None = None
    ast_complexity: object | None = None
    hotspot_score: object | None = None
    type_error_count: object | None = None
    annotation_ratio: object | None = None
    untyped_defs: object | None = None
    overlay_needed: object | None = None
    total_errors: object | None = None
    has_errors: object | None = None
    function_count: int | None = None
    high_risk_functions: int | None = None
    medium_risk_functions: int | None = None
    low_risk_functions: int | None = None
    max_risk_score: float | None = None
    functions: list[FunctionSummaryRow] = Field(default_factory=list)


class ModuleArchitectureRow(MappingModel):
    """Typed subset of ``docs.v_module_architecture``."""

    repo: str
    commit: str
    module: str
    rel_path: str | None = None
    tags: object | None = None
    owners: object | None = None
    import_fan_in: int | None = None
    import_fan_out: int | None = None
    symbol_fan_in: int | None = None
    symbol_fan_out: int | None = None
    module_coverage_ratio: float | None = None
    tested_function_count: int | None = None
    untested_function_count: int | None = None
    role: str | None = None
    role_confidence: float | None = None
    avg_risk_score: float | None = None
    max_risk_score: float | None = None


class SubsystemSummaryRow(MappingModel):
    """Summary row for ``docs.v_subsystem_summary``."""

    repo: str
    commit: str
    subsystem_id: str
    name: str
    description: str | None = None
    module_count: int
    modules_json: object | None = None
    entrypoints_json: list[object] | str | None = Field(default_factory=list)
    internal_edge_count: int | None = None
    external_edge_count: int | None = None
    fan_in: int | None = None
    fan_out: int | None = None
    function_count: int | None = None
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: int | None = None
    risk_level: str | None = None
    subsystem_disagree_count: int | None = None
    subsystem_member_count: int | None = None
    subsystem_agreement_ratio: float | None = None
    created_at: str | datetime | None = None


class RiskLevel(StrEnum):
    """Risk level categories for analytics payloads."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SubsystemProfileRow(MappingModel):
    """Profile row for ``docs.v_subsystem_profile``."""

    repo: str
    commit: str
    subsystem_id: str
    name: str
    description: str | None = None
    module_count: Annotated[int | None, Field(ge=0)] = None
    modules_json: object | None = None
    entrypoints_json: list[dict[str, str | list[str]]] = Field(default_factory=list)
    internal_edge_count: Annotated[int | None, Field(ge=0)] = None
    external_edge_count: Annotated[int | None, Field(ge=0)] = None
    fan_in: Annotated[int | None, Field(ge=0)] = None
    fan_out: Annotated[int | None, Field(ge=0)] = None
    function_count: Annotated[int | None, Field(ge=0)] = None
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: Annotated[int | None, Field(ge=0)] = None
    risk_level: RiskLevel | None = None
    import_in_degree: Annotated[float | None, Field(ge=0)] = None
    import_out_degree: Annotated[float | None, Field(ge=0)] = None
    import_pagerank: Annotated[float | None, Field(ge=0)] = None
    import_betweenness: Annotated[float | None, Field(ge=0)] = None
    import_closeness: Annotated[float | None, Field(ge=0)] = None
    import_layer: Annotated[int | None, Field(ge=0)] = None
    created_at: str | datetime | None = None

    @classmethod
    def _normalize_entrypoints(cls, value: object) -> list[dict[str, str | list[str]]]:
        return normalize_entrypoints_payload(value)

    @model_validator(mode="before")
    @classmethod
    def normalize_entrypoints(
        cls,
        value: dict[str, object] | list[object] | str | SubsystemProfileRow | None,
    ) -> object:
        """
        Coerce entrypoints_json payloads into a normalized list of dicts.

        Returns
        -------
        object
            Original payload with entrypoints_json normalized when present.
        """
        if isinstance(value, dict) and "entrypoints_json" in value:
            normalized = dict(value)
            normalized["entrypoints_json"] = cls._normalize_entrypoints(
                value.get("entrypoints_json")
            )
            return normalized
        return value


class SubsystemCoverageRow(MappingModel):
    """Coverage rollup row for ``docs.v_subsystem_coverage``."""

    repo: str
    commit: str
    subsystem_id: str
    name: str | None = None
    description: str | None = None
    module_count: Annotated[int | None, Field(ge=0)] = None
    function_count: Annotated[int | None, Field(ge=0)] = None
    risk_level: RiskLevel | None = None
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    test_count: Annotated[int | None, Field(ge=0)] = None
    passed_test_count: Annotated[int | None, Field(ge=0)] = None
    failed_test_count: Annotated[int | None, Field(ge=0)] = None
    skipped_test_count: Annotated[int | None, Field(ge=0)] = None
    xfail_test_count: Annotated[int | None, Field(ge=0)] = None
    flaky_test_count: Annotated[int | None, Field(ge=0)] = None
    total_functions_covered: Annotated[int | None, Field(ge=0)] = None
    avg_functions_covered: Annotated[float | None, Field(ge=0)] = None
    max_functions_covered: Annotated[float | None, Field(ge=0)] = None
    min_functions_covered: Annotated[float | None, Field(ge=0)] = None
    function_coverage_ratio: Annotated[float | None, Field(ge=0)] = None
    created_at: str | datetime | None = None


class ModuleWithSubsystemRow(MappingModel):
    """Membership row from ``docs.v_module_with_subsystem``."""

    repo: str
    commit: str
    subsystem_id: str
    subsystem_name: str | None = None
    module: str
    role: str | None = None
    rel_path: str | None = None
    tags: object | None = None
    owners: object | None = None
    import_fan_in: int | None = None
    import_fan_out: int | None = None
    symbol_fan_in: int | None = None
    symbol_fan_out: int | None = None
    risk_level: str | None = None
    avg_risk_score: float | None = None
    max_risk_score: float | None = None


class FunctionSummaryResponse(BaseModel):
    """Response wrapper for function summary lookups."""

    found: bool
    summary: FunctionSummaryRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class HighRiskFunctionsResponse(BaseModel):
    """Response wrapper for high-risk function listings."""

    functions: list[ViewRow]
    truncated: bool = False
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class CallGraphNeighborsResponse(BaseModel):
    """Incoming/outgoing call graph edges."""

    outgoing: list[CallGraphEdgeRow]
    incoming: list[CallGraphEdgeRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class TestsForFunctionResponse(BaseModel):
    """Tests that exercise a given function."""

    tests: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class GraphNeighborhoodResponse(BaseModel):
    """Nodes and edges for a bounded graph neighborhood."""

    nodes: list[ViewRow]
    edges: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ImportBoundaryResponse(BaseModel):
    """Edges crossing subsystem boundaries in the import graph."""

    nodes: list[ViewRow]
    edges: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FileSummaryResponse(BaseModel):
    """Summary of a file plus nested function rows."""

    found: bool
    file: FileSummaryRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FunctionProfileResponse(BaseModel):
    """Profile payload for a single function GOID."""

    found: bool
    profile: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FileProfileResponse(BaseModel):
    """Profile payload for a file path."""

    found: bool
    profile: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ModuleProfileResponse(BaseModel):
    """Profile payload for a module."""

    found: bool
    profile: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FunctionArchitectureResponse(BaseModel):
    """Architecture metrics for a function."""

    found: bool
    architecture: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ModuleArchitectureResponse(BaseModel):
    """Architecture metrics for a module."""

    found: bool
    architecture: ModuleArchitectureRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemSummaryResponse(BaseModel):
    """Summary of inferred subsystems."""

    subsystems: list[SubsystemSummaryRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ModuleSubsystemResponse(BaseModel):
    """Subsystem membership for a module."""

    found: bool
    memberships: list[ModuleWithSubsystemRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FileHintsResponse(BaseModel):
    """IDE-ready hints for a file path (module + subsystem context)."""

    found: bool
    hints: list[ViewRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemModulesResponse(BaseModel):
    """Subsystem detail payload with module membership rows."""

    found: bool
    subsystem: SubsystemSummaryRow | None = None
    modules: list[ModuleWithSubsystemRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemSearchResponse(BaseModel):
    """Search-oriented subsystem listing."""

    subsystems: list[SubsystemSummaryRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemProfileResponse(BaseModel):
    """Subsystem profile rows for docs view."""

    profiles: list[SubsystemProfileRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemCoverageResponse(BaseModel):
    """Subsystem coverage rollup rows for docs view."""

    coverage: list[SubsystemCoverageRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class DatasetDescriptor(BaseModel):
    """Metadata describing a browseable dataset."""

    name: str
    table: str
    family: str | None = None
    description: str
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: Literal["strict", "lenient"] | None = None
    capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Capability flags (validation, export, docs_view, read_only).",
    )


class DatasetSpecDescriptor(BaseModel):
    """Canonical dataset contract surfaced via HTTP and MCP."""

    name: str
    table_key: str
    is_view: bool
    schema_columns: list[str] = Field(default_factory=list)
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    has_row_binding: bool
    json_schema_id: str | None = None
    description: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: Literal["strict", "lenient"] | None = None
    upstream_dependencies: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Capability flags (validation, export, docs_view, read_only).",
    )


class DatasetSchemaColumn(BaseModel):
    """DuckDB column descriptor for dataset schemas."""

    name: str
    type: str
    nullable: bool


class DatasetSchemaResponse(BaseModel):
    """Composite schema detail payload for datasets."""

    dataset: str
    table_key: str
    duckdb_schema: list[DatasetSchemaColumn] = Field(default_factory=list)
    json_schema: dict[str, object] | None = None
    sample_rows: list[ViewRow] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: Literal["strict", "lenient"] | None = None


class DatasetRowsResponse(BaseModel):
    """Rows returned from a dataset slice."""

    dataset: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
