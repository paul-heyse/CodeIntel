"""Typed MCP request/response models and error payloads.

Transport Model Layer
---------------------
This module provides Pydantic models for HTTP/MCP serialization. These models
form the **Transport Layer** in the dual-model architecture.

Dual-Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~
The serving layer maintains two parallel model systems by design:

1. **Domain Models** (``domain_models.py``)
   - Pure Python dataclasses for business logic
   - Used within the Service layer
   - No serialization dependencies

2. **Transport Models** (this module)
   - Pydantic BaseModel subclasses
   - JSON serialization and validation
   - Provide ``from_domain()`` and ``to_domain()`` converters

Why Two Systems?
~~~~~~~~~~~~~~~~
- **Domain Purity**: Service layer stays independent of serialization
- **Validation at Boundaries**: Pydantic validation only at transport layer
- **Performance**: Dataclasses are faster for internal processing
- **Testability**: Domain models are easy to construct in tests

Model Correspondence
~~~~~~~~~~~~~~~~~~~~
Each transport model corresponds to a domain model in ``domain_models.py``:

| Transport Model | Domain Model | Converter Methods |
|-----------------|--------------|-------------------|
| ``FunctionSummaryResponse`` | ``dm.FunctionSummaryResult`` | ``from_domain()``, ``to_domain()`` |
| ``HighRiskFunctionsResponse`` | ``dm.HighRiskFunctionsResult`` | ``from_domain()``, ``to_domain()`` |
| ``CallGraphNeighborsResponse`` | ``dm.CallGraphNeighbors`` | ``from_domain()``, ``to_domain()`` |
| ``GraphNeighborhoodResponse`` | ``dm.GraphNeighborhood`` | ``from_domain()``, ``to_domain()`` |
| ... | ... | ... |

Conversion Flow
~~~~~~~~~~~~~~~
::

    [Client Request]
         │
         ▼
    Transport Model (Pydantic) ──from_domain()──▶ Response to Client
         │                                           ▲
         │ to_domain()                               │
         ▼                                           │
    Domain Model (dataclass) ◀──────────────────────┘
         │
         ▼
    [Service Layer Processing]

Usage Pattern
~~~~~~~~~~~~~
::


    domain_result = service.get_function_summary(...)
    response = FunctionSummaryResponse.from_domain(domain_result)
    return response


    http_response = self._http_call(...)
    return http_response.to_domain()

See Also
--------
- ``codeintel.serving.domain_models`` : Domain models (dataclasses)
- ``codeintel.serving.services.query_service`` : Service layer contract
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from codeintel.config.graph_helpers import GraphRunScope
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.view_utils import normalize_entrypoints_payload
from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail

_TIME_WINDOW_LEN = 2


class MappingModel(BaseModel):
    """Base model for typed row payloads.

    Use attribute access for field retrieval. Call `.model_dump()` when
    a dict representation is needed.
    """

    model_config = ConfigDict(extra="ignore")


class ViewRow(BaseModel):
    """Generic row wrapper for DuckDB view/table results.

    Use attribute access for field retrieval. Call `.model_dump()` when
    a dict representation is needed.
    """

    model_config = ConfigDict(extra="allow")


class ProblemDetail(BaseModel):
    """Problem Details payload for MCP error responses."""

    type: str = Field(default="about:blank")
    title: str
    detail: str | None = None
    status: int | None = None
    instance: str | None = None
    code: str | None = None
    extras: dict[str, object] | None = None

    @classmethod
    def from_domain(cls, detail: DomainProblemDetail) -> ProblemDetail:
        """
        Convert a domain ProblemDetail into the Pydantic transport model.

        Returns
        -------
        ProblemDetail
            Transport wrapper for the provided domain detail.
        """
        return cls(
            type=detail.type,
            title=detail.title,
            detail=detail.detail,
            status=detail.status,
            instance=detail.instance,
            code=detail.code,
            extras=detail.extras or {},
        )

    def to_domain(self) -> DomainProblemDetail:
        """
        Convert a Pydantic ProblemDetail into the domain dataclass.

        Returns
        -------
        DomainProblemDetail
            Domain representation of the problem payload.
        """
        return DomainProblemDetail(
            type=self.type,
            title=self.title,
            detail=self.detail,
            status=self.status,
            instance=self.instance or "",
            code=self.code,
            extras=dict(self.extras or {}),
        )


class Message(BaseModel):
    """Structured message attached to responses."""

    code: str
    severity: Literal["info", "warning", "error"] = "info"
    detail: str | None = None
    context: dict[str, object] | None = None

    def to_domain(self) -> dm.Message:
        """
        Convert to the domain Message dataclass.

        Returns
        -------
        dm.Message
            Domain representation of the message payload.
        """
        return dm.Message(
            code=self.code,
            severity=self.severity,
            detail=self.detail,
            context=dict(self.context or {}),
        )

    @classmethod
    def from_domain(cls, msg: dm.Message) -> Message:
        """
        Build a transport Message from the domain representation.

        Returns
        -------
        Message
            Transport message carrying the same fields.
        """
        return cls(
            code=msg.code,
            severity=msg.severity,
            detail=msg.detail,
            context=msg.context or {},
        )


class ResponseMeta(BaseModel):
    """Response metadata including clamping and messaging."""

    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = Field(default_factory=list)

    def to_domain(self) -> dm.ResponseMeta:
        """
        Convert to the domain ResponseMeta dataclass.

        Returns
        -------
        dm.ResponseMeta
            Domain representation of response metadata.
        """
        return dm.ResponseMeta(
            requested_limit=self.requested_limit,
            applied_limit=self.applied_limit,
            requested_offset=self.requested_offset,
            applied_offset=self.applied_offset,
            truncated=self.truncated,
            messages=[message.to_domain() for message in self.messages],
        )

    @classmethod
    def from_domain(cls, meta: dm.ResponseMeta) -> ResponseMeta:
        """
        Convert a domain ResponseMeta into the transport model.

        Returns
        -------
        ResponseMeta
            Transport metadata mirroring the domain payload.
        """
        return cls(
            requested_limit=meta.requested_limit,
            applied_limit=meta.applied_limit,
            requested_offset=meta.requested_offset,
            applied_offset=meta.applied_offset,
            truncated=meta.truncated,
            messages=[Message.from_domain(msg) for msg in meta.messages],
        )


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
    reason: Literal[
        "disabled",
        "missing_dependency",
        "missing_graph",
        "config_error",
        "incremental_skip",
        "unchanged",
    ]


class GraphPlanPluginMetadata(BaseModel):
    """Metadata for a plugin included in a plan response."""

    stage: str
    severity: str
    requires_isolation: bool
    isolation_kind: str | None = None
    scope_aware: bool = False
    supported_scopes: tuple[str, ...] = ()
    description: str | None = None
    enabled_by_default: bool | None = None
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    resource_hints: dict[str, int | None] | None = None
    options_model: str | None = None
    options_default: object | None = None
    version_hash: str | None = None
    contract_checkers: int | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()


class GraphPlanResponse(BaseModel):
    """Resolved graph metric plan including ordering and dependency graph."""

    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[GraphPlanSkipped, ...]
    dep_graph: dict[str, tuple[str, ...]]
    plugin_metadata: dict[str, GraphPlanPluginMetadata]

    def to_domain(self) -> dm.GraphPlan:
        """
        Convert to the domain GraphPlan representation.

        Returns
        -------
        dm.GraphPlan
            Domain plan payload with plugin metadata and dependency graph.
        """
        return dm.GraphPlan(
            plan_id=self.plan_id,
            ordered_plugins=self.ordered_plugins,
            skipped_plugins=[entry.model_dump() for entry in self.skipped_plugins],
            dep_graph={name: tuple(deps) for name, deps in self.dep_graph.items()},
            plugin_metadata={
                name: metadata.model_dump() for name, metadata in self.plugin_metadata.items()
            },
        )

    @classmethod
    def from_domain(cls, plan: dm.GraphPlan) -> GraphPlanResponse:
        """
        Convert a domain GraphPlan into the transport model.

        Returns
        -------
        GraphPlanResponse
            Transport plan payload.
        """
        return cls(
            plan_id=plan.plan_id,
            ordered_plugins=plan.ordered_plugins,
            skipped_plugins=tuple(
                GraphPlanSkipped.model_validate(entry) for entry in plan.skipped_plugins
            ),
            dep_graph={name: tuple(deps) for name, deps in plan.dep_graph.items()},
            plugin_metadata={
                name: GraphPlanPluginMetadata.model_validate(metadata)
                for name, metadata in plan.plugin_metadata.items()
            },
        )


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


def normalize_scope(scope: object | None) -> GraphScopePayload | None:
    """
    Normalize scope parameter to GraphScopePayload or None.

    This function consolidates the repeated ``isinstance(scope, GraphScopePayload)``
    checks that were scattered across backend methods. It handles three cases:

    1. ``scope`` is None → return None
    2. ``scope`` is already a GraphScopePayload → return as-is
    3. ``scope`` is a dict → validate and return GraphScopePayload

    Parameters
    ----------
    scope
        Raw scope parameter from backend method calls. May be None, a
        GraphScopePayload instance, or a dict representation.

    Returns
    -------
    GraphScopePayload | None
        Normalized scope payload, or None if input was None or invalid.

    Examples
    --------
    >>> normalize_scope(None)
    None
    >>> normalize_scope(GraphScopePayload(paths=("src/",)))
    GraphScopePayload(paths=('src/',), modules=(), time_window=None)
    >>> normalize_scope({"paths": ["src/"]})
    GraphScopePayload(paths=('src/',), modules=(), time_window=None)
    """
    if scope is None:
        return None
    if isinstance(scope, GraphScopePayload):
        return scope
    if isinstance(scope, dict):
        return GraphScopePayload.model_validate(scope)
    return None


class FunctionSummaryRow(MappingModel):
    """
    Typed row for ``docs.v_function_summary`` used by MCP consumers.

    Fields capture the most commonly consumed attributes; extra columns emitted
    by the view are preserved so forward-compatible extensions remain visible.
    """

    model_config = ConfigDict(extra="allow")

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


class FunctionProfileRow(MappingModel):
    """Profile row for ``analytics.function_profile``."""

    model_config = ConfigDict(extra="allow")

    repo: str
    commit: str
    function_goid_h128: int
    urn: str | None = None
    rel_path: str | None = None
    qualname: str | None = None
    risk_score: float | None = None
    coverage_ratio: float | None = None
    tested: bool | None = None
    test_count: int | None = None
    fan_in: int | None = None
    fan_out: int | None = None


class FileProfileRow(MappingModel):
    """Profile row for ``analytics.file_profile``."""

    model_config = ConfigDict(extra="allow")

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
    coverage_ratio: float | None = None
    max_risk_score: float | None = None


class ModuleProfileRow(MappingModel):
    """Profile row for ``analytics.module_profile``."""

    model_config = ConfigDict(extra="allow")

    repo: str
    commit: str
    module: str
    rel_path: str | None = None
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

    model_config = ConfigDict(extra="allow")

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

    model_config = ConfigDict(extra="allow")

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
    summary: FunctionSummaryRow | ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.FunctionSummaryResult:
        """
        Convert to the domain FunctionSummaryResult representation.

        Returns
        -------
        dm.FunctionSummaryResult
            Domain summary payload.
        """
        return dm.FunctionSummaryResult(
            found=self.found,
            summary=self.summary.model_dump() if self.summary is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FunctionSummaryResult) -> FunctionSummaryResponse:
        """
        Convert a domain FunctionSummaryResult into the transport model.

        Returns
        -------
        FunctionSummaryResponse
            Transport summary payload.
        """
        summary = None
        if result.summary is not None:
            try:
                summary = FunctionSummaryRow.model_validate(result.summary)
            except ValidationError:
                summary = ViewRow.model_validate(result.summary)
        return cls(found=result.found, summary=summary, meta=ResponseMeta.from_domain(result.meta))


class HighRiskFunctionsResponse(BaseModel):
    """Response wrapper for high-risk function listings."""

    functions: list[ViewRow]
    truncated: bool = False
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.HighRiskFunctionsResult:
        """
        Convert to the domain HighRiskFunctionsResult representation.

        Returns
        -------
        dm.HighRiskFunctionsResult
            Domain high-risk function payload.
        """
        return dm.HighRiskFunctionsResult(
            functions=[row.model_dump() for row in self.functions],
            truncated=self.truncated,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.HighRiskFunctionsResult) -> HighRiskFunctionsResponse:
        """
        Convert a domain HighRiskFunctionsResult into the transport model.

        Returns
        -------
        HighRiskFunctionsResponse
            Transport high-risk function payload.
        """
        return cls(
            functions=[ViewRow.model_validate(row) for row in result.functions],
            truncated=result.truncated,
            meta=ResponseMeta.from_domain(result.meta),
        )


class CallGraphNeighborsResponse(BaseModel):
    """Incoming/outgoing call graph edges."""

    outgoing: list[CallGraphEdgeRow | ViewRow]
    incoming: list[CallGraphEdgeRow | ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.CallGraphNeighbors:
        """
        Convert to the domain CallGraphNeighbors representation.

        Returns
        -------
        dm.CallGraphNeighbors
            Domain call graph neighbor payload.
        """
        return dm.CallGraphNeighbors(
            outgoing=[edge.model_dump() for edge in self.outgoing],
            incoming=[edge.model_dump() for edge in self.incoming],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.CallGraphNeighbors) -> CallGraphNeighborsResponse:
        """
        Convert a domain CallGraphNeighbors into the transport model.

        Returns
        -------
        CallGraphNeighborsResponse
            Transport call graph neighbor payload.
        """
        outgoing_edges: list[CallGraphEdgeRow | ViewRow] = []
        for edge in result.outgoing:
            try:
                outgoing_edges.append(CallGraphEdgeRow.model_validate(edge))
            except ValidationError:
                outgoing_edges.append(ViewRow.model_validate(edge))
        incoming_edges: list[CallGraphEdgeRow | ViewRow] = []
        for edge in result.incoming:
            try:
                incoming_edges.append(CallGraphEdgeRow.model_validate(edge))
            except ValidationError:
                incoming_edges.append(ViewRow.model_validate(edge))
        return cls(
            outgoing=outgoing_edges,
            incoming=incoming_edges,
            meta=ResponseMeta.from_domain(result.meta),
        )


class TestsForFunctionResponse(BaseModel):
    """Tests that exercise a given function."""

    tests: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.TestsForFunctionResult:
        """
        Convert to the domain TestsForFunctionResult representation.

        Returns
        -------
        dm.TestsForFunctionResult
            Domain tests payload.
        """
        return dm.TestsForFunctionResult(
            tests=[test.model_dump() for test in self.tests],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.TestsForFunctionResult) -> TestsForFunctionResponse:
        """
        Convert a domain TestsForFunctionResult into the transport model.

        Returns
        -------
        TestsForFunctionResponse
            Transport tests payload.
        """
        return cls(
            tests=[ViewRow.model_validate(test) for test in result.tests],
            meta=ResponseMeta.from_domain(result.meta),
        )


class GraphNeighborhoodResponse(BaseModel):
    """Nodes and edges for a bounded graph neighborhood."""

    nodes: list[ViewRow]
    edges: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.GraphNeighborhood:
        """
        Convert to the domain GraphNeighborhood representation.

        Returns
        -------
        dm.GraphNeighborhood
            Domain graph neighborhood payload.
        """
        return dm.GraphNeighborhood(
            nodes=[node.model_dump() for node in self.nodes],
            edges=[edge.model_dump() for edge in self.edges],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, neighborhood: dm.GraphNeighborhood) -> GraphNeighborhoodResponse:
        """
        Convert a domain GraphNeighborhood into the transport model.

        Returns
        -------
        GraphNeighborhoodResponse
            Transport graph neighborhood payload.
        """
        return cls(
            nodes=[ViewRow.model_validate(node) for node in neighborhood.nodes],
            edges=[ViewRow.model_validate(edge) for edge in neighborhood.edges],
            meta=ResponseMeta.from_domain(neighborhood.meta),
        )


class ImportBoundaryResponse(BaseModel):
    """Edges crossing subsystem boundaries in the import graph."""

    nodes: list[ViewRow]
    edges: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.ImportBoundary:
        """
        Convert to the domain ImportBoundary representation.

        Returns
        -------
        dm.ImportBoundary
            Domain import boundary payload.
        """
        return dm.ImportBoundary(
            nodes=[node.model_dump() for node in self.nodes],
            edges=[edge.model_dump() for edge in self.edges],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, boundary: dm.ImportBoundary) -> ImportBoundaryResponse:
        """
        Convert a domain ImportBoundary into the transport model.

        Returns
        -------
        ImportBoundaryResponse
            Transport import boundary payload.
        """
        return cls(
            nodes=[ViewRow.model_validate(node) for node in boundary.nodes],
            edges=[ViewRow.model_validate(edge) for edge in boundary.edges],
            meta=ResponseMeta.from_domain(boundary.meta),
        )


class FileSummaryResponse(BaseModel):
    """Summary of a file plus nested function rows."""

    found: bool
    file: FileSummaryRow | ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @model_validator(mode="before")
    @classmethod
    def _normalize_file(cls, values: object) -> object:
        """
        Preserve extra columns and avoid injecting defaults when functions are absent.

        Returns
        -------
        object
            Normalized mapping suitable for model validation.
        """
        if not isinstance(values, Mapping):
            return values
        file_value = values.get("file")
        if isinstance(file_value, Mapping) and "functions" not in file_value:
            normalized = dict(values)
            normalized["file"] = ViewRow.model_validate(file_value)
            return normalized
        return values

    def to_domain(self) -> dm.FileSummaryResult:
        """
        Convert to the domain FileSummaryResult representation.

        Returns
        -------
        dm.FileSummaryResult
            Domain file summary payload.
        """
        return dm.FileSummaryResult(
            found=self.found,
            file=self.file.model_dump() if self.file is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FileSummaryResult) -> FileSummaryResponse:
        """
        Convert a domain FileSummaryResult into the transport model.

        Returns
        -------
        FileSummaryResponse
            Transport file summary payload.
        """
        file_value: FileSummaryRow | ViewRow | None = None
        if result.file is not None:
            if isinstance(result.file, Mapping) and "functions" not in result.file:
                file_value = ViewRow.model_validate(result.file)
            else:
                try:
                    file_value = FileSummaryRow.model_validate(result.file)
                except ValidationError:
                    file_value = ViewRow.model_validate(result.file)
        return cls(
            found=result.found,
            file=file_value,
            meta=ResponseMeta.from_domain(result.meta),
        )


class FunctionProfileResponse(BaseModel):
    """Profile payload for a single function GOID."""

    found: bool
    profile: FunctionProfileRow | ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @model_validator(mode="before")
    @classmethod
    def _normalize_profile(cls, values: object) -> object:
        """
        Preserve profile payloads without injecting default fields.

        Returns
        -------
        object
            Normalized mapping suitable for model validation.
        """
        if not isinstance(values, Mapping):
            return values
        profile_value = values.get("profile")
        if isinstance(profile_value, Mapping) and not any(
            key in profile_value for key in ("test_count", "fan_in", "fan_out")
        ):
            normalized = dict(values)
            normalized["profile"] = ViewRow.model_validate(profile_value)
            return normalized
        return values

    def to_domain(self) -> dm.FunctionProfileResult:
        """
        Convert to the domain FunctionProfileResult representation.

        Returns
        -------
        dm.FunctionProfileResult
            Domain function profile payload.
        """
        return dm.FunctionProfileResult(
            found=self.found,
            profile=self.profile.model_dump() if self.profile is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FunctionProfileResult) -> FunctionProfileResponse:
        """
        Convert a domain FunctionProfileResult into the transport model.

        Returns
        -------
        FunctionProfileResponse
            Transport function profile payload.
        """
        profile_value: FunctionProfileRow | ViewRow | None = None
        if result.profile is not None:
            if isinstance(result.profile, Mapping) and not any(
                key in result.profile for key in ("test_count", "fan_in", "fan_out")
            ):
                profile_value = ViewRow.model_validate(result.profile)
            else:
                try:
                    profile_value = FunctionProfileRow.model_validate(result.profile)
                except ValidationError:
                    profile_value = ViewRow.model_validate(result.profile)
        return cls(
            found=result.found,
            profile=profile_value,
            meta=ResponseMeta.from_domain(result.meta),
        )


class FileProfileResponse(BaseModel):
    """Profile payload for a file path."""

    found: bool
    profile: FileProfileRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.FileProfileResult:
        """
        Convert to the domain FileProfileResult representation.

        Returns
        -------
        dm.FileProfileResult
            Domain file profile payload.
        """
        return dm.FileProfileResult(
            found=self.found,
            profile=self.profile.model_dump() if self.profile is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FileProfileResult) -> FileProfileResponse:
        """
        Convert a domain FileProfileResult into the transport model.

        Returns
        -------
        FileProfileResponse
            Transport file profile payload.
        """
        return cls(
            found=result.found,
            profile=(
                FileProfileRow.model_validate(result.profile)
                if result.profile is not None
                else None
            ),
            meta=ResponseMeta.from_domain(result.meta),
        )


class ModuleProfileResponse(BaseModel):
    """Profile payload for a module."""

    found: bool
    profile: ModuleProfileRow | ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @model_validator(mode="before")
    @classmethod
    def _normalize_profile(cls, values: object) -> object:
        """
        Preserve module profiles that do not include optional typed fields.

        Returns
        -------
        object
            Normalized mapping suitable for model validation.
        """
        if not isinstance(values, Mapping):
            return values
        profile_value = values.get("profile")
        if isinstance(profile_value, Mapping) and not any(
            key in profile_value for key in ("rel_path", "symbol_fan_in", "symbol_fan_out")
        ):
            normalized = dict(values)
            normalized["profile"] = ViewRow.model_validate(profile_value)
            return normalized
        return values

    def to_domain(self) -> dm.ModuleProfileResult:
        """
        Convert to the domain ModuleProfileResult representation.

        Returns
        -------
        dm.ModuleProfileResult
            Domain module profile payload.
        """
        return dm.ModuleProfileResult(
            found=self.found,
            profile=self.profile.model_dump() if self.profile is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.ModuleProfileResult) -> ModuleProfileResponse:
        """
        Convert a domain ModuleProfileResult into the transport model.

        Returns
        -------
        ModuleProfileResponse
            Transport module profile payload.
        """
        profile_value: ModuleProfileRow | ViewRow | None = None
        if result.profile is not None:
            if isinstance(result.profile, Mapping) and not any(
                key in result.profile for key in ("rel_path", "symbol_fan_in", "symbol_fan_out")
            ):
                profile_value = ViewRow.model_validate(result.profile)
            else:
                try:
                    profile_value = ModuleProfileRow.model_validate(result.profile)
                except ValidationError:
                    profile_value = ViewRow.model_validate(result.profile)
        return cls(
            found=result.found,
            profile=profile_value,
            meta=ResponseMeta.from_domain(result.meta),
        )


class FunctionArchitectureResponse(BaseModel):
    """Architecture metrics for a function."""

    found: bool
    architecture: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.FunctionArchitectureResult:
        """
        Convert to the domain FunctionArchitectureResult representation.

        Returns
        -------
        dm.FunctionArchitectureResult
            Domain function architecture payload.
        """
        return dm.FunctionArchitectureResult(
            found=self.found,
            architecture=self.architecture.model_dump() if self.architecture is not None else None,
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FunctionArchitectureResult) -> FunctionArchitectureResponse:
        """
        Convert a domain FunctionArchitectureResult into the transport model.

        Returns
        -------
        FunctionArchitectureResponse
            Transport function architecture payload.
        """
        return cls(
            found=result.found,
            architecture=(
                ViewRow.model_validate(result.architecture)
                if result.architecture is not None
                else None
            ),
            meta=ResponseMeta.from_domain(result.meta),
        )


class ModuleArchitectureResponse(BaseModel):
    """Architecture metrics for a module."""

    found: bool
    architecture: ModuleArchitectureRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.ModuleArchitectureResult:
        """
        Convert to the domain ModuleArchitectureResult representation.

        Returns
        -------
        dm.ModuleArchitectureResult
            Domain module architecture payload.
        """
        return dm.ModuleArchitectureResult(
            found=self.found,
            architecture=(
                self.architecture.model_dump(exclude_none=True)
                if self.architecture is not None
                else None
            ),
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.ModuleArchitectureResult) -> ModuleArchitectureResponse:
        """
        Convert a domain ModuleArchitectureResult into the transport model.

        Returns
        -------
        ModuleArchitectureResponse
            Transport module architecture payload.
        """
        return cls(
            found=result.found,
            architecture=(
                ModuleArchitectureRow.model_validate(result.architecture)
                if result.architecture is not None
                else None
            ),
            meta=ResponseMeta.from_domain(result.meta),
        )


class SubsystemSummaryResponse(BaseModel):
    """Summary of inferred subsystems."""

    subsystems: list[SubsystemSummaryRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.SubsystemSummaryResult:
        """
        Convert to the domain SubsystemSummaryResult representation.

        Returns
        -------
        dm.SubsystemSummaryResult
            Domain subsystem summary payload.
        """
        return dm.SubsystemSummaryResult(
            subsystems=[row.model_dump() for row in self.subsystems],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.SubsystemSummaryResult) -> SubsystemSummaryResponse:
        """
        Convert a domain SubsystemSummaryResult into the transport model.

        Returns
        -------
        SubsystemSummaryResponse
            Transport subsystem summary payload.
        """
        return cls(
            subsystems=[SubsystemSummaryRow.model_validate(row) for row in result.subsystems],
            meta=ResponseMeta.from_domain(result.meta),
        )


class ModuleSubsystemResponse(BaseModel):
    """Subsystem membership for a module."""

    found: bool
    memberships: list[ModuleWithSubsystemRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.ModuleSubsystemResult:
        """
        Convert to the domain ModuleSubsystemResult representation.

        Returns
        -------
        dm.ModuleSubsystemResult
            Domain module membership payload.
        """
        return dm.ModuleSubsystemResult(
            found=self.found,
            memberships=[row.model_dump() for row in self.memberships],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.ModuleSubsystemResult) -> ModuleSubsystemResponse:
        """
        Convert a domain ModuleSubsystemResult into the transport model.

        Returns
        -------
        ModuleSubsystemResponse
            Transport module membership payload.
        """
        return cls(
            found=result.found,
            memberships=[ModuleWithSubsystemRow.model_validate(row) for row in result.memberships],
            meta=ResponseMeta.from_domain(result.meta),
        )


class FileHintsResponse(BaseModel):
    """IDE-ready hints for a file path (module + subsystem context)."""

    found: bool
    hints: list[ViewRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.FileHintsResult:
        """
        Convert to the domain FileHintsResult representation.

        Returns
        -------
        dm.FileHintsResult
            Domain file hints payload.
        """
        return dm.FileHintsResult(
            found=self.found,
            hints=[hint.model_dump() for hint in self.hints],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.FileHintsResult) -> FileHintsResponse:
        """
        Convert a domain FileHintsResult into the transport model.

        Returns
        -------
        FileHintsResponse
            Transport file hints payload.
        """
        return cls(
            found=result.found,
            hints=[ViewRow.model_validate(hint) for hint in result.hints],
            meta=ResponseMeta.from_domain(result.meta),
        )


class SubsystemModulesResponse(BaseModel):
    """Subsystem detail payload with module membership rows."""

    found: bool
    subsystem: SubsystemSummaryRow | None = None
    modules: list[ModuleWithSubsystemRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.SubsystemModulesResult:
        """
        Convert to the domain SubsystemModulesResult representation.

        Returns
        -------
        dm.SubsystemModulesResult
            Domain subsystem modules payload.
        """
        return dm.SubsystemModulesResult(
            found=self.found,
            subsystem=self.subsystem.model_dump() if self.subsystem is not None else None,
            modules=[module.model_dump() for module in self.modules],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.SubsystemModulesResult) -> SubsystemModulesResponse:
        """
        Convert a domain SubsystemModulesResult into the transport model.

        Returns
        -------
        SubsystemModulesResponse
            Transport subsystem modules payload.
        """
        return cls(
            found=result.found,
            subsystem=(
                SubsystemSummaryRow.model_validate(result.subsystem)
                if result.subsystem is not None
                else None
            ),
            modules=[ModuleWithSubsystemRow.model_validate(module) for module in result.modules],
            meta=ResponseMeta.from_domain(result.meta),
        )


class SubsystemSearchResponse(BaseModel):
    """Search-oriented subsystem listing."""

    subsystems: list[SubsystemSummaryRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.SubsystemSearchResult:
        """
        Convert to the domain SubsystemSearchResult representation.

        Returns
        -------
        dm.SubsystemSearchResult
            Domain subsystem search payload.
        """
        return dm.SubsystemSearchResult(
            subsystems=[row.model_dump() for row in self.subsystems],
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.SubsystemSearchResult) -> SubsystemSearchResponse:
        """
        Convert a domain SubsystemSearchResult into the transport model.

        Returns
        -------
        SubsystemSearchResponse
            Transport subsystem search payload.
        """
        return cls(
            subsystems=[SubsystemSummaryRow.model_validate(row) for row in result.subsystems],
            meta=ResponseMeta.from_domain(result.meta),
        )


class SubsystemProfileResponse(BaseModel):
    """Subsystem profile rows for docs view."""

    profiles: list[SubsystemProfileRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.SubsystemProfileResult:
        """
        Convert to the domain SubsystemProfileResult representation.

        Returns
        -------
        dm.SubsystemProfileResult
            Domain subsystem profile payload.
        """
        return dm.SubsystemProfileResult(
            profiles=list(self.profiles),
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.SubsystemProfileResult) -> SubsystemProfileResponse:
        """
        Convert a domain SubsystemProfileResult into the transport model.

        Returns
        -------
        SubsystemProfileResponse
            Transport subsystem profile payload.
        """
        return cls(
            profiles=[SubsystemProfileRow.model_validate(row) for row in result.profiles],
            meta=ResponseMeta.from_domain(result.meta),
        )


class SubsystemCoverageResponse(BaseModel):
    """Subsystem coverage rollup rows for docs view."""

    coverage: list[SubsystemCoverageRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    def to_domain(self) -> dm.SubsystemCoverageResult:
        """
        Convert to the domain SubsystemCoverageResult representation.

        Returns
        -------
        dm.SubsystemCoverageResult
            Domain subsystem coverage payload.
        """
        return dm.SubsystemCoverageResult(
            coverage=list(self.coverage),
            meta=self.meta.to_domain(),
        )

    @classmethod
    def from_domain(cls, result: dm.SubsystemCoverageResult) -> SubsystemCoverageResponse:
        """
        Convert a domain SubsystemCoverageResult into the transport model.

        Returns
        -------
        SubsystemCoverageResponse
            Transport subsystem coverage payload.
        """
        return cls(
            coverage=[SubsystemCoverageRow.model_validate(row) for row in result.coverage],
            meta=ResponseMeta.from_domain(result.meta),
        )


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
    capabilities: Mapping[str, bool] = Field(
        default_factory=dict,
        description="Capability flags (validation, export, docs_view, read_only).",
    )


class DatasetSpecDescriptor(BaseModel):
    """Canonical dataset contract surfaced via HTTP and MCP."""

    name: str
    table_key: str
    family: str | None = None
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
    capabilities: Mapping[str, bool] = Field(
        default_factory=dict,
        description="Capability flags (validation, export, docs_view, read_only).",
    )


class DatasetSchemaColumn(BaseModel):
    """DuckDB column descriptor for dataset schemas."""

    name: str
    type: str
    nullable: bool


CapabilitiesValue = Decimal | bool | float | int | str
CapabilitiesMapping = Mapping[bytearray | bytes | str, CapabilitiesValue]


def _normalize_capabilities(capabilities: Mapping[Any, Any] | None) -> CapabilitiesMapping:
    """
    Normalize capability mappings to string keys and a strict value union.

    Returns
    -------
    dict[str, CapabilitiesValue]
        Normalized capability mapping with UTF-8 decoded keys/values.

    Raises
    ------
    TypeError
        If a capability value cannot be coerced into the allowed union.
    """
    if capabilities is None:
        return {}
    normalized: dict[str, CapabilitiesValue] = {}
    for key, value in capabilities.items():
        key_str = (
            str(key) if not isinstance(key, (bytes, bytearray)) else key.decode("utf-8", "ignore")
        )
        if isinstance(value, (bytes, bytearray)):
            value_obj: Any = value.decode("utf-8", "ignore")
        else:
            value_obj = value
        if not isinstance(value_obj, (Decimal, bool, float, int, str)):
            message = f"Unsupported capability value type: {type(value_obj).__name__}"
            raise TypeError(message)
        normalized[key_str] = value_obj
    return cast("CapabilitiesMapping", normalized)


class DatasetSchemaResponse(BaseModel):
    """Composite schema detail payload for datasets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: str
    table_key: str
    duckdb_schema: list[DatasetSchemaColumn] = Field(default_factory=list)
    json_schema: dict[str, object] | None = None
    sample_rows: list[ViewRow] = Field(default_factory=list)
    capabilities: CapabilitiesMapping = Field(default_factory=dict)
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    validation_profile: Literal["strict", "lenient"] | None = None
    meta: ResponseMeta | None = None

    def to_domain(self) -> dm.DatasetSchema:
        """
        Convert to the domain DatasetSchema representation.

        Returns
        -------
        dm.DatasetSchema
            Domain schema model with matching payload.
        """
        normalized_caps = cast(
            "Mapping[str, CapabilitiesValue]", _normalize_capabilities(self.capabilities)
        )
        return dm.DatasetSchema(
            dataset_name=self.dataset,
            table_key=self.table_key,
            duckdb_schema=[column.model_dump() for column in self.duckdb_schema],
            json_schema=self.json_schema,
            sample_rows=[row.model_dump() for row in self.sample_rows],
            capabilities={key: bool(value) for key, value in normalized_caps.items()},
            owner=self.owner,
            freshness_sla=self.freshness_sla,
            retention_policy=self.retention_policy,
            schema_version=self.schema_version,
            stable_id=self.stable_id,
            validation_profile=self.validation_profile,
            meta=self.meta.to_domain() if self.meta is not None else None,
        )

    @classmethod
    def from_domain(cls, schema: dm.DatasetSchema) -> DatasetSchemaResponse:
        """
        Convert a domain DatasetSchema into the transport model.

        Returns
        -------
        DatasetSchemaResponse
            Transport model reflecting the domain payload.
        """
        capabilities: CapabilitiesMapping = _normalize_capabilities(schema.capabilities)
        return cls(
            dataset=schema.dataset_name,
            table_key=schema.table_key,
            duckdb_schema=[
                DatasetSchemaColumn.model_validate(column) for column in schema.duckdb_schema
            ],
            json_schema=schema.json_schema,
            sample_rows=[ViewRow.model_validate(row) for row in schema.sample_rows],
            capabilities=capabilities,
            owner=schema.owner,
            freshness_sla=schema.freshness_sla,
            retention_policy=schema.retention_policy,
            schema_version=schema.schema_version,
            stable_id=schema.stable_id,
            validation_profile=schema.validation_profile,
            meta=ResponseMeta.from_domain(schema.meta) if schema.meta is not None else None,
        )


class DatasetRowsResponse(BaseModel):
    """Rows returned from a dataset slice."""

    dataset_name: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta | None = None

    @property
    def dataset(self) -> str:
        """Backward-compatible alias for dataset_name."""
        return self.dataset_name

    def to_domain(self) -> dm.DatasetRows:
        """
        Convert to the domain DatasetRows representation.

        Returns
        -------
        dm.DatasetRows
            Domain dataset rows payload.
        """
        return dm.DatasetRows(
            dataset_name=self.dataset_name,
            limit=self.limit,
            offset=self.offset,
            rows=[row.model_dump() for row in self.rows],
            meta=self.meta.to_domain() if self.meta is not None else dm.ResponseMeta(),
        )

    @classmethod
    def from_domain(cls, rows: dm.DatasetRows) -> DatasetRowsResponse:
        """
        Convert a domain DatasetRows into the transport model.

        Returns
        -------
        DatasetRowsResponse
            Transport model reflecting the domain rows.
        """
        return cls(
            dataset_name=rows.dataset_name,
            limit=rows.limit,
            offset=rows.offset,
            rows=[ViewRow.model_validate(row) for row in rows.rows],
            meta=ResponseMeta.from_domain(rows.meta),
        )


class DatasetMetaResponse(BaseModel):
    """Expose dataset metadata enriched with serving limits."""

    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None = None
    family: str | None = None
    is_docs_view: bool = False
    is_read_only: bool = False
    default_limit: int
    max_limit: int


class OperationMetaResponse(BaseModel):
    """Introspectable metadata for a single operation."""

    id: str
    category: str
    summary: str
    description: str | None = None
    http_method: str | None = None
    http_path: str | None = None
    tool_name: str | None = None
    output_model: str
    required_datasets: list[str] = Field(default_factory=list)
    required_graphs: list[str] = Field(default_factory=list)
    default_limit: int | None = None
    max_limit: int | None = None


class DataflowNodePayload(BaseModel):
    """HTTP/MCP payload representing a single dataflow node."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    id: str
    kind: Literal["table", "view", "operation", "graph"]
    family: str | None = None
    owner_package: str | None = None
    description: str | None = None


class DataflowEdgePayload(BaseModel):
    """HTTP/MCP payload representing a dataflow edge."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    src: str
    dst: str
    edge_type: Literal["builds", "reads", "exposes", "depends_on"]


class DataflowGraphResponse(BaseModel):
    """Bundle of dataflow nodes and edges."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    nodes: list[DataflowNodePayload]
    edges: list[DataflowEdgePayload]


class OperationPrereqDatasetStatus(BaseModel):
    """Status of a single dataset prerequisite check.

    Used by the debug endpoint to show whether each required dataset
    has rows for the requested repo/commit.
    """

    table_key: str = Field(description="Dataset table key (e.g., 'analytics.function_profile')")
    name: str = Field(description="Human-readable dataset name")
    has_rows: bool = Field(description="Whether the dataset has rows for the repo/commit")
    checked: bool = Field(description="Whether this dataset was successfully checked")
    error: str | None = Field(default=None, description="Error message if check failed")


class OperationPrereqRunSummary(BaseModel):
    """Summary of a pipeline run considered for prerequisite satisfaction.

    Used by the debug endpoint to show which runs were evaluated.
    """

    run_id: str = Field(description="Pipeline run identifier")
    kind: str = Field(description="Run kind (full, op_prereqs, etc.)")
    status: str = Field(description="Run status (succeeded, failed, etc.)")
    started_at: datetime | None = Field(default=None, description="When the run started")
    completed_at: datetime | None = Field(default=None, description="When the run completed")


class OperationPrereqDebugResponse(BaseModel):
    """Complete debug information for prerequisite checking.

    This response provides full observability into why prerequisites
    are or are not satisfied for an operation.
    """

    op_id: str = Field(description="Operation identifier (e.g., 'function.summary')")
    repo: str = Field(description="Repository slug")
    commit: str = Field(description="Commit SHA")
    required_datasets: list[str] = Field(
        default_factory=list,
        description="Directly required dataset table keys from operation config",
    )
    expanded_datasets: list[str] = Field(
        default_factory=list,
        description="All required datasets after transitive dependency expansion",
    )
    dataset_statuses: list[OperationPrereqDatasetStatus] = Field(
        default_factory=list,
        description="Status of each dataset check",
    )
    runs_considered: list[OperationPrereqRunSummary] = Field(
        default_factory=list,
        description="Recent pipeline runs considered for this repo/commit",
    )
    data_satisfied: bool = Field(
        description="Whether data-aware prerequisite check passed",
    )
    run_satisfied: bool = Field(
        description="Whether run-based prerequisite check passed",
    )
    overall_satisfied: bool = Field(
        description="Final determination of prerequisite satisfaction",
    )
