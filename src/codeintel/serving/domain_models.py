"""Transport-agnostic domain models for serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    """Domain-level diagnostic message attached to responses."""

    code: str
    severity: Literal["info", "warning", "error"]
    detail: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMeta:
    """Transport-agnostic metadata for paginated or bounded responses."""

    requested_limit: int | None = None
    applied_limit: int | None = None
    requested_offset: int | None = None
    applied_offset: int | None = None
    truncated: bool = False
    messages: list[Message] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        """
        Return a dictionary representation compatible with MCP responses.

        Returns
        -------
        dict[str, Any]
            Mapping of metadata values including messages.
        """
        return {
            "requested_limit": self.requested_limit,
            "applied_limit": self.applied_limit,
            "requested_offset": self.requested_offset,
            "applied_offset": self.applied_offset,
            "truncated": self.truncated,
            "messages": [
                {
                    "code": message.code,
                    "severity": message.severity,
                    "detail": message.detail,
                    "context": message.context,
                }
                for message in self.messages
            ],
        }


@dataclass
class FunctionSummary:
    """Core function summary information shared across transports."""

    urn: str
    goid_h128: int
    rel_path: str
    qualname: str
    short_summary: str | None
    long_summary: str | None
    is_test: bool
    meta: ResponseMeta


@dataclass
class HighRiskFunction:
    """Single row in a high-risk function listing."""

    goid_h128: int
    qualname: str
    rel_path: str
    risk_score: float
    is_tested: bool


@dataclass
class HighRiskFunctions:
    """Domain representation of high-risk functions listing."""

    functions: list[HighRiskFunction]
    meta: ResponseMeta


@dataclass
class FileSummary:
    """Summary of a file and its contained functions."""

    rel_path: str
    module: str | None
    functions: list[FunctionSummary]
    meta: ResponseMeta


@dataclass
class DatasetDescriptorDomain:
    """Domain-level description of a dataset."""

    name: str
    table: str
    description: str
    family: str | None = None
    owner: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    is_docs_view: bool = False
    is_read_only: bool = False


@dataclass
class DatasetRows:
    """Domain representation of dataset rows plus meta."""

    dataset_name: str
    limit: int
    offset: int
    rows: list[dict[str, Any]]
    meta: ResponseMeta

    def model_dump(self) -> dict[str, Any]:
        """
        Return a dictionary representation compatible with MCP models.

        Returns
        -------
        dict[str, Any]
            Mapping of dataset row payload with metadata.
        """
        return {
            "dataset": self.dataset_name,
            "limit": self.limit,
            "offset": self.offset,
            "rows": self.rows,
            "meta": self.meta.model_dump(),
        }


@dataclass
class DatasetSchema:
    """Domain representation of a dataset schema and sample rows."""

    dataset_name: str
    table_key: str
    duckdb_schema: list[dict[str, Any]]
    json_schema: dict[str, Any] | None
    sample_rows: list[dict[str, Any]]
    capabilities: dict[str, bool]
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    schema_version: str | None
    stable_id: str | None
    validation_profile: Literal["strict", "lenient"] | None
    meta: ResponseMeta | None = None


@dataclass
class GraphPlan:
    """Domain representation of a graph plugin execution plan."""

    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: list[dict[str, object]] = field(default_factory=list)
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    plugin_metadata: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class FunctionSummaryResult:
    """Domain payload for function summary lookups."""

    found: bool
    summary: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class HighRiskFunctionsResult:
    """Domain payload for high-risk function listings."""

    functions: list[dict[str, object]]
    truncated: bool
    meta: ResponseMeta


@dataclass
class CallGraphNeighbors:
    """Domain payload for call graph neighbor listings."""

    outgoing: list[dict[str, object]]
    incoming: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class TestsForFunctionResult:
    """Domain payload listing tests that exercise a function."""

    tests: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class GraphNeighborhood:
    """Domain payload for graph neighborhood responses."""

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class ImportBoundary:
    """Domain payload for import boundary graph responses."""

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class FileSummaryResult:
    """Domain payload for file summary lookups."""

    found: bool
    file: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class FunctionProfileResult:
    """Domain payload for function profile lookups."""

    found: bool
    profile: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class FileProfileResult:
    """Domain payload for file profile lookups."""

    found: bool
    profile: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class ModuleProfileResult:
    """Domain payload for module profile lookups."""

    found: bool
    profile: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class FunctionArchitectureResult:
    """Domain payload for function architecture lookups."""

    found: bool
    architecture: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class ModuleArchitectureResult:
    """Domain payload for module architecture lookups."""

    found: bool
    architecture: dict[str, object] | None
    meta: ResponseMeta


@dataclass
class SubsystemSummaryResult:
    """Domain payload for subsystem summary listings."""

    subsystems: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class ModuleSubsystemResult:
    """Domain payload for module→subsystem membership."""

    found: bool
    memberships: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class FileHintsResult:
    """Domain payload for IDE hint lookups."""

    found: bool
    hints: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class SubsystemModulesResult:
    """Domain payload for subsystem detail + module members."""

    found: bool
    subsystem: dict[str, object] | None
    modules: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class SubsystemSearchResult:
    """Domain payload for subsystem search listings."""

    subsystems: list[dict[str, object]]
    meta: ResponseMeta


@dataclass
class SubsystemProfileResult:
    """Domain payload for subsystem profile rows."""

    profiles: list[object]
    meta: ResponseMeta


@dataclass
class SubsystemCoverageResult:
    """Domain payload for subsystem coverage rollups."""

    coverage: list[object]
    meta: ResponseMeta


__all__ = [
    "CallGraphNeighbors",
    "DatasetDescriptorDomain",
    "DatasetRows",
    "DatasetSchema",
    "FileHintsResult",
    "FileProfileResult",
    "FileSummary",
    "FileSummaryResult",
    "FunctionArchitectureResult",
    "FunctionProfileResult",
    "FunctionSummary",
    "FunctionSummaryResult",
    "GraphNeighborhood",
    "GraphPlan",
    "HighRiskFunction",
    "HighRiskFunctions",
    "HighRiskFunctionsResult",
    "ImportBoundary",
    "Message",
    "ModuleArchitectureResult",
    "ModuleProfileResult",
    "ModuleSubsystemResult",
    "ResponseMeta",
    "SubsystemCoverageResult",
    "SubsystemModulesResult",
    "SubsystemProfileResult",
    "SubsystemSearchResult",
    "SubsystemSummaryResult",
    "TestsForFunctionResult",
]
