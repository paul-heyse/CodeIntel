"""Typed MCP request/response models and error payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class FunctionSummaryResponse(BaseModel):
    """Response wrapper for function summary lookups."""

    found: bool
    summary: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class HighRiskFunctionsResponse(BaseModel):
    """Response wrapper for high-risk function listings."""

    functions: list[ViewRow]
    truncated: bool = False
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class CallGraphNeighborsResponse(BaseModel):
    """Incoming/outgoing call graph edges."""

    outgoing: list[ViewRow]
    incoming: list[ViewRow]
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
    file: ViewRow | None = None
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
    architecture: ViewRow | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemSummaryResponse(BaseModel):
    """Summary of inferred subsystems."""

    subsystems: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ModuleSubsystemResponse(BaseModel):
    """Subsystem membership for a module."""

    found: bool
    memberships: list[ViewRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class FileHintsResponse(BaseModel):
    """IDE-ready hints for a file path (module + subsystem context)."""

    found: bool
    hints: list[ViewRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemModulesResponse(BaseModel):
    """Subsystem detail payload with module membership rows."""

    found: bool
    subsystem: ViewRow | None = None
    modules: list[ViewRow] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class SubsystemSearchResponse(BaseModel):
    """Search-oriented subsystem listing."""

    subsystems: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class DatasetDescriptor(BaseModel):
    """Metadata describing a browseable dataset."""

    name: str
    table: str
    description: str


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


class DatasetRowsResponse(BaseModel):
    """Rows returned from a dataset slice."""

    dataset: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
