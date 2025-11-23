"""Typed MCP request/response models and error payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ViewRow(BaseModel):
    """Generic row wrapper for DuckDB view/table results."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> object:
        """
        Allow dict-style access for compatibility with legacy code/tests.

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


class DatasetDescriptor(BaseModel):
    """Metadata describing a browseable dataset."""

    name: str
    table: str
    description: str


class DatasetRowsResponse(BaseModel):
    """Rows returned from a dataset slice."""

    dataset: str
    limit: int
    offset: int
    rows: list[ViewRow]
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
