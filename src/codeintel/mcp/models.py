"""Typed MCP request/response models and error payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """Problem Details payload for MCP error responses."""

    type: str = Field(default="about:blank")
    title: str
    detail: str | None = None
    status: int | None = None
    instance: str | None = None
    data: dict[str, Any] | None = None


class FunctionSummaryResponse(BaseModel):
    """Response wrapper for function summary lookups."""

    found: bool
    summary: dict[str, Any] | None = None


class HighRiskFunctionsResponse(BaseModel):
    """Response wrapper for high-risk function listings."""

    functions: list[dict[str, Any]]


class CallGraphNeighborsResponse(BaseModel):
    """Incoming/outgoing call graph edges."""

    outgoing: list[dict[str, Any]]
    incoming: list[dict[str, Any]]


class TestsForFunctionResponse(BaseModel):
    """Tests that exercise a given function."""

    tests: list[dict[str, Any]]


class FileSummaryResponse(BaseModel):
    """Summary of a file plus nested function rows."""

    found: bool
    file: dict[str, Any] | None = None


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
    rows: list[dict[str, Any]]
