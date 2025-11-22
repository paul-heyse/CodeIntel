"""Typed MCP request/response models and error payloads."""

from __future__ import annotations

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


class FunctionSummaryResponse(BaseModel):
    """Response wrapper for function summary lookups."""

    found: bool
    summary: ViewRow | None = None


class HighRiskFunctionsResponse(BaseModel):
    """Response wrapper for high-risk function listings."""

    functions: list[ViewRow]
    truncated: bool = False


class CallGraphNeighborsResponse(BaseModel):
    """Incoming/outgoing call graph edges."""

    outgoing: list[ViewRow]
    incoming: list[ViewRow]


class TestsForFunctionResponse(BaseModel):
    """Tests that exercise a given function."""

    tests: list[ViewRow]


class FileSummaryResponse(BaseModel):
    """Summary of a file plus nested function rows."""

    found: bool
    file: ViewRow | None = None


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
