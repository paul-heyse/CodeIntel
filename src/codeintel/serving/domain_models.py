"""Transport-agnostic domain models for serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Domain-level diagnostic message attached to responses."""

    code: str
    severity: str
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
    validation_profile: str | None
    meta: ResponseMeta | None = None


__all__ = [
    "DatasetDescriptorDomain",
    "DatasetRows",
    "DatasetSchema",
    "FileSummary",
    "FunctionSummary",
    "HighRiskFunction",
    "HighRiskFunctions",
    "Message",
    "ResponseMeta",
]
