"""Analytics table TypedDict row models and serializers.

This module provides TypedDict definitions for analytics DuckDB tables:
- CoverageLineRow for analytics.coverage_lines
- TypednessRow for analytics.typedness
- StaticDiagnosticRow for analytics.static_diagnostics
- FunctionValidationRow for analytics.function_validation
- GraphValidationRow for analytics.graph_validation
- HotspotRow for analytics.hotspots
- FunctionMetricsRow for analytics.function_metrics
- FunctionTypesRow for analytics.function_types
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, TypedDict, TypeVar

from codeintel.config.datasets.schemas import TABLE_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

_Column = TypeVar("_Column", bound=str)


def _serialize_row(
    row: Mapping[_Column, object],
    columns: Sequence[_Column],
) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Parameters
    ----------
    row
        Row data as a mapping from column name to value.
    columns
        Ordered sequence of column names.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


def _get_contract_columns(table_key: str) -> tuple[str, ...]:
    """Retrieve column names from the TableSchema for a given table key.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier.

    Returns
    -------
    tuple[str, ...]
        Column names in schema definition order.

    Raises
    ------
    ValueError
        If no schema is defined for the given table key.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    return tuple(schema.column_names())


class CoverageLineRow(TypedDict):
    """Row shape for analytics.coverage_lines inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    line
        Line number.
    is_executable
        Whether the line is executable.
    is_covered
        Whether the line is covered by tests.
    hits
        Number of times the line was hit.
    context_count
        Number of test contexts covering this line.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: datetime


def coverage_line_to_tuple(row: CoverageLineRow) -> tuple[object, ...]:
    """Serialize a CoverageLineRow into the INSERT column order.

    Parameters
    ----------
    row
        The coverage line row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by coverage_lines INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["line"],
        row["is_executable"],
        row["is_covered"],
        row["hits"],
        row["context_count"],
        row["created_at"],
    )


class TypednessRow(TypedDict):
    """Row shape for analytics.typedness inserts.

    Matches the analytics.typedness TableSchema with columns:
    repo, commit, path, type_error_count, annotation_ratio, untyped_defs, overlay_needed

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    path
        Relative file path.
    type_error_count
        Total number of static analysis errors (pyright + pyrefly + ruff).
    annotation_ratio
        JSON object with params/returns annotation ratios.
    untyped_defs
        Number of untyped function definitions.
    overlay_needed
        Whether overlay typing is needed.
    """

    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio: str  # JSON string
    untyped_defs: int
    overlay_needed: bool


def typedness_row_to_tuple(row: TypednessRow) -> tuple[object, ...]:
    """Serialize a TypednessRow into the INSERT column order.

    Order matches analytics.typedness schema:
    repo, commit, path, type_error_count, annotation_ratio, untyped_defs, overlay_needed

    Parameters
    ----------
    row
        The typedness row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by typedness INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["path"],
        row["type_error_count"],
        row["annotation_ratio"],
        row["untyped_defs"],
        row["overlay_needed"],
    )


class StaticDiagnosticRow(TypedDict):
    """Row shape for analytics.static_diagnostics inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    tool
        Name of the diagnostic tool (pyright, pyrefly, ruff).
    error_count
        Number of errors from this tool.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    tool: str
    error_count: int
    created_at: datetime


def static_diagnostic_to_tuple(row: StaticDiagnosticRow) -> tuple[object, ...]:
    """Serialize a StaticDiagnosticRow into the INSERT column order.

    Parameters
    ----------
    row
        The static diagnostic row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by static_diagnostics INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["tool"],
        row["error_count"],
        row["created_at"],
    )


class FunctionValidationRow(TypedDict):
    """Row shape for analytics.function_validation inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    function_goid_h128
        128-bit hash of the function GOID.
    rel_path
        Relative file path.
    qualname
        Fully qualified name.
    issue
        Validation issue type.
    detail
        Issue detail message.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime


def function_validation_row_to_tuple(row: FunctionValidationRow) -> tuple[object, ...]:
    """Serialize a FunctionValidationRow into the INSERT column order.

    Parameters
    ----------
    row
        The function validation row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by function_validation INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["function_goid_h128"],
        row["rel_path"],
        row["qualname"],
        row["issue"],
        row["detail"],
        row["created_at"],
    )


class GraphValidationRow(TypedDict):
    """Row shape for analytics.graph_validation inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    graph_name
        Name of the graph being validated.
    entity_id
        Entity identifier.
    issue
        Validation issue type.
    severity
        Issue severity level.
    rel_path
        Relative file path.
    detail
        Issue detail message.
    metadata
        Additional metadata (JSON).
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    graph_name: str
    entity_id: str
    issue: str
    severity: str | None
    rel_path: str | None
    detail: str
    metadata: object | None
    created_at: datetime


def graph_validation_row_to_tuple(row: GraphValidationRow) -> tuple[object, ...]:
    """Serialize a GraphValidationRow into the INSERT column order.

    Parameters
    ----------
    row
        The graph validation row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by graph_validation INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["graph_name"],
        row["entity_id"],
        row["issue"],
        row["severity"],
        row["rel_path"],
        row["detail"],
        row["metadata"],
        row["created_at"],
    )


class HotspotRow(TypedDict):
    """Row shape for analytics.hotspots inserts.

    Parameters
    ----------
    rel_path
        Relative file path.
    commit_count
        Number of commits touching this file.
    author_count
        Number of unique authors.
    lines_added
        Total lines added.
    lines_deleted
        Total lines deleted.
    complexity
        File complexity score.
    score
        Hotspot score.
    """

    rel_path: str
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    complexity: float
    score: float


def hotspot_row_to_tuple(row: HotspotRow) -> tuple[object, ...]:
    """Serialize a HotspotRow into the INSERT column order.

    Parameters
    ----------
    row
        The hotspot row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by hotspots INSERTs.
    """
    return (
        row["rel_path"],
        row["commit_count"],
        row["author_count"],
        row["lines_added"],
        row["lines_deleted"],
        row["complexity"],
        row["score"],
    )


class FunctionMetricsRow(TypedDict):
    """Row shape for analytics.function_metrics inserts.

    Parameters
    ----------
    function_goid_h128
        128-bit hash of the function GOID.
    urn
        Uniform Resource Name.
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    language
        Programming language.
    kind
        Function kind.
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Ending line number.
    loc
        Lines of code.
    logical_loc
        Logical lines of code.
    param_count
        Total parameter count.
    positional_params
        Positional parameter count.
    keyword_only_params
        Keyword-only parameter count.
    has_varargs
        Has *args parameter.
    has_varkw
        Has **kwargs parameter.
    is_async
        Is async function.
    is_generator
        Is generator function.
    return_count
        Number of return statements.
    yield_count
        Number of yield statements.
    raise_count
        Number of raise statements.
    cyclomatic_complexity
        Cyclomatic complexity score.
    max_nesting_depth
        Maximum nesting depth.
    stmt_count
        Statement count.
    decorator_count
        Decorator count.
    has_docstring
        Has a docstring.
    complexity_bucket
        Complexity bucket classification.
    created_at
        Creation timestamp.
    """

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int | None
    logical_loc: int | None
    param_count: int | None
    positional_params: int | None
    keyword_only_params: int | None
    has_varargs: bool
    has_varkw: bool
    is_async: bool
    is_generator: bool
    return_count: int | None
    yield_count: int | None
    raise_count: int | None
    cyclomatic_complexity: int | None
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    complexity_bucket: str | None
    created_at: datetime


FUNCTION_METRICS_COLUMNS: Final[tuple[str, ...]] = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "loc",
    "logical_loc",
    "param_count",
    "positional_params",
    "keyword_only_params",
    "has_varargs",
    "has_varkw",
    "is_async",
    "is_generator",
    "return_count",
    "yield_count",
    "raise_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "stmt_count",
    "decorator_count",
    "has_docstring",
    "complexity_bucket",
    "created_at",
)


def function_metrics_row_to_tuple(row: FunctionMetricsRow) -> tuple[object, ...]:
    """Serialize a FunctionMetricsRow into INSERT column order.

    Parameters
    ----------
    row
        The function metrics row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.function_metrics columns.
    """
    return _serialize_row(row, FUNCTION_METRICS_COLUMNS)


class FunctionTypesRow(TypedDict):
    """Row shape for analytics.function_types inserts.

    Parameters
    ----------
    function_goid_h128
        128-bit hash of the function GOID.
    urn
        Uniform Resource Name.
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    language
        Programming language.
    kind
        Function kind.
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Ending line number.
    total_params
        Total parameter count.
    annotated_params
        Number of annotated parameters.
    unannotated_params
        Number of unannotated parameters.
    param_typed_ratio
        Ratio of typed parameters.
    has_return_annotation
        Has return type annotation.
    return_type
        Return type string.
    return_type_source
        Source of return type info.
    type_comment
        Type comment if present.
    param_types
        Parameter types (JSON).
    fully_typed
        All parameters and return typed.
    partial_typed
        Some parameters typed.
    untyped
        No type annotations.
    typedness_bucket
        Typedness bucket classification.
    typedness_source
        Source of typedness info.
    created_at
        Creation timestamp.
    """

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    total_params: int | None
    annotated_params: int | None
    unannotated_params: int | None
    param_typed_ratio: float | None
    has_return_annotation: bool
    return_type: str | None
    return_type_source: str | None
    type_comment: str | None
    param_types: object
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    created_at: datetime


FUNCTION_TYPES_COLUMNS: Final[tuple[str, ...]] = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "total_params",
    "annotated_params",
    "unannotated_params",
    "param_typed_ratio",
    "has_return_annotation",
    "return_type",
    "return_type_source",
    "type_comment",
    "param_types",
    "fully_typed",
    "partial_typed",
    "untyped",
    "typedness_bucket",
    "typedness_source",
    "created_at",
)


def function_types_row_to_tuple(row: FunctionTypesRow) -> tuple[object, ...]:
    """Serialize a FunctionTypesRow into INSERT column order.

    Parameters
    ----------
    row
        The function types row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with analytics.function_types columns.
    """
    normalized = dict(row)
    param_types = normalized.get("param_types")
    if isinstance(param_types, dict):
        normalized["param_types"] = json.dumps(param_types)
    return _serialize_row(normalized, FUNCTION_TYPES_COLUMNS)


__all__ = [
    "FUNCTION_METRICS_COLUMNS",
    "FUNCTION_TYPES_COLUMNS",
    "CoverageLineRow",
    "FunctionMetricsRow",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GraphValidationRow",
    "HotspotRow",
    "StaticDiagnosticRow",
    "TypednessRow",
    "coverage_line_to_tuple",
    "function_metrics_row_to_tuple",
    "function_types_row_to_tuple",
    "function_validation_row_to_tuple",
    "graph_validation_row_to_tuple",
    "hotspot_row_to_tuple",
    "static_diagnostic_to_tuple",
    "typedness_row_to_tuple",
]
