"""Generated row models for insert helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

__all__ = [
    "CoreAstMetricsRow",
    "CoreAstNodesRow",
    "CoreCstNodesRow",
    "CoreDocstringsRow",
    "CoreFileLineIndexRow",
    "CoreFileStateRow",
    "CoreGoidCrosswalkRow",
    "CoreGoidsRow",
    "CoreIngestRunsRow",
    "CoreModulesRow",
    "CoreParseManifestRow",
    "CoreRepoMapRow",
    "CoreScipOccurrencesRow",
    "CoreScipSymbolsRow",
    "CoreSyntaxCallsRow",
    "CoreSyntaxDefsRow",
    "CoreSyntaxImportsRow",
    "CoreSyntaxRefsRow",
    "CoreSyntaxScopesRow",
    "CoreSyntaxSpansRow",
]


class CoreAstMetricsRow(TypedDict):
    """Row model for core.ast_metrics."""

    rel_path: str
    node_count: int
    function_count: int
    class_count: int
    avg_depth: float
    max_depth: int
    complexity: float
    generated_at: datetime


class CoreAstNodesRow(TypedDict):
    """Row model for core.ast_nodes."""

    path: str
    node_type: str
    name: str | None
    qualname: str | None
    lineno: int | None
    end_lineno: int | None
    decorator_start_line: int | None
    decorator_end_line: int | None
    col_offset: int | None
    end_col_offset: int | None
    parent_qualname: str | None
    decorators: object | None
    docstring: str | None
    hash: str


class CoreCstNodesRow(TypedDict):
    """Row model for core.cst_nodes."""

    path: str
    node_id: str
    kind: str
    span: object
    text_preview: str | None
    parents: object | None
    qnames: object | None


class CoreParseManifestRow(TypedDict):
    """Row model for core.parse_manifest."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    parse_ok: bool
    error_kind: str | None
    error_message: str | None
    error_line: int | None
    error_col: int | None
    error_snippet: str | None


class CoreSyntaxSpansRow(TypedDict):
    """Row model for core.syntax_spans."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    span_id: str
    span_kind: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


class CoreSyntaxScopesRow(TypedDict):
    """Row model for core.syntax_scopes."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    scope_id: str
    scope_kind: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    parent_scope_id: str | None


class CoreSyntaxDefsRow(TypedDict):
    """Row model for core.syntax_defs."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    def_id: str
    scope_id: str
    span_id: str
    def_kind: str
    name: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


class CoreSyntaxRefsRow(TypedDict):
    """Row model for core.syntax_refs."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    ref_id: str
    scope_id: str
    span_id: str
    ref_kind: str
    name: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


class CoreSyntaxCallsRow(TypedDict):
    """Row model for core.syntax_calls."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    scope_id: str
    span_id: str
    callee_span_id: str | None
    callee_text: str | None
    arg_count: int | None
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


class CoreSyntaxImportsRow(TypedDict):
    """Row model for core.syntax_imports."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    import_id: str
    scope_id: str
    span_id: str
    import_kind: str
    module: str | None
    name: str | None
    alias: str | None
    level: int | None
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


class CoreDocstringsRow(TypedDict):
    """Row model for core.docstrings."""

    repo: str
    commit: str
    rel_path: str
    module: str
    qualname: str
    kind: str
    lineno: int | None
    end_lineno: int | None
    raw_docstring: str | None
    style: str | None
    short_desc: str | None
    long_desc: str | None
    params: object | None
    returns: object | None
    raises: object | None
    examples: object | None
    created_at: datetime


class CoreFileLineIndexRow(TypedDict):
    """Row model for core.file_line_index."""

    repo: str
    commit: str
    rel_path: str
    line: int
    start_byte: int
    end_byte: int
    encoding: str


class CoreFileStateRow(TypedDict):
    """Row model for core.file_state."""

    repo: str
    commit: str
    rel_path: str
    language: str
    size_bytes: int
    mtime_ns: int
    content_hash: str


class CoreGoidCrosswalkRow(TypedDict):
    """Row model for core.goid_crosswalk."""

    repo: str
    commit: str
    goid: str
    lang: str
    module_path: str
    file_path: str
    start_line: int | None
    end_line: int | None
    scip_symbol: str | None
    ast_qualname: str | None
    cst_node_id: str | None
    chunk_id: str | None
    symbol_id: str | None
    updated_at: datetime


class CoreGoidsRow(TypedDict):
    """Row model for core.goids."""

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int | None
    end_line: int | None
    created_at: datetime


class CoreIngestRunsRow(TypedDict):
    """Row model for core.ingest_runs."""

    repo: str
    commit: str
    step: str
    run_id: str
    mode: str
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: str
    error_kind: str | None
    error_message: str | None
    datasets: object | None
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    modules_changed_ratio: float | None
    modules_deleted_ratio: float | None
    use_full_rebuild: bool | None


class CoreModulesRow(TypedDict):
    """Row model for core.modules."""

    module: str
    path: str
    repo: str | None
    commit: str | None
    language: str | None
    tags: object | None
    owners: object | None
    row_hash: str | None


class CoreRepoMapRow(TypedDict):
    """Row model for core.repo_map."""

    repo: str
    commit: str
    modules: object | None
    overlays: object | None
    generated_at: datetime | None


class CoreScipOccurrencesRow(TypedDict):
    """Row model for core.scip_occurrences."""

    repo: str
    commit: str
    rel_path: str
    symbol: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    roles: int
    position_encoding: int | None
    text_document_encoding: str | None
    start_byte: int | None
    end_byte: int | None
    created_at: datetime


class CoreScipSymbolsRow(TypedDict):
    """Row model for core.scip_symbols."""

    repo: str
    commit: str
    rel_path: str
    symbol: str
    documentation: str | None
    created_at: datetime
