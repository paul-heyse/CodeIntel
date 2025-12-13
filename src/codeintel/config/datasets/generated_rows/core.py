"""Generated row models for insert helpers."""

from __future__ import annotations

from typing import TypedDict

__all__ = [
    "CoreAstMetricsRow",
    "CoreAstNodesRow",
    "CoreCstNodesRow",
    "CoreDocstringsRow",
    "CoreFileStateRow",
    "CoreGoidCrosswalkRow",
    "CoreGoidsRow",
    "CoreIngestRunsRow",
    "CoreModulesRow",
    "CoreRepoMapRow",
    "CoreScipOccurrencesRow",
    "CoreScipSymbolsRow",
    "CoreTestResultsRow",
    "CoreTestSummaryRow",
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
    generated_at: str


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
    decorators: str | None
    docstring: str | None
    hash: str


class CoreCstNodesRow(TypedDict):
    """Row model for core.cst_nodes."""

    path: str
    node_id: str
    kind: str
    span: str
    text_preview: str | None
    parents: str | None
    qnames: str | None


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
    params: str | None
    returns: str | None
    raises: str | None
    examples: str | None
    created_at: str


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
    updated_at: str


class CoreGoidsRow(TypedDict):
    """Row model for core.goids."""

    goid_h128: float
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int | None
    end_line: int | None
    created_at: str


class CoreIngestRunsRow(TypedDict):
    """Row model for core.ingest_runs."""

    repo: str
    commit: str
    step: str
    run_id: str
    mode: str
    started_at: str
    finished_at: str | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: str
    error_kind: str | None
    error_message: str | None
    datasets: str | None
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
    tags: str | None
    owners: str | None


class CoreRepoMapRow(TypedDict):
    """Row model for core.repo_map."""

    repo: str
    commit: str
    modules: str | None
    overlays: str | None
    generated_at: str | None


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
    created_at: str


class CoreScipSymbolsRow(TypedDict):
    """Row model for core.scip_symbols."""

    repo: str
    commit: str
    rel_path: str
    symbol: str
    documentation: str | None
    created_at: str


class CoreTestResultsRow(TypedDict):
    """Row model for core.test_results."""

    repo: str
    commit: str
    nodeid: str
    rel_path: str
    outcome: str
    duration: float | None
    longrepr: str | None
    created_at: str


class CoreTestSummaryRow(TypedDict):
    """Row model for core.test_summary."""

    repo: str
    commit: str
    passed: int
    failed: int
    skipped: int
    error: int
    duration: float | None
    created_at: str
