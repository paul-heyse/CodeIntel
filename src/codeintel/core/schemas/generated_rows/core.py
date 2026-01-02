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
    "CoreSchemaInferenceErrorsRow",
    "CoreScipDiagnosticsRow",
    "CoreScipExternalSymbolsRow",
    "CoreScipModuleStateRow",
    "CoreScipOccurrenceSpanXrefRow",
    "CoreScipOccurrenceSyntaxXrefRow",
    "CoreScipOccurrencesRow",
    "CoreScipSymbolGoidXrefRow",
    "CoreScipSymbolInformationRow",
    "CoreScipSymbolRelationshipsRow",
    "CoreScipSymbolsRow",
    "CoreSyntaxCallsResolvedRow",
    "CoreSyntaxCallsRow",
    "CoreSyntaxDefsResolvedRow",
    "CoreSyntaxDefsRow",
    "CoreSyntaxEdgesRow",
    "CoreSyntaxImportsResolvedRow",
    "CoreSyntaxImportsRow",
    "CoreSyntaxNodesRow",
    "CoreSyntaxRefsResolvedRow",
    "CoreSyntaxRefsRow",
    "CoreSyntaxScopesRow",
    "CoreSyntaxSpansRow",
    "CoreTsCapturesRow",
    "CoreTsParseErrorsRow",
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


class CoreParseManifestRow(TypedDict):
    """Row model for core.parse_manifest."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    parse_ok: bool
    encoding: str | None
    default_indent: str | None
    default_newline: str | None
    has_trailing_newline: bool | None
    future_imports: object | None
    parser_backend: str | None
    libcst_version: str | None
    error_kind: str | None
    error_message: str | None
    error_line: int | None
    error_col: int | None
    error_snippet: str | None


class CoreRepoMapRow(TypedDict):
    """Row model for core.repo_map."""

    repo: str
    commit: str
    modules: object | None
    overlays: object | None
    generated_at: datetime | None


class CoreSchemaInferenceErrorsRow(TypedDict):
    """Row model for core.schema_inference_errors."""

    table_key: str
    repo: str
    commit: str
    error: str
    occurred_at: datetime
    run_id: str


class CoreScipDiagnosticsRow(TypedDict):
    """Row model for core.scip_diagnostics."""

    repo: str
    commit: str
    rel_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    position_encoding: int | None
    text_document_encoding: str | None
    severity: str
    code: str | None
    message: str
    source: str | None
    created_at: datetime


class CoreScipExternalSymbolsRow(TypedDict):
    """Row model for core.scip_external_symbols."""

    repo: str
    commit: str
    symbol: str
    package_manager: str | None
    package_name: str | None
    package_version: str | None
    created_at: datetime


class CoreScipModuleStateRow(TypedDict):
    """Row model for core.scip_module_state."""

    repo: str
    commit: str
    rel_path: str
    content_hash: str
    options_hash: str | None
    tool_version: str | None
    shard_path: str
    updated_at: datetime


class CoreScipOccurrenceSpanXrefRow(TypedDict):
    """Row model for core.scip_occurrence_span_xref."""

    repo: str
    commit: str
    rel_path: str
    scip_symbol: str
    roles: int
    is_definition: bool
    is_reference: bool
    is_import: bool
    is_write: bool
    is_read: bool
    enclosing_symbol: str | None
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    position_encoding: int | None
    text_document_encoding: str | None
    start_byte: int | None
    end_byte: int | None
    goid_h128: int | None
    created_at: datetime


class CoreScipOccurrenceSyntaxXrefRow(TypedDict):
    """Row model for core.scip_occurrence_syntax_xref."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    scip_symbol: str
    scip_occurrence_id: str
    occ_start_byte: int | None
    occ_end_byte: int | None
    occ_start_line: int
    occ_start_col: int
    occ_end_line: int
    occ_end_col: int
    syntax_node_id: str | None
    match_kind: str
    candidate_count: int


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


class CoreScipSymbolGoidXrefRow(TypedDict):
    """Row model for core.scip_symbol_goid_xref."""

    repo: str
    commit: str
    scip_symbol: str
    goid_h128: int | None
    def_rel_path: str | None
    def_start_line: int | None
    def_start_col: int | None
    def_end_line: int | None
    def_end_col: int | None
    position_encoding: int | None
    text_document_encoding: str | None
    created_at: datetime


class CoreScipSymbolInformationRow(TypedDict):
    """Row model for core.scip_symbol_information."""

    repo: str
    commit: str
    symbol: str
    documentation: str | None
    kind: int | None
    display_name: str | None
    signature: str | None
    enclosing_symbol: str | None
    created_at: datetime


class CoreScipSymbolRelationshipsRow(TypedDict):
    """Row model for core.scip_symbol_relationships."""

    repo: str
    commit: str
    symbol: str
    related_symbol: str
    relationship_kind: str
    created_at: datetime


class CoreScipSymbolsRow(TypedDict):
    """Row model for core.scip_symbols."""

    repo: str
    commit: str
    rel_path: str
    symbol: str
    documentation: str | None
    created_at: datetime


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
    extras_json: object | None


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
    extras_json: object | None


class CoreSyntaxEdgesRow(TypedDict):
    """Row model for core.syntax_edges."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    parent_node_id: str
    child_node_id: str
    edge_kind: str
    field_name: str | None
    child_ordinal: int


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
    extras_json: object | None


class CoreSyntaxNodesRow(TypedDict):
    """Row model for core.syntax_nodes."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    language: str
    node_id: str
    node_kind: str
    raw_kind: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None
    text_preview: str | None
    extras_json: object | None


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
    extras_json: object | None


class CoreSyntaxCallsResolvedRow(TypedDict):
    """Row model for core.syntax_calls_resolved."""

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
    scip_symbol: str | None
    scip_occurrence_id: str | None
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None
    goid_h128: int | None
    syntax_node_id: str | None
    match_kind: str | None
    candidate_count: int | None
    extras_json: object | None


class CoreSyntaxDefsResolvedRow(TypedDict):
    """Row model for core.syntax_defs_resolved."""

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
    scip_symbol: str | None
    scip_occurrence_id: str | None
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None
    goid_h128: int | None
    syntax_node_id: str | None
    match_kind: str | None
    candidate_count: int | None
    extras_json: object | None


class CoreSyntaxImportsResolvedRow(TypedDict):
    """Row model for core.syntax_imports_resolved."""

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
    scip_symbol: str | None
    scip_occurrence_id: str | None
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None
    goid_h128: int | None
    syntax_node_id: str | None
    match_kind: str | None
    candidate_count: int | None
    extras_json: object | None


class CoreSyntaxRefsResolvedRow(TypedDict):
    """Row model for core.syntax_refs_resolved."""

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
    scip_symbol: str | None
    scip_occurrence_id: str | None
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None
    goid_h128: int | None
    syntax_node_id: str | None
    match_kind: str | None
    candidate_count: int | None
    extras_json: object | None


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


class CoreTsCapturesRow(TypedDict):
    """Row model for core.ts_captures."""

    repo: str
    commit: str
    rel_path: str
    language: str
    query_pack: str
    capture_name: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    node_type: str
    text_preview: str | None
    extras: object | None


class CoreTsParseErrorsRow(TypedDict):
    """Row model for core.ts_parse_errors."""

    repo: str
    commit: str
    rel_path: str
    language: str
    error_type: str
    message: str | None
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
