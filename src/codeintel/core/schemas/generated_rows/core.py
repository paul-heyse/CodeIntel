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
    "CorePyBcBlocksRow",
    "CorePyBcCfgEdgesRow",
    "CorePyBcCodeUnitsRow",
    "CorePyBcDefuseEventsRow",
    "CorePyBcExceptionTableRow",
    "CorePyBcInstructionsRow",
    "CorePyCompilerMetadataRow",
    "CorePyInspectAnnotationsKvRow",
    "CorePyInspectClassAttrsRow",
    "CorePyInspectClassMroRow",
    "CorePyInspectMembersStaticRow",
    "CorePyInspectObjectsRow",
    "CorePyInspectRuntimeStateRow",
    "CorePyInspectSignatureParamsRow",
    "CorePyInspectSignaturesRow",
    "CorePyInspectSourceRow",
    "CorePyInspectUnwrapHopsRow",
    "CorePySymBindingsRow",
    "CorePySymFunctionPartitionsRow",
    "CorePySymNamespaceEdgesRow",
    "CorePySymResolutionEdgesRow",
    "CorePySymScopeEdgesRow",
    "CorePySymScopesRow",
    "CorePySymSymbolsRow",
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
    "CoreSyntaxCallArgsRow",
    "CoreSyntaxCallsResolvedRow",
    "CoreSyntaxCallsRow",
    "CoreSyntaxDefsResolvedRow",
    "CoreSyntaxDefsRow",
    "CoreSyntaxEdgesAugmentedRow",
    "CoreSyntaxEdgesRow",
    "CoreSyntaxFuncParamsRow",
    "CoreSyntaxImportsResolvedRow",
    "CoreSyntaxImportsRow",
    "CoreSyntaxNodesAugmentedRow",
    "CoreSyntaxNodesRow",
    "CoreSyntaxRefsResolvedRow",
    "CoreSyntaxRefsRow",
    "CoreSyntaxScopesRow",
    "CoreSyntaxSpansRow",
    "CoreTsCapturesRow",
    "CoreTsChangedRangesRow",
    "CoreTsEdgesRow",
    "CoreTsLanguageMetadataRow",
    "CoreTsNodesRow",
    "CoreTsParseErrorsRow",
    "CoreTsParseManifestRow",
    "CoreTsSyntaxNodeXrefRow",
    "CoreTsTokensRow",
    "CoreTsTriviaRow",
    "CoreTsWeldCoverageRow",
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
    start_byte: int | None
    end_byte: int | None
    parent_qualname: str | None
    decorators: object | None
    docstring: str | None
    ctx: str | None
    type_comment: str | None
    type_ignores: object | None
    identifier: str | None
    attribute: str | None
    imported: str | None
    asname: str | None
    module: str | None
    level: int | None
    constant_kind: str | None
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


class CorePyBcBlocksRow(TypedDict):
    """Row model for core.py_bc_blocks."""

    repo: str
    commit: str
    rel_path: str
    block_id: str
    code_unit_id: str
    start_offset: int | None
    end_offset: int | None
    start_label: str | None
    kind: str | None
    anchor_span_start_byte: int | None
    anchor_span_end_byte: int | None
    first_instr_index: int | None
    last_instr_index: int | None


class CorePyBcCfgEdgesRow(TypedDict):
    """Row model for core.py_bc_cfg_edges."""

    repo: str
    commit: str
    rel_path: str
    edge_id: str
    code_unit_id: str
    src_block_id: str
    dst_block_id: str
    kind: str
    cond_instr_id: str | None
    exc_entry_index: int | None


class CorePyBcCodeUnitsRow(TypedDict):
    """Row model for core.py_bc_code_units."""

    repo: str
    commit: str
    rel_path: str
    code_unit_id: str
    parent_code_unit_id: str | None
    qualpath: str | None
    co_name: str | None
    co_qualname: str | None
    kind: str | None
    co_firstlineno: int | None
    span_start_byte: int | None
    span_end_byte: int | None
    flags: int | None
    argcount: int | None
    posonlyargcount: int | None
    kwonlyargcount: int | None
    nlocals: int | None
    stacksize: int | None
    varnames: object | None
    names: object | None
    freevars: object | None
    cellvars: object | None
    bytecode_len: int | None
    exceptiontable_len: int | None
    python_version: str | None
    bytecode_magic: object | None
    optimize: int | None
    dont_inherit: bool | None


class CorePyBcDefuseEventsRow(TypedDict):
    """Row model for core.py_bc_defuse_events."""

    repo: str
    commit: str
    rel_path: str
    event_id: str
    code_unit_id: str
    instr_id: str
    instr_index: int | None
    event_kind: str
    space: str | None
    name: str | None
    confidence: float | None


class CorePyBcExceptionTableRow(TypedDict):
    """Row model for core.py_bc_exception_table."""

    repo: str
    commit: str
    rel_path: str
    code_unit_id: str
    exc_entry_index: int
    start_offset: int | None
    end_offset: int | None
    target_offset: int | None
    depth: int | None
    lasti: bool | None
    start_label: str | None
    end_label: str | None
    target_label: str | None


class CorePyBcInstructionsRow(TypedDict):
    """Row model for core.py_bc_instructions."""

    repo: str
    commit: str
    rel_path: str
    code_unit_id: str
    instr_id: str
    instr_physical_id: str | None
    instr_index: int | None
    start_offset: int | None
    offset: int | None
    cache_offset: int | None
    end_offset: int | None
    ext_arg_len: int | None
    op_len: int | None
    cache_len: int | None
    opcode: int | None
    opname: str | None
    baseopcode: int | None
    baseopname: str | None
    arg: int | None
    argrepr: str | None
    argval_kind: str | None
    argval_str: str | None
    argval_int: int | None
    argval_repr: str | None
    is_jump_target: bool | None
    jump_target_offset: int | None
    jump_target_label: str | None
    label: str | None
    starts_line: bool | None
    line_number: int | None
    pos: object | None
    span_start_byte: int | None
    span_end_byte: int | None
    cache_info: object | None
    cache_bytes: object | None
    op_bytes: object | None


class CorePyCompilerMetadataRow(TypedDict):
    """Row model for core.py_compiler_metadata."""

    repo: str
    commit: str
    run_id: str
    python_version: str | None
    magic_number: object | None
    optimize: int | None
    dont_inherit: bool | None
    flags: int | None


class CorePyInspectAnnotationsKvRow(TypedDict):
    """Row model for core.py_inspect_annotations_kv."""

    repo: str
    commit: str
    mode: str
    object_id: str
    eval_str: bool | None
    key: str
    value: object | None
    status: object | None


class CorePyInspectClassAttrsRow(TypedDict):
    """Row model for core.py_inspect_class_attrs."""

    repo: str
    commit: str
    mode: str
    class_object_id: str
    attr_name: str
    attr_kind: str | None
    defining_object_id: str | None
    value_kind: str | None
    value_object_id: str | None
    value_ref: object | None
    desc_is_data: bool | None
    desc_is_methoddesc: bool | None
    desc_is_getset: bool | None
    desc_is_member: bool | None
    status: object | None


class CorePyInspectClassMroRow(TypedDict):
    """Row model for core.py_inspect_class_mro."""

    repo: str
    commit: str
    mode: str
    class_object_id: str
    mro_index: int
    base_object_id: str | None
    base_kind: str | None
    status: object | None


class CorePyInspectMembersStaticRow(TypedDict):
    """Row model for core.py_inspect_members_static."""

    repo: str
    commit: str
    mode: str
    owner_object_id: str
    owner_kind: str | None
    attr_name: str
    value_kind: str | None
    value_object_id: str | None
    value_ref: object | None
    desc_kind: str | None
    desc_is_data: bool | None
    desc_is_methoddesc: bool | None
    desc_is_getset: bool | None
    desc_is_member: bool | None
    status: object | None


class CorePyInspectObjectsRow(TypedDict):
    """Row model for core.py_inspect_objects."""

    repo: str
    commit: str
    mode: str
    object_id: str
    object_addr: int | None
    kind: str | None
    module_name: str | None
    qualname: str | None
    name: str | None
    type_qualname: str | None
    is_builtin: bool | None
    is_callable: bool | None
    is_descriptor: bool | None
    has_wrapped: bool | None
    has_signature_override: bool | None
    has_annotations: bool | None
    status: object | None


class CorePyInspectRuntimeStateRow(TypedDict):
    """Row model for core.py_inspect_runtime_state."""

    repo: str
    commit: str
    mode: str
    object_id: str
    object_kind: str | None
    state_kind: str | None
    state: str | None
    frame_object_id: str | None
    frame_file: str | None
    frame_module: str | None
    frame_code_qualname: str | None
    frame_code_name: str | None
    frame_firstlineno: int | None
    frame_line: int | None
    frame_start_line: int | None
    frame_end_line: int | None
    frame_start_col: int | None
    frame_end_col: int | None
    frame_offset: int | None
    locals: object | None
    status: object | None


class CorePyInspectSignatureParamsRow(TypedDict):
    """Row model for core.py_inspect_signature_params."""

    repo: str
    commit: str
    mode: str
    signature_id: str
    param_index: int
    name: str | None
    kind: str | None
    default_present: bool | None
    default_value: object | None
    annotation_present: bool | None
    annotation_value: object | None
    status: object | None


class CorePyInspectSignaturesRow(TypedDict):
    """Row model for core.py_inspect_signatures."""

    repo: str
    commit: str
    mode: str
    signature_id: str
    object_id: str
    variant: str | None
    follow_wrapped: bool | None
    eval_str: bool | None
    effective_object_id: str | None
    sig_text: str | None
    sig_format: str | None
    return_annotation: object | None
    has_varargs: bool | None
    has_varkw: bool | None
    status: object | None


class CorePyInspectSourceRow(TypedDict):
    """Row model for core.py_inspect_source."""

    repo: str
    commit: str
    mode: str
    object_id: str
    file_name: str | None
    start_line: int | None
    line_count: int | None
    source_sha256: object | None
    source_preview: str | None
    status: object | None


class CorePyInspectUnwrapHopsRow(TypedDict):
    """Row model for core.py_inspect_unwrap_hops."""

    repo: str
    commit: str
    mode: str
    root_object_id: str
    hop: int
    object_id: str | None
    has_wrapped: bool | None
    has_signature_override: bool | None
    stop_reason: str | None
    status: object | None


class CorePySymBindingsRow(TypedDict):
    """Row model for core.py_sym_bindings."""

    repo: str
    commit: str
    rel_path: str
    binding_id: str
    scope_id: str
    name: str
    binding_kind: str
    declared_here: bool | None
    referenced_here: bool | None
    assigned_here: bool | None
    annotated_here: bool | None
    scoping_class: str | None


class CorePySymFunctionPartitionsRow(TypedDict):
    """Row model for core.py_sym_function_partitions."""

    repo: str
    commit: str
    rel_path: str
    scope_id: str
    parameters: object | None
    locals: object | None
    globals: object | None
    nonlocals: object | None
    frees: object | None


class CorePySymNamespaceEdgesRow(TypedDict):
    """Row model for core.py_sym_namespace_edges."""

    repo: str
    commit: str
    rel_path: str
    scope_id: str
    symbol_row_id: str
    name: str
    child_scope_id: str
    edge_kind: str
    is_ambiguous: bool | None


class CorePySymResolutionEdgesRow(TypedDict):
    """Row model for core.py_sym_resolution_edges."""

    repo: str
    commit: str
    rel_path: str
    edge_id: str
    src_binding_id: str
    dst_binding_id: str
    kind: str
    confidence: float | None
    reason: str | None


class CorePySymScopeEdgesRow(TypedDict):
    """Row model for core.py_sym_scope_edges."""

    repo: str
    commit: str
    rel_path: str
    parent_scope_id: str
    child_scope_id: str
    edge_kind: str


class CorePySymScopesRow(TypedDict):
    """Row model for core.py_sym_scopes."""

    repo: str
    commit: str
    rel_path: str
    scope_id: str
    scope_local_id: int | None
    parent_scope_id: str | None
    scope_type: str
    scope_name: str | None
    qualpath: str | None
    lineno: int | None
    is_nested: bool | None
    is_optimized: bool | None
    has_children: bool | None
    anchor_ast_node_id: str | None
    span_start_byte: int | None
    span_end_byte: int | None
    anchor_confidence: float | None
    anchor_reason: str | None


class CorePySymSymbolsRow(TypedDict):
    """Row model for core.py_sym_symbols."""

    repo: str
    commit: str
    rel_path: str
    scope_id: str
    symbol_row_id: str
    name: str
    is_referenced: bool | None
    is_assigned: bool | None
    is_imported: bool | None
    is_annotated: bool | None
    is_parameter: bool | None
    is_local: bool | None
    is_global: bool | None
    is_declared_global: bool | None
    is_nonlocal: bool | None
    is_free: bool | None
    is_namespace: bool | None
    namespace_count: int | None


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


class CoreSyntaxCallArgsRow(TypedDict):
    """Row model for core.syntax_call_args."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    arg_ordinal: int
    arg_kind: str
    arg_name: str | None
    arg_start_line: int
    arg_start_col: int
    arg_end_line: int
    arg_end_col: int
    arg_start_byte: int | None
    arg_end_byte: int | None
    arg_span_id: str | None
    arg_expr_node_id: str | None
    extras_json: object | None


class CoreSyntaxCallsRow(TypedDict):
    """Row model for core.syntax_calls."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    call_node_id: str | None
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
    callee_start_byte: int | None
    callee_end_byte: int | None
    extras_json: object | None


class CoreSyntaxCallsResolvedRow(TypedDict):
    """Row model for core.syntax_calls_resolved."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    call_node_id: str | None
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
    callee_start_byte: int | None
    callee_end_byte: int | None
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


class CoreSyntaxEdgesAugmentedRow(TypedDict):
    """Row model for core.syntax_edges_augmented."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    parent_node_id: str
    child_node_id: str
    edge_kind: str
    field_name: str | None
    child_ordinal: int


class CoreSyntaxFuncParamsRow(TypedDict):
    """Row model for core.syntax_func_params."""

    repo: str
    commit: str
    rel_path: str
    producer: str
    func_def_id: str
    param_def_id: str | None
    param_ordinal: int
    param_kind: str
    param_name: str
    param_start_line: int
    param_start_col: int
    param_end_line: int
    param_end_col: int
    param_start_byte: int | None
    param_end_byte: int | None
    param_span_id: str | None
    param_node_id: str | None
    extras_json: object | None


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


class CoreSyntaxNodesAugmentedRow(TypedDict):
    """Row model for core.syntax_nodes_augmented."""

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


class CoreTsChangedRangesRow(TypedDict):
    """Row model for core.ts_changed_ranges."""

    repo: str
    commit: str
    rel_path: str
    language: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int


class CoreTsEdgesRow(TypedDict):
    """Row model for core.ts_edges."""

    repo: str
    commit: str
    rel_path: str
    language: str
    parent_node_id: str
    child_node_id: str
    field_id: int | None
    field_name: str | None
    child_ordinal: int


class CoreTsLanguageMetadataRow(TypedDict):
    """Row model for core.ts_language_metadata."""

    repo: str
    commit: str
    language: str
    abi_version: int
    semantic_version: str
    node_kind_count: int
    field_count: int
    parse_state_count: int
    created_at: datetime


class CoreTsNodesRow(TypedDict):
    """Row model for core.ts_nodes."""

    repo: str
    commit: str
    rel_path: str
    language: str
    node_id: str
    node_type: str
    grammar_id: int | None
    kind_id: int | None
    is_named: bool
    is_missing: bool
    is_error: bool
    has_error: bool
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    parse_state: int | None
    next_parse_state: int | None
    text_preview: str | None
    extras_json: object | None


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
    extras_json: object | None


class CoreTsParseManifestRow(TypedDict):
    """Row model for core.ts_parse_manifest."""

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


class CoreTsSyntaxNodeXrefRow(TypedDict):
    """Row model for core.ts_syntax_node_xref."""

    repo: str
    commit: str
    rel_path: str
    language: str
    producer: str
    ts_node_id: str
    syntax_node_id: str | None
    match_kind: str
    candidate_count: int


class CoreTsTokensRow(TypedDict):
    """Row model for core.ts_tokens."""

    repo: str
    commit: str
    rel_path: str
    language: str
    token_id: str
    token_kind: str
    node_type: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
    extras_json: object | None


class CoreTsTriviaRow(TypedDict):
    """Row model for core.ts_trivia."""

    repo: str
    commit: str
    rel_path: str
    language: str
    trivia_id: str
    trivia_kind: str
    node_type: str
    start_byte: int
    end_byte: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    text_preview: str | None
    extras_json: object | None


class CoreTsWeldCoverageRow(TypedDict):
    """Row model for core.ts_weld_coverage."""

    repo: str
    commit: str
    rel_path: str
    language: str
    ts_node_count: int
    mapped_count: int
    coverage_ratio: float
