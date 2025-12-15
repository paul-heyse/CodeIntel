"""Generated row models for insert helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

_TYPE_HINTS_DATETIME = datetime

__all__ = [
    "AnalyticsBehavioralCoverageRow",
    "AnalyticsCfgBlockMetricsRow",
    "AnalyticsCfgFunctionMetricsExtRow",
    "AnalyticsCfgFunctionMetricsRow",
    "AnalyticsConfigDataFlowRow",
    "AnalyticsConfigGraphMetricsKeysRow",
    "AnalyticsConfigGraphMetricsModulesRow",
    "AnalyticsConfigProjectionKeyEdgesRow",
    "AnalyticsConfigProjectionModuleEdgesRow",
    "AnalyticsConfigValuesRow",
    "AnalyticsCoverageFunctionsRow",
    "AnalyticsCoverageLinesRow",
    "AnalyticsDataModelFieldsRow",
    "AnalyticsDataModelRelationshipsRow",
    "AnalyticsDataModelUsageRow",
    "AnalyticsDataModelsRow",
    "AnalyticsDfgBlockMetricsRow",
    "AnalyticsDfgFunctionMetricsExtRow",
    "AnalyticsDfgFunctionMetricsRow",
    "AnalyticsEntrypointTestsRow",
    "AnalyticsEntrypointsRow",
    "AnalyticsExternalDependenciesRow",
    "AnalyticsExternalDependencyCallsRow",
    "AnalyticsFileProfileRow",
    "AnalyticsFunctionAstFeaturesRow",
    "AnalyticsFunctionContractsRow",
    "AnalyticsFunctionEffectsRow",
    "AnalyticsFunctionHistoryRow",
    "AnalyticsFunctionMetricsRow",
    "AnalyticsFunctionProfileRow",
    "AnalyticsFunctionTypesRow",
    "AnalyticsFunctionValidationRow",
    "AnalyticsGoidRiskFactorsRow",
    "AnalyticsGraphMetricsFunctionsExtRow",
    "AnalyticsGraphMetricsFunctionsRow",
    "AnalyticsGraphMetricsModulesExtRow",
    "AnalyticsGraphMetricsModulesRow",
    "AnalyticsGraphStatsRow",
    "AnalyticsGraphValidationRow",
    "AnalyticsHistoryTimeseriesRow",
    "AnalyticsHotspotsRow",
    "AnalyticsModuleProfileRow",
    "AnalyticsSemanticRolesFunctionsRow",
    "AnalyticsSemanticRolesModulesRow",
    "AnalyticsStaticDiagnosticsRow",
    "AnalyticsSubsystemAgreementRow",
    "AnalyticsSubsystemCoverageCacheRow",
    "AnalyticsSubsystemGraphMetricsRow",
    "AnalyticsSubsystemModulesRow",
    "AnalyticsSubsystemProfileCacheRow",
    "AnalyticsSubsystemsRow",
    "AnalyticsSymbolGraphMetricsFunctionsRow",
    "AnalyticsSymbolGraphMetricsModulesRow",
    "AnalyticsTagsIndexRow",
    "AnalyticsTestCatalogRow",
    "AnalyticsTestCoverageEdgesRow",
    "AnalyticsTestGraphMetricsFunctionsRow",
    "AnalyticsTestGraphMetricsTestsRow",
    "AnalyticsTestProfileRow",
    "AnalyticsTypednessRow",
]


class AnalyticsBehavioralCoverageRow(TypedDict):
    """Row model for analytics.behavioral_coverage."""

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    rel_path: str
    qualname: str | None
    behavior_tags: object
    tag_source: str
    heuristic_version: str | None
    llm_model: str | None
    llm_run_id: str | None
    created_at: datetime


class AnalyticsCfgBlockMetricsRow(TypedDict):
    """Row model for analytics.cfg_block_metrics."""

    function_goid_h128: int
    repo: str
    commit: str
    block_idx: int
    is_entry: bool | None
    is_exit: bool | None
    is_branch: bool | None
    is_join: bool | None
    dom_depth: int | None
    dominates_exit: bool | None
    bc_betweenness: float | None
    bc_closeness: float | None
    bc_eigenvector: float | None
    in_loop_scc: bool | None
    loop_header: bool | None
    loop_nesting_depth: int | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsCfgFunctionMetricsRow(TypedDict):
    """Row model for analytics.cfg_function_metrics."""

    function_goid_h128: int
    repo: str
    commit: str
    rel_path: str
    module: str | None
    qualname: str | None
    cfg_block_count: int | None
    cfg_edge_count: int | None
    cfg_has_cycles: bool | None
    cfg_scc_count: int | None
    cfg_longest_path_len: int | None
    cfg_avg_shortest_path_len: float | None
    cfg_branching_factor_mean: float | None
    cfg_branching_factor_max: int | None
    cfg_linear_block_fraction: float | None
    cfg_dom_tree_height: int | None
    cfg_dominance_frontier_size_mean: float | None
    cfg_dominance_frontier_size_max: int | None
    cfg_loop_count: int | None
    cfg_loop_nesting_depth_max: int | None
    cfg_bc_betweenness_max: float | None
    cfg_bc_betweenness_mean: float | None
    cfg_bc_closeness_mean: float | None
    cfg_bc_eigenvector_max: float | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsCfgFunctionMetricsExtRow(TypedDict):
    """Row model for analytics.cfg_function_metrics_ext."""

    function_goid_h128: int
    repo: str
    commit: str
    unreachable_block_count: int | None
    loop_header_count: int | None
    true_edge_count: int | None
    false_edge_count: int | None
    back_edge_count: int | None
    exception_edge_count: int | None
    fallthrough_edge_count: int | None
    loop_edge_count: int | None
    entry_exit_simple_paths: int | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsConfigDataFlowRow(TypedDict):
    """Row model for analytics.config_data_flow."""

    repo: str
    commit: str
    config_key: str
    config_path: str
    function_goid_h128: int
    usage_kind: str
    evidence_json: object | None
    call_chain_id: str
    call_chain_json: object | None
    created_at: datetime


class AnalyticsConfigGraphMetricsKeysRow(TypedDict):
    """Row model for analytics.config_graph_metrics_keys."""

    repo: str
    commit: str
    config_key: str
    degree: int | None
    weighted_degree: float | None
    betweenness: float | None
    closeness: float | None
    community_id: int | None
    created_at: datetime


class AnalyticsConfigGraphMetricsModulesRow(TypedDict):
    """Row model for analytics.config_graph_metrics_modules."""

    repo: str
    commit: str
    module: str
    degree: int | None
    weighted_degree: float | None
    betweenness: float | None
    closeness: float | None
    community_id: int | None
    created_at: datetime


class AnalyticsConfigProjectionKeyEdgesRow(TypedDict):
    """Row model for analytics.config_projection_key_edges."""

    repo: str
    commit: str
    src_key: str
    dst_key: str
    weight: float | None
    created_at: datetime


class AnalyticsConfigProjectionModuleEdgesRow(TypedDict):
    """Row model for analytics.config_projection_module_edges."""

    repo: str
    commit: str
    src_module: str
    dst_module: str
    weight: float | None
    created_at: datetime


class AnalyticsConfigValuesRow(TypedDict):
    """Row model for analytics.config_values."""

    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    reference_paths: object | None
    reference_modules: object | None
    reference_count: int


class AnalyticsCoverageFunctionsRow(TypedDict):
    """Row model for analytics.coverage_functions."""

    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    executable_lines: int | None
    covered_lines: int | None
    coverage_ratio: float | None
    tested: bool | None
    untested_reason: str | None
    created_at: datetime | None


class AnalyticsCoverageLinesRow(TypedDict):
    """Row model for analytics.coverage_lines."""

    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: datetime


class AnalyticsDataModelFieldsRow(TypedDict):
    """Row model for analytics.data_model_fields."""

    repo: str
    commit: str
    model_id: str
    field_name: str
    field_type: str | None
    required: bool
    has_default: bool
    default_expr: str | None
    constraints_json: object
    source: str
    rel_path: str
    lineno: int | None
    created_at: datetime


class AnalyticsDataModelRelationshipsRow(TypedDict):
    """Row model for analytics.data_model_relationships."""

    repo: str
    commit: str
    source_model_id: str
    target_model_id: str
    target_module: str | None
    target_model_name: str | None
    field_name: str
    relationship_kind: str
    multiplicity: str | None
    via: str | None
    evidence_json: object | None
    rel_path: str
    lineno: int | None
    created_at: datetime


class AnalyticsDataModelUsageRow(TypedDict):
    """Row model for analytics.data_model_usage."""

    repo: str
    commit: str
    model_id: str
    function_goid_h128: int
    usage_kinds_json: object
    evidence_json: object | None
    context_json: object | None
    created_at: datetime


class AnalyticsDataModelsRow(TypedDict):
    """Row model for analytics.data_models."""

    repo: str
    commit: str
    model_id: str
    goid_h128: int | None
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    base_classes_json: object | None
    doc_short: str | None
    doc_long: str | None
    created_at: datetime


class AnalyticsDfgBlockMetricsRow(TypedDict):
    """Row model for analytics.dfg_block_metrics."""

    function_goid_h128: int
    repo: str
    commit: str
    block_idx: int
    dfg_in_degree: int | None
    dfg_out_degree: int | None
    dfg_phi_in_degree: int | None
    dfg_phi_out_degree: int | None
    dfg_bc_betweenness: float | None
    dfg_bc_closeness: float | None
    dfg_bc_eigenvector: float | None
    dfg_in_chain: bool | None
    dfg_in_scc: bool | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsDfgFunctionMetricsRow(TypedDict):
    """Row model for analytics.dfg_function_metrics."""

    function_goid_h128: int
    repo: str
    commit: str
    rel_path: str
    module: str | None
    qualname: str | None
    dfg_block_count: int | None
    dfg_edge_count: int | None
    dfg_phi_edge_count: int | None
    dfg_symbol_count: int | None
    dfg_component_count: int | None
    dfg_scc_count: int | None
    dfg_has_cycles: bool | None
    dfg_longest_chain_len: int | None
    dfg_avg_shortest_path_len: float | None
    dfg_avg_in_degree: float | None
    dfg_avg_out_degree: float | None
    dfg_max_in_degree: int | None
    dfg_max_out_degree: int | None
    dfg_branchy_block_fraction: float | None
    dfg_bc_betweenness_max: float | None
    dfg_bc_betweenness_mean: float | None
    dfg_bc_eigenvector_max: float | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsDfgFunctionMetricsExtRow(TypedDict):
    """Row model for analytics.dfg_function_metrics_ext."""

    function_goid_h128: int
    repo: str
    commit: str
    data_flow_edge_count: int | None
    intra_block_edge_count: int | None
    use_kind_phi_count: int | None
    use_kind_data_flow_count: int | None
    use_kind_intra_block_count: int | None
    use_kind_other_count: int | None
    phi_edge_ratio: float | None
    entry_exit_simple_paths: int | None
    created_at: datetime
    metrics_version: int | None


class AnalyticsEntrypointTestsRow(TypedDict):
    """Row model for analytics.entrypoint_tests."""

    repo: str
    commit: str
    entrypoint_id: str
    test_id: str
    test_goid_h128: int | None
    coverage_ratio: float | None
    status: str | None
    duration_ms: float | None
    created_at: datetime


class AnalyticsEntrypointsRow(TypedDict):
    """Row model for analytics.entrypoints."""

    repo: str
    commit: str
    entrypoint_id: str
    kind: str
    framework: str | None
    handler_goid_h128: int
    handler_urn: str
    handler_rel_path: str
    handler_module: str
    handler_qualname: str
    http_method: str | None
    route_path: str | None
    status_codes: object | None
    auth_required: bool | None
    command_name: str | None
    arguments_schema: object | None
    schedule: str | None
    trigger: str | None
    extra: object | None
    subsystem_id: str | None
    subsystem_name: str | None
    tags: object | None
    owners: object | None
    tests_touching: int | None
    failing_tests: int | None
    slow_tests: int | None
    flaky_tests: int | None
    entrypoint_coverage_ratio: float | None
    last_test_status: str | None
    created_at: datetime


class AnalyticsExternalDependenciesRow(TypedDict):
    """Row model for analytics.external_dependencies."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str | None
    category: str | None
    language: str | None
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: object
    usage_modes: object
    config_keys: object | None
    risk_level: str | None
    created_at: datetime


class AnalyticsExternalDependencyCallsRow(TypedDict):
    """Row model for analytics.external_dependency_calls."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str | None
    language: str | None
    severity: str | None
    criticality: float | None
    risk_score: float | None
    matched_pattern: str | None
    function_goid_h128: int
    function_urn: str
    rel_path: str
    module: str
    qualname: str
    callsite_count: int
    modes: object
    evidence_json: object | None
    created_at: datetime


class AnalyticsFileProfileRow(TypedDict):
    """Row model for analytics.file_profile."""

    repo: str | None
    commit: str | None
    rel_path: str | None
    module: str | None
    language: str | None
    node_count: int | None
    function_count: int | None
    class_count: int | None
    avg_depth: float | None
    max_depth: int | None
    ast_complexity: float | None
    hotspot_score: float | None
    commit_count: int | None
    author_count: int | None
    lines_added: int | None
    lines_deleted: int | None
    annotation_ratio: float | None
    untyped_defs: int | None
    overlay_needed: bool | None
    type_error_count: int | None
    static_error_count: int | None
    has_static_errors: bool | None
    total_functions: int | None
    public_functions: int | None
    avg_loc: float | None
    max_loc: int | None
    avg_cyclomatic_complexity: float | None
    max_cyclomatic_complexity: int | None
    high_risk_function_count: int | None
    medium_risk_function_count: int | None
    max_risk_score: float | None
    file_coverage_ratio: float | None
    tested_function_count: int | None
    untested_function_count: int | None
    tests_touching: int | None
    tags: object | None
    owners: object | None
    created_at: datetime | None


class AnalyticsFunctionAstFeaturesRow(TypedDict):
    """Row model for analytics.function_ast_features."""

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    is_async: bool
    uses_network: bool
    uses_db: bool
    uses_filesystem: bool
    uses_subprocess: bool
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool
    http_client_libs: object
    http_server_libs: object
    db_libs: object
    message_libs: object
    config_read_count: int
    feature_flag_count: int
    decorators: object
    libraries_used: object
    created_at: datetime


class AnalyticsFunctionContractsRow(TypedDict):
    """Row model for analytics.function_contracts."""

    repo: str
    commit: str
    function_goid_h128: int
    preconditions_json: object | None
    postconditions_json: object | None
    raises_json: object | None
    param_nullability_json: object | None
    return_nullability: str | None
    contract_confidence: float | None
    created_at: datetime


class AnalyticsFunctionEffectsRow(TypedDict):
    """Row model for analytics.function_effects."""

    repo: str
    commit: str
    function_goid_h128: int
    is_pure: bool
    uses_io: bool
    touches_db: bool
    uses_time: bool
    uses_randomness: bool
    modifies_globals: bool
    modifies_closure: bool
    spawns_threads_or_tasks: bool
    has_transitive_effects: bool
    purity_confidence: float | None
    effects_json: object | None
    created_at: datetime


class AnalyticsFunctionHistoryRow(TypedDict):
    """Row model for analytics.function_history."""

    repo: str
    commit: str
    function_goid_h128: int
    urn: str
    rel_path: str
    module: str
    qualname: str
    created_in_commit: str | None
    created_at: datetime | None
    last_modified_commit: str | None
    last_modified_at: datetime | None
    age_days: int | None
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    churn_score: float
    stability_bucket: str
    history_window_start: datetime | None
    history_window_end: datetime | None
    created_at_row: datetime


class AnalyticsFunctionMetricsRow(TypedDict):
    """Row model for analytics.function_metrics."""

    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
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
    has_varargs: bool | None
    has_varkw: bool | None
    is_async: bool | None
    is_generator: bool | None
    return_count: int | None
    yield_count: int | None
    raise_count: int | None
    cyclomatic_complexity: int | None
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool | None
    complexity_bucket: str | None
    created_at: datetime | None


class AnalyticsFunctionProfileRow(TypedDict):
    """Row model for analytics.function_profile."""

    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
    module: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int | None
    logical_loc: int | None
    cyclomatic_complexity: int | None
    complexity_bucket: str | None
    param_count: int | None
    positional_params: int | None
    keyword_params: int | None
    vararg: bool | None
    kwarg: bool | None
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool | None
    total_params: int | None
    annotated_params: int | None
    return_type: str | None
    param_types: object | None
    fully_typed: bool | None
    partial_typed: bool | None
    untyped: bool | None
    typedness_bucket: str | None
    typedness_source: str | None
    file_typed_ratio: float | None
    static_error_count: int | None
    has_static_errors: bool | None
    executable_lines: int | None
    covered_lines: int | None
    coverage_ratio: float | None
    tested: bool | None
    untested_reason: str | None
    tests_touching: int | None
    failing_tests: int | None
    slow_tests: int | None
    flaky_tests: int | None
    last_test_status: str | None
    dominant_test_status: str | None
    slow_test_threshold_ms: float | None
    created_in_commit: str | None
    created_at_history: datetime | None
    last_modified_commit: str | None
    last_modified_at: datetime | None
    age_days: int | None
    commit_count: int | None
    author_count: int | None
    lines_added: int | None
    lines_deleted: int | None
    churn_score: float | None
    stability_bucket: str | None
    call_fan_in: int | None
    call_fan_out: int | None
    call_edge_in_count: int | None
    call_edge_out_count: int | None
    call_is_leaf: bool | None
    call_is_entrypoint: bool | None
    call_is_public: bool | None
    risk_score: float | None
    risk_level: str | None
    risk_component_coverage: float | None
    risk_component_complexity: float | None
    risk_component_static: float | None
    risk_component_hotspot: float | None
    is_pure: bool | None
    uses_io: bool | None
    touches_db: bool | None
    uses_time: bool | None
    uses_randomness: bool | None
    modifies_globals: bool | None
    modifies_closure: bool | None
    spawns_threads_or_tasks: bool | None
    has_transitive_effects: bool | None
    purity_confidence: float | None
    param_nullability_json: object | None
    return_nullability: str | None
    has_preconditions: bool | None
    has_postconditions: bool | None
    has_raises: bool | None
    contract_confidence: float | None
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: object | None
    tags: object | None
    owners: object | None
    doc_short: str | None
    doc_long: str | None
    doc_params: object | None
    doc_returns: object | None
    created_at: datetime | None


class AnalyticsFunctionTypesRow(TypedDict):
    """Row model for analytics.function_types."""

    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    total_params: int | None
    annotated_params: int | None
    unannotated_params: int | None
    param_typed_ratio: float | None
    has_return_annotation: bool | None
    return_type: str | None
    return_type_source: str | None
    type_comment: str | None
    param_types: object | None
    fully_typed: bool | None
    partial_typed: bool | None
    untyped: bool | None
    typedness_bucket: str | None
    typedness_source: str | None
    created_at: datetime | None


class AnalyticsFunctionValidationRow(TypedDict):
    """Row model for analytics.function_validation."""

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime


class AnalyticsGoidRiskFactorsRow(TypedDict):
    """Row model for analytics.goid_risk_factors."""

    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    loc: int | None
    logical_loc: int | None
    cyclomatic_complexity: int | None
    complexity_bucket: str | None
    typedness_bucket: str | None
    typedness_source: str | None
    hotspot_score: float | None
    file_typed_ratio: float | None
    static_error_count: int | None
    has_static_errors: bool | None
    executable_lines: int | None
    covered_lines: int | None
    coverage_ratio: float | None
    tested: bool | None
    test_count: int | None
    failing_test_count: int | None
    last_test_status: str | None
    risk_score: float | None
    risk_level: str | None
    tags: object | None
    owners: object | None
    created_at: datetime | None


class AnalyticsGraphMetricsFunctionsRow(TypedDict):
    """Row model for analytics.graph_metrics_functions."""

    repo: str
    commit: str
    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    call_in_degree: int
    call_out_degree: int
    call_pagerank: float | None
    call_betweenness: float | None
    call_closeness: float | None
    call_cycle_member: bool
    call_cycle_id: int | None
    call_layer: int | None
    created_at: datetime


class AnalyticsGraphMetricsFunctionsExtRow(TypedDict):
    """Row model for analytics.graph_metrics_functions_ext."""

    repo: str
    commit: str
    function_goid_h128: int
    call_betweenness: float | None
    call_closeness: float | None
    call_eigenvector: float | None
    call_harmonic: float | None
    call_core_number: int | None
    call_clustering_coeff: float | None
    call_triangle_count: int | None
    call_is_articulation: bool | None
    call_articulation_impact: int | None
    call_is_bridge_endpoint: bool | None
    call_component_id: int | None
    call_component_size: int | None
    call_scc_id: int | None
    call_scc_size: int | None
    call_ancestor_count: int | None
    call_descendant_count: int | None
    call_community_id: int | None
    created_at: datetime


class AnalyticsGraphMetricsModulesRow(TypedDict):
    """Row model for analytics.graph_metrics_modules."""

    repo: str
    commit: str
    module: str
    import_fan_in: int
    import_fan_out: int
    import_in_degree: int
    import_out_degree: int
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_cycle_member: bool
    import_cycle_id: int | None
    import_layer: int | None
    symbol_fan_in: int
    symbol_fan_out: int
    created_at: datetime


class AnalyticsGraphMetricsModulesExtRow(TypedDict):
    """Row model for analytics.graph_metrics_modules_ext."""

    repo: str
    commit: str
    module: str
    import_betweenness: float | None
    import_closeness: float | None
    import_eigenvector: float | None
    import_harmonic: float | None
    import_k_core: int | None
    import_constraint: float | None
    import_effective_size: float | None
    import_rich_club: bool | None
    import_shell_index: int | None
    import_community_id: int | None
    import_component_id: int | None
    import_component_size: int | None
    import_scc_id: int | None
    import_scc_size: int | None
    created_at: datetime


class AnalyticsGraphStatsRow(TypedDict):
    """Row model for analytics.graph_stats."""

    graph_name: str
    repo: str
    commit: str
    node_count: int | None
    edge_count: int | None
    weak_component_count: int | None
    scc_count: int | None
    component_layers: int | None
    avg_clustering: float | None
    diameter_estimate: float | None
    avg_shortest_path_estimate: float | None
    created_at: datetime


class AnalyticsGraphValidationRow(TypedDict):
    """Row model for analytics.graph_validation."""

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


class AnalyticsHistoryTimeseriesRow(TypedDict):
    """Row model for analytics.history_timeseries."""

    repo: str
    entity_kind: str
    entity_stable_id: str
    function_goid_h128: int | None
    module: str | None
    rel_path: str
    language: str
    qualname: str | None
    commit: str
    commit_ts: datetime
    loc: int | None
    cyclomatic_complexity: int | None
    coverage_ratio: float | None
    static_error_count: int | None
    typedness_bucket: str | None
    risk_score: float | None
    risk_level: str | None
    bucket_label: str | None
    created_at_row: datetime


class AnalyticsHotspotsRow(TypedDict):
    """Row model for analytics.hotspots."""

    rel_path: str | None
    commit_count: int | None
    author_count: int | None
    lines_added: int | None
    lines_deleted: int | None
    complexity: float | None
    score: float | None


class AnalyticsModuleProfileRow(TypedDict):
    """Row model for analytics.module_profile."""

    repo: str | None
    commit: str | None
    module: str | None
    path: str | None
    language: str | None
    file_count: int | None
    total_loc: int | None
    total_logical_loc: int | None
    function_count: int | None
    class_count: int | None
    avg_file_complexity: float | None
    max_file_complexity: float | None
    high_risk_function_count: int | None
    medium_risk_function_count: int | None
    low_risk_function_count: int | None
    max_risk_score: float | None
    avg_risk_score: float | None
    module_coverage_ratio: float | None
    tested_function_count: int | None
    untested_function_count: int | None
    import_fan_in: int | None
    import_fan_out: int | None
    cycle_group: int | None
    in_cycle: bool | None
    role: str | None
    role_confidence: float | None
    role_sources_json: object | None
    tags: object | None
    owners: object | None
    created_at: datetime | None


class AnalyticsSemanticRolesFunctionsRow(TypedDict):
    """Row model for analytics.semantic_roles_functions."""

    repo: str
    commit: str
    function_goid_h128: int
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: object | None
    created_at: datetime


class AnalyticsSemanticRolesModulesRow(TypedDict):
    """Row model for analytics.semantic_roles_modules."""

    repo: str
    commit: str
    module: str
    role: str | None
    role_confidence: float | None
    role_sources_json: object | None
    created_at: datetime


class AnalyticsStaticDiagnosticsRow(TypedDict):
    """Row model for analytics.static_diagnostics."""

    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool


class AnalyticsSubsystemAgreementRow(TypedDict):
    """Row model for analytics.subsystem_agreement."""

    repo: str
    commit: str
    module: str
    subsystem_id: str | None
    import_community_id: int | None
    agrees: bool | None
    created_at: datetime


class AnalyticsSubsystemCoverageCacheRow(TypedDict):
    """Row model for analytics.subsystem_coverage_cache."""

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    function_count: int | None
    risk_level: str | None
    avg_risk_score: float | None
    max_risk_score: float | None
    test_count: int | None
    passed_test_count: int | None
    failed_test_count: int | None
    skipped_test_count: int | None
    xfail_test_count: int | None
    flaky_test_count: int | None
    total_functions_covered: int | None
    avg_functions_covered: float | None
    max_functions_covered: float | None
    min_functions_covered: float | None
    function_coverage_ratio: float | None
    created_at: datetime | None


class AnalyticsSubsystemGraphMetricsRow(TypedDict):
    """Row model for analytics.subsystem_graph_metrics."""

    repo: str
    commit: str
    subsystem_id: str
    import_in_degree: float | None
    import_out_degree: float | None
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_layer: int | None
    created_at: datetime


class AnalyticsSubsystemModulesRow(TypedDict):
    """Row model for analytics.subsystem_modules."""

    repo: str
    commit: str
    subsystem_id: str
    module: str
    role: str | None


class AnalyticsSubsystemProfileCacheRow(TypedDict):
    """Row model for analytics.subsystem_profile_cache."""

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    modules_json: object | None
    entrypoints_json: object | None
    internal_edge_count: int | None
    external_edge_count: int | None
    fan_in: int | None
    fan_out: int | None
    function_count: int | None
    avg_risk_score: float | None
    max_risk_score: float | None
    high_risk_function_count: int | None
    risk_level: str | None
    import_in_degree: float | None
    import_out_degree: float | None
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_layer: int | None
    created_at: datetime | None


class AnalyticsSubsystemsRow(TypedDict):
    """Row model for analytics.subsystems."""

    repo: str
    commit: str
    subsystem_id: str
    name: str
    description: str | None
    module_count: int
    modules_json: object
    entrypoints_json: object | None
    internal_edge_count: int
    external_edge_count: int
    fan_in: int
    fan_out: int
    function_count: int
    avg_risk_score: float | None
    max_risk_score: float | None
    high_risk_function_count: int
    risk_level: str | None
    created_at: datetime


class AnalyticsSymbolGraphMetricsFunctionsRow(TypedDict):
    """Row model for analytics.symbol_graph_metrics_functions."""

    repo: str
    commit: str
    function_goid_h128: int
    symbol_betweenness: float | None
    symbol_closeness: float | None
    symbol_eigenvector: float | None
    symbol_harmonic: float | None
    symbol_k_core: int | None
    symbol_constraint: float | None
    symbol_effective_size: float | None
    symbol_community_id: int | None
    symbol_component_id: int | None
    symbol_component_size: int | None
    created_at: datetime


class AnalyticsSymbolGraphMetricsModulesRow(TypedDict):
    """Row model for analytics.symbol_graph_metrics_modules."""

    repo: str
    commit: str
    module: str
    symbol_betweenness: float | None
    symbol_closeness: float | None
    symbol_eigenvector: float | None
    symbol_harmonic: float | None
    symbol_k_core: int | None
    symbol_constraint: float | None
    symbol_effective_size: float | None
    symbol_community_id: int | None
    symbol_component_id: int | None
    symbol_component_size: int | None
    created_at: datetime


class AnalyticsTagsIndexRow(TypedDict):
    """Row model for analytics.tags_index."""

    tag: str
    description: str | None
    includes: object | None
    excludes: object | None
    matches: object | None


class AnalyticsTestCatalogRow(TypedDict):
    """Row model for analytics.test_catalog."""

    test_id: str
    test_goid_h128: float | None
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    qualname: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: object | None
    parametrized: bool | None
    flaky: bool | None
    created_at: datetime | None


class AnalyticsTestCoverageEdgesRow(TypedDict):
    """Row model for analytics.test_coverage_edges."""

    test_id: str | None
    test_goid_h128: int | None
    function_goid_h128: int | None
    urn: str | None
    repo: str | None
    commit: str | None
    rel_path: str | None
    qualname: str | None
    covered_lines: int | None
    executable_lines: int | None
    coverage_ratio: float | None
    last_status: str | None
    created_at: datetime | None


class AnalyticsTestGraphMetricsFunctionsRow(TypedDict):
    """Row model for analytics.test_graph_metrics_functions."""

    function_goid_h128: int
    repo: str
    commit: str
    tests_degree: int | None
    tests_weighted_degree: float | None
    tests_degree_centrality: float | None
    proj_degree: int | None
    proj_weight: float | None
    proj_clustering: float | None
    proj_betweenness: float | None
    tests_risk_weighted_degree: float | None
    created_at: datetime


class AnalyticsTestGraphMetricsTestsRow(TypedDict):
    """Row model for analytics.test_graph_metrics_tests."""

    test_id: str
    repo: str
    commit: str
    degree: int | None
    weighted_degree: float | None
    degree_centrality: float | None
    proj_degree: int | None
    proj_weight: float | None
    proj_clustering: float | None
    proj_betweenness: float | None
    risk_weighted_degree: float | None
    created_at: datetime


class AnalyticsTestProfileRow(TypedDict):
    """Row model for analytics.test_profile."""

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: object | None
    flaky: bool | None
    last_run_at: datetime | None
    functions_covered: object | None
    functions_covered_count: int | None
    primary_function_goids: object | None
    subsystems_covered: object | None
    subsystems_covered_count: int | None
    primary_subsystem_id: str | None
    assert_count: int | None
    raise_count: int | None
    uses_parametrize: bool | None
    uses_fixtures: bool | None
    io_bound: bool | None
    uses_network: bool | None
    uses_db: bool | None
    uses_filesystem: bool | None
    uses_subprocess: bool | None
    flakiness_score: float | None
    importance_score: float | None
    notes: str | None
    tg_degree: int | None
    tg_weighted_degree: float | None
    tg_proj_degree: int | None
    tg_proj_weight: float | None
    tg_proj_clustering: float | None
    tg_proj_betweenness: float | None
    created_at: datetime


class AnalyticsTypednessRow(TypedDict):
    """Row model for analytics.typedness."""

    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio: object
    untyped_defs: int
    overlay_needed: bool
