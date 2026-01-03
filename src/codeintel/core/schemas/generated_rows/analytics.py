"""Generated row models for insert helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

__all__ = [
    "AnalyticsCfgBlockMetricsRow",
    "AnalyticsCfgFunctionMetricsExtRow",
    "AnalyticsCfgFunctionMetricsRow",
    "AnalyticsConfigDataFlowRow",
    "AnalyticsConfigGraphMetricsKeysRow",
    "AnalyticsConfigGraphMetricsModulesRow",
    "AnalyticsConfigProjectionKeyEdgesRow",
    "AnalyticsConfigProjectionModuleEdgesRow",
    "AnalyticsConfigValuesRow",
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
    "AnalyticsFunctionAstFeaturesRow",
    "AnalyticsFunctionContractsRow",
    "AnalyticsFunctionEffectsRow",
    "AnalyticsFunctionTypesRow",
    "AnalyticsFunctionValidationRow",
    "AnalyticsGraphMetricsFunctionsExtRow",
    "AnalyticsGraphMetricsFunctionsRow",
    "AnalyticsGraphMetricsModulesExtRow",
    "AnalyticsGraphMetricsModulesRow",
    "AnalyticsGraphStatsRow",
    "AnalyticsGraphValidationRow",
    "AnalyticsHelloExampleRow",
    "AnalyticsSemanticRolesFunctionsRow",
    "AnalyticsSemanticRolesModulesRow",
    "AnalyticsStaticDiagnosticsRow",
    "AnalyticsSubsystemAgreementRow",
    "AnalyticsSubsystemGraphMetricsRow",
    "AnalyticsSubsystemModulesRow",
    "AnalyticsSubsystemProfileCacheRow",
    "AnalyticsSubsystemsRow",
    "AnalyticsSymbolGraphMetricsFunctionsRow",
    "AnalyticsSymbolGraphMetricsModulesRow",
    "AnalyticsTagsIndexRow",
    "AnalyticsTestCatalogRow",
]


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
    evidence_json: bytes | None
    call_chain_id: str
    call_chain_json: bytes | None
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
    reference_paths: bytes | None
    reference_modules: bytes | None
    reference_count: int


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
    constraints_json: bytes
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
    evidence_json: bytes | None
    rel_path: str
    lineno: int | None
    created_at: datetime


class AnalyticsDataModelUsageRow(TypedDict):
    """Row model for analytics.data_model_usage."""

    repo: str
    commit: str
    model_id: str
    function_goid_h128: int
    usage_kinds_json: bytes
    evidence_json: bytes | None
    context_json: bytes | None
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
    base_classes_json: bytes | None
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
    status_codes: bytes | None
    auth_required: bool | None
    command_name: str | None
    arguments_schema: bytes | None
    schedule: str | None
    trigger: str | None
    extra: bytes | None
    subsystem_id: str | None
    subsystem_name: str | None
    tags: bytes | None
    owners: bytes | None
    tests_touching: int | None
    failing_tests: int | None
    slow_tests: int | None
    flaky_tests: int | None
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
    modules_json: bytes
    usage_modes: bytes
    config_keys: bytes | None
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
    modes: bytes
    evidence_json: bytes | None
    created_at: datetime


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
    http_client_libs: bytes
    http_server_libs: bytes
    db_libs: bytes
    message_libs: bytes
    config_read_count: int
    feature_flag_count: int
    decorators: bytes
    libraries_used: bytes
    created_at: datetime


class AnalyticsFunctionContractsRow(TypedDict):
    """Row model for analytics.function_contracts."""

    repo: str
    commit: str
    function_goid_h128: int
    preconditions_json: bytes | None
    postconditions_json: bytes | None
    raises_json: bytes | None
    param_nullability_json: bytes | None
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
    effects_json: bytes | None
    created_at: datetime


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
    return_type: str | None
    type_comment: str | None
    param_types: bytes | None
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
    metadata: bytes | None
    created_at: datetime


class AnalyticsHelloExampleRow(TypedDict):
    """Row model for analytics.hello_example."""

    message: str
    value: int


class AnalyticsSemanticRolesFunctionsRow(TypedDict):
    """Row model for analytics.semantic_roles_functions."""

    repo: str
    commit: str
    function_goid_h128: int
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: bytes | None
    created_at: datetime


class AnalyticsSemanticRolesModulesRow(TypedDict):
    """Row model for analytics.semantic_roles_modules."""

    repo: str
    commit: str
    module: str
    role: str | None
    role_confidence: float | None
    role_sources_json: bytes | None
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
    modules_json: bytes | None
    entrypoints_json: bytes | None
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
    modules_json: bytes
    entrypoints_json: bytes | None
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
    includes: bytes | None
    excludes: bytes | None
    matches: bytes | None


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
    markers: bytes | None
    parametrized: bool | None
    flaky: bool | None
    created_at: datetime | None
