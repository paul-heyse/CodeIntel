"""Test environment orchestration functions.

This module provides functions for creating and managing test environments,
including file I/O, coverage generation, provisioning, and database seeding.
"""

from __future__ import annotations

from tests._helpers.orchestration.coverage_orchestration import (
    compute_coverage_edges,
    create_coverage_edge_env,
    generate_coverage_artifact,
    seed_coverage_rows,
)
from tests._helpers.orchestration.entrypoints_orchestration import (
    AppSeeds,
    make_coverage_seed_from_app,
    seed_app_modules_and_goids,
)
from tests._helpers.orchestration.graph_orchestration import (
    build_seeded_graph_engine,
    build_span_graph_components,
    collect_span_snapshot,
    create_span_test_env,
    generate_span_coverage,
)
from tests._helpers.orchestration.pipeline_orchestration import (
    build_graph_and_symbols,
    create_pipeline_env,
    generate_pipeline_coverage,
    load_coverage,
)
from tests._helpers.orchestration.provisioning import (
    build_callgraph_fixture_repo,
    docs_views_ready_gateway,
    graph_metrics_ready_gateway,
    make_repo_context,
    provision_docs_export_ready,
    provision_existing_repo,
    provision_gateway_with_repo,
    provision_graph_ready_repo,
    provision_ingested_repo,
    provisioned_gateway,
)
from tests._helpers.orchestration.repo_writers import (
    write_callgraph_alias_repo,
    write_coverage_driver,
    write_graph_metrics_repo,
    write_sample_repo,
)
from tests._helpers.orchestration.seeding import (
    seed_call_graph_scoping,
    seed_callgraph_goids,
    seed_cfg_dfg_for_metrics,
    seed_docs_export_invalid_profile,
    seed_function_graph_cycle,
    seed_graph_validation_gaps,
    seed_module_graph_inputs,
)
from tests._helpers.orchestration.seeding_docs import (
    seed_docs_export_minimal,
    seed_mcp_backend,
    seed_profile_data,
)

__all__ = [
    "AppSeeds",
    "build_callgraph_fixture_repo",
    "build_graph_and_symbols",
    "build_seeded_graph_engine",
    "build_span_graph_components",
    "collect_span_snapshot",
    "compute_coverage_edges",
    "create_coverage_edge_env",
    "create_pipeline_env",
    "create_span_test_env",
    "docs_views_ready_gateway",
    "generate_coverage_artifact",
    "generate_pipeline_coverage",
    "generate_span_coverage",
    "graph_metrics_ready_gateway",
    "load_coverage",
    "make_coverage_seed_from_app",
    "make_repo_context",
    "provision_docs_export_ready",
    "provision_existing_repo",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provision_ingested_repo",
    "provisioned_gateway",
    "seed_app_modules_and_goids",
    "seed_call_graph_scoping",
    "seed_callgraph_goids",
    "seed_cfg_dfg_for_metrics",
    "seed_coverage_rows",
    "seed_docs_export_invalid_profile",
    "seed_docs_export_minimal",
    "seed_function_graph_cycle",
    "seed_graph_validation_gaps",
    "seed_mcp_backend",
    "seed_module_graph_inputs",
    "seed_profile_data",
    "write_callgraph_alias_repo",
    "write_coverage_driver",
    "write_graph_metrics_repo",
    "write_sample_repo",
]
