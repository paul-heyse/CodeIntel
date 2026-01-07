"""Test environment orchestration functions.

This module provides functions for creating and managing test environments,
including file I/O, provisioning, database seeding, and gateway setup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.fixtures.repos import (
    write_callgraph_alias_repo,
    write_graph_metrics_repo,
    write_sample_repo,
)
from tests._helpers.orchestration.entrypoints_orchestration import (
    AppSeeds,
    seed_app_modules_and_goids,
)
from tests._helpers.orchestration.gateway import (
    DuckDBConnection,
    GatewayFactory,
    seed_tables,
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
    "DuckDBConnection",
    "GatewayFactory",
    "GitRepoContext",
    "ToolingContext",
    "ToolingOutputs",
    "build_callgraph_fixture_repo",
    "build_seeded_graph_engine",
    "build_span_graph_components",
    "build_tooling_context",
    "collect_span_snapshot",
    "create_span_test_env",
    "docs_views_ready_gateway",
    "graph_metrics_ready_gateway",
    "init_git_repo_with_history",
    "make_repo_context",
    "make_tools_config",
    "provision_docs_export_ready",
    "provision_existing_repo",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provision_hamilton_repo",
    "provision_ingested_repo",
    "provisioned_gateway",
    "run_static_tooling",
    "seed_app_modules_and_goids",
    "seed_call_graph_scoping",
    "seed_callgraph_goids",
    "seed_cfg_dfg_for_metrics",
    "seed_docs_export_invalid_profile",
    "seed_docs_export_minimal",
    "seed_function_graph_cycle",
    "seed_graph_validation_gaps",
    "seed_mcp_backend",
    "seed_module_graph_inputs",
    "seed_profile_data",
    "seed_tables",
    "write_callgraph_alias_repo",
    "write_graph_metrics_repo",
    "write_sample_repo",
]

if TYPE_CHECKING:
    from tests._helpers.orchestration.graph_orchestration import (
        build_seeded_graph_engine,
        build_span_graph_components,
        collect_span_snapshot,
        create_span_test_env,
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
        provision_hamilton_repo,
        provision_ingested_repo,
        provisioned_gateway,
    )
    from tests._helpers.orchestration.tooling import (
        GitRepoContext,
        ToolingContext,
        ToolingOutputs,
        build_tooling_context,
        init_git_repo_with_history,
        make_tools_config,
        run_static_tooling,
    )

_LAZY_PROVISIONING = {
    "build_callgraph_fixture_repo": "tests._helpers.orchestration.provisioning",
    "docs_views_ready_gateway": "tests._helpers.orchestration.provisioning",
    "graph_metrics_ready_gateway": "tests._helpers.orchestration.provisioning",
    "make_repo_context": "tests._helpers.orchestration.provisioning",
    "provision_docs_export_ready": "tests._helpers.orchestration.provisioning",
    "provision_existing_repo": "tests._helpers.orchestration.provisioning",
    "provision_gateway_with_repo": "tests._helpers.orchestration.provisioning",
    "provision_graph_ready_repo": "tests._helpers.orchestration.provisioning",
    "provision_hamilton_repo": "tests._helpers.orchestration.provisioning",
    "provision_ingested_repo": "tests._helpers.orchestration.provisioning",
    "provisioned_gateway": "tests._helpers.orchestration.provisioning",
}

_LAZY_GRAPH = {
    "build_seeded_graph_engine": "tests._helpers.orchestration.graph_orchestration",
    "build_span_graph_components": "tests._helpers.orchestration.graph_orchestration",
    "collect_span_snapshot": "tests._helpers.orchestration.graph_orchestration",
    "create_span_test_env": "tests._helpers.orchestration.graph_orchestration",
}

_LAZY_TOOLING = {
    "GitRepoContext": "tests._helpers.orchestration.tooling",
    "ToolingContext": "tests._helpers.orchestration.tooling",
    "ToolingOutputs": "tests._helpers.orchestration.tooling",
    "build_tooling_context": "tests._helpers.orchestration.tooling",
    "init_git_repo_with_history": "tests._helpers.orchestration.tooling",
    "make_tools_config": "tests._helpers.orchestration.tooling",
    "run_static_tooling": "tests._helpers.orchestration.tooling",
}

if TYPE_CHECKING:
    from tests._helpers.orchestration import provisioning as _provisioning

    build_callgraph_fixture_repo = _provisioning.build_callgraph_fixture_repo
    docs_views_ready_gateway = _provisioning.docs_views_ready_gateway
    graph_metrics_ready_gateway = _provisioning.graph_metrics_ready_gateway
    make_repo_context = _provisioning.make_repo_context
    provision_docs_export_ready = _provisioning.provision_docs_export_ready
    provision_existing_repo = _provisioning.provision_existing_repo
    provision_gateway_with_repo = _provisioning.provision_gateway_with_repo
    provision_graph_ready_repo = _provisioning.provision_graph_ready_repo
    provision_hamilton_repo = _provisioning.provision_hamilton_repo
    provision_ingested_repo = _provisioning.provision_ingested_repo
    provisioned_gateway = _provisioning.provisioned_gateway


def __getattr__(name: str) -> object:
    if name in _LAZY_PROVISIONING:
        module_name = _LAZY_PROVISIONING[name]
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    if name in _LAZY_GRAPH:
        module_name = _LAZY_GRAPH[name]
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    if name in _LAZY_TOOLING:
        module_name = _LAZY_TOOLING[name]
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    message = f"module {__name__} has no attribute {name}"
    raise AttributeError(message)
