"""Shared test helpers package.

This package intentionally avoids importing heavy test/build dependencies at
import time. Many test modules import submodules such as
``tests._helpers.assertions.expectation_assertions``; Python executes this
``__init__`` first, so eager imports can make unrelated tests fail during
collection.

To keep test collection resilient, names re-exported from ``tests._helpers`` are
resolved lazily via ``__getattr__``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

_MODULE_EXPORTS: Final[dict[str, tuple[str, ...]]] = {
    "tests._helpers.build": (
        "ManifestParams",
        "RecordingProviders",
        "make_build_config",
        "make_build_paths",
        "make_snapshot",
        "sample_manifest",
        "sample_target_graph",
        "write_build_config",
    ),
    "tests._helpers.catalogs": ("ensure_catalog_with_goids", "seed_goids_from_catalog"),
    "tests._helpers.build_config_overrides": (
        "reload_build_config",
        "write_build_config_sections",
    ),
    "tests._helpers.cli_context": (
        "CliTestContext",
        "cli_test_context_with_seeds",
        "create_cli_test_context",
        "make_command_context",
        "params",
    ),
    "tests._helpers.configs.provisioning_config": (
        "CallgraphFixtureOptions",
        "GatewayOptions",
        "GraphMetricsGatewayOptions",
        "ProvisionedGateway",
        "ProvisioningConfig",
        "provisioning_gateway_options",
    ),
    "tests._helpers.fixtures.snapshots": (
        "DEFAULT_VARIANT",
        "GOLDEN_VARIANT",
        "METRICS_VARIANT",
        "SNAPSHOT_VARIANTS",
        "SPAN_VARIANT",
        "SnapshotVariants",
        "SnapshotVariant",
    ),
    "tests._helpers.context": (
        "QueryRow",
        "SeedPack",
        "TestContext",
    ),
    "tests._helpers.env": ("build_test_gateway", "create_provisioned_test_env", "create_test_env"),
    "tests._helpers.evidence": ("build_entrypoint_evidence",),
    "tests._helpers.fakes.httpx_clients": ("RecordingAsyncClient",),
    "tests._helpers.fixtures.repos": (
        "RepoFixture",
        "RepoFixtureSpec",
        "RepoFixtureWriter",
        "write_callgraph_alias_repo",
        "write_canonical_repo",
        "write_generated_noise_fixture",
        "write_graph_metrics_repo",
        "write_large_file_fixture",
        "write_monorepo_fixture",
        "write_sample_repo",
        "write_scoped_paths_fixture",
        "write_tree",
    ),
    "tests._helpers.fixtures.graphs": (
        "chain_graph",
        "complete_digraph",
        "cyclic_graph",
        "diamond_graph",
        "disconnected_graph",
        "star_graph",
    ),
    "tests._helpers.gateway": ("GatewayFactory", "analytics_gateway"),
    "tests._helpers.hamilton_harness_artifacts": ("HarnessArtifacts",),
    "tests._helpers.hamilton_manifest_priming": ("ManifestPriming",),
    "tests._helpers.hamilton_execution": (
        "HamiltonTestBuilder",
        "HamiltonTestContext",
        "execute_hamilton_target",
        "execute_hamilton_targets",
        "execute_hamilton_target_async",
    ),
    "tests._helpers.immutability": ("assert_all_frozen", "assert_frozen"),
    "tests._helpers.manifests": (
        "assert_succeeded",
        "load_manifest_index",
        "prime_manifest",
        "prime_modules_manifest",
    ),
    "tests._helpers.modules_expectations": (
        "module_paths_expected_from_repo_tree",
        "modules_expected_from_env",
        "modules_expected_from_repo_tree",
    ),
    "tests._helpers.orchestration.provisioning": (
        "build_callgraph_fixture_repo",
        "docs_views_ready_gateway",
        "graph_metrics_ready_gateway",
        "provision_docs_export_ready",
        "provision_gateway_with_repo",
        "provision_graph_ready_repo",
        "provision_hamilton_repo",
        "provisioned_gateway",
    ),
    "tests._helpers.orchestration.seeding": (
        "seed_call_graph_scoping",
        "seed_docs_export_invalid_profile",
        "seed_function_graph_cycle",
        "seed_graph_validation_gaps",
        "seed_module_graph_inputs",
    ),
    "tests._helpers.orchestration.seeding_docs": (
        "seed_docs_export_minimal",
        "seed_mcp_backend",
        "seed_profile_data",
    ),
    "tests._helpers.orchestration.repo_registry": (
        "RepoFixtureEntry",
        "build_repo_fixture",
        "get_repo_fixture",
        "list_repo_fixtures",
    ),
    "tests._helpers.fixtures.rows": ("function_meta", "function_metrics_row", "module_row"),
    "tests._helpers.scenarios": (
        "ScenarioConfig",
        "TestScenario",
        "full_context",
        "graph_context",
        "minimal_context",
    ),
    "tests._helpers.seeds": (
        "CLI_CORE_PACK",
        "CORE_PACK",
        "DATA_MODELS_PACK",
        "FUNCTION_TYPES_PACK",
        "GRAPH_HANDLER_PACK",
        "GRAPH_PACK",
        "METRICS_PACK",
        "OPERATION_REGISTRY_PACK",
        "STORAGE_PROFILE_PACK",
        "SUBSYSTEM_ANALYTICS_PACK",
        "SUBSYSTEM_HANDLER_PACK",
    ),
    "tests._helpers.tooling_audit": (
        "ToolCall",
        "ToolCallLog",
        "assert_tool_called",
        "require_tooling",
    ),
    "tests._helpers.tool_payloads": ("pytest_report_payload",),
}

_EXPORT_TO_MODULE: Final[dict[str, str]] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

_ALL_EXPORTS: Final[tuple[str, ...]] = tuple(sorted(_EXPORT_TO_MODULE))

if TYPE_CHECKING:
    from tests._helpers.build import (
        ManifestParams,
        RecordingProviders,
        make_build_config,
        make_build_paths,
        make_snapshot,
        sample_manifest,
        sample_target_graph,
        write_build_config,
    )
    from tests._helpers.build_config_overrides import (
        reload_build_config,
        write_build_config_sections,
    )
    from tests._helpers.catalogs import ensure_catalog_with_goids, seed_goids_from_catalog
    from tests._helpers.cli_context import (
        CliTestContext,
        cli_test_context_with_seeds,
        create_cli_test_context,
        make_command_context,
        params,
    )
    from tests._helpers.configs.provisioning_config import (
        CallgraphFixtureOptions,
        GatewayOptions,
        GraphMetricsGatewayOptions,
        ProvisionedGateway,
        ProvisioningConfig,
        provisioning_gateway_options,
    )
    from tests._helpers.context import QueryRow, SeedPack, TestContext
    from tests._helpers.env import build_test_gateway, create_provisioned_test_env, create_test_env
    from tests._helpers.evidence import build_entrypoint_evidence
    from tests._helpers.fakes.httpx_clients import RecordingAsyncClient
    from tests._helpers.fixtures.graphs import (
        chain_graph,
        complete_digraph,
        cyclic_graph,
        diamond_graph,
        disconnected_graph,
        star_graph,
    )
    from tests._helpers.fixtures.repos import (
        RepoFixture,
        RepoFixtureSpec,
        RepoFixtureWriter,
        write_callgraph_alias_repo,
        write_canonical_repo,
        write_generated_noise_fixture,
        write_graph_metrics_repo,
        write_large_file_fixture,
        write_monorepo_fixture,
        write_sample_repo,
        write_scoped_paths_fixture,
        write_tree,
    )
    from tests._helpers.fixtures.rows import function_meta, function_metrics_row, module_row
    from tests._helpers.fixtures.snapshots import (
        DEFAULT_VARIANT,
        SNAPSHOT_VARIANTS,
        SnapshotVariant,
        SnapshotVariants,
    )
    from tests._helpers.gateway import GatewayFactory, analytics_gateway
    from tests._helpers.hamilton_execution import (
        HamiltonTestBuilder,
        HamiltonTestContext,
        execute_hamilton_target,
        execute_hamilton_target_async,
        execute_hamilton_targets,
    )
    from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
    from tests._helpers.hamilton_manifest_priming import ManifestPriming
    from tests._helpers.immutability import assert_all_frozen, assert_frozen
    from tests._helpers.manifests import (
        assert_succeeded,
        load_manifest_index,
        prime_manifest,
        prime_modules_manifest,
    )
    from tests._helpers.modules_expectations import (
        module_paths_expected_from_repo_tree,
        modules_expected_from_env,
        modules_expected_from_repo_tree,
    )
    from tests._helpers.orchestration.provisioning import (
        build_callgraph_fixture_repo,
        docs_views_ready_gateway,
        graph_metrics_ready_gateway,
        provision_docs_export_ready,
        provision_gateway_with_repo,
        provision_graph_ready_repo,
        provision_hamilton_repo,
        provisioned_gateway,
    )
    from tests._helpers.orchestration.repo_registry import (
        RepoFixtureEntry,
        build_repo_fixture,
        get_repo_fixture,
        list_repo_fixtures,
    )
    from tests._helpers.orchestration.seeding import (
        seed_call_graph_scoping,
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
    from tests._helpers.scenarios import (
        ScenarioConfig,
        TestScenario,
        full_context,
        graph_context,
        minimal_context,
    )
    from tests._helpers.seeds import (
        CLI_CORE_PACK,
        CORE_PACK,
        DATA_MODELS_PACK,
        FUNCTION_TYPES_PACK,
        GRAPH_HANDLER_PACK,
        GRAPH_PACK,
        METRICS_PACK,
        OPERATION_REGISTRY_PACK,
        STORAGE_PROFILE_PACK,
        SUBSYSTEM_ANALYTICS_PACK,
        SUBSYSTEM_HANDLER_PACK,
    )
    from tests._helpers.tool_payloads import (
        pytest_report_payload,
    )
    from tests._helpers.tooling_audit import (
        ToolCall,
        ToolCallLog,
        assert_tool_called,
        require_tooling,
    )

    _TYPE_CHECKING_EXPORTS = (
        CallgraphFixtureOptions,
        CLI_CORE_PACK,
        CORE_PACK,
        DEFAULT_VARIANT,
        SnapshotVariant,
        SnapshotVariants,
        SNAPSHOT_VARIANTS,
        DATA_MODELS_PACK,
        FUNCTION_TYPES_PACK,
        GRAPH_HANDLER_PACK,
        GRAPH_PACK,
        GatewayFactory,
        GatewayOptions,
        GraphMetricsGatewayOptions,
        HarnessArtifacts,
        HamiltonTestBuilder,
        HamiltonTestContext,
        ManifestPriming,
        METRICS_PACK,
        ManifestParams,
        OPERATION_REGISTRY_PACK,
        ProvisionedGateway,
        ProvisioningConfig,
        QueryRow,
        RecordingAsyncClient,
        RecordingProviders,
        ScenarioConfig,
        STORAGE_PROFILE_PACK,
        SUBSYSTEM_ANALYTICS_PACK,
        SUBSYSTEM_HANDLER_PACK,
        SeedPack,
        TestContext,
        TestScenario,
        CliTestContext,
        analytics_gateway,
        assert_all_frozen,
        assert_frozen,
        build_callgraph_fixture_repo,
        build_entrypoint_evidence,
        build_test_gateway,
        chain_graph,
        cli_test_context_with_seeds,
        complete_digraph,
        create_cli_test_context,
        create_provisioned_test_env,
        create_test_env,
        cyclic_graph,
        diamond_graph,
        disconnected_graph,
        docs_views_ready_gateway,
        execute_hamilton_target,
        execute_hamilton_targets,
        execute_hamilton_target_async,
        ensure_catalog_with_goids,
        full_context,
        function_meta,
        function_metrics_row,
        graph_context,
        graph_metrics_ready_gateway,
        make_build_config,
        make_build_paths,
        make_command_context,
        make_snapshot,
        minimal_context,
        module_paths_expected_from_repo_tree,
        module_row,
        modules_expected_from_env,
        modules_expected_from_repo_tree,
        params,
        provision_docs_export_ready,
        provision_gateway_with_repo,
        provision_graph_ready_repo,
        provision_hamilton_repo,
        provisioned_gateway,
        provisioning_gateway_options,
        sample_manifest,
        sample_target_graph,
        seed_call_graph_scoping,
        seed_docs_export_invalid_profile,
        seed_docs_export_minimal,
        seed_function_graph_cycle,
        seed_goids_from_catalog,
        seed_graph_validation_gaps,
        seed_mcp_backend,
        seed_module_graph_inputs,
        seed_profile_data,
        star_graph,
        RepoFixture,
        RepoFixtureEntry,
        RepoFixtureSpec,
        RepoFixtureWriter,
        write_callgraph_alias_repo,
        write_canonical_repo,
        write_generated_noise_fixture,
        write_graph_metrics_repo,
        write_large_file_fixture,
        write_monorepo_fixture,
        write_sample_repo,
        write_scoped_paths_fixture,
        write_tree,
        ToolCall,
        ToolCallLog,
        assert_tool_called,
        require_tooling,
        build_repo_fixture,
        get_repo_fixture,
        list_repo_fixtures,
        pytest_report_payload,
        reload_build_config,
        assert_succeeded,
        load_manifest_index,
        prime_manifest,
        prime_modules_manifest,
        write_build_config,
        write_build_config_sections,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve re-exported test helpers.

    Parameters
    ----------
    name
        Attribute name requested from this package.

    Returns
    -------
    object
        Resolved attribute value.

    Raises
    ------
    AttributeError
        If ``name`` is not a known export.
    """
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ALL_EXPORTS))
