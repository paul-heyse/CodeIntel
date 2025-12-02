"""Tests for the ingestion plugin registry wiring and metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    default_code_profile,
    default_config_profile,
)
from codeintel.ingestion.plugins import (
    DEFAULT_INGEST_PLUGINS,
    IngestRuntimeScratch,
    get_ingest_registry,
    plan_ingest_plugins,
)
from codeintel.ingestion.plugins.protocol import IngestStage
from codeintel.ingestion.plugins.registry import IngestPluginRegistry, PlanOptions
from codeintel.ingestion.resources.registry import ResourceRegistry
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway


def test_registry_includes_all_expected_plugins() -> None:
    """Ensure the default registry is synchronized with the expected plugin set."""
    registry = get_ingest_registry()
    names = set(registry.list_names())
    expected = {
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "typing_ingest",
        "coverage_ingest",
        "tests_ingest",
        "docstrings_ingest",
        "config_ingest",
    }
    missing = expected - names
    if missing:
        pytest.fail(f"Missing ingestion plugins in registry: {sorted(missing)}")
    # Extra plugins are allowed (could be from entry points)


def test_metadata_exposes_tables_and_deps() -> None:
    """Verify registry metadata surfaces dependencies and tables accurately."""
    registry = get_ingest_registry()

    repo_scan = registry.get("repo_scan")
    meta = repo_scan.metadata
    if "core.modules" not in meta.produces_tables:
        pytest.fail("repo_scan metadata missing core.modules table")
    if meta.depends_on != ():
        pytest.fail(f"repo_scan depends_on should be empty, found: {meta.depends_on}")

    scip = registry.get("scip_ingest")
    scip_meta = scip.metadata
    if "core.scip_symbols" not in scip_meta.produces_tables:
        pytest.fail("scip_ingest metadata missing core.scip_symbols table")
    if "core.goid_crosswalk" not in scip_meta.produces_tables:
        pytest.fail("scip_ingest metadata missing core.goid_crosswalk table")
    if "repo_scan" not in scip_meta.depends_on:
        pytest.fail(f"scip_ingest depends_on incorrect: {scip_meta.depends_on}")

    docstrings = registry.get("docstrings_ingest")
    doc_meta = docstrings.metadata
    if "core.docstrings" not in doc_meta.produces_tables:
        pytest.fail("docstrings_ingest metadata missing core.docstrings table")
    if "repo_scan" not in doc_meta.depends_on:
        pytest.fail(f"docstrings_ingest should depend on repo_scan, found {doc_meta.depends_on}")


def test_plan_respects_dependencies() -> None:
    """Confirm plan ordering respects declared prerequisites."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=(
                "repo_scan",
                "scip_ingest",
                "ast_extract",
                "cst_extract",
                "docstrings_ingest",
            ),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )
    order = plan.ordered_names
    positions = {name: order.index(name) for name in order}

    if positions["repo_scan"] >= positions["scip_ingest"]:
        pytest.fail("repo_scan must precede scip_ingest")
    if positions["repo_scan"] >= positions["ast_extract"]:
        pytest.fail("repo_scan must precede ast_extract")
    if positions["repo_scan"] >= positions["cst_extract"]:
        pytest.fail("repo_scan must precede cst_extract")
    if positions["repo_scan"] >= positions["docstrings_ingest"]:
        pytest.fail("repo_scan must precede docstrings_ingest")


def test_custom_plugin_registry_execution(tmp_path: Path) -> None:
    """Smoke test exercising dependency expansion with custom plugins."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = BuildPaths.from_repo_root(repo_root)
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    code_profile = default_code_profile(repo_root)
    config_profile = default_config_profile(repo_root)
    tools = ToolsConfig.default()

    executed: list[str] = []

    # Create test plugins using class-based architecture
    @dataclass
    class AlphaPlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "alpha"
        plugin_description: ClassVar[str] = "First step"
        plugin_stage: ClassVar[IngestStage] = "scan"
        depends_on: ClassVar[tuple[str, ...]] = ()

        def compute(  # noqa: PLR6301
            self,
            ctx: IngestExecutionContext,
        ) -> Mapping[str, int] | None:
            _ = ctx  # Unused but required by protocol
            executed.append("alpha")
            return None

    @dataclass
    class BravoPlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "bravo"
        plugin_description: ClassVar[str] = "Second step"
        plugin_stage: ClassVar[IngestStage] = "parse"
        depends_on: ClassVar[tuple[str, ...]] = ("alpha",)

        def compute(  # noqa: PLR6301
            self,
            ctx: IngestExecutionContext,
        ) -> Mapping[str, int] | None:
            _ = ctx  # Unused but required by protocol
            executed.append("bravo")
            return None

    @dataclass
    class CharliePlugin(BaseIngestPlugin):
        plugin_name: ClassVar[str] = "charlie"
        plugin_description: ClassVar[str] = "Final step"
        plugin_stage: ClassVar[IngestStage] = "enrich"
        depends_on: ClassVar[tuple[str, ...]] = ("bravo",)

        def compute(  # noqa: PLR6301
            self,
            ctx: IngestExecutionContext,
        ) -> Mapping[str, int] | None:
            _ = ctx  # Unused but required by protocol
            executed.append("charlie")
            return None

    # Create a custom registry with our test plugins
    registry = IngestPluginRegistry()
    registry.register(AlphaPlugin())
    registry.register(BravoPlugin())
    registry.register(CharliePlugin())

    # Plan and execute all plugins (registry doesn't auto-expand dependencies)
    plan = registry.plan(
        PlanOptions(
            plugin_names=("alpha", "bravo", "charlie"),
            defaults=("alpha", "bravo", "charlie"),
        )
    )

    gateway = open_ingestion_gateway()
    try:
        scratch = IngestRuntimeScratch()
        resources = ResourceRegistry()
        for plugin in plan.plugins:
            ctx = IngestExecutionContext(
                gateway=gateway,
                snapshot=snapshot,
                paths=paths,
                tools=tools,
                code_profile=code_profile,
                config_profile=config_profile,
                resources=resources,
                scratch=scratch,
                plugin_name=plugin.metadata.name,
            )
            plugin.execute(ctx)
    finally:
        gateway.close()

    if executed != ["alpha", "bravo", "charlie"]:
        pytest.fail(f"Unexpected execution order: {executed}")


def test_disabled_plugins_are_skipped() -> None:
    """Verify disabled plugins are excluded from the plan."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=DEFAULT_INGEST_PLUGINS,
            disabled=("scip_ingest", "typing_ingest"),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )

    # Disabled plugins should not be in the plan
    ordered = plan.ordered_names
    if "scip_ingest" in ordered:
        pytest.fail("scip_ingest should be excluded when disabled")
    if "typing_ingest" in ordered:
        pytest.fail("typing_ingest should be excluded when disabled")

    # They should appear in skipped_plugins
    skipped_names = {s.name for s in plan.skipped_plugins}
    if "scip_ingest" not in skipped_names:
        pytest.fail("scip_ingest should be in skipped_plugins")
    if "typing_ingest" not in skipped_names:
        pytest.fail("typing_ingest should be in skipped_plugins")
