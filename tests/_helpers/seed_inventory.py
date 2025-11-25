"""Lightweight inventory linking seed helpers/builders to tables and consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SeedInventoryEntry:
    """Describes a seeder/builder and its downstream consumers."""

    name: str
    kind: str
    tables: list[str]
    consumers: list[str]
    notes: str | None = None


SEED_INVENTORY: Final[list[SeedInventoryEntry]] = [
    SeedInventoryEntry(
        name="provision_ingested_repo/provision_graph_ready_repo",
        kind="provisioner",
        tables=[
            "core.repo_map",
            "core.modules",
            "core.goids",
            "graph.call_graph_nodes",
            "graph.call_graph_edges",
            "graph.import_graph_edges",
            "graph.cfg_blocks",
            "graph.cfg_edges",
            "graph.dfg_edges",
            "analytics.coverage_functions",
            "analytics.function_metrics",
            "analytics.function_types",
            "analytics.function_validation",
        ],
        consumers=[
            "tests/graphs/*",
            "tests/storage/*",
            "tests/server/test_fastapi_endpoints.py",
            "tests/test_pipeline_smoke.py",
        ],
        notes="Production-style ingestion baseline; graph_ready adds analytics graph metrics.",
    ),
    SeedInventoryEntry(
        name="provision_docs_export_ready/seed_docs_export_minimal",
        kind="provisioner",
        tables=[
            "core.repo_map",
            "core.modules",
            "core.goids",
            "core.goid_crosswalk",
            "graph.call_graph_nodes",
            "graph.call_graph_edges",
            "graph.cfg_blocks",
            "graph.import_graph_edges",
            "graph.symbol_use_edges",
            "analytics.test_catalog",
            "analytics.test_coverage_edges",
        ],
        consumers=[
            "tests/docs_export/*",
        ],
        notes="Docs-export smoke/validation seeds with minimal GOID and coverage rows.",
    ),
    SeedInventoryEntry(
        name="seed_mcp_backend",
        kind="seeder",
        tables=[
            "analytics.goid_risk_factors",
            "analytics.function_metrics",
            "analytics.function_validation",
            "graph.call_graph_edges",
            "analytics.test_catalog",
            "analytics.test_coverage_edges",
        ],
        consumers=[
            "tests/mcp/test_backend.py",
        ],
        notes="MCP backend smoke rows (GOID 1) for dataset/tests endpoints.",
    ),
    SeedInventoryEntry(
        name="seed_graph_validation_gaps",
        kind="seeder",
        tables=[
            "core.goids",
            "graph.call_graph_edges",
            "graph.symbol_use_edges",
            "analytics.function_validation",
        ],
        consumers=[
            "tests/graphs/test_validation.py",
            "tests/graphs/test_graph_validation_catalog.py",
        ],
    ),
    SeedInventoryEntry(
        name="builders (RepoMapRow, ModuleRow, GoidRow, CallGraphEdgeRow, ...)",
        kind="builder",
        tables=[
            "core.repo_map",
            "core.modules",
            "core.goids",
            "graph.call_graph_nodes",
            "graph.call_graph_edges",
            "graph.import_graph_edges",
            "graph.symbol_use_edges",
            "graph.cfg_blocks",
            "graph.cfg_edges",
            "graph.dfg_edges",
            "analytics.*",
        ],
        consumers=[
            "tests/_helpers/test_helpers_roundtrip.py",
            "tests/analytics/*",
            "tests/storage/*",
            "tests/docs_export/*",
            "tests/server/*",
        ],
        notes="Typed row factories; insert helpers centralize schema writes.",
    ),
]


def inventory() -> list[SeedInventoryEntry]:
    """
    Return the current seed inventory for tooling or inspection.

    Returns
    -------
    list[SeedInventoryEntry]
        Inventory entries describing seeders/builders and consumers.
    """
    return SEED_INVENTORY
