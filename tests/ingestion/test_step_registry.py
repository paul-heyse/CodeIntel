"""Tests for the ingestion step registry wiring and metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.runner import IngestionContext, run_ingest_steps
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
from codeintel.ingestion.steps import (
    DEFAULT_REGISTRY,
    IngestionContextProtocol,
    IngestStepRegistry,
)
from tests._helpers.gateway import open_ingestion_gateway


def test_registry_includes_all_expected_steps() -> None:
    """Ensure the default registry is synchronized with the expected step set."""
    names = set(DEFAULT_REGISTRY.step_names())
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
    extra = names - expected
    if missing:
        pytest.fail(f"Missing ingestion steps in registry: {sorted(missing)}")
    if extra:
        pytest.fail(f"Unexpected ingestion steps in registry: {sorted(extra)}")


def test_metadata_exposes_tables_and_deps() -> None:
    """Verify registry metadata surfaces dependencies and tables accurately."""
    meta_by_name = {meta.name: meta for meta in DEFAULT_REGISTRY.all_metadata()}

    repo_scan = meta_by_name["repo_scan"]
    if "core.modules" not in repo_scan.produces_tables:
        pytest.fail("repo_scan metadata missing core.modules table")
    if repo_scan.requires != ():
        pytest.fail(f"repo_scan requires should be empty, found: {repo_scan.requires}")

    scip = meta_by_name["scip_ingest"]
    if "core.scip_symbols" not in scip.produces_tables:
        pytest.fail("scip_ingest metadata missing core.scip_symbols table")
    if "core.goid_crosswalk" not in scip.produces_tables:
        pytest.fail("scip_ingest metadata missing core.goid_crosswalk table")
    if scip.requires != ("repo_scan",):
        pytest.fail(f"scip_ingest requires incorrect: {scip.requires}")

    docstrings = meta_by_name["docstrings_ingest"]
    if "core.docstrings" not in docstrings.produces_tables:
        pytest.fail("docstrings_ingest metadata missing core.docstrings table")
    if "repo_scan" not in docstrings.requires:
        pytest.fail(f"docstrings_ingest should depend on repo_scan, found {docstrings.requires}")


def test_topological_order_respects_dependencies() -> None:
    """Confirm registry ordering respects declared prerequisites."""
    names = ["repo_scan", "scip_ingest", "ast_extract", "cst_extract", "docstrings_ingest"]
    order = DEFAULT_REGISTRY.topological_order(names)
    positions = {name: order.index(name) for name in names}

    if positions["repo_scan"] >= positions["scip_ingest"]:
        pytest.fail("repo_scan must precede scip_ingest")
    if positions["repo_scan"] >= positions["ast_extract"]:
        pytest.fail("repo_scan must precede ast_extract")
    if positions["repo_scan"] >= positions["cst_extract"]:
        pytest.fail("repo_scan must precede cst_extract")
    if positions["repo_scan"] >= positions["docstrings_ingest"]:
        pytest.fail("repo_scan must precede docstrings_ingest")


@dataclass
class RecordingStep:
    """Lightweight ingest step for exercising run_ingest_steps ordering."""

    name: str
    description: str
    produces_tables: Sequence[str]
    requires: Sequence[str]
    executed: list[str]

    def run(self, ctx: IngestionContextProtocol) -> None:
        """Record execution order for the orchestrated run."""
        _ = ctx.snapshot.repo  # Reference context to satisfy typing and linting.
        self.executed.append(self.name)


def test_run_ingest_steps_executes_dependencies(tmp_path: Path) -> None:
    """Smoke test exercising dependency expansion within run_ingest_steps."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = BuildPaths.from_repo_root(repo_root)
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    code_profile = default_code_profile(repo_root)
    config_profile = default_config_profile(repo_root)
    tools = ToolsConfig.default()

    executed: list[str] = []
    registry = IngestStepRegistry(
        _steps={
            "alpha": RecordingStep(
                name="alpha",
                description="First step",
                produces_tables=(),
                requires=(),
                executed=executed,
            ),
            "bravo": RecordingStep(
                name="bravo",
                description="Second step",
                produces_tables=(),
                requires=("alpha",),
                executed=executed,
            ),
            "charlie": RecordingStep(
                name="charlie",
                description="Final step",
                produces_tables=(),
                requires=("bravo",),
                executed=executed,
            ),
        },
        _sequence=("alpha", "bravo", "charlie"),
    )

    gateway = open_ingestion_gateway()
    try:
        ctx = IngestionContext(
            snapshot=snapshot,
            paths=paths,
            gateway=gateway,
            tools=tools,
            code_profile_cfg=code_profile,
            config_profile_cfg=config_profile,
        )
        run_ingest_steps(ctx, ["charlie"], registry=registry)
    finally:
        gateway.close()

    if executed != ["alpha", "bravo", "charlie"]:
        pytest.fail(f"Unexpected execution order: {executed}")
