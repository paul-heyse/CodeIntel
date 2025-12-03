"""Tests for unified pipeline orchestration.

These tests validate the declarative pipeline spec, planner, and executor
using real infrastructure and lightweight fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.pipeline.executor import run_pipeline
from codeintel.pipeline.planner import PipelinePlanOptions, build_pipeline_plan
from codeintel.pipeline.spec import (
    ANALYTICS_ONLY,
    FULL_PIPELINE,
    GRAPHS_ONLY,
    INGEST_ONLY,
    PipelineSpec,
    PipelineStage,
    get_pipeline_spec,
    list_pipeline_specs,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fixtures import ProvisionedGateway, provisioned_gateway

if TYPE_CHECKING:
    from collections.abc import Iterator


# -----------------------------------------------------------------------------
# Spec Module Tests
# -----------------------------------------------------------------------------


def test_full_pipeline_has_all_stages() -> None:
    """FULL_PIPELINE includes ingestion, graphs, and analytics."""
    modules = {stage.module for stage in FULL_PIPELINE.stages}
    assert modules == {"ingestion", "graphs", "analytics"}


def test_ingest_only_has_one_stage() -> None:
    """INGEST_ONLY has exactly one ingestion stage."""
    assert len(INGEST_ONLY.stages) == 1
    assert INGEST_ONLY.stages[0].module == "ingestion"


def test_graphs_only_has_one_stage() -> None:
    """GRAPHS_ONLY has exactly one graphs stage."""
    assert len(GRAPHS_ONLY.stages) == 1
    assert GRAPHS_ONLY.stages[0].module == "graphs"


def test_analytics_only_has_one_stage() -> None:
    """ANALYTICS_ONLY has exactly one analytics stage."""
    assert len(ANALYTICS_ONLY.stages) == 1
    assert ANALYTICS_ONLY.stages[0].module == "analytics"


def test_get_pipeline_spec_found() -> None:
    """get_pipeline_spec returns the correct spec for known IDs."""
    assert get_pipeline_spec("full") is FULL_PIPELINE
    assert get_pipeline_spec("ingest") is INGEST_ONLY
    assert get_pipeline_spec("graphs") is GRAPHS_ONLY
    assert get_pipeline_spec("analytics") is ANALYTICS_ONLY


def test_get_pipeline_spec_not_found() -> None:
    """get_pipeline_spec raises KeyError for unknown IDs."""
    with pytest.raises(KeyError):
        get_pipeline_spec("nonexistent")


def test_list_pipeline_specs() -> None:
    """list_pipeline_specs returns all registered spec IDs."""
    specs = list_pipeline_specs()
    assert "full" in specs
    assert "ingest" in specs
    assert "graphs" in specs
    assert "analytics" in specs


def test_stage_required_defaults_to_true() -> None:
    """PipelineStage.required defaults to True."""
    stage = PipelineStage(module="ingestion", name="test")
    assert stage.required is True


def test_custom_pipeline_spec() -> None:
    """Custom PipelineSpec can be constructed."""
    expected_stage_count = 2
    spec = PipelineSpec(
        id="custom",
        description="Custom pipeline",
        stages=(
            PipelineStage(module="ingestion", name="builtin.default"),
            PipelineStage(module="graphs", name="builtin.full", required=False),
        ),
    )
    assert spec.id == "custom"
    assert len(spec.stages) == expected_stage_count
    assert spec.stages[0].required is True
    assert spec.stages[1].required is False


# -----------------------------------------------------------------------------
# Planner Module Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a test snapshot reference.

    Parameters
    ----------
    tmp_path
        Pytest tmp_path fixture.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root,
    )


@pytest.fixture
def paths(tmp_path: Path) -> BuildPaths:
    """Create test build paths.

    Parameters
    ----------
    tmp_path
        Pytest tmp_path fixture.

    Returns
    -------
    BuildPaths
        Test build paths.
    """
    return BuildPaths.from_repo_root(tmp_path / "repo")


@pytest.fixture
def tools() -> ToolsConfig:
    """Create minimal tools config.

    Returns
    -------
    ToolsConfig
        Test tools configuration.
    """
    return ToolsConfig.model_validate({})


def test_build_plan_for_full_pipeline(
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    fresh_gateway: StorageGateway,
) -> None:
    """build_pipeline_plan creates plans for all stages in FULL_PIPELINE."""
    plan = build_pipeline_plan(
        spec=FULL_PIPELINE,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
            trigger="cli",
        ),
    )

    assert plan.spec is FULL_PIPELINE
    assert plan.run_context.kind == "full"
    assert plan.run_context.run_id.startswith("full-")
    assert plan.ingestion is not None
    assert plan.graphs is not None
    assert plan.analytics is not None


def test_build_plan_for_ingest_only(
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    fresh_gateway: StorageGateway,
) -> None:
    """build_pipeline_plan creates only ingestion plan for INGEST_ONLY."""
    plan = build_pipeline_plan(
        spec=INGEST_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
            trigger="cli",
        ),
    )

    assert plan.run_context.kind == "ingest"
    assert plan.ingestion is not None
    assert plan.graphs is None
    assert plan.analytics is None


def test_build_plan_for_graphs_only(
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    fresh_gateway: StorageGateway,
) -> None:
    """build_pipeline_plan creates only graphs plan for GRAPHS_ONLY."""
    plan = build_pipeline_plan(
        spec=GRAPHS_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
            trigger="cli",
        ),
    )

    assert plan.run_context.kind == "graphs"
    assert plan.ingestion is None
    assert plan.graphs is not None
    assert plan.analytics is None


def test_build_plan_for_analytics_only(
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    fresh_gateway: StorageGateway,
) -> None:
    """build_pipeline_plan creates only analytics plan for ANALYTICS_ONLY."""
    plan = build_pipeline_plan(
        spec=ANALYTICS_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
            trigger="cli",
        ),
    )

    assert plan.run_context.kind == "analytics"
    assert plan.ingestion is None
    assert plan.graphs is None
    assert plan.analytics is not None


def test_plan_run_id_uses_kind_prefix(
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    fresh_gateway: StorageGateway,
) -> None:
    """Run IDs are prefixed with the inferred run kind."""
    ingest_plan = build_pipeline_plan(
        spec=INGEST_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
        ),
    )
    assert ingest_plan.run_context.run_id.startswith("ingest-")

    graphs_plan = build_pipeline_plan(
        spec=GRAPHS_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=fresh_gateway,
            tools=tools,
        ),
    )
    assert graphs_plan.run_context.run_id.startswith("graphs-")


# -----------------------------------------------------------------------------
# Executor Module Tests (Integration)
# -----------------------------------------------------------------------------


@pytest.fixture
def provisioned_ctx(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway with ingested data for executor tests.

    Parameters
    ----------
    tmp_path
        Pytest tmp_path fixture.

    Yields
    ------
    ProvisionedGateway
        Provisioned gateway context.
    """
    with provisioned_gateway(tmp_path / "repo") as ctx:
        yield ctx


@pytest.mark.integration
def test_ingest_only_records_run(
    provisioned_ctx: ProvisionedGateway,
) -> None:
    """Ingestion-only pipeline records run and steps in tracking tables."""
    snapshot = SnapshotRef(
        repo=provisioned_ctx.repo,
        commit=provisioned_ctx.commit,
        repo_root=provisioned_ctx.repo_root,
    )
    paths = BuildPaths.from_repo_root(provisioned_ctx.repo_root)
    tools = ToolsConfig.model_validate({})

    result = run_pipeline(
        spec=INGEST_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=provisioned_ctx.gateway,
            tools=tools,
            trigger="cli",
        ),
    )

    # Verify run record
    assert result is not None
    assert result.run_id is not None
    assert result.pipeline_name == "ingest"
    assert result.status in {"succeeded", "failed"}

    # Verify steps were recorded
    steps = provisioned_ctx.gateway.runs.fetch_steps(result.run_id)
    assert len(steps) >= 1


@pytest.mark.integration
def test_run_tracking_persisted(
    provisioned_ctx: ProvisionedGateway,
) -> None:
    """Pipeline runs are persisted and fetchable from tracking tables."""
    snapshot = SnapshotRef(
        repo=provisioned_ctx.repo,
        commit=provisioned_ctx.commit,
        repo_root=provisioned_ctx.repo_root,
    )
    paths = BuildPaths.from_repo_root(provisioned_ctx.repo_root)
    tools = ToolsConfig.model_validate({})

    result = run_pipeline(
        spec=INGEST_ONLY,
        options=PipelinePlanOptions(
            snapshot=snapshot,
            paths=paths,
            gateway=provisioned_ctx.gateway,
            tools=tools,
        ),
    )

    # Fetch the run back
    fetched = provisioned_ctx.gateway.runs.fetch_run(result.run_id)
    assert fetched is not None
    assert fetched.run_id == result.run_id
    assert fetched.pipeline_name == result.pipeline_name


# -----------------------------------------------------------------------------
# Failure Handling Tests
# -----------------------------------------------------------------------------


def test_optional_stage_spec() -> None:
    """Verify optional stages can be specified."""
    spec = PipelineSpec(
        id="with-optional",
        description="Pipeline with optional stage",
        stages=(
            PipelineStage(
                module="ingestion",
                name="builtin.default",
                required=True,
            ),
            PipelineStage(
                module="graphs",
                name="builtin.full",
                required=False,  # Optional
            ),
        ),
    )

    assert spec.stages[0].required is True
    assert spec.stages[1].required is False


def test_stage_descriptions_preserved() -> None:
    """Stage descriptions are preserved in spec."""
    for stage in FULL_PIPELINE.stages:
        assert stage.description
        assert isinstance(stage.description, str)


# -----------------------------------------------------------------------------
# Public API Export Tests
# -----------------------------------------------------------------------------


def test_spec_exports() -> None:
    """Pipeline spec types are exported from codeintel.pipeline."""
    # ruff: noqa: PLC0415
    from codeintel.pipeline import (
        ANALYTICS_ONLY,
        FULL_PIPELINE,
        GRAPHS_ONLY,
        INGEST_ONLY,
        PIPELINE_SPECS,
        PipelineSpec,
        PipelineStage,
        get_pipeline_spec,
        list_pipeline_specs,
    )

    # Just verify imports work
    assert PipelineSpec is not None
    assert PipelineStage is not None
    assert FULL_PIPELINE is not None
    assert INGEST_ONLY is not None
    assert GRAPHS_ONLY is not None
    assert ANALYTICS_ONLY is not None
    assert PIPELINE_SPECS is not None
    assert get_pipeline_spec is not None
    assert list_pipeline_specs is not None


def test_planner_exports() -> None:
    """Pipeline planner types are exported from codeintel.pipeline."""
    # ruff: noqa: PLC0415
    from codeintel.pipeline import (
        AnalyticsStagePlan,
        GraphsStagePlan,
        IngestionStagePlan,
        PipelinePlan,
        build_pipeline_plan,
    )

    assert AnalyticsStagePlan is not None
    assert GraphsStagePlan is not None
    assert IngestionStagePlan is not None
    assert PipelinePlan is not None
    assert build_pipeline_plan is not None


def test_executor_exports() -> None:
    """Pipeline executor function is exported from codeintel.pipeline."""
    # ruff: noqa: PLC0415
    from codeintel.pipeline import run_pipeline

    assert run_pipeline is not None
    assert callable(run_pipeline)
