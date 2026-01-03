"""Pytest fixtures shared across ingestion tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.orchestration.tooling import tooling_outputs_session
from tests._helpers.schemas import ensure_schema_service

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from tests._helpers.orchestration.tooling import ToolingOutputs


@pytest.fixture
def ingestion_snapshot(tmp_path_factory: pytest.TempPathFactory) -> SnapshotRef:
    """Provide a default snapshot reference for ingestion tests.

    Returns
    -------
    SnapshotRef
        Snapshot reference for the test repository.
    """
    repo_root = tmp_path_factory.mktemp("ingestion-repo")
    return DEFAULT_VARIANT.to_snapshot(repo_root=repo_root)


@pytest.fixture
def ingestion_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a dataset root for parquet-backed ingestion outputs.

    Returns
    -------
    Path
        Root directory for parquet datasets.
    """
    return tmp_path_factory.mktemp("ingestion-datasets")


@pytest.fixture
def ingestion_ctx_bundle(
    ingestion_snapshot: SnapshotRef,
    ingestion_dataset_root: Path,
) -> SimpleNamespace:
    """Provision a reusable ingestion context for parquet-backed tests.

    Returns
    -------
    SimpleNamespace
        Namespace containing repo_root, snapshot, dataset_root, and build_dir.
    """
    repo_root = ingestion_snapshot.repo_root
    build_dir = ingestion_dataset_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        repo_root=repo_root,
        snapshot=ingestion_snapshot,
        dataset_root=ingestion_dataset_root,
        build_dir=build_dir,
    )


@pytest.fixture
def tooling_outputs(tooling_outputs_session: ToolingOutputs) -> ToolingOutputs:
    """Alias tooling outputs for ingestion tests (session-scoped under the hood).

    Returns
    -------
    ToolingOutputs
        Session-scoped tooling outputs fixture.
    """
    return tooling_outputs_session


@pytest.fixture(autouse=True)
def ingestion_schema_service() -> None:
    """Ensure schema service is available for parquet helpers."""
    ensure_schema_service()


__all__ = [
    "ingestion_ctx_bundle",
    "ingestion_dataset_root",
    "ingestion_snapshot",
    "tooling_outputs",
    "tooling_outputs_session",
]
