"""Ingestion step configuration models and builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from codeintel.config.primitives import BuildPaths, SnapshotRef, ToolBinaries

if TYPE_CHECKING:
    from codeintel.ingestion.scip_ingest import ScipIngestResult


@dataclass(frozen=True)
class RepoScanStepConfig:
    """Configuration for repository scanning into core.modules."""

    snapshot: SnapshotRef
    paths: BuildPaths
    tags_index_path: Path | None = None
    tool_runner: object | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def resolved_tags_index_path(self) -> Path | None:
        """Optional tags index path, resolved when provided."""
        return self.tags_index_path.resolve() if self.tags_index_path is not None else None


@dataclass(frozen=True)
class ScipIngestStepConfig:
    """Configuration for SCIP ingestion step."""

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def build_dir(self) -> Path:
        """Build directory."""
        return self.paths.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory."""
        return self.paths.document_output_dir

    @property
    def scip_python_bin(self) -> str:
        """Path to scip-python binary."""
        return self.binaries.scip_python_bin

    @property
    def scip_bin(self) -> str:
        """Path to scip binary."""
        return self.binaries.scip_bin


@dataclass(frozen=True)
class PyAstIngestStepConfig:
    """Configuration for stdlib AST ingestion."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class CoverageIngestStepConfig:
    """Configuration for ingesting coverage lines."""

    snapshot: SnapshotRef
    paths: BuildPaths
    coverage_file: Path | None = None
    tool_runner: object | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def resolved_coverage_path(self) -> Path:
        """Resolved coverage JSON path."""
        return self.coverage_file or self.paths.coverage_json


@dataclass(frozen=True)
class TestsIngestStepConfig:
    """Configuration for pytest report ingestion."""

    snapshot: SnapshotRef
    paths: BuildPaths
    pytest_report_path: Path | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def resolved_report_path(self) -> Path:
        """Resolved pytest report path."""
        return self.pytest_report_path or self.paths.pytest_report


@dataclass(frozen=True)
class TypingIngestStepConfig:
    """Configuration for typedness ingestion."""

    snapshot: SnapshotRef
    paths: BuildPaths
    tool_runner: object | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class DocstringStepConfig:
    """Configuration for docstring ingestion step."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class ConfigIngestStepConfig:
    """Configuration for config ingestion step."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


class _IngestionOwner(Protocol):
    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries


class IngestionStepBuilder:
    """Provides ingestion-related config builders used by ConfigBuilder."""

    def __init__(self, owner: _IngestionOwner) -> None:
        self._owner = owner

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference shared across built configs."""
        return self._owner.snapshot

    @property
    def paths(self) -> BuildPaths:
        """Resolved build paths for the snapshot."""
        return self._owner.paths

    @property
    def binaries(self) -> ToolBinaries:
        """Tool binaries available for ingestion tasks."""
        return self._owner.binaries

    def scip_ingest(
        self,
        *,
        scip_runner: Callable[..., ScipIngestResult] | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
    ) -> ScipIngestStepConfig:
        """
        Build SCIP ingestion configuration.

        Returns
        -------
        ScipIngestStepConfig
            Configuration for SCIP index generation.
        """
        return ScipIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            binaries=self.binaries,
            scip_runner=scip_runner,
            artifact_writer=artifact_writer,
        )

    def docstring(self) -> DocstringStepConfig:
        """
        Build docstring ingestion configuration.

        Returns
        -------
        DocstringStepConfig
            Configuration for docstring extraction.
        """
        return DocstringStepConfig(snapshot=self.snapshot)

    def repo_scan(
        self,
        *,
        tags_index_path: Path | None = None,
        tool_runner: object | None = None,
    ) -> RepoScanStepConfig:
        """
        Build repository scan configuration.

        Returns
        -------
        RepoScanStepConfig
            Configuration for module discovery.
        """
        return RepoScanStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            tags_index_path=tags_index_path,
            tool_runner=tool_runner,
        )

    def coverage_ingest(
        self,
        *,
        coverage_file: Path | None = None,
        tool_runner: object | None = None,
    ) -> CoverageIngestStepConfig:
        """
        Build coverage ingestion configuration.

        Returns
        -------
        CoverageIngestStepConfig
            Configuration for coverage line ingestion.
        """
        resolved = coverage_file or self.paths.coverage_json
        return CoverageIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            coverage_file=resolved,
            tool_runner=tool_runner,
        )

    def tests_ingest(
        self,
        *,
        pytest_report_path: Path | None = None,
    ) -> TestsIngestStepConfig:
        """
        Build tests ingestion configuration.

        Returns
        -------
        TestsIngestStepConfig
            Configuration for pytest catalog ingestion.
        """
        resolved = pytest_report_path or self.paths.pytest_report
        return TestsIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            pytest_report_path=resolved,
        )

    def typing_ingest(
        self,
        *,
        tool_runner: object | None = None,
    ) -> TypingIngestStepConfig:
        """
        Build typing ingestion configuration.

        Returns
        -------
        TypingIngestStepConfig
            Configuration for typedness and diagnostics ingestion.
        """
        return TypingIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            tool_runner=tool_runner,
        )

    def config_ingest(self) -> ConfigIngestStepConfig:
        """
        Build config-values ingestion configuration.

        Returns
        -------
        ConfigIngestStepConfig
            Configuration for config-values ingestion.
        """
        return ConfigIngestStepConfig(snapshot=self.snapshot)

    def py_ast_ingest(self) -> PyAstIngestStepConfig:
        """
        Build stdlib AST ingestion configuration.

        Returns
        -------
        PyAstIngestStepConfig
            Configuration for AST ingestion.
        """
        return PyAstIngestStepConfig(snapshot=self.snapshot)


__all__ = [
    "ConfigIngestStepConfig",
    "CoverageIngestStepConfig",
    "DocstringStepConfig",
    "IngestionStepBuilder",
    "PyAstIngestStepConfig",
    "RepoScanStepConfig",
    "ScipIngestStepConfig",
    "TestsIngestStepConfig",
    "TypingIngestStepConfig",
]
