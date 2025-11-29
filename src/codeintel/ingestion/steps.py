"""Ingestion step interface and registry."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from codeintel.config import DocstringStepConfig, ScipIngestStepConfig, ToolBinaries
from codeintel.config.builder import (
    ConfigIngestStepConfig,
    CoverageIngestStepConfig,
    RepoScanStepConfig,
    TestsIngestStepConfig,
    TypingIngestStepConfig,
)
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion import change_tracker as change_tracker_module
from codeintel.ingestion import (
    config_ingest,
    coverage_ingest,
    cst_extract,
    docstrings_ingest,
    py_ast_extract,
    repo_scan,
    scip_ingest,
    tests_ingest,
    typing_ingest,
)
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class IngestionContextProtocol(Protocol):
    """Structural subset of IngestionContext used by ingestion steps."""

    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    tool_runner: ToolRunner | None
    tool_service: ToolService | None
    scip_runner: Callable[..., ScipIngestResult] | None
    artifact_writer: Callable[[Path, Path, Path], None] | None
    change_tracker: change_tracker_module.ChangeTracker | None

    @property
    def active_tools(self) -> ToolsConfig:
        """Return the active tools configuration."""
        ...

    @property
    def code_profile(self) -> ScanProfile:
        """Return the configured code scan profile."""
        ...

    @property
    def config_profile(self) -> ScanProfile:
        """Return the configured config scan profile."""
        ...


@runtime_checkable
class IngestStep(Protocol):
    """Protocol implemented by ingestion steps."""

    @property
    def name(self) -> str:
        """Stable step identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary of the step."""
        ...

    @property
    def produces_tables(self) -> Sequence[str]:
        """DuckDB tables populated by this step."""
        ...

    @property
    def requires(self) -> Sequence[str]:
        """Names of steps this step depends on."""
        ...

    def run(self, ctx: IngestionContextProtocol) -> object | None:  # pragma: no cover - protocol
        """Execute the ingestion step against the provided context."""
        ...


@dataclass(frozen=True)
class IngestStepMetadata:
    """
    Machine-readable metadata for an ingestion step.

    Parameters
    ----------
    name
        Unique step identifier (e.g. ``"repo_scan"``).
    description
        Human-readable description of what the step does.
    produces_tables
        DuckDB tables populated by this step.
    requires
        Names of steps this step depends on.
    """

    name: str
    description: str
    produces_tables: tuple[str, ...]
    requires: tuple[str, ...]


def _require_change_tracker(
    ctx: IngestionContextProtocol,
    step_name: str,
) -> change_tracker_module.ChangeTracker:
    """
    Ensure a change tracker is available for steps that require repo_scan.

    Parameters
    ----------
    ctx
        Ingestion context carrying the shared change tracker.
    step_name
        Name of the step performing the check, used for error reporting.

    Raises
    ------
    RuntimeError
        If the change tracker has not been populated by ``repo_scan``.

    Returns
    -------
    change_tracker_module.ChangeTracker
        Previously populated change tracker.
    """
    tracker = ctx.change_tracker
    if tracker is None:
        message = (
            f"change_tracker is not set; run repo_scan before incremental ingest for {step_name}"
        )
        raise RuntimeError(message)
    return tracker


@dataclass(frozen=True)
class RepoScanStep:
    """Scan repository tree into core tables and change-tracker state."""

    name: str = "repo_scan"
    description: str = "Scan repository modules and build change-tracker state."
    produces_tables: tuple[str, ...] = (
        "core.file_state",
        "core.modules",
        "core.repo_map",
        "analytics.tags_index",
    )
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> change_tracker_module.ChangeTracker:
        log.debug("Running ingestion step %s", self.name)
        cfg = RepoScanStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        tracker = repo_scan.ingest_repo(
            ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
        )
        ctx.change_tracker = tracker
        return tracker


@dataclass(frozen=True)
class ScipIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    description: str = "Run scip-python and persist symbols and GOID crosswalk."
    produces_tables: tuple[str, ...] = (
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    )
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContextProtocol) -> ScipIngestResult:
        log.debug("Running ingestion step %s", self.name)
        binaries = ToolBinaries(
            scip_python_bin=ctx.active_tools.scip_python_bin,
            scip_bin=ctx.active_tools.scip_bin,
            pyright_bin=ctx.active_tools.pyright_bin,
            pyrefly_bin=ctx.active_tools.pyrefly_bin,
            ruff_bin=ctx.active_tools.ruff_bin,
            coverage_bin=ctx.active_tools.coverage_bin,
            pytest_bin=ctx.active_tools.pytest_bin,
            git_bin=ctx.active_tools.git_bin,
            default_timeout_s=ctx.active_tools.default_timeout_s,
        )
        cfg = ScipIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            binaries=binaries,
            scip_runner=ctx.scip_runner,
            artifact_writer=ctx.artifact_writer,
        )
        tracker = _require_change_tracker(ctx, self.name)

        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        return scip_ingest.ingest_scip(
            ctx.gateway,
            cfg=cfg,
            tracker=tracker,
            tool_service=service,
        )


@dataclass(frozen=True)
class CstExtractStep:
    """Parse CST and persist rows."""

    name: str = "cst_extract"
    description: str = "Parse CST via LibCST and write rows into core.cst_nodes."
    produces_tables: tuple[str, ...] = ("core.cst_nodes",)
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        tracker = _require_change_tracker(ctx, self.name)
        executor_kind = os.getenv("CODEINTEL_CST_EXECUTOR", "process")
        cst_extract.ingest_cst(tracker, executor_kind=executor_kind)


@dataclass(frozen=True)
class AstExtractStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    description: str = "Parse Python AST and persist rows + metrics into core.ast_* tables."
    produces_tables: tuple[str, ...] = ("core.ast_nodes", "core.ast_metrics")
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        tracker = _require_change_tracker(ctx, self.name)
        py_ast_extract.ingest_python_ast(tracker)


@dataclass(frozen=True)
class TypingIngestStep:
    """Compute typedness and static diagnostics."""

    name: str = "typing_ingest"
    description: str = "Populate analytics.typedness and analytics.static_diagnostics."
    produces_tables: tuple[str, ...] = ("analytics.typedness", "analytics.static_diagnostics")
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = TypingIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        typing_ingest.ingest_typing_signals(
            gateway=ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
            tools=ctx.active_tools,
            tool_service=service,
        )


@dataclass(frozen=True)
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    description: str = "Load coverage.py data into analytics.coverage_lines."
    produces_tables: tuple[str, ...] = ("analytics.coverage_lines",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = CoverageIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            coverage_file=ctx.active_tools.coverage_file,  # type: ignore[arg-type]
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        coverage_ingest.ingest_coverage_lines(
            gateway=ctx.gateway,
            cfg=cfg,
            tools=ctx.active_tools,
            tool_service=service,
            json_output_path=ctx.paths.coverage_json,
        )


@dataclass(frozen=True)
class TestsIngestStep:
    """Load pytest results into analytics.test_catalog."""

    name: str = "tests_ingest"
    description: str = "Ingest pytest JSON report into analytics.test_catalog."
    produces_tables: tuple[str, ...] = ("analytics.test_catalog",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = TestsIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            pytest_report_path=ctx.paths.pytest_report,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        tests_ingest.ingest_tests(
            gateway=ctx.gateway,
            cfg=cfg,
            report_path=ctx.paths.pytest_report,
            tools=ctx.active_tools,
            tool_service=service,
        )


@dataclass(frozen=True)
class DocstringsIngestStep:
    """Extract and persist docstrings."""

    name: str = "docstrings_ingest"
    description: str = "Extract docstrings and persist structured rows into core.docstrings."
    produces_tables: tuple[str, ...] = ("core.docstrings",)
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = DocstringStepConfig(snapshot=ctx.snapshot)
        docstrings_ingest.ingest_docstrings(
            ctx.gateway,
            cfg,
            code_profile=ctx.code_profile,
        )


@dataclass(frozen=True)
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    description: str = "Flatten config files into analytics.config_values."
    produces_tables: tuple[str, ...] = ("analytics.config_values",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContextProtocol) -> None:
        log.debug("Running ingestion step %s", self.name)
        cfg = ConfigIngestStepConfig(snapshot=ctx.snapshot)
        config_ingest.ingest_config_values(
            ctx.gateway,
            cfg=cfg,
            config_profile=ctx.config_profile,
        )


@dataclass(frozen=True)
class IngestStepRegistry:
    """
    Registry of ingestion steps with dependency-aware ordering helpers.

    Parameters
    ----------
    _steps
        Mapping of step names to step instances.
    _sequence
        Ordered sequence of step names (defines default ordering).
    """

    _steps: Mapping[str, IngestStep]
    _sequence: tuple[str, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[IngestStep]:
        """
        Iterate over steps in default sequence order.

        Yields
        ------
        IngestStep
            Registered steps in the registry's default order.
        """
        for name in self._sequence:
            yield self._steps[name]

    def __contains__(self, name: str) -> bool:
        """
        Return True if a step with this name is registered.

        Parameters
        ----------
        name
            Step identifier to check.

        Returns
        -------
        bool
            True when the step is present in the registry.
        """
        return name in self._steps

    def __len__(self) -> int:
        """
        Return the number of registered steps.

        Returns
        -------
        int
            Count of steps tracked by the registry.
        """
        return len(self._steps)

    def step_names(self) -> tuple[str, ...]:
        """
        Return the tuple of step names in default sequence order.

        Returns
        -------
        tuple[str, ...]
            Step names corresponding to the registry's iteration order.
        """
        return self._sequence

    def get(self, name: str) -> IngestStep:
        """
        Look up a step by name.

        Raises
        ------
        KeyError
            If the step name is not registered.

        Returns
        -------
        IngestStep
            Registered step matching ``name``.
        """
        try:
            return self._steps[name]
        except KeyError as exc:  # pragma: no cover - trivial
            message = f"Unknown ingestion step: {name}"
            raise KeyError(message) from exc

    def metadata_for(self, name: str) -> IngestStepMetadata:
        """
        Return metadata for a single step.

        Parameters
        ----------
        name
            Step identifier to describe.

        Returns
        -------
        IngestStepMetadata
            Metadata snapshot for the requested step.
        """
        step = self.get(name)
        return IngestStepMetadata(
            name=step.name,
            description=step.description,
            produces_tables=tuple(step.produces_tables),
            requires=tuple(step.requires),
        )

    def all_metadata(self) -> list[IngestStepMetadata]:
        """
        Return metadata for all steps in default order.

        Returns
        -------
        list[IngestStepMetadata]
            Metadata entries mirroring the registry order.
        """
        return [self.metadata_for(name) for name in self._sequence]

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """
        Return mapping of step name -> direct dependencies.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Direct dependency map keyed by step name.
        """
        return {name: tuple(self._steps[name].requires) for name in self._steps}

    def _expand_recursive(self, name: str, expanded: set[str]) -> None:
        """
        Recursively expand dependencies for a name.

        Parameters
        ----------
        name
            Step to expand.
        expanded
            Accumulator tracking discovered steps.
        """
        if name in expanded:
            return
        step = self.get(name)
        for dep in step.requires:
            self._expand_recursive(dep, expanded)
        expanded.add(name)

    def expand_with_deps(self, names: Sequence[str]) -> set[str]:
        """
        Expand a set of step names to include all transitive dependencies.

        Parameters
        ----------
        names
            Step names requested by the caller.

        Returns
        -------
        set[str]
            Expanded set including all transitive dependencies.
        """
        expanded: set[str] = set()
        for name in names:
            self._expand_recursive(name, expanded)
        return expanded

    def topological_order(self, names: Sequence[str]) -> list[str]:
        """
        Return a topological ordering of the requested steps.

        Raises
        ------
        RuntimeError
            If a dependency cycle is detected.
        KeyError
            If any step name is not registered.

        Returns
        -------
        list[str]
            Names ordered so dependencies precede dependents.
        """
        for name in names:
            if name not in self._steps:
                message = f"Unknown ingestion step: {name}"
                raise KeyError(message)

        deps = {name: set(self._steps[name].requires) & set(names) for name in names}
        remaining = set(names)
        ordered: list[str] = []
        no_deps = [name for name in names if not deps[name]]

        while no_deps:
            name = no_deps.pop()
            ordered.append(name)
            remaining.discard(name)
            for other in list(remaining):
                deps[other].discard(name)
                if not deps[other]:
                    no_deps.append(other)

        if remaining:
            message = f"Circular dependencies detected in ingestion steps: {sorted(remaining)}"
            raise RuntimeError(message)
        return ordered


def _build_default_registry() -> IngestStepRegistry:
    """
    Construct the default registry with all built-in ingestion steps.

    Returns
    -------
    IngestStepRegistry
        Registry populated with the standard step set.
    """
    steps: dict[str, IngestStep] = {
        "repo_scan": RepoScanStep(),
        "scip_ingest": ScipIngestStep(),
        "cst_extract": CstExtractStep(),
        "ast_extract": AstExtractStep(),
        "typing_ingest": TypingIngestStep(),
        "coverage_ingest": CoverageIngestStep(),
        "tests_ingest": TestsIngestStep(),
        "docstrings_ingest": DocstringsIngestStep(),
        "config_ingest": ConfigIngestStep(),
    }
    sequence = tuple(steps.keys())
    return IngestStepRegistry(_steps=steps, _sequence=sequence)


DEFAULT_REGISTRY: IngestStepRegistry = _build_default_registry()


def get_ingest_step(name: str) -> IngestStep:
    """
    Return an ingestion step from the default registry.

    Parameters
    ----------
    name
        Step identifier to retrieve.

    Returns
    -------
    IngestStep
        Registered step matching ``name``.
    """
    return DEFAULT_REGISTRY.get(name)


__all__ = [
    "DEFAULT_REGISTRY",
    "IngestStep",
    "IngestStepMetadata",
    "IngestStepRegistry",
    "IngestionContextProtocol",
    "get_ingest_step",
]
