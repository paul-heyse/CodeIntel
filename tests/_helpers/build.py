"""Helpers for build-system unit tests.

Migration note
--------------
Targets should be defined via contracts (``OutputContract`` / ``OutputTarget``
factories like ``from_tables``) and referenced by ``table_keys``. Avoid adding
new call sites that pass ``tables=`` directly; the contract is the source of
truth for outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.build.config import CONFIG_FILE_NAME, BuildConfig
from codeintel.build.executor import StageExecutionResult
from codeintel.build.manifest import OutputManifest
from codeintel.build.plan import MODULE_ORDER, BuildPlan, PlanStage, PlanStep
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugin_registry import PluginRegistryStore
from codeintel.build.result import TargetResult
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.fakes.configs import create_test_build_paths, create_test_snapshot
from tests._helpers.fakes.fake_providers import (
    FakeCoverageCollector,
    FakeGitHistoryProvider,
    FakeScipIndexer,
    FakeTestReporter,
    FakeToolRunner,
    FakeTypeChecker,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.config.primitives import BuildPaths, SnapshotRef


def make_snapshot(
    tmp_path: Path | None = None,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> SnapshotRef:
    """Create a SnapshotRef with sensible defaults.

    Parameters
    ----------
    tmp_path
        Temporary path for repo_root. If None, uses current directory.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SnapshotRef
        Snapshot reference configured for tests.
    """
    effective_path = tmp_path if tmp_path is not None else Path.cwd()
    return create_test_snapshot(effective_path, repo=repo, commit=commit)


def make_build_paths(tmp_path: Path | None = None) -> BuildPaths:
    """Create BuildPaths rooted at tmp_path (or current directory).

    Parameters
    ----------
    tmp_path
        Temporary path for build directory. If None, uses current directory.

    Returns
    -------
    BuildPaths
        Build paths configured for tests.
    """
    effective_path = tmp_path if tmp_path is not None else Path.cwd()
    return create_test_build_paths(effective_path)


def make_build_config(
    data: Mapping[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> BuildConfig:
    """Create a BuildConfig from a mapping.

    Returns
    -------
    BuildConfig
        Parsed configuration.
    """
    return (
        BuildConfig.from_dict(dict(data), config_path=config_path)
        if data is not None
        else BuildConfig.empty()
    )


def write_build_config(project_root: Path, sections: Mapping[str, Mapping[str, Any]]) -> Path:
    """Render a minimal TOML config file under project_root.

    Returns
    -------
    Path
        Path to the written configuration file.
    """
    lines: list[str] = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format_toml_value(value)}")
        lines.append("")
    config_path = project_root / CONFIG_FILE_NAME
    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return config_path


def sample_target_graph(
    targets: Sequence[OutputTarget] | None = None,
) -> TargetGraph:
    """Build a small target graph for tests.

    Returns
    -------
    TargetGraph
        Graph populated with default targets unless provided.
    """
    graph = TargetGraph()
    for target in targets or _default_targets():
        graph.register(target)
    return graph


def make_plugin_registry_store(
    loader: Callable[[PluginRegistryStore], None] | None = None,
) -> PluginRegistryStore:
    """Create a plugin registry store with an optional loader.

    Returns
    -------
    PluginRegistryStore
        Registry store configured with the provided loader.
    """
    return PluginRegistryStore(loader=loader)


def sample_build_plan(
    graph: TargetGraph | None = None,
    *,
    requested: Sequence[str] | None = None,
    reasons: Mapping[str, str] | None = None,
) -> BuildPlan:
    """Build a simple BuildPlan derived from a graph.

    Returns
    -------
    BuildPlan
        Plan with stages grouped by module order.
    """
    graph = graph or sample_target_graph()
    reason_map = dict(reasons or {})
    requested_targets = tuple(requested or tuple(t.name for t in graph.all_targets))
    stages: list[PlanStage] = []
    for module in MODULE_ORDER:
        steps = [
            PlanStep(
                target=target.name,
                module=target.module,
                plugin=target.plugin,
                estimated_duration_ms=target.estimated_duration_ms,
                dependencies=target.dependencies,
                reason=reason_map.get(target.name, "requested"),
            )
            for target in graph.all_targets
            if target.module == module and target.name in requested_targets
        ]
        if steps:
            stages.append(PlanStage(module=module, steps=tuple(steps)))
    return BuildPlan(
        requested_targets=requested_targets,
        stages=tuple(stages),
        skipped_targets=(),
        blocked_targets=(),
    )


@dataclass(frozen=True)
class ManifestParams:
    """Parameters for constructing manifest fixtures."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    plugin: str = "test_plugin"
    input_hash: str = "input-hash"
    computed_at: datetime | None = None
    duration_ms: float = 1.0
    row_count: int | None = None
    output_hash: str | None = None
    options_hash: str | None = None


def sample_manifest(target: str, params: ManifestParams | None = None) -> OutputManifest:
    """Create an OutputManifest with defaulted timestamps and hashes.

    Parameters
    ----------
    target
        Target name for the manifest.
    params
        Optional manifest parameters; defaults will be used otherwise.

    Returns
    -------
    OutputManifest
        Manifest ready for insertion into tracking tables.
    """
    cfg = params or ManifestParams()
    return OutputManifest(
        target=target,
        repo=cfg.repo,
        commit=cfg.commit,
        plugin=cfg.plugin,
        computed_at=cfg.computed_at or datetime.now(tz=UTC),
        duration_ms=cfg.duration_ms,
        input_hash=cfg.input_hash,
        output_hash=cfg.output_hash,
        row_count=cfg.row_count,
        options_hash=cfg.options_hash,
    )


@dataclass
class RecordingPlugin(TargetPlugin):
    """Minimal plugin that records execution contexts."""

    plugin_name: ClassVar[str] = "recording_plugin"
    plugin_version: ClassVar[str] = "1.0.0"
    plugin_description: ClassVar[str] = "Recording test plugin"
    result: TargetResult = field(default_factory=TargetResult.succeeded)
    validation_errors: tuple[str, ...] = ()
    calls: list[TargetExecutionContext] = field(default_factory=list)

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Record context and return preconfigured result.

        Parameters
        ----------
        ctx
            Execution context provided by the build system.

        Returns
        -------
        TargetResult
            Preconfigured result for deterministic tests.
        """
        self.calls.append(ctx)
        return self.result

    def validate_context(self, ctx: TargetExecutionContext) -> list[str]:
        """Return configured validation errors (if any).

        Parameters
        ----------
        ctx
            Context to validate.

        Returns
        -------
        list[str]
            Validation errors configured for this plugin.
        """
        _ = ctx
        return list(self.validation_errors)


@dataclass
class RecordingProviders:
    """Container of fake providers with recording hooks."""

    tool_runner: FakeToolRunner = field(default_factory=FakeToolRunner)
    scip_indexer: FakeScipIndexer = field(default_factory=FakeScipIndexer)
    type_checker: FakeTypeChecker = field(default_factory=FakeTypeChecker)
    coverage_collector: FakeCoverageCollector = field(default_factory=FakeCoverageCollector)
    test_reporter: FakeTestReporter = field(default_factory=FakeTestReporter)
    git_history: FakeGitHistoryProvider = field(default_factory=FakeGitHistoryProvider)

    def as_dict(self) -> dict[str, object]:
        """Return providers as a mapping keyed by provider name.

        Returns
        -------
        dict[str, object]
            Mapping of provider name to fake implementation.
        """
        return {
            "tool_runner": self.tool_runner,
            "scip_indexer": self.scip_indexer,
            "type_checker": self.type_checker,
            "coverage_collector": self.coverage_collector,
            "test_reporter": self.test_reporter,
            "git_history": self.git_history,
        }


@dataclass
class RecordingExecutor:
    """Record plan stages and produce StageExecutionResult fixtures."""

    executed_stages: list[PlanStage] = field(default_factory=list)
    results: list[StageExecutionResult] = field(default_factory=list)

    def record(
        self,
        stage: PlanStage,
        *,
        failed: Sequence[str] | None = None,
        error: str | None = None,
        durations_ms: Mapping[str, float] | None = None,
        row_counts: Mapping[str, int | None] | None = None,
    ) -> StageExecutionResult:
        """Record a stage and return a matching StageExecutionResult.

        Parameters
        ----------
        stage
            Stage to record.
        failed
            Optional iterable of targets to mark as failed.
        error
            Optional stage-level error message.
        durations_ms
            Optional duration overrides keyed by target.
        row_counts
            Optional row count overrides keyed by target.

        Returns
        -------
        StageExecutionResult
            Execution result matching the provided inputs.
        """
        failed_set = set(failed or ())
        completed = tuple(step.target for step in stage.steps if step.target not in failed_set)
        failed_targets = tuple(step.target for step in stage.steps if step.target in failed_set)
        durations: dict[str, float] = {}
        rows: dict[str, int | None] = {}
        for step in stage.steps:
            durations[step.target] = (
                durations_ms[step.target]
                if durations_ms and step.target in durations_ms
                else float(step.estimated_duration_ms or 0)
            )
            rows[step.target] = row_counts.get(step.target) if row_counts else None
        result = StageExecutionResult(
            module=stage.module,
            completed=completed,
            failed=failed_targets,
            durations_ms=durations,
            row_counts=rows,
            error=error,
        )
        self.executed_stages.append(stage)
        self.results.append(result)
        return result


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    message = f"Unsupported TOML value type: {type(value).__name__}"
    raise TypeError(message)


def _default_targets() -> tuple[OutputTarget, ...]:
    ingestion_modules = OutputTarget.from_tables(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        options=TargetOptions(description="Repository module index"),
    )
    ast_target = OutputTarget.from_tables(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        options=TargetOptions(dependencies=("modules",), description="AST extraction"),
    )
    goids_target = OutputTarget.from_tables(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        options=TargetOptions(dependencies=("ast",), description="GOID construction"),
    )
    metrics_target = OutputTarget.from_tables(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        options=TargetOptions(dependencies=("goids",), description="Function metrics"),
    )
    return (ingestion_modules, ast_target, goids_target, metrics_target)


__all__ = [
    "ManifestParams",
    "RecordingExecutor",
    "RecordingPlugin",
    "RecordingProviders",
    "make_build_config",
    "make_build_paths",
    "make_snapshot",
    "sample_build_plan",
    "sample_manifest",
    "sample_target_graph",
    "write_build_config",
]
