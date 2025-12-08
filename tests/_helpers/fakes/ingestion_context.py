"""Fakes and builders for ingestion plugin contexts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.targets import OutputTarget
from tests._helpers.env import DEFAULT_COMMIT, DEFAULT_REPO, create_test_env
from tests._helpers.fakes.contexts import (
    BuilderOptions,
    TargetResourceOverrides,
    build_target_execution_context,
)

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway


@dataclass
class RecordingGateway:
    """Gateway stub that records SQL executed via its .con property."""

    result_rows: list[tuple[object, ...]] = field(default_factory=list)
    executions: list[tuple[str, list[object]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.con = _RecordingConnection(self)


class _RecordingConnection:
    def __init__(self, gateway: RecordingGateway) -> None:
        self._gateway = gateway

    def execute(self, sql: str, params: Iterable[object]) -> _RecordingConnection:
        self._gateway.executions.append((sql, list(params)))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._gateway.result_rows


@dataclass
class RecordingResources:
    """Minimal resources bundle for plugin contexts."""

    modules: tuple[str, ...] = ()
    type_checker: object | None = None


@dataclass(frozen=True)
class LegacyTargetContextOptions:
    """Configuration for legacy RecordingContext construction."""

    modules: tuple[str, ...] | None = None
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT)
    type_checker: object | None = None
    gateway: StorageGateway | RecordingGateway | None = None
    use_real_gateway: bool = True
    tmp_path: Path | None = None

    @classmethod
    def with_modules(cls, modules: Iterable[str], **kwargs: object) -> LegacyTargetContextOptions:
        """Convenience constructor to set modules explicitly."""
        return cls(modules=tuple(modules), **kwargs)


@dataclass
class RecordingContext:
    """Lightweight stand-in for legacy ingestion tests."""

    repo_root: Path
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    gateway: StorageGateway | RecordingGateway = field(default_factory=RecordingGateway)
    resources: RecordingResources = field(default_factory=RecordingResources)


def make_target_context(
    repo_root: Path,
    options: LegacyTargetContextOptions | None = None,
    *,
    modules: Iterable[str] | None = None,
    gateway: StorageGateway | RecordingGateway | None = None,
    type_checker: object | None = None,
    use_real_gateway: bool | None = None,
    tmp_path: Path | None = None,
    snapshot: tuple[str, str] | None = None,
) -> RecordingContext:
    """Construct a recording context matching plugin expectations.

    Returns
    -------
    RecordingContext
        Context populated with provided options.
    """
    if options is not None and any(
        val is not None
        for val in (modules, gateway, type_checker, use_real_gateway, tmp_path, snapshot)
    ):
        message = "Pass either options or individual overrides, not both."
        raise ValueError(message)

    if options is None:
        opts = LegacyTargetContextOptions(
            modules=tuple(modules) if modules is not None else None,
            snapshot=snapshot or (DEFAULT_REPO, DEFAULT_COMMIT),
            type_checker=type_checker,
            gateway=gateway,
            use_real_gateway=use_real_gateway if use_real_gateway is not None else True,
            tmp_path=tmp_path,
        )
    else:
        opts = options

    repo, commit = opts.snapshot
    gateway_obj = opts.gateway
    if gateway_obj is None:
        if opts.use_real_gateway:
            base_path = opts.tmp_path if opts.tmp_path is not None else repo_root
            env_ctx = create_test_env(base_path, repo=repo, commit=commit, repo_root=repo_root)
            gateway_obj = env_ctx.gateway
        else:
            gateway_obj = RecordingGateway()

    resources = RecordingResources(
        modules=opts.modules or (),
        type_checker=opts.type_checker,
    )
    return RecordingContext(
        repo_root=repo_root,
        repo=repo,
        commit=commit,
        gateway=gateway_obj,
        resources=resources,
    )


def make_target_context_from_modules(
    repo_root: Path,
    modules: Iterable[str],
    *,
    snapshot: tuple[str, str] | None = None,
    gateway: StorageGateway | RecordingGateway | None = None,
    type_checker: object | None = None,
    use_real_gateway: bool | None = None,
    tmp_path: Path | None = None,
) -> RecordingContext:
    """Helper to build a RecordingContext when only modules and a few overrides are needed."""
    opts = LegacyTargetContextOptions(
        modules=tuple(modules),
        snapshot=snapshot or (DEFAULT_REPO, DEFAULT_COMMIT),
        type_checker=type_checker,
        gateway=gateway,
        use_real_gateway=True if use_real_gateway is None else use_real_gateway,
        tmp_path=tmp_path,
    )
    return make_target_context(repo_root=repo_root, options=opts)


def build_target_context(
    target: OutputTarget,
    *,
    tmp_path: Path | None = None,
    builder_options: BuilderOptions | None = None,
    resources: TargetResourceOverrides | None = None,
) -> TargetExecutionContext:
    """Build a production TargetExecutionContext for ingestion plugins.

    Returns
    -------
    TargetExecutionContext
        Configured target execution context.
    """
    overrides = resources or TargetResourceOverrides()
    opts = builder_options or BuilderOptions()
    return build_target_execution_context(
        target,
        tmp_path=tmp_path,
        options=opts,
        resources=TargetResourceOverrides(
            modules=overrides.modules,
            type_checker=overrides.type_checker,
            providers=overrides.providers,
            change_tracker=overrides.change_tracker,
            graph_runtime=overrides.graph_runtime,
            catalog=overrides.catalog,
            test_reporter=overrides.test_reporter,
            coverage_collector=overrides.coverage_collector,
            scip_indexer=overrides.scip_indexer,
            tool_runner=overrides.tool_runner,
            git_history=overrides.git_history,
        ),
        parameters=None,
    )


def build_repo_tree(root: Path, files: Mapping[str, str]) -> Path:
    """Write a set of files relative to root and return the root path.

    Returns
    -------
    Path
        Repository root containing the written files.
    """
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


__all__ = [
    "LegacyTargetContextOptions",
    "RecordingContext",
    "RecordingGateway",
    "RecordingResources",
    "build_repo_tree",
    "build_target_context",
    "make_target_context",
    "make_target_context_from_modules",
]
