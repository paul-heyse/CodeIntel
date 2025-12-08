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
    gateway: StorageGateway | RecordingGateway | None = None


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
        """Create options with explicit modules.

        Returns
        -------
        LegacyTargetContextOptions
            Options with the provided modules applied.
        """
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
    **overrides: object,
) -> RecordingContext:
    """Construct a recording context matching plugin expectations.

    Parameters
    ----------
    repo_root
        Repository root directory.
    options
        Pre-built options bundle. Mutually exclusive with overrides.
    overrides
        Optional overrides: modules, gateway, type_checker, use_real_gateway, tmp_path, snapshot.

    Returns
    -------
    RecordingContext
        Context populated with provided options.

    Raises
    ------
    ValueError
        If both options and overrides are provided or unexpected keys are supplied.
    """
    allowed_keys = {
        "modules",
        "gateway",
        "type_checker",
        "use_real_gateway",
        "tmp_path",
        "snapshot",
    }
    unexpected_keys = set(overrides) - allowed_keys
    if options is not None and overrides:
        message = "Pass either options or individual overrides, not both."
        raise ValueError(message)
    if unexpected_keys:
        unexpected_list = ", ".join(sorted(unexpected_keys))
        message = f"Unexpected overrides: {unexpected_list}"
        raise ValueError(message)

    if options is None:
        opts = LegacyTargetContextOptions(
            modules=tuple(overrides["modules"]) if "modules" in overrides else None,
            snapshot=overrides.get("snapshot", (DEFAULT_REPO, DEFAULT_COMMIT)),
            type_checker=overrides.get("type_checker"),
            gateway=overrides.get("gateway"),
            use_real_gateway=overrides.get("use_real_gateway", True),
            tmp_path=overrides.get("tmp_path"),
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
        gateway=gateway_obj,
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
    **overrides: object,
) -> RecordingContext:
    """Build a RecordingContext with explicit modules and optional overrides.

    Parameters
    ----------
    repo_root
        Repository root directory.
    modules
        Module paths to include in the context.
    overrides
        Optional overrides forwarded to `make_target_context`.

    Returns
    -------
    RecordingContext
        Context populated with provided modules and overrides.
    """
    opts = LegacyTargetContextOptions(
        modules=tuple(modules),
        snapshot=overrides.get("snapshot", (DEFAULT_REPO, DEFAULT_COMMIT)),
        type_checker=overrides.get("type_checker"),
        gateway=overrides.get("gateway"),
        use_real_gateway=overrides.get("use_real_gateway", True),
        tmp_path=overrides.get("tmp_path"),
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
    "_RecordingConnection",
    "build_repo_tree",
    "build_target_context",
    "make_target_context",
    "make_target_context_from_modules",
]
