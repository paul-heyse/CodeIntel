"""Fakes and builders for ingestion plugin contexts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    from codeintel.build.protocols import TypeChecker
    from codeintel.storage.gateway import StorageGateway
else:  # pragma: no cover - runtime placeholder
    TypeChecker = object  # type: ignore[misc,assignment]


# Valid override keys for make_target_context
_VALID_OVERRIDE_KEYS = frozenset(
    {
        "modules",
        "snapshot",
        "type_checker",
        "gateway",
        "use_real_gateway",
        "tmp_path",
        "options",
    }
)


@dataclass
class RecordingGateway:
    """Gateway stub that records SQL executed via its .con property."""

    result_rows: list[tuple[object, ...]] = field(default_factory=list)
    executions: list[tuple[str, list[object]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the connection attribute."""
        self.con = _RecordingConnection(self)


class _RecordingConnection:
    """Stub connection that records SQL and returns canned results."""

    def __init__(self, gateway: RecordingGateway) -> None:
        self._gateway = gateway

    def execute(self, sql: str, params: Iterable[object]) -> _RecordingConnection:
        """Record an SQL execution and return self for chaining.

        Returns
        -------
        _RecordingConnection
            Self for method chaining.
        """
        self._gateway.executions.append((sql, list(params)))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        """Return the canned result rows.

        Returns
        -------
        list[tuple[object, ...]]
            List of result row tuples.
        """
        return self._gateway.result_rows


@dataclass
class RecordingResources:
    """Minimal resources bundle for plugin contexts."""

    modules: tuple[str, ...] = ()
    type_checker: object | None = None
    gateway: StorageGateway | RecordingGateway | None = None


@dataclass(frozen=True)
class TargetContextOverrides:
    """Typed overrides for building ingestion target contexts."""

    modules: tuple[str, ...] = ()
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT)
    type_checker: object | None = None
    gateway: StorageGateway | RecordingGateway | None = None
    use_real_gateway: bool = True
    tmp_path: Path | None = None


# Legacy alias for backward compatibility with tests
LegacyTargetContextOptions = TargetContextOverrides


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
    overrides: TargetContextOverrides | None = None,
    *,
    modules: Sequence[str] | None = None,
    snapshot: tuple[str, str] | None = None,
    type_checker: object | None = None,
    gateway: StorageGateway | RecordingGateway | None = None,
    use_real_gateway: bool | None = None,
    tmp_path: Path | None = None,
    options: TargetContextOverrides | None = None,
) -> RecordingContext:
    """Construct a recording context matching plugin expectations.

    This function supports two API styles:
    1. Dataclass style: pass a `TargetContextOverrides` object
    2. Keyword style: pass individual kwargs (modules, gateway, etc.)

    Parameters
    ----------
    repo_root
        Repository root directory.
    overrides
        Optional overrides bundle (dataclass style API).
    modules
        Module paths to include in resources (keyword style).
    snapshot
        Tuple of (repo, commit) identifiers.
    type_checker
        Type checker instance for resources.
    gateway
        Pre-existing gateway to use.
    use_real_gateway
        Whether to create a real gateway (True) or use RecordingGateway (False).
    tmp_path
        Temporary path for gateway setup.
    options
        Alias for `overrides` (for backward compatibility).

    Returns
    -------
    RecordingContext
        Context populated with provided options.

    Raises
    ------
    ValueError
        If both `overrides` and keyword overrides are provided, or if unknown
        keyword arguments are passed.
    """
    # Check for mixed API usage
    has_kwargs = any(
        v is not None
        for v in [modules, snapshot, type_checker, gateway, use_real_gateway, tmp_path]
    )
    opts_arg = overrides or options

    if opts_arg is not None and has_kwargs:
        message = "Use either options/overrides OR keyword arguments, not both"
        raise ValueError(message)

    # Build overrides from kwargs if no dataclass provided
    if opts_arg is None and has_kwargs:
        opts = TargetContextOverrides(
            modules=tuple(modules) if modules else (),
            snapshot=snapshot if snapshot else (DEFAULT_REPO, DEFAULT_COMMIT),
            type_checker=type_checker,
            gateway=gateway,
            use_real_gateway=use_real_gateway if use_real_gateway is not None else True,
            tmp_path=tmp_path,
        )
    else:
        opts = opts_arg or TargetContextOverrides()

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
    overrides: TargetContextOverrides | None = None,
    *,
    snapshot: tuple[str, str] | None = None,
    type_checker: object | None = None,
    gateway: StorageGateway | RecordingGateway | None = None,
    use_real_gateway: bool | None = None,
    tmp_path: Path | None = None,
) -> RecordingContext:
    """Build a RecordingContext with explicit modules and optional overrides.

    This is a convenience wrapper around `make_target_context` that accepts
    modules as a positional argument.

    Parameters
    ----------
    repo_root
        Repository root directory.
    modules
        Module paths to include in the context.
    overrides
        Optional overrides bundle (dataclass style API).
    snapshot
        Tuple of (repo, commit) identifiers.
    type_checker
        Type checker instance for resources.
    gateway
        Pre-existing gateway to use.
    use_real_gateway
        Whether to create a real gateway (True) or use RecordingGateway (False).
    tmp_path
        Temporary path for gateway setup.

    Returns
    -------
    RecordingContext
        Context populated with provided modules and overrides.
    """
    base_overrides = overrides or TargetContextOverrides()

    # Merge kwargs with base overrides, preferring explicit kwargs
    merged = TargetContextOverrides(
        modules=tuple(modules),
        snapshot=snapshot if snapshot else base_overrides.snapshot,
        type_checker=type_checker if type_checker is not None else base_overrides.type_checker,
        gateway=gateway if gateway is not None else base_overrides.gateway,
        use_real_gateway=(
            use_real_gateway if use_real_gateway is not None else base_overrides.use_real_gateway
        ),
        tmp_path=tmp_path if tmp_path is not None else base_overrides.tmp_path,
    )
    return make_target_context(repo_root=repo_root, overrides=merged)


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
        resources=overrides,
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
    "TargetContextOverrides",
    "_RecordingConnection",
    "build_repo_tree",
    "build_target_context",
    "make_target_context",
    "make_target_context_from_modules",
]
