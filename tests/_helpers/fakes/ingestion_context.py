"""Fakes and builders for ingestion plugin contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.targets import OutputTarget
from tests._helpers.env import DEFAULT_COMMIT, DEFAULT_REPO, create_test_env
from tests._helpers.fakes.contexts import (
    BuilderOptions,
    RecordingGateway,
    TargetResourceOverrides,
    build_target_execution_context,
)

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway


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


@dataclass
class RecordingContext:
    """Lightweight stand-in for legacy ingestion tests."""

    repo_root: Path
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    gateway: StorageGateway | RecordingGateway | None = None
    resources: RecordingResources = field(default_factory=RecordingResources)


def make_target_context(
    repo_root: Path,
    overrides: TargetContextOverrides | None = None,
    **kwargs: object,
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
    **kwargs
        Optional keyword overrides (modules, snapshot, type_checker, gateway,
        use_real_gateway, tmp_path, or legacy ``options`` alias).

    Returns
    -------
    RecordingContext
        Context populated with provided options.

    Raises
    ------
    ValueError
        If an unsupported override key is provided.
    """
    opts_arg = overrides or kwargs.pop("options", None)
    recognized = {
        "modules",
        "snapshot",
        "type_checker",
        "gateway",
        "use_real_gateway",
        "tmp_path",
    }
    unknown_keys = set(kwargs) - recognized
    if unknown_keys:
        message = f"Unsupported override keys: {sorted(unknown_keys)}"
        raise ValueError(message)

    if opts_arg is None and kwargs:
        modules_val = kwargs.get("modules")
        use_real_gateway_val = kwargs.get("use_real_gateway")
        opts = TargetContextOverrides(
            modules=tuple(cast("Sequence[str]", modules_val)) if modules_val is not None else (),
            snapshot=cast(
                "tuple[str, str] | None",
                kwargs.get("snapshot"),
            )
            or (DEFAULT_REPO, DEFAULT_COMMIT),
            type_checker=kwargs.get("type_checker"),
            gateway=cast(
                "StorageGateway | RecordingGateway | None",
                kwargs.get("gateway"),
            ),
            use_real_gateway=bool(use_real_gateway_val)
            if use_real_gateway_val is not None
            else True,
            tmp_path=cast("Path | None", kwargs.get("tmp_path")),
        )
    else:
        opts = cast("TargetContextOverrides", opts_arg or TargetContextOverrides())

    repo, commit = opts.snapshot
    gateway_obj = opts.gateway
    if gateway_obj is None:
        base_path = opts.tmp_path if opts.tmp_path is not None else repo_root
        env_ctx = create_test_env(base_path, repo=repo, commit=commit, repo_root=repo_root)
        gateway_obj = env_ctx.gateway

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


def build_target_context(
    target: OutputTarget,
    tmp_path: Path,
    *,
    builder_options: BuilderOptions | None = None,
    resources: TargetResourceOverrides | None = None,
) -> TargetExecutionContext:
    """Build a production TargetExecutionContext for ingestion plugins.

    Parameters
    ----------
    target
        Output target for the context.
    tmp_path
        Temporary directory for test isolation (required).
    builder_options
        Optional builder configuration.
    resources
        Optional resource overrides.

    Returns
    -------
    TargetExecutionContext
        Configured target execution context.
    """
    overrides = resources or TargetResourceOverrides()
    opts = builder_options or BuilderOptions()
    return build_target_execution_context(
        target,
        tmp_path,
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
    "RecordingContext",
    "RecordingGateway",
    "RecordingResources",
    "TargetContextOverrides",
    "build_repo_tree",
    "build_target_context",
    "make_target_context",
]
