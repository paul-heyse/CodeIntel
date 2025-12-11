"""Compatibility shims for bridging old and new patterns.

Provide adapters that allow legacy HandlerContext-based handlers to work
with the new Deps-based Command[T] pattern, enabling gradual migration.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeGuard

from codeintel.cli.deps.protocols import StorageAccess

if TYPE_CHECKING:
    from codeintel.cli.core.command import Command
    from codeintel.cli.core.results import CliResult
    from codeintel.cli.deps.container import Deps
    from codeintel.cli.handlers.context import HandlerContext
    from codeintel.storage.gateway import StorageGateway


class _HandlerContextStorageAdapter:
    """Adapter that wraps HandlerContext to satisfy StorageAccess protocol.

    Allow new-pattern code to access storage through a HandlerContext.

    Parameters
    ----------
    ctx
        The legacy HandlerContext to wrap.
    """

    def __init__(self, ctx: HandlerContext) -> None:
        """Initialize adapter with HandlerContext."""
        self._ctx = ctx

    @property
    def gateway(self) -> StorageGateway:
        """Get the storage gateway from context.

        Returns
        -------
        StorageGateway
            The context's gateway.
        """
        return self._ctx.gateway

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Get write-enabled gateway from context.

        Yields
        ------
        StorageGateway
            Write-enabled gateway.
        """
        with self._ctx.write_gateway() as gw:
            yield gw


class _DataclassCommand(Protocol):
    __dataclass_fields__: dict[str, dataclasses.Field[object]]
    __operation_id__: str


def _is_dataclass_command(value: object) -> TypeGuard[_DataclassCommand]:
    """Return True when the value is a dataclass command instance.

    Returns
    -------
    bool
        True if the object is a dataclass command instance.
    """
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def deps_from_handler_context(ctx: HandlerContext) -> Deps:
    """Create Deps from legacy HandlerContext.

    Bridge old handlers to new dependency model during migration.

    Parameters
    ----------
    ctx
        Legacy HandlerContext.

    Returns
    -------
    Deps
        New-style dependency container wrapping the context.
    """
    from codeintel.cli.deps.container import Deps
    from codeintel.cli.jobs import get_job_manager

    storage: StorageAccess | None = None
    if ctx._gateway is not None:  # noqa: SLF001
        storage = _HandlerContextStorageAdapter(ctx)

    return Deps(
        config=ctx.config,
        logger=ctx.logger,
        jobs=get_job_manager(),
        _storage=storage,
        _serving=None,
    )


def handler_context_from_deps[T](deps: Deps, cmd: Command[T]) -> HandlerContext:
    """Create HandlerContext from Deps for legacy handler compatibility.

    Allow new Command[T] instances to delegate to legacy handlers
    during migration.

    Parameters
    ----------
    deps
        New-style dependency container.
    cmd
        Command instance to extract parameters from.

    Returns
    -------
    HandlerContext
        Legacy context compatible with old handlers.
    """
    from codeintel.cli.handlers.context import HandlerContext
    from codeintel.cli.rendering.types import OutputFormat

    if not _is_dataclass_command(cmd):
        msg = "handler_context_from_deps requires a dataclass command instance"
        raise TypeError(msg)

    params: dict[str, object] = {}
    flags_output_format = OutputFormat.TEXT
    flags_verbosity = 0
    flags_project_root = None

    for fld in dataclasses.fields(cmd):
        value = getattr(cmd, fld.name)
        if fld.name == "flags" and value is not None:
            # Extract SharedFlags values
            if hasattr(value, "output_format"):
                flags_output_format = value.output_format
            if hasattr(value, "verbose"):
                flags_verbosity = value.verbose
            if hasattr(value, "project_root"):
                flags_project_root = value.project_root
        else:
            params[fld.name] = value

    return HandlerContext(
        config=deps.config,
        operation_id=cmd.__operation_id__,
        output_format=flags_output_format,
        verbosity=flags_verbosity,
        project_root=flags_project_root,
        _params=params,
    )


def wrap_legacy_handler[T](
    handler: Callable[[HandlerContext], CliResult[T]],
) -> Callable[[Command[T], Deps], CliResult[T]]:
    """Wrap a legacy handler to work with new Command pattern.

    For gradual migration of complex handlers that are difficult
    to rewrite all at once.

    Parameters
    ----------
    handler
        Legacy handler function that takes HandlerContext.

    Returns
    -------
    Callable[[Command[T], Deps], CliResult[T]]
        Wrapper function that can be called with Command and Deps.
    """

    def wrapped(cmd: Command[T], deps: Deps) -> CliResult[T]:
        ctx = handler_context_from_deps(deps, cmd)
        return handler(ctx)

    return wrapped


__all__ = [
    "deps_from_handler_context",
    "handler_context_from_deps",
    "wrap_legacy_handler",
]
