"""OperationSpec builders for tests."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.execution.registry import OperationSpec

ContextHandler = Callable[[CommandContext], CliResult[Any]]
ZeroArgHandler = Callable[[], CliResult[Any]]
Handler = ContextHandler | ZeroArgHandler


def make_operation_spec(
    operation_id: str,
    handler: Handler,
    *,
    name: str | None = None,
    description: str | None = None,
    group: str = "test",
) -> OperationSpec:
    """Build an OperationSpec with sane defaults for tests.

    Returns
    -------
    OperationSpec
        Spec constructed from the provided handler and metadata.
    """
    sig = inspect.signature(handler)
    expects_ctx = len(sig.parameters) > 0

    def wrapped(ctx: CommandContext) -> CliResult[Any]:
        if expects_ctx:
            ctx_handler = cast("ContextHandler", handler)
            return ctx_handler(ctx)
        zero_handler = cast("ZeroArgHandler", handler)
        return zero_handler()

    return OperationSpec(
        operation_id=operation_id,
        name=name or operation_id,
        description=description or operation_id,
        handler=wrapped,
        group=group,
    )


__all__ = ["make_operation_spec"]
