"""Lightweight CLI CommandContext stubs for tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from codeintel.cli.context import CommandContext, CommandContextBuilder
from codeintel.storage.gateway import StorageGateway


@contextmanager
def make_stub_command_context(
    params: dict[str, object] | None = None,
    *,
    operation_id: str = "cli.stub",
    gateway: StorageGateway | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[CommandContext]:
    """
    Create a minimal CommandContext for tests.

    Parameters
    ----------
    params
        Handler parameters.
    operation_id
        Operation identifier.
    gateway
        Optional gateway to inject; if omitted, no gateway is attached.
    extra
        Optional attributes to attach to the context (e.g., runtime stubs).

    Yields
    ------
    Iterator[CommandContext]
        Built command context configured per arguments.
    """
    builder = CommandContextBuilder().with_params(params or {}).with_operation_id(operation_id)
    if gateway is not None:
        builder = builder.with_injected_gateway(gateway)

    with builder.build() as ctx:
        if extra:
            ctx.__dict__.update(extra)
        yield ctx


__all__ = ["make_stub_command_context"]
