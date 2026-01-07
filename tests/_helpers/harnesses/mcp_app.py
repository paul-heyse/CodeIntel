"""FastMCP-focused harness for serving tests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp.client import Client

from codeintel.serving.settings import ServingSettings
from tests._helpers.harnesses.serving_app import (
    ServingAppHarness,
    ServingSettingsOverrides,
)
from tests._helpers.serving_snapshot_factory import ServingSnapshot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from fastmcp.client import FastMCPTransport

    from codeintel.serving.context import ServingContext
    from codeintel.serving.mcp.protocols import SemanticKernelProtocol


@dataclass(frozen=True)
class McpAppHarness:
    """Harness for MCP-only serving clients."""

    snapshot: ServingSnapshot
    settings: ServingSettings

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ServingSnapshot,
        *,
        settings_overrides: ServingSettingsOverrides | None = None,
    ) -> McpAppHarness:
        """Create a harness from a serving snapshot.

        Parameters
        ----------
        snapshot
            Serving snapshot created by ServingSnapshotFactory.
        settings_overrides
            Optional setting overrides to apply.

        Returns
        -------
        McpAppHarness
            Harness configured with resolved settings.
        """
        base = ServingAppHarness.from_snapshot(snapshot, settings_overrides=settings_overrides)
        return cls(snapshot=base.snapshot, settings=base.settings)

    @asynccontextmanager
    async def client(
        self,
        *,
        settings_overrides: ServingSettingsOverrides | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        kernel_builder: Callable[[ServingContext], SemanticKernelProtocol] | None = None,
    ) -> AsyncIterator[Client[FastMCPTransport]]:
        """Create a FastMCP client bound to the serving MCP server.

        Parameters
        ----------
        settings_overrides
            Optional ServingSettings overrides.
        client_kwargs
            Optional keyword arguments passed to fastmcp.client.Client.
        kernel_builder
            Optional callback to build a kernel from the runtime.

        Yields
        ------
        Client
            Active FastMCP client.
        """
        harness = ServingAppHarness(snapshot=self.snapshot, settings=self.settings)
        async with harness.mcp_client(
            settings_overrides=settings_overrides,
            client_kwargs=client_kwargs,
            kernel_builder=kernel_builder,
        ) as client:
            yield client


__all__ = ["McpAppHarness"]
