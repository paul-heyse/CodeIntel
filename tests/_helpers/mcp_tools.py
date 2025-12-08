"""Helpers for constructing MCP tool contexts in tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.tool_context import McpToolContext, ModelResolver
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations import Operation
from tests._helpers.mcp_registrar import RecordingMcpRegistrar

if TYPE_CHECKING:
    from codeintel.serving.mcp.tool_builder import McpToolRegistrar


@dataclass
class McpTestContext:
    """Bundled MCP context for tests."""

    mcp: RecordingMcpRegistrar
    backend: QueryBackendOrService
    operations: Iterable[Operation]
    model_resolver: ModelResolver | None = None
    config: ServingConfig | None = None

    def register(self, categories: set[str]) -> None:
        """Register tools for the given categories."""
        context = McpToolContext(
            mcp=cast("McpToolRegistrar", self.mcp),
            backend=self.backend,
            operations=self.operations,
            model_resolver=self.model_resolver,
            config=self.config,
        )
        context.register_categories(categories)


def make_mcp_context(
    backend: QueryBackendOrService,
    operations: Iterable[Operation],
    *,
    model_resolver: ModelResolver | None = None,
    config: ServingConfig | None = None,
) -> McpTestContext:
    """Create a RecordingMcp plus registration context for tests.

    Returns
    -------
    McpTestContext
        Context that can register tools for given categories.
    """
    return McpTestContext(
        mcp=RecordingMcpRegistrar(),
        backend=backend,
        operations=operations,
        model_resolver=model_resolver,
        config=config,
    )


__all__ = ["McpTestContext", "make_mcp_context"]
