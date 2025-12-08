"""Composable MCP tool registration context.

This context allows tests and utilities to register MCP tools with
custom operation catalogs, model resolvers, and backends without
patching global state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.serialization import ResponseFactory
from codeintel.serving.mcp.tool_builder import (
    McpToolRegistrar,
    ToolRegistrationOptions,
    register_tools_for_category,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations import Operation

ModelResolver = Callable[[str], ResponseFactory | None]


@dataclass
class McpToolContext:
    """Context for registering MCP tools against a backend."""

    mcp: McpToolRegistrar
    backend: QueryBackendOrService
    operations: Iterable[Operation]
    model_resolver: ModelResolver | None = None
    config: ServingConfig | None = None

    def register_categories(self, categories: set[str]) -> None:
        """Register tools for the given categories using injected operations."""
        register_tools_for_category(
            self.mcp,
            self.backend,
            categories,
            config=self.config,
            options=ToolRegistrationOptions(
                operations=self.operations,
                model_resolver=self._resolve_model,
            ),
        )

    def _resolve_model(self, name: str) -> ResponseFactory | None:
        """Resolve response models through injected resolver when available.

        Returns
        -------
        ResponseFactory | None
            Resolved response model factory if provided by the resolver.
        """
        if self.model_resolver is None:
            return None
        return self.model_resolver(name)


__all__ = ["McpToolContext", "ModelResolver"]
