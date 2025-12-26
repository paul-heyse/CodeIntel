"""Lifecycle helpers for observability bootstrap and teardown."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.observability.mcp import McpOpenTelemetryMiddleware
from codeintel.observability.runtime import (
    ObservabilityRuntime,
    ObservabilityShutdownResult,
    bootstrap_observability,
    get_observability,
    resolve_observability_config,
    shutdown_observability,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import FastAPI


@dataclass(slots=True)
class ObservabilityLifecycle:
    """Shared lifecycle controller for observability runtime state."""

    default_service_name: str
    runtime: ObservabilityRuntime | None = None

    def bootstrap(
        self,
        settings: ObservabilitySettings,
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> ObservabilityRuntime:
        """Bootstrap observability and store the runtime handle.

        Returns
        -------
        ObservabilityRuntime
            Initialized observability runtime.
        """
        resolved = resolve_observability_config(
            settings,
            default_service_name=self.default_service_name,
            overrides=overrides,
        )
        self.runtime = bootstrap_observability(resolved)
        return self.runtime

    def install_logging(
        self, runtime: ObservabilityRuntime | None = None
    ) -> logging.Handler | None:
        """Return the log handler attached to the runtime, if any.

        Returns
        -------
        logging.Handler | None
            Log handler attached to the runtime, if available.
        """
        active = runtime or self.runtime or get_observability()
        return active.log_handler

    @staticmethod
    def attach_fastapi(app: FastAPI) -> None:
        """Attach FastAPI instrumentation to an application."""
        FastAPIInstrumentor.instrument_app(app)

    @staticmethod
    def attach_mcp() -> McpOpenTelemetryMiddleware:
        """Return the MCP middleware instance for the serving stack.

        Returns
        -------
        McpOpenTelemetryMiddleware
            Middleware instance for MCP instrumentation.
        """
        return McpOpenTelemetryMiddleware()

    def shutdown(self) -> ObservabilityShutdownResult | None:
        """Shutdown the observability runtime and clear local state.

        Returns
        -------
        ObservabilityShutdownResult | None
            Shutdown result when telemetry was enabled.
        """
        result = shutdown_observability()
        self.runtime = None
        return result


__all__ = ["ObservabilityLifecycle"]
