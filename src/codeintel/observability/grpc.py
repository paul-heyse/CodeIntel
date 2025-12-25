"""gRPC observability plugin lifecycle."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.config.settings import GrpcObservabilitySettings
from codeintel.observability.instrumentation_registry import InstrumentationRegistry

if TYPE_CHECKING:
    from opentelemetry.metrics import MeterProvider

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GrpcObservabilityHandle:
    """Handle for an active grpcio-observability plugin."""

    plugin: object

    def shutdown(self) -> None:
        """Deregister the global grpcio-observability plugin.

        Returns
        -------
        None
            None.
        """
        deregister = getattr(self.plugin, "deregister_global", None)
        if callable(deregister):
            deregister()


def register_grpc_observability(
    settings: GrpcObservabilitySettings,
    *,
    meter_provider: MeterProvider,
    registry: InstrumentationRegistry,
) -> GrpcObservabilityHandle | None:
    """Register grpcio-observability when enabled and supported.

    Returns
    -------
    GrpcObservabilityHandle | None
        Handle for the registered plugin, or ``None`` when unavailable.
    """
    if not settings.enabled:
        registry.record_suppressed("grpcio-observability")
        return None

    if sys.platform != "linux":
        registry.record_unavailable(
            "grpcio-observability",
            detail="grpcio-observability is only supported on Linux",
        )
        return None

    try:
        grpc_observability = importlib.import_module("grpc_observability")
    except ModuleNotFoundError as exc:
        registry.record_unavailable("grpcio-observability", detail=str(exc))
        return None

    if settings.other_method_label != "other":
        LOG.warning(
            "grpcio-observability uses a fixed other label; got %s",
            settings.other_method_label,
        )
    if settings.other_target_label != "other":
        LOG.warning(
            "grpcio-observability uses a fixed other label; got %s",
            settings.other_target_label,
        )

    method_filter = _build_allowlist_filter(settings.method_allowlist)
    target_filter = _build_allowlist_filter(settings.target_allowlist)

    plugin = grpc_observability.OpenTelemetryPlugin(
        meter_provider=meter_provider,
        generic_method_attribute_filter=method_filter,
        target_attribute_filter=target_filter,
    )
    try:
        plugin.register_global()
    except RuntimeError as exc:
        registry.record_error("grpcio-observability", detail=str(exc))
        return None

    registry.record_enabled("grpcio-observability")
    return GrpcObservabilityHandle(plugin=plugin)


def _build_allowlist_filter(allowlist: tuple[str, ...]) -> Callable[[str], bool]:
    values = {value.strip() for value in allowlist if value.strip()}
    if not values:
        return lambda _value: False
    return lambda value: value in values


__all__ = [
    "GrpcObservabilityHandle",
    "register_grpc_observability",
]
