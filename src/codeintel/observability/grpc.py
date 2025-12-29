"""gRPC observability plugin lifecycle."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.core.config.settings import GrpcObservabilitySettings
from codeintel.observability.registry import InstrumentationRegistry

if TYPE_CHECKING:
    from opentelemetry.metrics import MeterProvider

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GrpcObservabilityHandle:
    """Handle for an active grpcio-observability plugin."""

    plugin: _GrpcObservabilityPlugin

    def shutdown(self) -> None:
        """Deregister the global grpcio-observability plugin."""
        deregister = getattr(self.plugin, "deregister_global", None)
        if callable(deregister):
            deregister()


class _GrpcObservabilityPlugin(Protocol):
    def register_global(self) -> None:
        """Register the global grpcio-observability plugin."""
        ...

    def deregister_global(self) -> None:
        """Deregister the global grpcio-observability plugin."""
        ...


def _load_grpc_observability_module(
    *,
    registry: InstrumentationRegistry,
    module_loader: Callable[[str], ModuleType] | None = None,
) -> ModuleType | None:
    loader = module_loader or importlib.import_module
    try:
        module = loader("grpc_observability")
    except ModuleNotFoundError as exc:
        registry.record_unavailable("grpcio-observability", detail=str(exc))
        return None
    if module is None:
        registry.record_unavailable(
            "grpcio-observability",
            detail="grpcio-observability module is unavailable",
        )
        return None
    return module


def _build_grpc_observability_plugin(
    *,
    module: ModuleType,
    settings: GrpcObservabilitySettings,
    meter_provider: MeterProvider,
    registry: InstrumentationRegistry,
) -> _GrpcObservabilityPlugin | None:
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

    plugin_cls = getattr(module, "OpenTelemetryPlugin", None)
    if not isinstance(plugin_cls, type):
        registry.record_unavailable(
            "grpcio-observability",
            detail="grpcio-observability plugin is unavailable",
        )
        return None

    plugin_type = cast("Callable[..., _GrpcObservabilityPlugin]", plugin_cls)
    return plugin_type(
        meter_provider=meter_provider,
        generic_method_attribute_filter=method_filter,
        target_attribute_filter=target_filter,
    )


def register_grpc_observability(
    settings: GrpcObservabilitySettings,
    *,
    meter_provider: MeterProvider,
    registry: InstrumentationRegistry,
    platform_override: str | None = None,
    module_loader: Callable[[str], ModuleType] | None = None,
) -> GrpcObservabilityHandle | None:
    """Register grpcio-observability when enabled and supported.

    Parameters
    ----------
    settings
        gRPC observability settings.
    meter_provider
        Meter provider used to emit grpcio-observability metrics.
    registry
        Instrumentation registry for reporting status.
    platform_override
        Optional platform string override for testing.
    module_loader
        Optional module loader override for testing.

    Returns
    -------
    GrpcObservabilityHandle | None
        Handle for the registered plugin, or ``None`` when unavailable.
    """
    if not settings.enabled:
        registry.record_suppressed("grpcio-observability")
        return None

    active_platform = platform_override or sys.platform
    if active_platform != "linux":
        registry.record_unavailable(
            "grpcio-observability",
            detail="grpcio-observability is only supported on Linux",
        )
        return None

    module = _load_grpc_observability_module(
        registry=registry,
        module_loader=module_loader,
    )
    if module is None:
        return None

    plugin = _build_grpc_observability_plugin(
        module=module,
        settings=settings,
        meter_provider=meter_provider,
        registry=registry,
    )
    if plugin is None:
        return None

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
