"""gRPC observability plugin tests."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

import pytest
from opentelemetry.sdk.metrics import MeterProvider

from codeintel.core.config.settings import GrpcObservabilitySettings
from codeintel.observability.grpc import register_grpc_observability
from codeintel.observability.instrumentation_registry import InstrumentationRegistry


class _StubPlugin:
    def __init__(
        self,
        *,
        meter_provider: object,
        generic_method_attribute_filter: Callable[[str], bool],
        target_attribute_filter: Callable[[str], bool],
    ) -> None:
        self.meter_provider = meter_provider
        self.generic_method_attribute_filter = generic_method_attribute_filter
        self.target_attribute_filter = target_attribute_filter
        self.registered = False

    def register_global(self) -> None:
        self.registered = True

    def deregister_global(self) -> None:
        self.registered = False


class _StubModule(ModuleType):
    OpenTelemetryPlugin = _StubPlugin


def _make_stub_module() -> ModuleType:
    return _StubModule("grpc_observability")


def test_grpc_observability_disabled_records_suppressed() -> None:
    """Disabled settings should record a suppressed status."""
    registry = InstrumentationRegistry()
    settings = GrpcObservabilitySettings(enabled=False)

    handle = register_grpc_observability(
        settings,
        meter_provider=MeterProvider(),
        registry=registry,
    )

    assert handle is None
    records = registry.snapshot()
    assert len(records) == 1
    assert records[0].status == "suppressed"


def test_grpc_observability_unavailable_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Linux platforms should report grpcio-observability as unavailable."""
    registry = InstrumentationRegistry()
    settings = GrpcObservabilitySettings(enabled=True)
    monkeypatch.setattr(sys, "platform", "darwin")

    handle = register_grpc_observability(
        settings,
        meter_provider=MeterProvider(),
        registry=registry,
    )

    assert handle is None
    records = registry.snapshot()
    assert len(records) == 1
    assert records[0].status == "unavailable"


def test_grpc_observability_registers_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled settings should register the plugin and apply filters."""
    registry = InstrumentationRegistry()
    settings = GrpcObservabilitySettings(
        enabled=True,
        method_allowlist=("service/Method",),
        target_allowlist=("127.0.0.1",),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "grpc_observability", _make_stub_module())

    handle = register_grpc_observability(
        settings,
        meter_provider=MeterProvider(),
        registry=registry,
    )

    assert handle is not None
    plugin = cast("_StubPlugin", handle.plugin)
    assert plugin.registered is True
    assert plugin.generic_method_attribute_filter("service/Method") is True
    assert plugin.generic_method_attribute_filter("other") is False
    assert plugin.target_attribute_filter("127.0.0.1") is True
    assert plugin.target_attribute_filter("other") is False

    records = registry.snapshot()
    assert len(records) == 1
    assert records[0].status == "enabled"
