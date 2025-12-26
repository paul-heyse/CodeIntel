"""Observability test fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from codeintel.observability.attribute_sanitizer import SpanAttributeValue
from codeintel.observability.attribute_schema import AttributeRegistry, default_attribute_registry


@dataclass(frozen=True, slots=True)
class TelemetryContract:
    """Validate telemetry attributes against the registered schema."""

    registry: AttributeRegistry

    def assert_valid_attributes(self, attributes: Mapping[str, SpanAttributeValue]) -> None:
        """Assert all attributes match the registry schema."""
        for key, value in attributes.items():
            schema = self.registry.resolve(key)
            assert schema is not None, f"Unknown telemetry attribute key: {key}"
            assert schema.is_value_allowed(value), f"Invalid value for {key}: {value}"


@pytest.fixture
def telemetry_contract() -> TelemetryContract:
    """Provide a telemetry contract fixture with the default registry.

    Returns
    -------
    TelemetryContract
        Contract using the default attribute registry.
    """
    return TelemetryContract(registry=default_attribute_registry())
