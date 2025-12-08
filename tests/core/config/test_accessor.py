"""Test ConfigAccessor protocol from codeintel.core.config_protocol.

This module tests the configuration accessor protocol including:
- Protocol structural checks (runtime_checkable)
- Integration with ConfigProvider from plugins.context
- Integration with ConfigRegistry
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.config.accessor import ConfigAccessor
from codeintel.core.config.registry import ConfigRegistry
from codeintel.core.plugins.execution.context import ConfigProvider
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)

# =============================================================================
# Test Configuration Classes
# =============================================================================


@dataclass
class TestConfig:
    """Test configuration class."""

    value: str


@dataclass
class AnotherConfig:
    """Another test configuration class."""

    number: int


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


def test_config_provider_implements_accessor() -> None:
    """Verify that ConfigProvider implements ConfigAccessor protocol."""
    provider = ConfigProvider()

    expect_is_instance(provider, ConfigAccessor)


def test_config_registry_implements_accessor() -> None:
    """Verify that ConfigRegistry implements ConfigAccessor protocol."""
    registry = ConfigRegistry()

    expect_is_instance(registry, ConfigAccessor)


def test_config_accessor_is_runtime_checkable() -> None:
    """Verify that ConfigAccessor is a runtime_checkable protocol."""
    # The protocol should be checkable at runtime
    expect_true(
        hasattr(ConfigAccessor, "__protocol_attrs__") or hasattr(ConfigAccessor, "__subclasshook__")
    )

    # Non-conforming classes should not pass isinstance check
    class NotAnAccessor:
        pass

    expect_true(not isinstance(NotAnAccessor(), ConfigAccessor))


# =============================================================================
# ConfigProvider Implementation Tests
# =============================================================================


def test_config_provider_get_registered(config_provider: ConfigProvider) -> None:
    """Verify that ConfigProvider.get() returns registered config."""
    config = TestConfig(value="test")
    config_provider.register(TestConfig, config)

    result = config_provider.get(TestConfig)

    expect_true(result is config)


def test_config_provider_get_raises_for_missing() -> None:
    """Verify that ConfigProvider.get() raises ValueError for missing config."""
    provider = ConfigProvider()

    with pytest.raises(ValueError, match="not available"):
        provider.get(TestConfig)


def test_config_provider_get_optional_returns_config() -> None:
    """Verify that ConfigProvider.get_optional() returns config when registered."""
    provider = ConfigProvider()
    config = TestConfig(value="test")
    provider.register(TestConfig, config)

    result = provider.get_optional(TestConfig)

    expect_true(result is config)


def test_config_provider_get_optional_returns_none() -> None:
    """Verify that ConfigProvider.get_optional() returns None when not registered."""
    provider = ConfigProvider()

    result = provider.get_optional(TestConfig)

    expect_true(result is None)


def test_config_provider_has_returns_true() -> None:
    """Verify that ConfigProvider.has() returns True for registered config."""
    provider = ConfigProvider()
    provider.register(TestConfig, TestConfig(value="test"))

    expect_true(provider.has(TestConfig))


def test_config_provider_has_returns_false() -> None:
    """Verify that ConfigProvider.has() returns False for unregistered config."""
    provider = ConfigProvider()

    expect_true(not provider.has(TestConfig))


def test_config_provider_register_adds_config() -> None:
    """Verify that ConfigProvider.register() adds a configuration."""
    provider = ConfigProvider()
    config = TestConfig(value="test")

    provider.register(TestConfig, config)

    expect_true(provider.has(TestConfig))
    expect_true(provider.get(TestConfig) is config)


def test_config_provider_initialized_with_configs() -> None:
    """Verify that ConfigProvider can be initialized with a config mapping."""
    config = TestConfig(value="initial")
    another = AnotherConfig(number=42)

    provider = ConfigProvider(
        {
            TestConfig: config,
            AnotherConfig: another,
        }
    )

    expect_true(provider.get(TestConfig) is config)
    expect_true(provider.get(AnotherConfig) is another)


# =============================================================================
# ConfigRegistry Implementation Tests (Protocol Compliance)
# =============================================================================


def test_config_registry_get_as_accessor() -> None:
    """Verify that ConfigRegistry.get() works through ConfigAccessor interface."""
    registry = ConfigRegistry()
    config = TestConfig(value="test")
    registry.register(TestConfig, config)

    # Access through protocol type hint
    accessor: ConfigAccessor = registry
    result = accessor.get(TestConfig)

    expect_true(result is config)


def test_config_registry_get_optional_as_accessor() -> None:
    """Verify that ConfigRegistry.get_optional() works through ConfigAccessor interface."""
    registry = ConfigRegistry()

    accessor: ConfigAccessor = registry
    result = accessor.get_optional(TestConfig)

    expect_true(result is None)


def test_config_registry_has_as_accessor() -> None:
    """Verify that ConfigRegistry.has() works through ConfigAccessor interface."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(value="test"))

    accessor: ConfigAccessor = registry

    expect_true(accessor.has(TestConfig))
    expect_true(not accessor.has(AnotherConfig))


def test_config_registry_register_as_accessor() -> None:
    """Verify that ConfigRegistry.register() works through ConfigAccessor interface."""
    registry = ConfigRegistry()
    config = TestConfig(value="test")

    accessor: ConfigAccessor = registry
    accessor.register(TestConfig, config)

    expect_true(accessor.has(TestConfig))


# =============================================================================
# Polymorphic Usage Tests
# =============================================================================


def test_function_accepting_accessor_with_provider() -> None:
    """Verify that functions can accept ConfigAccessor and use ConfigProvider."""

    def get_config_value(accessor: ConfigAccessor) -> str:
        config = accessor.get(TestConfig)
        return config.value

    provider = ConfigProvider({TestConfig: TestConfig(value="from_provider")})

    result = get_config_value(provider)

    expect_equal(result, "from_provider")


def test_function_accepting_accessor_with_registry() -> None:
    """Verify that functions can accept ConfigAccessor and use ConfigRegistry."""

    def get_config_value(accessor: ConfigAccessor) -> str:
        config = accessor.get(TestConfig)
        return config.value

    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(value="from_registry"))

    result = get_config_value(registry)

    expect_equal(result, "from_registry")


def test_optional_config_retrieval_polymorphic() -> None:
    """Verify that optional config retrieval works polymorphically."""

    def get_optional_number(accessor: ConfigAccessor) -> int | None:
        config = accessor.get_optional(AnotherConfig)
        return config.number if config else None

    # With provider that has the config
    provider_with = ConfigProvider({AnotherConfig: AnotherConfig(number=42)})
    expect_equal(get_optional_number(provider_with), 42)

    # With provider that doesn't have the config
    provider_without = ConfigProvider()
    expect_true(get_optional_number(provider_without) is None)

    # With registry
    registry = ConfigRegistry()
    expect_true(get_optional_number(registry) is None)

    registry.register(AnotherConfig, AnotherConfig(number=100))
    expect_equal(get_optional_number(registry), 100)


def test_config_presence_check_polymorphic() -> None:
    """Verify that config presence check works polymorphically."""

    def has_test_config(accessor: ConfigAccessor) -> bool:
        return accessor.has(TestConfig)

    provider = ConfigProvider()
    registry = ConfigRegistry()

    expect_true(not has_test_config(provider))
    expect_true(not has_test_config(registry))

    provider.register(TestConfig, TestConfig(value="p"))
    registry.register(TestConfig, TestConfig(value="r"))

    expect_true(has_test_config(provider))
    expect_true(has_test_config(registry))
