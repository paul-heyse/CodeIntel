"""Test config registry from codeintel.core.config_registry.

This module tests:
- ConfigRegistry operations (register, get, get_optional, has, remove, clear)
- ConfigNotFoundError, ConfigTypeError, ConfigValidationError
- Validator registration and validate_all()
- Iteration, len, contains, as_mapping, copy
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.config_registry import (
    ConfigNotFoundError,
    ConfigRegistry,
    ConfigTypeError,
    ConfigValidationError,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class TestConfig:
    """Test configuration class."""

    host: str
    port: int


@dataclass
class AnotherConfig:
    """Another test configuration class."""

    name: str


# =============================================================================
# ConfigRegistry Basic Tests
# =============================================================================


def test_config_registry_empty_by_default() -> None:
    """Verify new registry is empty."""
    registry = ConfigRegistry()

    assert len(registry) == 0
    assert not registry.has(TestConfig)


def test_config_registry_register_and_get() -> None:
    """Verify register and get work correctly."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)

    registry.register(TestConfig, config)

    assert registry.has(TestConfig)
    assert registry.get(TestConfig) is config


def test_config_registry_get_raises_config_not_found() -> None:
    """Verify get raises ConfigNotFoundError for missing config."""
    registry = ConfigRegistry()

    with pytest.raises(ConfigNotFoundError) as exc_info:
        registry.get(TestConfig)

    assert exc_info.value.config_type is TestConfig
    assert "TestConfig" in str(exc_info.value)


def test_config_registry_get_optional_returns_none() -> None:
    """Verify get_optional returns None for missing config."""
    registry = ConfigRegistry()

    result = registry.get_optional(TestConfig)

    assert result is None


def test_config_registry_get_optional_returns_config() -> None:
    """Verify get_optional returns config when present."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    result = registry.get_optional(TestConfig)

    assert result is config


def test_config_registry_has() -> None:
    """Verify has returns correct boolean."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)

    assert not registry.has(TestConfig)

    registry.register(TestConfig, config)

    assert registry.has(TestConfig)


def test_config_registry_remove_existing() -> None:
    """Verify remove returns True and removes config."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    result = registry.remove(TestConfig)

    assert result is True
    assert not registry.has(TestConfig)


def test_config_registry_remove_missing() -> None:
    """Verify remove returns False for missing config."""
    registry = ConfigRegistry()

    result = registry.remove(TestConfig)

    assert result is False


def test_config_registry_clear() -> None:
    """Verify clear removes all configs."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    registry.clear()

    assert len(registry) == 0
    assert not registry.has(TestConfig)
    assert not registry.has(AnotherConfig)


# =============================================================================
# Type Safety Tests
# =============================================================================


def test_config_registry_register_type_mismatch() -> None:
    """Verify register raises ConfigTypeError for type mismatch."""
    registry = ConfigRegistry()
    wrong_config = AnotherConfig(name="test")

    with pytest.raises(ConfigTypeError) as exc_info:
        registry.register(TestConfig, wrong_config)  # type: ignore[arg-type]

    assert exc_info.value.config_type is TestConfig
    assert exc_info.value.actual_type is AnotherConfig


def test_config_registry_multiple_types() -> None:
    """Verify registry can hold multiple config types."""
    registry = ConfigRegistry()
    test_config = TestConfig(host="localhost", port=5432)
    another_config = AnotherConfig(name="test")

    registry.register(TestConfig, test_config)
    registry.register(AnotherConfig, another_config)

    assert registry.get(TestConfig) is test_config
    assert registry.get(AnotherConfig) is another_config
    assert len(registry) == 2


# =============================================================================
# Validator Tests
# =============================================================================


def test_config_registry_register_validator() -> None:
    """Verify validator is called on register."""
    registry = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry.register_validator(TestConfig, validator)
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    assert len(validated) == 1
    assert validated[0] is config


def test_config_registry_validator_raises() -> None:
    """Verify validator failure raises ConfigValidationError."""
    registry = ConfigRegistry()

    def validator(config: TestConfig) -> None:
        if config.port < 1:
            msg = "Port must be positive"
            raise ValueError(msg)

    registry.register_validator(TestConfig, validator)

    with pytest.raises(ConfigValidationError) as exc_info:
        registry.register(TestConfig, TestConfig(host="localhost", port=0))

    assert exc_info.value.config_type is TestConfig
    assert "Port must be positive" in str(exc_info.value)


def test_config_registry_validate_all() -> None:
    """Verify validate_all runs all validators."""
    registry = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry.register_validator(TestConfig, validator)
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    # Clear and re-validate
    validated.clear()
    registry.validate_all()

    assert len(validated) == 1


def test_config_registry_validate_all_raises() -> None:
    """Verify validate_all raises on validation failure."""
    registry = ConfigRegistry()
    call_count = [0]

    def validator(_config: TestConfig) -> None:
        call_count[0] += 1
        if call_count[0] > 1:
            msg = "Validation failed on re-check"
            raise ValueError(msg)

    registry.register_validator(TestConfig, validator)
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)  # First call passes

    with pytest.raises(ConfigValidationError):
        registry.validate_all()  # Second call fails


# =============================================================================
# Iteration and Container Protocol Tests
# =============================================================================


def test_config_registry_len() -> None:
    """Verify len returns correct count."""
    registry = ConfigRegistry()

    assert len(registry) == 0

    registry.register(TestConfig, TestConfig(host="localhost", port=5432))

    assert len(registry) == 1


def test_config_registry_iter() -> None:
    """Verify iteration over types."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    types = list(registry)

    assert len(types) == 2
    assert TestConfig in types
    assert AnotherConfig in types


def test_config_registry_contains() -> None:
    """Verify in operator works."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))

    assert TestConfig in registry
    assert AnotherConfig not in registry


def test_config_registry_types() -> None:
    """Verify types() returns frozenset."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    types = registry.types()

    assert isinstance(types, frozenset)
    assert TestConfig in types
    assert AnotherConfig in types


# =============================================================================
# Mapping and Copy Tests
# =============================================================================


def test_config_registry_as_mapping() -> None:
    """Verify as_mapping returns correct mapping."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    mapping = registry.as_mapping()

    assert TestConfig in mapping
    assert mapping[TestConfig] is config


def test_config_registry_copy() -> None:
    """Verify copy creates independent registry."""
    registry1 = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry1.register(TestConfig, config)

    registry2 = registry1.copy()

    # Both have the config
    assert registry2.has(TestConfig)
    assert registry2.get(TestConfig) is config

    # Modifying one doesn't affect the other
    registry1.remove(TestConfig)
    assert not registry1.has(TestConfig)
    assert registry2.has(TestConfig)


def test_config_registry_copy_includes_validators() -> None:
    """Verify copy includes validators."""
    registry1 = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry1.register_validator(TestConfig, validator)

    registry2 = registry1.copy()

    # Register config in copy - validator should run
    config = TestConfig(host="localhost", port=5432)
    registry2.register(TestConfig, config)

    assert len(validated) == 1


# =============================================================================
# Error Classes Tests
# =============================================================================


def test_config_not_found_error() -> None:
    """Verify ConfigNotFoundError attributes."""
    error = ConfigNotFoundError(TestConfig)

    assert error.config_type is TestConfig
    assert "TestConfig" in str(error)


def test_config_type_error() -> None:
    """Verify ConfigTypeError attributes."""
    error = ConfigTypeError(TestConfig, AnotherConfig)

    assert error.config_type is TestConfig
    assert error.actual_type is AnotherConfig
    assert "TestConfig" in str(error)
    assert "AnotherConfig" in str(error)


def test_config_validation_error() -> None:
    """Verify ConfigValidationError attributes."""
    error = ConfigValidationError(TestConfig, "Invalid value")

    assert error.config_type is TestConfig
    assert error.reason == "Invalid value"
    assert "TestConfig" in str(error)
    assert "Invalid value" in str(error)
