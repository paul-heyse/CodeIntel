"""Test config registry from codeintel.core.config_registry.

This module tests:
- ConfigRegistry operations (register, get, get_optional, has, remove, clear)
- ConfigNotFoundError, ConfigTypeError, ConfigValidationError
- Validator registration and validate_all()
- Iteration, len, contains, as_mapping, copy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from codeintel.core.config.registry import (
    ConfigNotFoundError,
    ConfigRegistry,
    ConfigTypeError,
    ConfigValidationError,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_true,
)


@dataclass
class TestConfig:
    """Test configuration class."""

    host: str
    port: int


@dataclass
class AnotherConfig:
    """Another test configuration class."""

    name: str


def test_config_registry_empty_by_default() -> None:
    """Verify new registry is empty."""
    registry = ConfigRegistry()

    expect_equal(len(registry), 0)
    expect_false(registry.has(TestConfig))


def test_config_registry_register_and_get() -> None:
    """Verify register and get work correctly."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)

    registry.register(TestConfig, config)

    expect_true(registry.has(TestConfig))
    expect_true(registry.get(TestConfig) is config)


def test_config_registry_get_raises_config_not_found() -> None:
    """Verify get raises ConfigNotFoundError for missing config."""
    registry = ConfigRegistry()

    with pytest.raises(ConfigNotFoundError) as exc_info:
        registry.get(TestConfig)

    expect_true(exc_info.value.config_type is TestConfig)
    expect_in("TestConfig", str(exc_info.value))


def test_config_registry_get_optional_returns_none() -> None:
    """Verify get_optional returns None for missing config."""
    registry = ConfigRegistry()

    result = registry.get_optional(TestConfig)

    expect_is_none(result)


def test_config_registry_get_optional_returns_config() -> None:
    """Verify get_optional returns config when present."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    result = registry.get_optional(TestConfig)

    expect_true(result is config)


def test_config_registry_has() -> None:
    """Verify has returns correct boolean."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)

    expect_false(registry.has(TestConfig))

    registry.register(TestConfig, config)

    expect_true(registry.has(TestConfig))


def test_config_registry_remove_existing() -> None:
    """Verify remove returns True and removes config."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    result = registry.remove(TestConfig)

    expect_true(result is True)
    expect_false(registry.has(TestConfig))


def test_config_registry_remove_missing() -> None:
    """Verify remove returns False for missing config."""
    registry = ConfigRegistry()

    result = registry.remove(TestConfig)

    expect_false(result)


def test_config_registry_clear() -> None:
    """Verify clear removes all configs."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    registry.clear()

    expect_equal(len(registry), 0)
    expect_false(registry.has(TestConfig))
    expect_false(registry.has(AnotherConfig))


def test_config_registry_register_type_mismatch() -> None:
    """Verify register raises ConfigTypeError for type mismatch."""
    registry = ConfigRegistry()
    wrong_config = AnotherConfig(name="test")

    with pytest.raises(ConfigTypeError) as exc_info:
        registry.register(TestConfig, cast("TestConfig", wrong_config))

    expect_true(exc_info.value.config_type is TestConfig)
    expect_true(exc_info.value.actual_type is AnotherConfig)


def test_config_registry_multiple_types() -> None:
    """Verify registry can hold multiple config types."""
    registry = ConfigRegistry()
    test_config = TestConfig(host="localhost", port=5432)
    another_config = AnotherConfig(name="test")

    registry.register(TestConfig, test_config)
    registry.register(AnotherConfig, another_config)

    expect_true(registry.get(TestConfig) is test_config)
    expect_true(registry.get(AnotherConfig) is another_config)
    expect_equal(len(registry), 2)


def test_config_registry_register_validator() -> None:
    """Verify validator is called on register."""
    registry = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry.register_validator(TestConfig, validator)
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    expect_equal(len(validated), 1)
    expect_true(validated[0] is config)


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

    expect_true(exc_info.value.config_type is TestConfig)
    expect_in("Port must be positive", str(exc_info.value))


def test_config_registry_validate_all() -> None:
    """Verify validate_all runs all validators."""
    registry = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry.register_validator(TestConfig, validator)
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    validated.clear()
    registry.validate_all()

    expect_equal(len(validated), 1)


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
    registry.register(TestConfig, config)

    with pytest.raises(ConfigValidationError):
        registry.validate_all()


def test_config_registry_len() -> None:
    """Verify len returns correct count."""
    registry = ConfigRegistry()

    expect_equal(len(registry), 0)

    registry.register(TestConfig, TestConfig(host="localhost", port=5432))

    expect_equal(len(registry), 1)


def test_config_registry_iter() -> None:
    """Verify iteration over types."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    types = list(registry)

    expect_equal(len(types), 2)
    expect_true(TestConfig in types)
    expect_true(AnotherConfig in types)


def test_config_registry_contains() -> None:
    """Verify in operator works."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))

    expect_true(TestConfig in registry)
    expect_false(AnotherConfig in registry)


def test_config_registry_types() -> None:
    """Verify types() returns frozenset."""
    registry = ConfigRegistry()
    registry.register(TestConfig, TestConfig(host="localhost", port=5432))
    registry.register(AnotherConfig, AnotherConfig(name="test"))

    types = registry.types()

    expect_true(isinstance(types, frozenset))
    expect_true(TestConfig in types)
    expect_true(AnotherConfig in types)


def test_config_registry_as_mapping() -> None:
    """Verify as_mapping returns correct mapping."""
    registry = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry.register(TestConfig, config)

    mapping = registry.as_mapping()

    expect_true(TestConfig in mapping)
    expect_true(mapping[TestConfig] is config)


def test_config_registry_copy() -> None:
    """Verify copy creates independent registry."""
    registry1 = ConfigRegistry()
    config = TestConfig(host="localhost", port=5432)
    registry1.register(TestConfig, config)

    registry2 = registry1.copy()

    expect_true(registry2.has(TestConfig))
    expect_true(registry2.get(TestConfig) is config)

    registry1.remove(TestConfig)
    expect_false(registry1.has(TestConfig))
    expect_true(registry2.has(TestConfig))


def test_config_registry_copy_includes_validators() -> None:
    """Verify copy includes validators."""
    registry1 = ConfigRegistry()
    validated = []

    def validator(config: TestConfig) -> None:
        validated.append(config)

    registry1.register_validator(TestConfig, validator)

    registry2 = registry1.copy()

    config = TestConfig(host="localhost", port=5432)
    registry2.register(TestConfig, config)

    expect_equal(len(validated), 1)


def test_config_not_found_error() -> None:
    """Verify ConfigNotFoundError attributes."""
    error = ConfigNotFoundError(TestConfig)

    expect_true(error.config_type is TestConfig)
    expect_in("TestConfig", str(error))


def test_config_type_error() -> None:
    """Verify ConfigTypeError attributes."""
    error = ConfigTypeError(TestConfig, AnotherConfig)

    expect_true(error.config_type is TestConfig)
    expect_true(error.actual_type is AnotherConfig)
    expect_in("TestConfig", str(error))
    expect_in("AnotherConfig", str(error))


def test_config_validation_error() -> None:
    """Verify ConfigValidationError attributes."""
    error = ConfigValidationError(TestConfig, "Invalid value")

    expect_true(error.config_type is TestConfig)
    expect_equal(error.reason, "Invalid value")
    expect_in("TestConfig", str(error))
    expect_in("Invalid value", str(error))
