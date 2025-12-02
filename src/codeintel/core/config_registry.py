"""Type-safe configuration registry with runtime validation.

This module provides a generic configuration registry that maintains
type safety at runtime while using explicit cast() for type narrowing.
The registry enforces that values are instances of their key types,
enabling safe generic retrieval without type: ignore comments.

Architecture
------------
The ConfigRegistry uses a dict[type[object], object] internally but
maintains a stronger invariant: for every (key, value) pair,
isinstance(value, key) is True. This is enforced by register() and
verified by runtime isinstance checks.

The get() and get_optional() methods use explicit cast() for type
narrowing. These casts are documented as safe because:
1. register() enforces the isinstance invariant
2. All mutation methods maintain this invariant
3. The registry is the single point of access

Example
-------
>>> from dataclasses import dataclass
>>> @dataclass
... class DatabaseConfig:
...     host: str
...     port: int
>>> registry = ConfigRegistry()
>>> db_config = DatabaseConfig(host="localhost", port=5432)
>>> registry.register(DatabaseConfig, db_config)
>>> retrieved = registry.get(DatabaseConfig)
>>> assert retrieved.host == "localhost"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

T = TypeVar("T")


class ConfigNotFoundError(KeyError):
    """Requested configuration type is not registered.

    Attributes
    ----------
    config_type
        The type that was not found in the registry.
    """

    def __init__(self, config_type: type[object]) -> None:
        """Initialize with the missing config type.

        Parameters
        ----------
        config_type
            The type that was not found.
        """
        self.config_type = config_type
        super().__init__(f"Configuration not found: {config_type.__name__}")


class ConfigValidationError(ValueError):
    """Configuration failed validation.

    Attributes
    ----------
    config_type
        The type that failed validation.
    reason
        Description of the validation failure.
    """

    def __init__(self, config_type: type[object], reason: str) -> None:
        """Initialize with validation failure details.

        Parameters
        ----------
        config_type
            The type that failed validation.
        reason
            Description of the validation failure.
        """
        self.config_type = config_type
        self.reason = reason
        super().__init__(f"Config {config_type.__name__} invalid: {reason}")


class ConfigTypeError(TypeError):
    """Configuration value is not an instance of the declared type.

    Attributes
    ----------
    config_type
        The expected type.
    actual_type
        The actual type of the value.
    """

    def __init__(self, config_type: type[object], actual_type: type[object]) -> None:
        """Initialize with type mismatch details.

        Parameters
        ----------
        config_type
            The expected type.
        actual_type
            The actual type of the value.
        """
        self.config_type = config_type
        self.actual_type = actual_type
        super().__init__(
            f"Config must be instance of {config_type.__name__}, got {actual_type.__name__}"
        )


@dataclass
class ConfigRegistry:
    """Type-safe registry for configuration objects.

    Maintain the invariant that values are instances of their key types.
    Use runtime isinstance checks for safety and explicit cast() for
    type narrowing (compliant with project type guidelines).

    The registry provides:
    - Type-safe get/get_optional with generic return types
    - Runtime isinstance validation on register
    - Optional validation hooks per config type
    - Introspection of registered types

    Attributes
    ----------
    _store
        Internal mapping of config types to instances.
    _validators
        Optional validation hooks per config type.

    Example
    -------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class AppConfig:
    ...     debug: bool
    >>> registry = ConfigRegistry()
    >>> registry.register(AppConfig, AppConfig(debug=True))
    >>> config = registry.get(AppConfig)
    >>> assert config.debug is True
    """

    _store: dict[type[object], object] = field(default_factory=dict, repr=False)
    _validators: dict[type[object], Callable[[object], None]] = field(
        default_factory=dict, repr=False
    )

    def register(self, config_type: type[T], config: T) -> None:
        """Register a configuration instance with runtime type checking.

        Enforce the invariant that config is an instance of config_type.
        Run any registered validator for this type.

        Parameters
        ----------
        config_type
            The type key for registration.
        config
            The configuration instance (must be instance of config_type).

        Raises
        ------
        ConfigTypeError
            If config is not an instance of config_type.
        ConfigValidationError
            If the config has a validator that fails.
        """
        # Runtime safety: verify the invariant
        if not isinstance(config, config_type):
            raise ConfigTypeError(config_type, type(config))

        # Run validator if registered
        if config_type in self._validators:
            try:
                self._validators[config_type](config)
            except Exception as exc:
                raise ConfigValidationError(config_type, str(exc)) from exc

        self._store[config_type] = config

    def get(self, config_type: type[T]) -> T:
        """Retrieve a required configuration.

        Type Safety Note
        ----------------
        Use explicit cast() for type narrowing. The cast is safe because:
        1. register() enforces isinstance(config, config_type)
        2. The invariant is maintained by all mutation methods

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T
            The configuration instance.

        Raises
        ------
        ConfigNotFoundError
            If the config type is not registered.
        """
        if config_type not in self._store:
            raise ConfigNotFoundError(config_type)
        # Safe cast: register() guarantees isinstance(value, key)
        return cast("T", self._store[config_type])

    def get_optional(self, config_type: type[T]) -> T | None:
        """Retrieve an optional configuration.

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T | None
            The configuration instance or None if not registered.
        """
        value = self._store.get(config_type)
        if value is None:
            return None
        # Safe cast: register() guarantees isinstance(value, key)
        return cast("T", value)

    def has(self, config_type: type[object]) -> bool:
        """Check if a configuration type is registered.

        Parameters
        ----------
        config_type
            The type to check.

        Returns
        -------
        bool
            True if registered.
        """
        return config_type in self._store

    def remove(self, config_type: type[object]) -> bool:
        """Remove a configuration if present.

        Parameters
        ----------
        config_type
            The type to remove.

        Returns
        -------
        bool
            True if the config was present and removed.
        """
        if config_type in self._store:
            del self._store[config_type]
            return True
        return False

    def register_validator(
        self,
        config_type: type[T],
        validator: Callable[[T], None],
    ) -> None:
        """Register a validation hook for a config type.

        The validator is called on register() and validate_all().
        The validator accepts T but is stored internally as Callable[[object], None].
        This is safe because register() guarantees isinstance(config, config_type)
        before the validator is called, so the validator always receives the
        correct type at runtime.

        Parameters
        ----------
        config_type
            The type to validate.
        validator
            Function that raises on invalid config.
        """
        # Safe cast: register() enforces isinstance before calling validator
        self._validators[config_type] = cast("Callable[[object], None]", validator)

    def validate_all(self) -> None:
        """Re-validate all registered configurations.

        Raises
        ------
        ConfigValidationError
            If any config fails validation.
        """
        for config_type, config in self._store.items():
            if config_type in self._validators:
                try:
                    self._validators[config_type](config)
                except Exception as exc:
                    raise ConfigValidationError(config_type, str(exc)) from exc

    def types(self) -> frozenset[type[object]]:
        """Return all registered configuration types.

        Returns
        -------
        frozenset[type[object]]
            Registered types.
        """
        return frozenset(self._store.keys())

    def __len__(self) -> int:
        """Return number of registered configurations.

        Returns
        -------
        int
            Count of registered configs.
        """
        return len(self._store)

    def __iter__(self) -> Iterator[type[object]]:
        """Iterate over registered types.

        Returns
        -------
        Iterator[type[object]]
            Iterator over registered type keys.
        """
        return iter(self._store.keys())

    def __contains__(self, config_type: type[object]) -> bool:
        """Check if a type is registered.

        Parameters
        ----------
        config_type
            Type to check.

        Returns
        -------
        bool
            True if registered.
        """
        return config_type in self._store

    def as_mapping(self) -> Mapping[type[object], object]:
        """Return read-only view of the registry.

        Returns
        -------
        Mapping[type[object], object]
            Immutable view.
        """
        return dict(self._store)

    def clear(self) -> None:
        """Remove all registered configurations."""
        self._store.clear()

    def copy(self) -> ConfigRegistry:
        """Create a shallow copy of the registry.

        Returns
        -------
        ConfigRegistry
            New registry with same configs.
        """
        new_registry = ConfigRegistry()
        new_registry._store = dict(self._store)
        new_registry._validators = dict(self._validators)
        return new_registry


__all__ = [
    "ConfigNotFoundError",
    "ConfigRegistry",
    "ConfigTypeError",
    "ConfigValidationError",
]
