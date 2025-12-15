"""Base options implementation.

This module provides a base class for options that implements
the OptionsProtocol with default behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Self

from codeintel.core.options.protocol import ValidationResult


@dataclass(frozen=True)
class BaseOptions:
    """Base options class implementing OptionsProtocol.

    Inherit from this class to get default implementations of
    validate(), with_defaults(), and to_dict().

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class MyOptions(BaseOptions):
    ...     timeout_ms: int = 5000
    ...     retry_count: int = 3
    ...
    ...     def validate(self) -> ValidationResult:
    ...         if self.timeout_ms <= 0:
    ...             return ValidationResult.failure("timeout_ms must be positive")
    ...         return super().validate()
    """

    def validate(self) -> ValidationResult:
        """Validate options and return any issues.

        Default implementation always returns success. Override in
        subclasses to add specific validation logic.

        Returns
        -------
        ValidationResult
            Validation result (always ok=True by default).
        """
        # Base implementation validates that self is a dataclass (always true here)
        _ = fields(self)  # Ensure self is used
        return ValidationResult.success()

    def with_defaults(self, defaults: Self) -> Self:
        """Merge with default values, preferring self's non-None values.

        For each field, if self's value is None, use the default's value.
        Otherwise, keep self's value.

        Note: Subclasses should override this method with explicit field
        handling for proper type safety.

        Parameters
        ----------
        defaults
            Default options to merge from.

        Returns
        -------
        Self
            New options with defaults filled in.
        """
        changes: dict[str, object] = {}
        for dataclass_field in fields(self):
            self_value = getattr(self, dataclass_field.name)
            if self_value is not None:
                continue
            changes[dataclass_field.name] = getattr(defaults, dataclass_field.name)
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary for logging/debugging.

        Returns
        -------
        dict[str, object]
            Dictionary representation of options.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Create options from a dictionary.

        Parameters
        ----------
        data
            Dictionary with option values.

        Returns
        -------
        Self
            New options instance.
        """
        # Filter to only fields that exist on the class
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


__all__ = [
    "BaseOptions",
]
