"""Type-safe parameter access for build targets.

This module provides the TargetParameters class for accessing
configuration parameters in a type-safe manner. Parameters are
loaded from the global build config file and can be overridden
per-target.

Example
-------
>>> from codeintel.build.parameters import TargetParameters
>>> params = TargetParameters(
...     {
...         "max_commits": 2000,
...         "scoring_weights": {"frequency": 0.4, "recency": 0.3},
...     }
... )
>>> max_commits = params.get_typed("max_commits", int, default=1000)
>>> weights = params.get_typed("scoring_weights", dict)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

__all__ = [
    "ParameterError",
    "TargetParameters",
]


T = TypeVar("T")


class ParameterError(ValueError):
    """Error accessing a parameter with wrong type or missing required value.

    Attributes
    ----------
    parameter
        The parameter name that caused the error.
    expected_type
        The expected type for the parameter.
    actual_type
        The actual type of the value (if type mismatch).
    """

    def __init__(
        self,
        parameter: str,
        expected_type: type[object],
        actual_type: type[object] | None = None,
        message: str | None = None,
    ) -> None:
        self.parameter = parameter
        self.expected_type = expected_type
        self.actual_type = actual_type
        if message:
            super().__init__(message)
        elif actual_type:
            super().__init__(
                f"Parameter '{parameter}' expected {expected_type.__name__}, "
                f"got {actual_type.__name__}"
            )
        else:
            super().__init__(
                f"Required parameter '{parameter}' not found (expected {expected_type.__name__})"
            )


@dataclass(frozen=True)
class TargetParameters(Mapping[str, object]):
    """Type-safe parameter container for target execution.

    Parameters are stored as a frozen dict and accessed via the get()
    method with type validation. This replaces the many step config
    classes that previously held tuning parameters.

    Parameters
    ----------
    values
        Dictionary of parameter name to value mappings.

    Examples
    --------
    >>> params = TargetParameters({"max_commits": 2000, "enabled": True})
    >>> params.get_typed("max_commits", int)
    2000
    >>> params.get_typed("missing", str, default="fallback")
    'fallback'
    """

    _values: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        """Return a raw parameter value.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        object
            Raw parameter value.
        """
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter keys.

        Returns
        -------
        Iterator[str]
            Iterator over parameter keys.
        """
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return len(self._values)

    def get_typed(self, key: str, type_: type[T], default: T | None = None) -> T:
        """Get a parameter with type validation.

        Parameters
        ----------
        key
            Parameter name to look up.
        type_
            Expected type for the parameter value.
        default
            Default value if parameter is not set.
            If None and parameter is missing, raises ParameterError.

        Returns
        -------
        T
            The parameter value cast to the expected type.

        Raises
        ------
        ParameterError
            If parameter is missing (and no default) or wrong type.

        Examples
        --------
        >>> params = TargetParameters({"count": 10})
        >>> params.get_typed("count", int)
        10
        >>> params.get_typed("missing", str, default="default")
        'default'
        """
        value = self._values.get(key)

        if value is None:
            if default is not None:
                return default
            raise ParameterError(key, type_)

        if not isinstance(value, type_):
            raise ParameterError(key, type_, type(value))

        return value

    def get_optional(self, key: str, type_: type[T]) -> T | None:
        """Get a parameter that may not exist.

        Unlike get_typed(), this returns None for missing parameters
        without requiring a default value.

        Parameters
        ----------
        key
            Parameter name to look up.
        type_
            Expected type for the parameter value.

        Returns
        -------
        T | None
            The parameter value if present and correctly typed, None otherwise.

        Raises
        ------
        ParameterError
            If parameter exists but has wrong type.
        """
        value = self._values.get(key)
        if value is None:
            return None
        if not isinstance(value, type_):
            raise ParameterError(key, type_, type(value))
        return value

    def has(self, key: str) -> bool:
        """Check if a parameter is set.

        Parameters
        ----------
        key
            Parameter name to check.

        Returns
        -------
        bool
            True if the parameter exists in the values dict.
        """
        return key in self._values

    def key_set(self) -> frozenset[str]:
        """Return all parameter names as a stable, hashable set.

        Returns
        -------
        frozenset[str]
            Set of all parameter names.
        """
        return frozenset(self._values)

    def as_dict(self) -> dict[str, object]:
        """Return a copy of parameters as a dict.

        Returns
        -------
        dict[str, object]
            Copy of the underlying mapping.
        """
        return dict(self._values)

    def merge(self, other: TargetParameters) -> TargetParameters:
        """Create new parameters with values from other merged in.

        Values from `other` override values from `self`.

        Parameters
        ----------
        other
            Parameters to merge in.

        Returns
        -------
        TargetParameters
            New instance with merged values.
        """
        merged = dict(self._values)
        merged.update(other._values)
        return TargetParameters(merged)

    @classmethod
    def empty(cls) -> TargetParameters:
        """Create empty parameters.

        Returns
        -------
        TargetParameters
            Instance with no parameters set.
        """
        return cls({})


EMPTY_PARAMETERS = TargetParameters({})
