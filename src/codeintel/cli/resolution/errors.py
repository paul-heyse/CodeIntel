"""Resolution error types for the CLI resolution layer.

This module defines exceptions raised during runtime and gateway resolution.
These errors indicate that the CLI could not resolve the project configuration
from either a project file (config/codeintel.yaml) or explicit CLI parameters.
"""

from __future__ import annotations


class ResolutionError(Exception):
    """Raised when runtime or gateway resolution fails.

    This exception indicates that the CLI could not resolve the project
    configuration from either a project file (config/codeintel.yaml) or explicit
    CLI parameters.

    Parameters
    ----------
    message
        Human-readable error description.
    missing_params
        Optional list of missing required parameters.

    Examples
    --------
    >>> raise ResolutionError("No config/codeintel.yaml found")
    >>> raise ResolutionError(
    ...     "Missing required parameters",
    ...     missing_params=["repo", "commit"],
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        missing_params: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.missing_params = missing_params or []

    def __str__(self) -> str:
        """Format error message with missing parameters if present.

        Returns
        -------
        str
            Error message, optionally including missing parameters.
        """
        if self.missing_params:
            params = ", ".join(self.missing_params)
            return f"{self.message}. Missing: {params}"
        return self.message


__all__ = ["ResolutionError"]
