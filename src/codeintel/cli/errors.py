"""CLI-specific error types and exit code semantics."""

from __future__ import annotations

from dataclasses import dataclass

CLI_EXIT_SUCCESS = 0
CLI_EXIT_VALIDATION = 1
CLI_EXIT_USAGE = 2


@dataclass(frozen=True)
class DocsValidationError(Exception):
    """Raised when docs export validation fails."""

    message: str

    def __post_init__(self) -> None:
        """Populate base Exception args for default string rendering."""
        object.__setattr__(self, "args", (self.message,))


__all__ = [
    "CLI_EXIT_SUCCESS",
    "CLI_EXIT_USAGE",
    "CLI_EXIT_VALIDATION",
    "DocsValidationError",
]
