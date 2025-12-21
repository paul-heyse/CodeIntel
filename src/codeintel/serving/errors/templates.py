"""Error catalog templates and safe rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.core.errors.taxonomy import ErrorCode
from codeintel.serving.errors.models import ErrorKind

if TYPE_CHECKING:
    from collections.abc import Mapping


def safe_format(template: str, params: Mapping[str, Any] | None) -> str:
    """Format a template with a safe fallback for missing keys.

    Returns
    -------
    str
        Rendered template with missing keys preserved.
    """
    if not params:
        return template

    class SafeDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict(params))


@dataclass(frozen=True, slots=True)
class ErrorInfoTemplate:
    """Template for an error code in the catalog."""

    code: str
    error_code: ErrorCode
    kind: ErrorKind
    message: str
    hint: str | None = None
    retryable: bool = False

    def render_message(self, params: Mapping[str, Any] | None = None) -> str:
        """Render the message template with parameter substitution.

        Returns
        -------
        str
            Rendered message with placeholders substituted.
        """
        return safe_format(self.message, params)

    def render_hint(self, params: Mapping[str, Any] | None = None) -> str | None:
        """Render the hint template with parameter substitution.

        Returns
        -------
        str | None
            Rendered hint text, or ``None`` when no hint is defined.
        """
        if self.hint is None:
            return None
        return safe_format(self.hint, params)


__all__ = ["ErrorInfoTemplate", "safe_format"]
