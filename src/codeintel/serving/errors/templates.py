"""Error catalog templates and safe rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.serving.errors.models import ErrorKind

if TYPE_CHECKING:
    from collections.abc import Mapping


def safe_format(template: str, params: Mapping[str, Any] | None) -> str:
    """Format a template with a safe fallback for missing keys."""
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
    kind: ErrorKind
    message: str
    hint: str | None = None
    retryable: bool = False
    http_status: int | None = None

    def render_message(self, params: Mapping[str, Any] | None = None) -> str:
        return safe_format(self.message, params)

    def render_hint(self, params: Mapping[str, Any] | None = None) -> str | None:
        if self.hint is None:
            return None
        return safe_format(self.hint, params)


__all__ = ["ErrorInfoTemplate", "safe_format"]
