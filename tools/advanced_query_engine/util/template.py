"""String template helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Missing:
    """Placeholder for missing template fields."""

    key: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return the placeholder string representation.

        Returns
        -------
        str
            String representation of the missing placeholder.
        """
        return f"<missing:{self.key}>"


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> Missing:
        return Missing(key)


def safe_format(template: str, values: Mapping[str, object]) -> str:
    """Format a template using Missing placeholders for absent keys.

    Parameters
    ----------
    template:
        Format string to render.
    values:
        Mapping of template variables to values.

    Returns
    -------
    str
        Formatted string with placeholders for missing keys.
    """
    return template.format_map(SafeFormatDict(values))


__all__ = ["Missing", "safe_format"]
