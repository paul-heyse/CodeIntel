from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Missing:
    key: str

    def __str__(self) -> str:
        return f"<missing:{self.key}>"

    __repr__ = __str__


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> Any:
        return Missing(key)


def safe_format(template: str, values: Mapping[str, Any]) -> str:
    """Format using str.format_map with a Missing() placeholder for absent keys."""
    return template.format_map(SafeFormatDict(values))
