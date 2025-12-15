"""Schema manifest types for build-time schema products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


@dataclass(frozen=True)
class SchemaManifest:
    """Stable manifest of table schemas compiled for a build selection."""

    version: str
    tables: tuple[TableSchema, ...]

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable manifest representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        return {
            "version": self.version,
            "tables": [table.to_json_obj() for table in self.tables],
        }


__all__ = ["SchemaManifest"]
