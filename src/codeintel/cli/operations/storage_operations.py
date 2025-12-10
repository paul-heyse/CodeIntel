"""Storage command operation specifications.

Define and register operations for the storage command group including
info and query commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.cli.core import CliResult
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection import register_operation


@dataclass
class StorageInfoResult:
    """Result for storage info command.

    Parameters
    ----------
    db_path
        Path to the database file.
    db_size_bytes
        Size of the database in bytes.
    table_count
        Number of tables.
    view_count
        Number of views.
    """

    db_path: str | None
    db_size_bytes: int
    table_count: int
    view_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "db_path": self.db_path,
            "db_size_bytes": self.db_size_bytes,
            "table_count": self.table_count,
            "view_count": self.view_count,
        }


def _storage_info_handler() -> CliResult[StorageInfoResult]:
    """Show storage info handler.

    Returns
    -------
    CliResult[StorageInfoResult]
        Storage information.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        StorageInfoResult(
            db_path=None,
            db_size_bytes=0,
            table_count=0,
            view_count=0,
        )
    )


# Storage Info Operation
STORAGE_INFO_SPEC: OperationSpec[StorageInfoResult] = register_operation(
    OperationSpec(
        operation_id="storage.info",
        handler=_storage_info_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Show storage information",
    )
)

__all__ = [
    "STORAGE_INFO_SPEC",
    "StorageInfoResult",
]
