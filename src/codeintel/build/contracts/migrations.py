"""Contract migration registry for dataset compatibility."""

from __future__ import annotations

from typing import Protocol

import pyarrow as pa


class ContractMigration(Protocol):
    """Callable that migrates table data between contract versions."""

    def __call__(
        self,
        table: pa.Table,
        *,
        table_key: str,
        from_version: str,
        to_version: str,
    ) -> pa.Table:
        """Return a migrated table for the target contract version."""
        ...


_CONTRACT_MIGRATIONS: dict[str, ContractMigration] = {}


def register_contract_migration(
    *,
    table_key: str,
    migration: ContractMigration,
) -> None:
    """Register a contract migration for a table key.

    Raises
    ------
    ValueError
        If table_key is empty.
    """
    if not table_key:
        msg = "table_key must be a non-empty string"
        raise ValueError(msg)
    _CONTRACT_MIGRATIONS[table_key] = migration


def get_contract_migration(*, table_key: str) -> ContractMigration | None:
    """Return the migration for a table key, if registered.

    Returns
    -------
    ContractMigration | None
        Migration when registered, otherwise None.
    """
    if not table_key:
        return None
    return _CONTRACT_MIGRATIONS.get(table_key)


def apply_contract_migration(
    table: pa.Table,
    *,
    table_key: str,
    from_version: str | None,
    to_version: str | None,
) -> pa.Table:
    """Apply a registered migration when contract versions differ.

    Returns
    -------
    pa.Table
        Migrated table or the original table when no migration applies.
    """
    if from_version is None or to_version is None or from_version == to_version:
        return table
    migration = get_contract_migration(table_key=table_key)
    if migration is None:
        return table
    return migration(
        table,
        table_key=table_key,
        from_version=from_version,
        to_version=to_version,
    )


__all__ = [
    "ContractMigration",
    "apply_contract_migration",
    "get_contract_migration",
    "register_contract_migration",
]
