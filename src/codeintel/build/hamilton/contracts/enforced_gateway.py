"""Build-time gateway wrappers for strict contract enforcement.

These wrappers are only used during Hamilton execution when strict contracts
are enabled. They intercept writes performed via the IbisGateway and validate
table_key targets against the active ContractEnforcer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as it
    import pandas as pd

    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.ibis_adapter import IbisGateway, OnConflict, WriteResult


class ContractEnforcingIbisGateway:
    """IbisGateway wrapper that enforces ContractEnforcer on writes."""

    def __init__(self, inner: IbisGateway) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the wrapped gateway.

        Returns
        -------
        object
            Attribute value from the wrapped gateway.
        """
        return getattr(self._inner, name)

    def write(
        self,
        table_key: str,
        data: it.Table | pd.DataFrame | Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None = None,
        on_conflict: OnConflict | None = None,
    ) -> WriteResult:
        """Validate table_key against the active contract, then delegate write.

        Returns
        -------
        WriteResult
            Result from the wrapped gateway write operation.
        """
        ContractEnforcer.validate_table_write(table_key)
        return self._inner.write(table_key, data, columns=columns, on_conflict=on_conflict)


class ContractEnforcingStorageGateway:
    """StorageGateway wrapper that replaces .ibis with an enforcing wrapper."""

    def __init__(self, inner: StorageGateway) -> None:
        self._inner = inner
        self.ibis = ContractEnforcingIbisGateway(inner.ibis)

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the wrapped gateway.

        Returns
        -------
        object
            Attribute value from the wrapped gateway.
        """
        return getattr(self._inner, name)


__all__ = [
    "ContractEnforcingIbisGateway",
    "ContractEnforcingStorageGateway",
]
