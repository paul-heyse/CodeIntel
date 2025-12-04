"""Default implementations of analytics support providers."""

from __future__ import annotations

from codeintel.analytics.core.contracts import ContractValidator, OutputContractSpec
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.validation import safe_count_rows


class DefaultAnalyticsSupportProvider:
    """Use storage helpers to implement analytics runtime conveniences."""

    def __init__(self) -> None:
        self._row_count_fn = safe_count_rows
        self._validator_factory = ContractValidator

    def compute_row_counts(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        tables: tuple[str, ...],
    ) -> dict[str, int]:
        """Return row counts for the provided tables scoped to the snapshot.

        Parameters
        ----------
        gateway
            Storage gateway for executing queries.
        snapshot
            Snapshot used to scope counts by repo and commit.
        tables
            Fully qualified table names to count.

        Returns
        -------
        dict[str, int]
            Mapping of table names to row counts.
        """
        if not tables:
            return {}
        counts = self._row_count_fn(
            gateway.con,
            repo=snapshot.repo,
            commit=snapshot.commit,
            tables=tables,
        )
        return counts or {}

    def validate_contracts(
        self,
        gateway: StorageGateway,
        contracts: tuple[object, ...],
        snapshot: SnapshotRef,
    ) -> tuple[bool, tuple[str, ...]]:
        """Validate output contracts for the given snapshot.

        Parameters
        ----------
        gateway
            Storage gateway for querying contract results.
        contracts
            Contract specifications to validate.
        snapshot
            Snapshot context for scoping validation queries.

        Returns
        -------
        tuple[bool, tuple[str, ...]]
            Tuple of overall validity flag and validation error messages.
        """
        if not contracts:
            return True, ()

        valid_contracts = [c for c in contracts if isinstance(c, OutputContractSpec)]
        if not valid_contracts:
            return True, ()

        validator = self._validator_factory(gateway)
        result = validator.validate(valid_contracts, snapshot)
        errors = tuple(v.message for v in result.violations)
        return result.valid, errors


__all__ = ["DefaultAnalyticsSupportProvider"]
