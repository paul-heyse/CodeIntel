"""Contract enforcement for strict mode.

When strict_contracts is enabled, all writes are validated against
the target's DAG-declared outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from codeintel.build.errors import ContractViolationError

if TYPE_CHECKING:
    from collections.abc import Iterator


class ContractEnforcer:
    """Enforces write operations against target contracts.

    When active, intercepts table/artifact writes and validates
    they are within the current target's declared contract.

    This is a class-level singleton that tracks the current execution
    context. Use the `for_target()` context manager to set the active
    target for a block of execution.
    """

    _current_target: ContextVar[str | None] = ContextVar(
        "codeintel_contract_enforcer_target", default=None
    )
    _strict: ContextVar[bool] = ContextVar("codeintel_contract_enforcer_strict", default=False)
    _allowed_tables: ContextVar[frozenset[str]] = ContextVar(
        "codeintel_contract_enforcer_allowed_tables", default=frozenset()
    )
    _allowed_artifacts: ContextVar[frozenset[str]] = ContextVar(
        "codeintel_contract_enforcer_allowed_artifacts", default=frozenset()
    )

    @classmethod
    def activate(
        cls,
        target_name: str,
        *,
        strict: bool,
        allowed_tables: frozenset[str],
        allowed_artifacts: frozenset[str],
    ) -> None:
        """Activate enforcement for a specific target.

        Parameters
        ----------
        target_name
            Target name whose declared outputs should be enforced.
        strict
            When True, enforce the target contract and raise on violations.
        allowed_tables
            Table keys allowed for the target.
        allowed_artifacts
            Artifact names allowed for the target.
        """
        cls._current_target.set(target_name)
        cls._strict.set(strict)
        cls._allowed_tables.set(allowed_tables)
        cls._allowed_artifacts.set(allowed_artifacts)

    @classmethod
    def deactivate(cls) -> None:
        """Deactivate enforcement for subsequent writes."""
        cls._current_target.set(None)
        cls._strict.set(False)
        cls._allowed_tables.set(frozenset())
        cls._allowed_artifacts.set(frozenset())

    @classmethod
    @contextmanager
    def for_target(
        cls,
        target_name: str,
        *,
        strict: bool,
        allowed_tables: frozenset[str],
        allowed_artifacts: frozenset[str],
    ) -> Iterator[None]:
        """Context manager for contract enforcement during target execution.

        Parameters
        ----------
        target_name
            Target name being executed.
        strict
            If True, raise on contract violations.
        allowed_tables
            Table keys allowed for the target.
        allowed_artifacts
            Artifact names allowed for the target.

        Examples
        --------
        >>> from codeintel.build.target_metadata import get_target_metadata_service
        >>> catalog = get_target_metadata_service().system.catalog
        >>> with ContractEnforcer.for_target(
        ...     "risk_factors",
        ...     strict=True,
        ...     allowed_tables=frozenset(),
        ...     allowed_artifacts=frozenset(),
        ... ):
        ...     # All writes in this block are validated
        ...     pass
        """
        token_target = cls._current_target.set(target_name)
        token_strict = cls._strict.set(strict)
        token_tables = cls._allowed_tables.set(allowed_tables)
        token_artifacts = cls._allowed_artifacts.set(allowed_artifacts)

        try:
            yield
        finally:
            cls._current_target.reset(token_target)
            cls._strict.reset(token_strict)
            cls._allowed_tables.reset(token_tables)
            cls._allowed_artifacts.reset(token_artifacts)

    @classmethod
    def validate_table_write(cls, table_key: str) -> None:
        """Validate that a table write is within contract.

        Parameters
        ----------
        table_key
            Table being written to.

        Raises
        ------
        ContractViolationError
            If strict mode and write is outside contract.

        Examples
        --------
        >>> ContractEnforcer.validate_table_write("analytics.function_metrics")
        >>> # Raises ContractViolationError if strict mode and not in contract
        """
        if not cls._strict.get():
            return

        current_target = cls._current_target.get()
        if current_target is None:
            return

        allowed_tables = cls._allowed_tables.get()
        if table_key not in allowed_tables:
            raise ContractViolationError(
                target=current_target,
                table_key=table_key,
                allowed_tables=set(allowed_tables),
            )

    @classmethod
    def validate_artifact_write(cls, artifact_name: str) -> None:
        """Validate that an artifact write is within contract.

        Parameters
        ----------
        artifact_name
            Artifact name being written to.

        Raises
        ------
        ContractViolationError
            If strict mode and write is outside contract.

        Examples
        --------
        >>> ContractEnforcer.validate_artifact_write("index.scip")
        >>> # Raises ContractViolationError if strict mode and not in contract
        """
        if not cls._strict.get():
            return

        current_target = cls._current_target.get()
        if current_target is None:
            return

        artifact_names = cls._allowed_artifacts.get()
        if artifact_name not in artifact_names:
            raise ContractViolationError(
                target=current_target,
                artifact_name=artifact_name,
            )


__all__ = [
    "ContractEnforcer",
]
