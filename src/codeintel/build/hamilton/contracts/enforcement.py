"""Contract enforcement for strict mode.

When strict_contracts is enabled, all writes are validated against
the target's declared OutputContract.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from codeintel.build.errors import ContractViolationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.build.targets import OutputTarget


class ContractEnforcer:
    """Enforces write operations against target contracts.

    When active, intercepts table/artifact writes and validates
    they are within the current target's declared contract.

    This is a class-level singleton that tracks the current execution
    context. Use the `for_target()` context manager to set the active
    target for a block of execution.
    """

    _current_target: ContextVar[OutputTarget | None] = ContextVar(
        "codeintel_contract_enforcer_target", default=None
    )
    _strict: ContextVar[bool] = ContextVar("codeintel_contract_enforcer_strict", default=False)

    @classmethod
    def activate(cls, target: OutputTarget, *, strict: bool) -> None:
        """Activate enforcement for a specific target.

        Parameters
        ----------
        target
            Target whose contract should be enforced for subsequent writes.
        strict
            When True, enforce the target contract and raise on violations.
        """
        cls._current_target.set(target)
        cls._strict.set(strict)

    @classmethod
    def deactivate(cls) -> None:
        """Deactivate enforcement for subsequent writes."""
        cls._current_target.set(None)
        cls._strict.set(False)

    @classmethod
    @contextmanager
    def for_target(
        cls,
        target: OutputTarget,
        *,
        strict: bool,
    ) -> Iterator[None]:
        """Context manager for contract enforcement during target execution.

        Parameters
        ----------
        target
            Target being executed.
        strict
            If True, raise on contract violations.

        Examples
        --------
        >>> from codeintel.build.registry import get_target_graph
        >>> graph = get_target_graph()
        >>> target = graph.get("risk_factors")
        >>> with ContractEnforcer.for_target(target, strict=True):
        ...     # All writes in this block are validated
        ...     pass
        """
        token_target = cls._current_target.set(target)
        token_strict = cls._strict.set(strict)

        try:
            yield
        finally:
            cls._current_target.reset(token_target)
            cls._strict.reset(token_strict)

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

        allowed_tables = set(current_target.contract.table_keys)
        if table_key not in allowed_tables:
            raise ContractViolationError(
                target=current_target.name,
                table_key=table_key,
                allowed_tables=allowed_tables,
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

        artifact_names = {artifact.name for artifact in current_target.contract.artifacts}
        if artifact_name not in artifact_names:
            raise ContractViolationError(
                target=current_target.name,
                artifact_name=artifact_name,
            )


__all__ = [
    "ContractEnforcer",
]
