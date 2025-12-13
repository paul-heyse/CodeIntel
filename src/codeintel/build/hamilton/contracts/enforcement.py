"""Contract enforcement for strict mode.

When strict_contracts is enabled, all writes are validated against
the target's declared OutputContract.
"""

from __future__ import annotations

from contextlib import contextmanager
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

    _current_target: OutputTarget | None = None
    _strict: bool = False

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
        old_target = cls._current_target
        old_strict = cls._strict

        cls._current_target = target
        cls._strict = strict

        try:
            yield
        finally:
            cls._current_target = old_target
            cls._strict = old_strict

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
        if not cls._strict or cls._current_target is None:
            return

        if not cls._current_target.contract:
            # No contract defined - allow write (backward compatibility)
            return

        allowed_tables = set(cls._current_target.contract.table_keys or [])
        if table_key not in allowed_tables:
            raise ContractViolationError(
                target=cls._current_target.name,
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
        if not cls._strict or cls._current_target is None:
            return

        if not cls._current_target.contract:
            # No contract defined - allow write (backward compatibility)
            return

        artifact_names = {
            artifact.name for artifact in (cls._current_target.contract.artifacts or [])
        }
        if artifact_name not in artifact_names:
            raise ContractViolationError(
                target=cls._current_target.name,
                artifact_name=artifact_name,
            )


__all__ = [
    "ContractEnforcer",
]
