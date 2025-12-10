"""Base class for CLI commands using the new pattern.

The Command[T] base class provides type-safe command execution with explicit
dependency injection through the Deps container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from codeintel.cli.core.results import CliResult
    from codeintel.cli.deps import Deps


class Command[T](ABC):
    """Base class for type-safe CLI commands.

    Commands are dataclasses that define their parameters as fields and
    implement execute() to perform the operation. The type parameter T
    specifies the exact return type, enabling end-to-end type safety.

    Class Attributes
    ----------------
    __operation_id__
        Unique identifier for this operation (e.g., "jobs.list").
    __require_storage__
        If True, storage access is required and provided via deps.storage.
    __require_serving__
        If True, serving access is required and provided via deps.serving.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from codeintel.cli.core.command import Command
    >>> from codeintel.cli.core.results import CliResult
    >>> from codeintel.cli.core.result_types import ListResult
    >>> from codeintel.cli.deps import Deps
    >>>
    >>> @dataclass(frozen=True)
    ... class ListItems(Command[ListResult[str]]):
    ...     __operation_id__ = "items.list"
    ...     __require_storage__ = True
    ...
    ...     limit: int = 10
    ...
    ...     def execute(self, deps: Deps) -> CliResult[ListResult[str]]:
    ...         items = ["item1", "item2"][: self.limit]
    ...         return CliResult.ok(ListResult.from_items(items))
    """

    __operation_id__: ClassVar[str]
    __require_storage__: ClassVar[bool] = False
    __require_serving__: ClassVar[bool] = False

    @abstractmethod
    def execute(self, deps: Deps) -> CliResult[T]:
        """Execute the command with provided dependencies.

        Parameters
        ----------
        deps
            Dependency container with config, logger, jobs, and optional
            storage/serving access based on class requirements.

        Returns
        -------
        CliResult[T]
            Result containing typed data on success or ProblemDetail on failure.
        """
        ...


__all__ = [
    "Command",
]
