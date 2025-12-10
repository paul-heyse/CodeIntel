"""Core operation protocol and @operation decorator.

This module defines the fundamental Operation protocol that all operations
implement, plus the @operation decorator for registration.
"""

from __future__ import annotations

import typing
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

TYPE_PARAM_COUNT = 2

if TYPE_CHECKING:
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


class Capability:
    """Standard operation capabilities.

    Capabilities define what resources an operation requires access to.
    Used for capability-based security and plugin sandboxing.
    """

    STORAGE_READ = "storage:read"
    STORAGE_WRITE = "storage:write"
    RUNTIME = "runtime"
    SERVING = "serving"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    NETWORK = "network"
    FILESYSTEM_READ = "filesystem:read"
    FILESYSTEM_WRITE = "filesystem:write"


TParams = typing.TypeVar("TParams", contravariant=True)
TResult = typing.TypeVar("TResult", covariant=True)


@runtime_checkable
class Operation[TParams, TResult](Protocol):
    """Protocol for operations.

    Operations are the atomic units of business logic in CodeIntel.
    They receive typed parameters and return typed results.

    Type Parameters
    ---------------
    TParams
        Parameter dataclass type (frozen, validated).
    TResult
        Result data type (auto-serializable via @result_type).

    Attributes
    ----------
    __operation_id__ : ClassVar[str]
        Unique operation identifier (e.g., "jobs.list").
    __params_type__ : ClassVar[type[TParams]]
        The parameter dataclass type.
    __result_type__ : ClassVar[type[TResult]]
        The result dataclass type.
    __capabilities__ : ClassVar[frozenset[str]]
        Required capabilities for this operation.

    Example
    -------
    >>> from dataclasses import dataclass
    >>> from codeintel.operations import Operation, operation, Result, OpContext
    >>> from codeintel.operations.result_types import ListResult
    >>>
    >>> @dataclass(frozen=True)
    ... class ListJobsParams:
    ...     limit: int = 20
    >>>
    >>> @operation("jobs.list", capabilities={Capability.JOBS_READ})
    ... class ListJobs(Operation[ListJobsParams, ListResult[str]]):
    ...     def execute(self, params: ListJobsParams, ctx: OpContext) -> Result[ListResult[str]]:
    ...         # Implementation here
    ...         ...
    """

    __operation_id__: ClassVar[str]
    __params_type__: ClassVar[type[object]]
    __result_type__: ClassVar[type[object]]
    __capabilities__: ClassVar[frozenset[str]]

    @abstractmethod
    def execute(self, params: TParams, ctx: OpContext) -> Result[TResult]:
        """Execute the operation.

        Parameters
        ----------
        params
            Validated parameter instance.
        ctx
            Operation context with resources.

        Returns
        -------
        Result[TResult]
            Success with result data or failure with error.
        """
        ...


@dataclass(frozen=True)
class OperationSpec:
    """Specification for a registered operation.

    Created by @operation decorator and stored in OperationRegistry.

    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list").
    name
        Class name of the operation.
    description
        Short description from docstring.
    params_type
        The parameter dataclass type.
    result_type
        The result dataclass type.
    operation_class
        The operation class itself.
    group
        Operation group derived from operation_id prefix.
    capabilities
        Required capabilities.
    require_storage
        Whether storage access is required.
    require_runtime
        Whether runtime resolution is required.
    require_serving
        Whether serving access is required.
    hidden
        Whether to hide from help/discovery.
    tags
        Optional tags for filtering.
    """

    operation_id: str
    name: str
    description: str
    params_type: type[object]
    result_type: type[object]
    operation_class: type[Operation[typing.Any, typing.Any]]
    group: str
    capabilities: frozenset[str] = frozenset()
    require_storage: bool = False
    require_runtime: bool = False
    require_serving: bool = False
    hidden: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize for introspection.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the spec.
        """
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "capabilities": sorted(self.capabilities),
            "require_storage": self.require_storage,
            "require_runtime": self.require_runtime,
            "require_serving": self.require_serving,
            "tags": list(self.tags),
            "hidden": self.hidden,
        }


def operation[TParams, TResult](
    operation_id: str,
    *,
    capabilities: frozenset[str] | set[str] = frozenset(),
    hidden: bool = False,
    tags: tuple[str, ...] = (),
) -> Callable[[type[Operation[TParams, TResult]]], type[Operation[TParams, TResult]]]:
    """Register an operation class.

    The decorator:
    1. Extracts params and result types from class signature
    2. Sets class-level metadata
    3. Registers with OperationRegistry
    4. Validates the class structure

    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list", "datasets.describe").
    capabilities
        Required capabilities (e.g., {"storage:read"}).
    hidden
        If True, hide from help/discovery.
    tags
        Optional tags for filtering.

    Returns
    -------
    Callable[[type[Operation[P, R]]], type[Operation[P, R]]]
        Class decorator.

    Example
    -------
    >>> @operation("jobs.list", capabilities={Capability.JOBS_READ})
    ... class ListJobs(Operation[ListJobsParams, ListResult[JobInfo]]):
    ...     def execute(self, params, ctx): ...
    """

    def decorator(cls: type[Operation[TParams, TResult]]) -> type[Operation[TParams, TResult]]:
        # Extract type parameters from class
        params_type, result_type = _extract_type_params(cls)

        # Validate class structure
        if not hasattr(cls, "execute"):
            msg = f"{cls.__name__} must define execute()"
            raise TypeError(msg)

        # Normalize capabilities to frozenset
        caps = frozenset(capabilities)

        # Set class-level metadata
        cls.__operation_id__ = operation_id
        cls.__params_type__ = params_type
        cls.__result_type__ = result_type
        cls.__capabilities__ = caps

        # Derive resource requirements from capabilities
        require_storage = Capability.STORAGE_READ in caps or Capability.STORAGE_WRITE in caps
        require_runtime = Capability.RUNTIME in caps
        require_serving = Capability.SERVING in caps

        # Extract description from docstring
        description = cls.__doc__ or f"Execute {operation_id}"
        description = description.strip().split("\n", maxsplit=1)[0].strip()

        # Extract group from operation_id
        group = operation_id.split(".", maxsplit=1)[0]

        # Create and register spec
        spec = OperationSpec(
            operation_id=operation_id,
            name=cls.__name__,
            description=description,
            params_type=params_type,
            result_type=result_type,
            operation_class=cls,
            group=group,
            capabilities=caps,
            require_storage=require_storage,
            require_runtime=require_runtime,
            require_serving=require_serving,
            hidden=hidden,
            tags=tags,
        )

        # Import here to avoid circular imports (intentional deferred import)
        from codeintel.operations.registry import get_default_registry  # noqa: PLC0415

        get_default_registry().register(spec)

        return cls

    return decorator


def _extract_type_params(cls: type[object]) -> tuple[type[object], type[object]]:
    """Extract TParams and TResult from Operation[TParams, TResult].

    Parameters
    ----------
    cls
        The operation class to extract types from.

    Returns
    -------
    tuple[type[object], type[object]]
        Tuple of (params_type, result_type).

    Raises
    ------
    TypeError
        If type parameters cannot be extracted.
    """
    for base in getattr(cls, "__orig_bases__", ()):
        origin = typing.get_origin(base)
        # Check if this base is Operation or a subclass/protocol
        if origin is not None:
            origin_name = getattr(origin, "__name__", "")
            if origin_name == "Operation":
                args = typing.get_args(base)
                if len(args) == TYPE_PARAM_COUNT:
                    return args[0], args[1]

    msg = (
        f"{cls.__name__} must explicitly specify type parameters: Operation[ParamsType, ResultType]"
    )
    raise TypeError(msg)


__all__ = [
    "Capability",
    "Operation",
    "OperationSpec",
    "operation",
]
