"""Unified operation infrastructure for CodeIntel.

This package provides entry-point agnostic operations that can be invoked
from CLI, HTTP, plugins, MCP, or background jobs through adapters.

Public API
----------
Operation : Protocol
    Base protocol for all operations.
operation : decorator
    Register an operation class with the registry.
Result : dataclass
    Typed result container for operation outcomes.
result_type : decorator
    Add automatic serialization to result dataclasses.
OpContext : dataclass
    Execution context providing access to resources.
OpContextBuilder : class
    Fluent builder for constructing OpContext.
OperationPipeline : dataclass
    Middleware pipeline for operation execution.
OperationRegistry : dataclass
    Registry for operation discovery and lookup.
Capability : class
    Standard capability constants.

Example
-------
>>> from codeintel.operations import Operation, operation, Result, OpContext
>>> from dataclasses import dataclass
>>>
>>> @dataclass(frozen=True)
... class GreetParams:
...     name: str = "World"
>>>
>>> @operation("hello.greet")
... class Greet(Operation[GreetParams, str]):
...     def execute(self, params: GreetParams, ctx: OpContext) -> Result[str]:
...         return Result.ok(f"Hello, {params.name}!")
"""

from __future__ import annotations

from codeintel.operations.base import Capability, Operation, OperationSpec, operation
from codeintel.operations.registry import (
    OperationRegistry,
    create_isolated_registry,
    get_default_registry,
)
from codeintel.operations.result import Result, Serializable, result_type

__all__ = [
    "Capability",
    "Operation",
    "OperationRegistry",
    "OperationSpec",
    "Result",
    "Serializable",
    "create_isolated_registry",
    "get_default_registry",
    "operation",
    "result_type",
]
