"""Parsing data types for CST/AST operations.

This module defines data classes for representing parsed source code,
allowing pure computation to work with parsed representations
without coupling to specific parsing libraries.

Note
----
This module provides graph-specific parsing types optimized for call graph
analysis with LibCST/AST. For SCIP-based parsing with richer type annotation
data, see ``codeintel.core.parsing.models``.

The ``ParsedFunction`` here is a lightweight representation focused on
call graph edges, while ``core.parsing.ParsedFunction`` includes full
type annotation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast

    import libcst as cst


@dataclass(frozen=True)
class ParsedFunction:
    """Parsed function representation.

    Attributes
    ----------
    name
        Function name.
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Ending line number.
    is_async
        Whether the function is async.
    decorator_names
        Names of decorators applied to the function.
    parameters
        Parameter names.
    """

    name: str
    qualname: str
    start_line: int
    end_line: int
    is_async: bool = False
    decorator_names: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()


@dataclass
class ParsedModule:
    """Parsed module representation.

    Attributes
    ----------
    source
        Original source code.
    functions
        Parsed function representations.
    imports
        Import statements as (module, names) tuples.
    cst_module
        Optional LibCST module for advanced operations.
    ast_module
        Optional AST module for advanced operations.
    """

    source: str
    functions: tuple[ParsedFunction, ...] = ()
    imports: tuple[tuple[str, tuple[str, ...]], ...] = ()
    cst_module: cst.Module | None = None
    ast_module: ast.Module | None = None
    _import_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def import_aliases(self) -> dict[str, str]:
        """Mapping of local names to imported module paths.

        Returns
        -------
        dict[str, str]
            Import alias mapping.
        """
        return self._import_aliases


@dataclass(frozen=True)
class ParseError:
    """Representation of a parsing error.

    Attributes
    ----------
    message
        Error message.
    line
        Line number where error occurred.
    column
        Column number where error occurred.
    """

    message: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class ParseResult:
    """Result of a parsing operation.

    Attributes
    ----------
    module
        Parsed module if successful.
    error
        Parse error if failed.
    success
        Whether parsing succeeded.
    """

    module: ParsedModule | None
    error: ParseError | None = None
    success: bool = True

    @classmethod
    def ok(cls, module: ParsedModule) -> ParseResult:
        """Create a successful parse result.

        Parameters
        ----------
        module
            Successfully parsed module.

        Returns
        -------
        ParseResult
            Successful result.
        """
        return cls(module=module, success=True)

    @classmethod
    def fail(cls, error: ParseError) -> ParseResult:
        """Create a failed parse result.

        Parameters
        ----------
        error
            Parse error that occurred.

        Returns
        -------
        ParseResult
            Failed result with error.
        """
        return cls(module=None, error=error, success=False)


__all__ = [
    "ParseError",
    "ParseResult",
    "ParsedFunction",
    "ParsedModule",
]
