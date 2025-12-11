"""Parsing port interface for CST/AST operations.

This module defines the ParsingPort protocol that abstracts source code
parsing, allowing pure computation to work with parsed representations
without coupling to specific parsing libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

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


@runtime_checkable
class ParsingPort(Protocol):
    """Protocol for source code parsing operations.

    Implementations parse source code into structured representations
    that can be consumed by pure computation functions.
    """

    def parse_module(self, source: str) -> ParseResult:
        """Parse a module from source code.

        Parameters
        ----------
        source
            Source code string.

        Returns
        -------
        ParseResult
            Parsed module or error.
        """
        ...

    def parse_function(
        self,
        source: str,
        start_line: int,
        end_line: int,
    ) -> ParsedFunction | None:
        """Parse a single function from source within a line range.

        Parameters
        ----------
        source
            Source code string.
        start_line
            Starting line number.
        end_line
            Ending line number.

        Returns
        -------
        ParsedFunction | None
            Parsed function if found, None otherwise.
        """
        ...

    def extract_imports(self, source: str) -> Sequence[tuple[str, tuple[str, ...]]]:
        """Extract import statements from source.

        Parameters
        ----------
        source
            Source code string.

        Returns
        -------
        Sequence[tuple[str, tuple[str, ...]]]
            Sequence of (module, imported_names) tuples.
        """
        ...

    def extract_call_sites(
        self,
        module: ParsedModule,
        function_span: tuple[int, int],
    ) -> Sequence[tuple[str, int]]:
        """Extract call sites within a function.

        Parameters
        ----------
        module
            Parsed module containing the function.
        function_span
            (start_line, end_line) of the function.

        Returns
        -------
        Sequence[tuple[str, int]]
            Sequence of (callee_name, line_number) tuples.
        """
        ...


__all__ = [
    "ParseError",
    "ParseResult",
    "ParsedFunction",
    "ParsedModule",
    "ParsingPort",
]
