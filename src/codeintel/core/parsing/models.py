"""Parsed code models for analytics and tooling.

This module provides the canonical parsed function and module types
used across the codebase for analyzing code structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ast
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from codeintel.core.parsing.ast_index import AstSpanIndex
    from codeintel.core.parsing.source_span import SourceSpan


@dataclass(frozen=True)
class ParsedFunction:
    """Language-agnostic parsed function representation.

    This type captures the essential metadata about a parsed function
    for use in analytics, type checking, and documentation generation.

    Attributes
    ----------
    path
        Path to the source file containing the function.
    qualname
        Fully qualified name of the function.
    function_goid_h128
        Optional GOID hash identifying the function. Populated when
        the function is registered in the catalog.
    span
        Source span defining the function's location.
    ast
        Language-specific AST node (e.g., ast.FunctionDef for Python).
    docstring
        Extracted docstring text, if present.
    param_annotations
        Mapping of parameter names to their type annotations.
    return_annotation
        Return type annotation, if present.
    param_any_flags
        Mapping indicating whether each parameter is annotated as Any.
    return_is_any
        Whether the return type is annotated as Any.

    Examples
    --------
    >>> from pathlib import Path
    >>> span = SourceSpan(Path("src/main.py"), 10, 0, 25, 40)
    >>> func = ParsedFunction(
    ...     path=Path("src/main.py"),
    ...     qualname="module.MyClass.method",
    ...     function_goid_h128=12345,
    ...     span=span,
    ...     ast=None,
    ...     docstring="Do something useful.",
    ...     param_annotations={"x": "int"},
    ...     return_annotation="str",
    ...     param_any_flags={"x": False},
    ...     return_is_any=False,
    ... )
    >>> func.qualname
    'module.MyClass.method'
    """

    path: Path
    qualname: str
    function_goid_h128: int | None
    span: SourceSpan
    ast: Any  # Language-specific AST node
    docstring: str | None
    param_annotations: Mapping[str, Any]
    return_annotation: Any | None
    param_any_flags: Mapping[str, bool]
    return_is_any: bool

    @property
    def local_name(self) -> str:
        """Extract the local function name from the qualname.

        Returns
        -------
        str
            Unqualified function name.
        """
        return self.qualname.rsplit(".", maxsplit=1)[-1]

    @property
    def name(self) -> str:
        """Extract the function name (alias for local_name).

        Returns
        -------
        str
            Unqualified function name.
        """
        return self.local_name

    @property
    def start_line(self) -> int:
        """Get the starting line number from the span.

        Returns
        -------
        int
            Starting line number (1-based).
        """
        return self.span.start_line

    @property
    def end_line(self) -> int:
        """Get the ending line number from the span.

        Returns
        -------
        int
            Ending line number (1-based).
        """
        return self.span.end_line


@dataclass(frozen=True)
class ParsedModule:
    """Parsed module contents and extracted functions.

    This type represents a fully parsed Python module with its
    AST and all discovered functions.

    Attributes
    ----------
    path
        Path to the source file.
    source
        Complete source code as a string.
    lines
        Source code split into lines for line-based access.
    module_ast
        Root AST node of the module.
    span_index
        Index for efficient AST node lookup by span.
    functions
        Sequence of all functions parsed from the module.

    Examples
    --------
    >>> from pathlib import Path
    >>> # module = parse_module(Path("src/main.py"))
    >>> # len(module.functions)  # Number of functions in the module
    """

    path: Path
    source: str
    lines: Sequence[str]
    module_ast: ast.AST
    span_index: AstSpanIndex
    functions: Sequence[ParsedFunction]

    @property
    def line_count(self) -> int:
        """Return the total number of lines in the module.

        Returns
        -------
        int
            Number of source lines.
        """
        return len(self.lines)


__all__ = [
    "ParsedFunction",
    "ParsedModule",
]
