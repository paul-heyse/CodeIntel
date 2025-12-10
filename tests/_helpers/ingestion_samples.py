"""Shared source snippets for ingestion parsing tests."""

from __future__ import annotations

from textwrap import dedent

SIMPLE_MODULE = dedent(
    '''
    """Module docstring."""

    def foo(x: int) -> int:
        """Function docstring."""
        return x + 1

    class Bar:
        """Class docstring."""

        def baz(self) -> None:
            """Method docstring."""
            return None
    '''
).strip()

MULTILINE_FUNCTION = dedent(
    '''
    def complex_function(
        arg1: int,
        arg2: str,
        arg3: float,
    ) -> dict[str, int]:
        """Multi-line function."""
        result = {}
        for i in range(arg1):
            result[f"{arg2}_{i}"] = int(arg3)
        return result
    '''
).strip()

SYNTAX_ERROR_CODE = dedent(
    """
    def broken(
        return "missing colon"
    """
).strip()

UNICODE_MODULE = dedent(
    '''
    """Unicode test: café, naïve, 日本語."""

    def grüß() -> str:
        """Return greeting."""
        return "Hallo"
    '''
).strip()

__all__ = [
    "MULTILINE_FUNCTION",
    "SIMPLE_MODULE",
    "SYNTAX_ERROR_CODE",
    "UNICODE_MODULE",
]
