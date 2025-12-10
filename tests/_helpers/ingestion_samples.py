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

TYPED_SOURCE = dedent(
    """\
    def fn(x: int) -> int:
        return x
    """
).strip()

NESTED_CLASS_FUNCTION = dedent(
    """
    class Outer:
        def inner(self) -> None:
            pass
    """
).strip()

DECORATED_FUNCTION = dedent(
    """\
    @dec1
    @dec2("x")
    def foo() -> int:
        return 1
    """
).strip()

__all__ = [
    "DECORATED_FUNCTION",
    "MULTILINE_FUNCTION",
    "NESTED_CLASS_FUNCTION",
    "SIMPLE_MODULE",
    "SYNTAX_ERROR_CODE",
    "TYPED_SOURCE",
    "UNICODE_MODULE",
]
