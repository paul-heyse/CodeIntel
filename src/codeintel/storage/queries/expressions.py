"""DuckDB expression helpers for common filters."""

from __future__ import annotations

from collections.abc import Iterable

from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression, Expression


class ExpressionBuilder:
    """Builder for DuckDB expressions."""

    @staticmethod
    def col(name: str) -> Expression:
        """Return a column expression for a column name.

        Returns
        -------
        Expression
            Column expression.
        """
        return ColumnExpression(name)

    @staticmethod
    def lit(value: object) -> Expression:
        """Return a constant expression for a literal value.

        Returns
        -------
        Expression
            Constant expression.
        """
        return ConstantExpression(value)

    @staticmethod
    def eq(column: str, value: object) -> Expression:
        """Return an equality expression for a column and literal value.

        Returns
        -------
        Expression
            Equality expression.
        """
        return ExpressionBuilder.col(column) == ExpressionBuilder.lit(value)

    @staticmethod
    def and_all(expressions: Iterable[Expression]) -> Expression:
        """Combine expressions with AND, requiring at least one expression.

        Returns
        -------
        Expression
            Combined expression.

        Raises
        ------
        ValueError
            If no expressions are provided.
        """
        iterator = iter(expressions)
        first = next(iterator, None)
        if first is None:
            msg = "and_all requires at least one expression"
            raise ValueError(msg)
        combined = first
        for expr in iterator:
            combined &= expr
        return combined

    @staticmethod
    def snapshot_filter(*, repo: str, commit: str) -> Expression:
        """Return a repo/commit filter expression for snapshot-scoped tables.

        Returns
        -------
        Expression
            Snapshot filter expression.
        """
        return ExpressionBuilder.eq("repo", repo) & ExpressionBuilder.eq("commit", commit)


_BUILDER = ExpressionBuilder()


def col(name: str) -> Expression:
    """Return a column expression for a column name.

    Returns
    -------
    Expression
        Column expression.
    """
    return _BUILDER.col(name)


def lit(value: object) -> Expression:
    """Return a constant expression for a literal value.

    Returns
    -------
    Expression
        Constant expression.
    """
    return _BUILDER.lit(value)


def eq(column: str, value: object) -> Expression:
    """Return an equality expression for a column and literal value.

    Returns
    -------
    Expression
        Equality expression.
    """
    return _BUILDER.eq(column, value)


def and_all(expressions: Iterable[Expression]) -> Expression:
    """Combine expressions with AND, requiring at least one expression.

    Returns
    -------
    Expression
        Combined expression.
    """
    return _BUILDER.and_all(expressions)


def snapshot_filter(*, repo: str, commit: str) -> Expression:
    """Return a repo/commit filter expression for snapshot-scoped tables.

    Returns
    -------
    Expression
        Snapshot filter expression.
    """
    return _BUILDER.snapshot_filter(repo=repo, commit=commit)


__all__ = [
    "ExpressionBuilder",
    "and_all",
    "col",
    "eq",
    "lit",
    "snapshot_filter",
]
