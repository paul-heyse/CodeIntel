"""Expression vocabulary for Arrow datasets and plans."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import pyarrow.compute as pc

if TYPE_CHECKING:
    from pyarrow.compute import Expression
else:
    Expression = object

ExpressionInput = pc.Expression | str | tuple[str, ...]


def _as_expression(value: ExpressionInput) -> pc.Expression:
    if isinstance(value, pc.Expression):
        return value
    return pc.field(value)


class ExprVocab:
    """Provide convenience constructors for Arrow compute expressions."""

    @staticmethod
    def field(name: str | tuple[str, ...]) -> pc.Expression:
        """Return a field reference expression.

        Parameters
        ----------
        name
            Field name or nested field path.

        Returns
        -------
        pyarrow.compute.Expression
            Field reference expression.
        """
        return pc.field(name)

    @staticmethod
    def scalar(value: object) -> pc.Expression:
        """Return a literal scalar expression.

        Parameters
        ----------
        value
            Literal value to wrap as an expression.

        Returns
        -------
        pyarrow.compute.Expression
            Literal scalar expression.
        """
        return pc.scalar(value)

    @staticmethod
    def cast(expr: pc.Expression, dtype: str) -> pc.Expression:
        """Cast an expression to a target Arrow type.

        Parameters
        ----------
        expr
            Expression to cast.
        dtype
            Arrow type string (e.g. "int64").

        Returns
        -------
        pyarrow.compute.Expression
            Casted expression.
        """
        return expr.cast(dtype)

    @staticmethod
    def is_valid(value: ExpressionInput) -> pc.Expression:
        """Return an is-valid expression.

        Parameters
        ----------
        value
            Field name, nested path, or expression to test.

        Returns
        -------
        pyarrow.compute.Expression
            Expression evaluating to True for non-null values.
        """
        expr = _as_expression(value)
        return expr.is_valid()

    @staticmethod
    def is_null(value: ExpressionInput) -> pc.Expression:
        """Return an is-null expression.

        Parameters
        ----------
        value
            Field name, nested path, or expression to test.

        Returns
        -------
        pyarrow.compute.Expression
            Expression evaluating to True for null values.
        """
        expr = _as_expression(value)
        return expr.is_null()

    @staticmethod
    def in_(value: ExpressionInput, values: Sequence[object]) -> pc.Expression:
        """Return a membership expression.

        Parameters
        ----------
        value
            Field name, nested path, or expression to test.
        values
            Sequence of membership values.

        Returns
        -------
        pyarrow.compute.Expression
            Expression evaluating membership in the provided values.
        """
        expr = _as_expression(value)
        return expr.isin(list(values))

    @staticmethod
    def and_(*exprs: pc.Expression) -> pc.Expression:
        """Combine expressions using AND.

        Parameters
        ----------
        *exprs
            Expressions to combine.

        Returns
        -------
        pyarrow.compute.Expression
            Combined expression.

        Raises
        ------
        ValueError
            If no expressions are provided.
        """
        if not exprs:
            msg = "and_ requires at least one expression"
            raise ValueError(msg)
        combined = exprs[0]
        for expr in exprs[1:]:
            combined &= expr
        return combined

    @staticmethod
    def or_(*exprs: pc.Expression) -> pc.Expression:
        """Combine expressions using OR.

        Parameters
        ----------
        *exprs
            Expressions to combine.

        Returns
        -------
        pyarrow.compute.Expression
            Combined expression.

        Raises
        ------
        ValueError
            If no expressions are provided.
        """
        if not exprs:
            msg = "or_ requires at least one expression"
            raise ValueError(msg)
        combined = exprs[0]
        for expr in exprs[1:]:
            combined |= expr
        return combined

    @staticmethod
    def not_(expr: pc.Expression) -> pc.Expression:
        """Negate an expression.

        Parameters
        ----------
        expr
            Expression to negate.

        Returns
        -------
        pyarrow.compute.Expression
            Negated expression.
        """
        return ~expr

    @staticmethod
    def coalesce(values: Iterable[ExpressionInput]) -> pc.Expression:
        """Return a coalesce expression over the provided inputs.

        Parameters
        ----------
        values
            Sequence of field names or expressions.

        Returns
        -------
        pyarrow.compute.Expression
            Expression that selects the first non-null input.

        Raises
        ------
        ValueError
            If no expressions are provided.
        TypeError
            If Arrow does not return an expression.
        """
        expressions = [_as_expression(value) for value in values]
        if not expressions:
            msg = "coalesce requires at least one input"
            raise ValueError(msg)
        result = pc.call_function("coalesce", list(expressions))
        if isinstance(result, pc.Expression):
            return result
        msg = "Arrow compute coalesce did not return an expression."
        raise TypeError(msg)

    @staticmethod
    def if_else(
        condition: pc.Expression,
        if_true: pc.Expression,
        if_false: pc.Expression,
    ) -> pc.Expression:
        """Return a conditional expression.

        Parameters
        ----------
        condition
            Boolean expression to test.
        if_true
            Expression to return when condition is True.
        if_false
            Expression to return when condition is False.

        Returns
        -------
        pyarrow.compute.Expression
            Conditional expression.

        Raises
        ------
        TypeError
            If Arrow does not return an expression.
        """
        result = pc.call_function("if_else", [condition, if_true, if_false])
        if isinstance(result, pc.Expression):
            return result
        msg = "Arrow compute if_else did not return an expression."
        raise TypeError(msg)

    @staticmethod
    def utf8_trim(expr: pc.Expression) -> pc.Expression:
        """Return a utf8_trim expression.

        Parameters
        ----------
        expr
            Expression to trim.

        Returns
        -------
        pyarrow.compute.Expression
            Trimmed expression.

        Raises
        ------
        TypeError
            If Arrow does not return an expression.
        """
        result = pc.call_function("utf8_trim", [expr])
        if isinstance(result, pc.Expression):
            return result
        msg = "Arrow compute utf8_trim did not return an expression."
        raise TypeError(msg)


E = ExprVocab

__all__ = ["E", "ExprVocab", "Expression"]
