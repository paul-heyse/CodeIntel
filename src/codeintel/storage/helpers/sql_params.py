"""SQL parameter interpolation helpers for relation-only query APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal

_PARAM_RE = re.compile(r"\$(\w+)")


def render_sql(sql: str, params: Mapping[str, object] | None) -> str:
    """Render a SQL string with $name parameters replaced by literals.

    Parameters
    ----------
    sql
        SQL string containing $name placeholders.
    params
        Mapping of parameter names to literal values.

    Returns
    -------
    str
        SQL string with parameters substituted as safe literals.

    Raises
    ------
    KeyError
        If a placeholder is missing from params.
    """
    if params is None or not params:
        return sql
    active_params: Mapping[str, object] = params
    missing = {key for key in _PARAM_RE.findall(sql) if key not in active_params}
    if missing:
        missing_list = ", ".join(sorted(missing))
        msg = f"Missing SQL parameter(s): {missing_list}"
        raise KeyError(msg)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return _sql_literal(active_params[key])

    return _PARAM_RE.sub(replace, sql)


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return f"'{value.isoformat()}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


__all__ = ["render_sql"]
