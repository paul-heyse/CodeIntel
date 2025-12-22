"""Opt-in db.query.parameter attribute emission."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

ScalarAttr = str | bool | int | float

_DUCKDB_NAMED_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True, slots=True)
class DbQueryParameterConfig:
    """Configuration for db.query.parameter emission."""

    enabled: bool = False
    allowed_keys: frozenset[str] = frozenset()
    require_key_in_sql: bool = True
    max_string_len: int = 80
    hash_string_values_for_keys: frozenset[str] = frozenset()
    hash_len: int = 16
    disable_on_batch: bool = True

    def is_effectively_enabled(self) -> bool:
        """Return whether parameter emission is configured and allowed.

        Returns
        -------
        bool
            True when emission is enabled and the allowlist is non-empty.

        Examples
        --------
        >>> config = DbQueryParameterConfig(enabled=True, allowed_keys=frozenset({"id"}))
        >>> config.is_effectively_enabled()
        True

        Notes
        -----
        This is a lightweight guard that does not validate SQL and does not raise.
        """
        return self.enabled and bool(self.allowed_keys)


def emit_db_query_parameters(
    *,
    sql: str,
    params: object | None,
    db_system_name: str,
    config: DbQueryParameterConfig,
    is_batch: bool = False,
) -> dict[str, ScalarAttr]:
    """Emit db.query.parameter attributes for safe scalar parameters.

    Parameters
    ----------
    sql
        SQL statement text.
    params
        Query parameters supplied to the database driver.
    db_system_name
        Database system identifier (for example, ``duckdb``).
    config
        Emission controls and allowlist configuration.
    is_batch
        Whether the query is part of a batch operation.

    Returns
    -------
    dict[str, ScalarAttr]
        Mapping of attribute keys to scalar values for observability.

    Examples
    --------
    >>> config = DbQueryParameterConfig(enabled=True, allowed_keys=frozenset({"id"}))
    >>> emit_db_query_parameters(
    ...     sql="SELECT * FROM t WHERE id = $id",
    ...     params={"id": 1},
    ...     db_system_name="duckdb",
    ...     config=config,
    ... )
    {'db.query.parameter.id': 1}

    Notes
    -----
    Only scalar parameter values are emitted, and allowlists are enforced.
    """
    if not config.is_effectively_enabled():
        return {}
    if config.disable_on_batch and is_batch:
        return {}
    if not isinstance(params, Mapping):
        return {}
    if not all(isinstance(key, str) for key in params.keys()):
        return {}

    keys_in_sql: set[str] = set()
    if config.require_key_in_sql:
        keys_in_sql = _extract_named_param_keys(sql, db_system_name=db_system_name)
        if not keys_in_sql:
            return {}

    attrs: dict[str, ScalarAttr] = {}
    for key in config.allowed_keys:
        if key not in params:
            continue
        if config.require_key_in_sql and key not in keys_in_sql:
            continue
        raw = _coerce_scalar(params[key], max_string_len=config.max_string_len)
        if raw is None:
            continue
        if isinstance(raw, str) and key in config.hash_string_values_for_keys:
            raw = _hash_str(raw, config.hash_len)
        attrs[f"db.query.parameter.{key}"] = raw

    return attrs


def _extract_named_param_keys(sql: str, *, db_system_name: str) -> set[str]:
    system = (db_system_name or "").lower()
    if system == "duckdb":
        return set(_DUCKDB_NAMED_PARAM_RE.findall(sql))
    return set()


def _coerce_scalar(value: object, *, max_string_len: int) -> ScalarAttr | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return _truncate(value, max_string_len)
    return None


def _truncate(value: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(value) <= max_len:
        return value
    if max_len == 1:
        return "."
    return f"{value[: max_len - 3]}..."


def _hash_str(value: str, hash_len: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if hash_len <= 0:
        return ""
    return digest[:hash_len]


__all__ = ["DbQueryParameterConfig", "emit_db_query_parameters"]
