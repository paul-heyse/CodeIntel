"""Lightweight contract checking helpers for graph metric plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from codeintel.storage.gateway import StorageGateway

SAFE_TABLE_QUERIES: dict[str, str] = {
    "analytics.graph_metrics_functions": (
        "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?"
    ),
    "analytics.graph_metrics_modules": (
        "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?"
    ),
    "analytics.graph_stats": (
        "SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?"
    ),
}
SAFE_TABLE_COLUMNS: dict[str, set[str]] = {
    "analytics.graph_metrics_functions": {
        "repo",
        "commit",
        "node_id",
        "out_degree",
        "in_degree",
        "degree",
        "eccentricity",
        "pagerank",
        "betweenness",
        "closeness",
        "is_sink",
        "fan_in",
        "fan_out",
        "updated_at",
    },
    "analytics.graph_metrics_modules": {
        "repo",
        "commit",
        "module",
        "in_degree",
        "out_degree",
        "degree",
        "pagerank",
        "betweenness",
        "closeness",
        "is_sink",
        "fan_in",
        "fan_out",
        "updated_at",
    },
    "analytics.graph_stats": {"repo", "commit", "metric", "value", "updated_at"},
}


@dataclass(frozen=True)
class PluginContractResult:
    """Result of a plugin contract check."""

    name: str
    status: Literal["passed", "failed", "soft_failed"]
    message: str | None = None


class _ContractContext(Protocol):
    """Narrow context interface required by contract checkers."""

    @property
    def gateway(self) -> StorageGateway:
        """Storage gateway providing DB access."""
        ...

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...


ContractChecker = Callable[[_ContractContext], PluginContractResult]


def run_contract_checkers(
    *,
    ctx: _ContractContext,
    checkers: tuple[ContractChecker, ...],
) -> tuple[PluginContractResult, ...]:
    """
    Execute contract checkers and aggregate results.

    Parameters
    ----------
    ctx:
        Execution context providing gateway/repo/commit.
    checkers:
        Sequence of callables returning PluginContractResult.

    Returns
    -------
    tuple[PluginContractResult, ...]
    """
    results: list[PluginContractResult] = []
    for checker in checkers:
        result = checker(ctx)
        results.append(result)
    return tuple(results)


def _split_table(table: str) -> tuple[str, str]:
    schema, name = table.split(".", maxsplit=1) if "." in table else ("analytics", table)
    return schema, name


def _ensure_safe_table(table: str) -> None:
    if table not in SAFE_TABLE_COLUMNS:
        message = f"Unsafe or unknown table requested in contract: {table}"
        raise ValueError(message)


def assert_table_not_empty(
    gateway: StorageGateway,
    *,
    table: str,
    repo: str,
    commit: str,
    name: str | None = None,
) -> PluginContractResult:
    """
    Ensure a table has at least one row for repo/commit.

    Returns
    -------
    PluginContractResult
        Contract outcome with status and optional message.
    """
    resolved_name = name or f"{table}_not_empty"
    query = SAFE_TABLE_QUERIES.get(table)
    if query is None:
        message = f"Unsafe or unknown table requested in contract: {table}"
        return PluginContractResult(name=resolved_name, status="failed", message=message)
    row = gateway.con.execute(query, [repo, commit]).fetchone()
    count = int(row[0]) if row is not None else 0
    if count > 0:
        return PluginContractResult(name=resolved_name, status="passed")
    return PluginContractResult(
        name=resolved_name,
        status="failed",
        message=f"{table} is empty for {repo}@{commit}",
    )


def assert_table_exists(
    gateway: StorageGateway,
    *,
    table: str,
    name: str | None = None,
) -> PluginContractResult:
    """
    Ensure a table exists in the analytics schema.

    Returns
    -------
    PluginContractResult
        Contract outcome describing existence.
    """
    resolved_name = name or f"{table}_exists"
    _ensure_safe_table(table)
    schema, table_name = _split_table(table)
    query = (
        "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1"
    )
    row = gateway.con.execute(query, [schema, table_name]).fetchone()
    if row is None:
        message = f"{table} is missing"
        return PluginContractResult(name=resolved_name, status="failed", message=message)
    return PluginContractResult(name=resolved_name, status="passed")


def assert_columns_present(
    gateway: StorageGateway,
    *,
    table: str,
    expected_columns: set[str],
    name: str | None = None,
) -> PluginContractResult:
    """
    Ensure required columns exist on a table.

    Returns
    -------
    PluginContractResult
        Contract outcome describing column presence.
    """
    resolved_name = name or f"{table}_columns_present"
    _ensure_safe_table(table)
    schema, table_name = _split_table(table)
    allowed = SAFE_TABLE_COLUMNS.get(table, set())
    if not expected_columns.issubset(allowed):
        missing = expected_columns.difference(allowed)
        message = f"Columns not allowed for contract: {missing}"
        return PluginContractResult(name=resolved_name, status="failed", message=message)
    query = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = ? AND table_name = ?"
    )
    rows = gateway.con.execute(query, [schema, table_name]).fetchall()
    present = {row[0] for row in rows}
    missing_columns = expected_columns.difference(present)
    if missing_columns:
        message = f"Missing columns on {table}: {sorted(missing_columns)}"
        return PluginContractResult(name=resolved_name, status="failed", message=message)
    return PluginContractResult(name=resolved_name, status="passed")


def assert_not_null_fraction(  # noqa: PLR0913
    gateway: StorageGateway,
    *,
    table: str,
    column: str,
    repo: str,
    commit: str,
    min_fraction: float,
    name: str | None = None,
) -> PluginContractResult:
    """
    Ensure a column has a minimum non-null fraction for repo/commit rows.

    Returns
    -------
    PluginContractResult
        Contract outcome describing non-null fraction status.
    """
    resolved_name = name or f"{table}.{column}_not_null_fraction"
    _ensure_safe_table(table)
    allowed = SAFE_TABLE_COLUMNS.get(table, set())
    if column not in allowed:
        message = f"Column {column} is not allowed for {table}"
        return PluginContractResult(name=resolved_name, status="failed", message=message)
    row = gateway.con.execute(
        f"SELECT AVG(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) "  # noqa: S608
        f"FROM {table} WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    fraction = float(row[0]) if row is not None and row[0] is not None else 0.0
    if fraction >= min_fraction:
        return PluginContractResult(name=resolved_name, status="passed", message=str(fraction))
    message = f"{table}.{column} non-null fraction {fraction:.2f} below {min_fraction}"
    return PluginContractResult(name=resolved_name, status="failed", message=message)


def table_not_empty_checker(table: str, *, name: str | None = None) -> ContractChecker:
    """
    Build a contract checker that asserts a table has rows for a repo/commit.

    Returns
    -------
    ContractChecker
        Callable that validates row presence for the configured table.
    """

    def _checker(ctx: _ContractContext) -> PluginContractResult:
        return assert_table_not_empty(
            ctx.gateway,
            table=table,
            repo=ctx.repo,
            commit=ctx.commit,
            name=name,
        )

    return _checker


def table_exists_checker(table: str, *, name: str | None = None) -> ContractChecker:
    """
    Build a contract checker that asserts table existence.

    Returns
    -------
    ContractChecker
        Callable that validates table presence.
    """

    def _checker(ctx: _ContractContext) -> PluginContractResult:
        return assert_table_exists(ctx.gateway, table=table, name=name)

    return _checker


def columns_present_checker(
    table: str, *, expected_columns: set[str], name: str | None = None
) -> ContractChecker:
    """
    Build a contract checker that asserts required columns exist.

    Returns
    -------
    ContractChecker
        Callable that validates required columns for a table.
    """

    def _checker(ctx: _ContractContext) -> PluginContractResult:
        return assert_columns_present(
            ctx.gateway,
            table=table,
            expected_columns=expected_columns,
            name=name,
        )

    return _checker


def not_null_fraction_checker(
    table: str,
    *,
    column: str,
    min_fraction: float,
    name: str | None = None,
) -> ContractChecker:
    """
    Build a contract checker that validates non-null fraction for a column.

    Returns
    -------
    ContractChecker
        Callable that validates non-null fraction for the configured column.
    """

    def _checker(ctx: _ContractContext) -> PluginContractResult:
        return assert_not_null_fraction(
            ctx.gateway,
            table=table,
            column=column,
            repo=ctx.repo,
            commit=ctx.commit,
            min_fraction=min_fraction,
            name=name,
        )

    return _checker
