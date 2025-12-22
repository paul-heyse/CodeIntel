"""SQLGlot toolkit for DuckDB dialect operations.

This module centralizes SQLGlot-based utilities used across the storage layer:

- parsing and canonicalization
- scope-aware physical table reference extraction (CTE-safe)
- stable SQL fingerprinting
- low-cardinality query summaries for observability

Keeping these primitives in one place prevents semantic drift between modules
that need to reason about compiled SQL (view dependencies, diffs, perimeters).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sqlglot import diff as semantic_diff
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.lineage import lineage as build_lineage
from sqlglot.optimizer import build_scope, normalize_identifiers, optimize, qualify
from sqlglot.optimizer.scope import traverse_scope

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlglot.lineage import Node

__all__ = [
    "ParseError",
    "canonical_sql_duckdb",
    "canonicalize_expression_duckdb",
    "extract_column_lineage_duckdb",
    "extract_table_keys_duckdb",
    "extract_table_refs",
    "fingerprint_canonical_sql",
    "fingerprint_sql_duckdb",
    "fingerprint_sql_duckdb_safe",
    "normalize_sql_for_hash",
    "parse_one_duckdb",
    "render_sql_duckdb",
    "semantic_diff_sql_duckdb",
    "summarize_sql_duckdb",
]

SchemaMapping = Mapping[str, Mapping[str, str]]
_MIN_LINEAGE_PARTS = 2
_SCHEMA_QUALIFIED_PARTS = 3
_MAX_QUERY_SUMMARY_CHARS = 255
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_SINGLE_QUOTED_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_HEX_LITERAL_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_SQL_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SQL_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS_RE = re.compile(r"\s+")


def parse_one_duckdb(sql: str) -> exp.Expression:
    """Parse a DuckDB SQL string into a SQLGlot AST.

    Parameters
    ----------
    sql
        SQL string to parse.

    Returns
    -------
    sqlglot.expressions.Expression
        Parsed AST root.
    """
    return parse_one(sql, dialect=DUCKDB_DIALECT)


def canonicalize_expression_duckdb(
    root: exp.Expression,
    *,
    schema: SchemaMapping | None = None,
) -> exp.Expression:
    """Canonicalize a SQLGlot expression for stable DuckDB rendering.

    Parameters
    ----------
    root
        SQLGlot expression to canonicalize.
    schema
        Optional schema mapping to improve qualification/optimization.

    Returns
    -------
    sqlglot.expressions.Expression
        Canonicalized AST.
    """
    normalized_schema = _normalize_schema_mapping(schema)
    normalized = normalize_identifiers.normalize_identifiers(root, dialect=DUCKDB_DIALECT)
    qualified = qualify.qualify(
        normalized,
        dialect=DUCKDB_DIALECT,
        schema=normalized_schema,
        validate_qualify_columns=False,
        identify=False,
    )
    try:
        return optimize(qualified, dialect=DUCKDB_DIALECT, schema=normalized_schema)
    except (SqlglotError, TypeError, ValueError):
        return qualified


def render_sql_duckdb(root: exp.Expression) -> str:
    """Render a SQLGlot expression using the DuckDB dialect.

    Parameters
    ----------
    root
        SQLGlot expression to render.

    Returns
    -------
    str
        Rendered DuckDB SQL.
    """
    return root.sql(dialect=DUCKDB_DIALECT)


def canonical_sql_duckdb(sql: str, *, schema: SchemaMapping | None = None) -> str:
    """Return a canonicalized DuckDB SQL string.

    Canonicalization performs parse → normalize → qualify → optimize → render.

    Returns
    -------
    str
        Canonicalized SQL string.
    """
    root = parse_one_duckdb(sql)
    canonical = canonicalize_expression_duckdb(root, schema=schema)
    return render_sql_duckdb(canonical)


def fingerprint_sql_duckdb(sql: str, *, schema: SchemaMapping | None = None) -> str:
    """Return a stable SHA-256 fingerprint for a SQL string.

    Returns
    -------
    str
        Fingerprint of the canonicalized SQL.
    """
    canon = canonical_sql_duckdb(sql, schema=schema)
    return fingerprint_canonical_sql(canon)


def fingerprint_sql_duckdb_safe(sql: str, *, schema: SchemaMapping | None = None) -> str:
    """Return a stable fingerprint with a fallback to raw SQL hashing.

    Returns
    -------
    str
        Stable fingerprint for the SQL text.
    """
    try:
        return fingerprint_sql_duckdb(sql, schema=schema)
    except (ParseError, SqlglotError, TypeError, ValueError):
        normalized = normalize_sql_for_hash(sql)
        if normalized:
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def fingerprint_canonical_sql(canon: str) -> str:
    """Return a stable SHA-256 fingerprint for canonical SQL text.

    Parameters
    ----------
    canon
        Canonical SQL string.

    Returns
    -------
    str
        Stable fingerprint of the text.
    """
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def normalize_sql_for_hash(sql: str) -> str:
    """Normalize SQL text for stable hashing when parsing fails.

    This removes comments, replaces common literal forms with ``?``, and collapses
    whitespace to keep hashes stable across different inputs.

    Parameters
    ----------
    sql
        SQL statement text to normalize.

    Returns
    -------
    str
        Normalized SQL text suitable for hashing.

    Examples
    --------
    >>> normalize_sql_for_hash("SELECT 1")
    'SELECT ?'

    Notes
    -----
    The normalization is heuristic and intentionally lossy.
    """
    sql = _SQL_BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _SQL_LINE_COMMENT_RE.sub(" ", sql)
    sql = _SQL_SINGLE_QUOTED_STRING_RE.sub("?", sql)
    sql = _SQL_HEX_LITERAL_RE.sub("?", sql)
    sql = _SQL_UUID_RE.sub("?", sql)
    sql = _SQL_NUMBER_RE.sub("?", sql)
    return _WS_RE.sub(" ", sql).strip()


def summarize_sql_duckdb(
    sql: str,
    *,
    max_len: int = _MAX_QUERY_SUMMARY_CHARS,
    max_targets: int = 6,
) -> str | None:
    """Generate a low-cardinality summary for a DuckDB SQL string.

    The summary is intended for observability grouping (``db.query.summary``) and
    keeps cardinality low by excluding literals and aliases.

    Parameters
    ----------
    sql
        SQL statement text.
    max_len
        Maximum length of the returned summary.
    max_targets
        Maximum number of target tokens to include.

    Returns
    -------
    str | None
        Summary text, or None when the input is empty.

    Examples
    --------
    >>> summarize_sql_duckdb("SELECT 1") is not None
    True

    Notes
    -----
    When parsing fails, a token-based fallback summary is used.
    """
    stripped = sql.strip()
    if not stripped:
        return None

    try:
        root = parse_one_duckdb(stripped)
    except (ParseError, SqlglotError, TypeError, ValueError):
        return _fallback_query_summary(stripped, max_len=max_len)

    parts = _query_summary_parts_from_root(
        root,
        raw_sql=stripped,
        max_targets=max_targets,
    )
    return _truncate_query_summary_parts(parts, max_len=max_len)


def _fallback_query_summary(sql: str, *, max_len: int) -> str | None:
    tokens = _FALLBACK_TOKEN_RE.findall(sql)
    if not tokens:
        return None
    parts = [tokens[0]]
    if len(tokens) > 1:
        parts.append(tokens[1])
    return _truncate_query_summary_parts(parts, max_len=max_len)


def _query_summary_parts_from_root(
    root: exp.Expression,
    *,
    raw_sql: str,
    max_targets: int,
) -> list[str]:
    if isinstance(root, exp.With) and isinstance(root.this, exp.Expression):
        return _query_summary_parts_from_root(
            root.this,
            raw_sql=raw_sql,
            max_targets=max_targets,
        )

    if isinstance(root, exp.Insert):
        return _query_summary_parts_for_insert(
            root,
            raw_sql=raw_sql,
            max_targets=max_targets,
        )

    if isinstance(root, exp.Create):
        return _query_summary_parts_for_create(
            root,
            raw_sql=raw_sql,
            max_targets=max_targets,
        )

    operation = _operation_name_for_root(root)
    parts: list[str] = [operation] if operation else []
    parts.extend(
        _query_summary_targets_for_expression(
            root,
            raw_sql=raw_sql,
            max_targets=max_targets,
        )
    )
    return parts


def _query_summary_parts_for_insert(
    root: exp.Insert,
    *,
    raw_sql: str,
    max_targets: int,
) -> list[str]:
    parts: list[str] = ["INSERT"]
    target = _format_table_for_summary(getattr(root, "this", None))
    if target:
        parts.append(target)

    nested = getattr(root, "expression", None)
    if isinstance(nested, exp.Expression):
        nested_op = _operation_name_for_root(nested) or "SELECT"
        parts.append(nested_op)
        exclude = {target.lower()} if target else set()
        parts.extend(
            _query_summary_targets_for_expression(
                nested,
                raw_sql=raw_sql,
                max_targets=max_targets,
                exclude=exclude,
            )
        )
    return parts


def _query_summary_parts_for_create(
    root: exp.Create,
    *,
    raw_sql: str,
    max_targets: int,
) -> list[str]:
    parts: list[str] = ["CREATE"]
    target = _format_table_for_summary(getattr(root, "this", None))
    if target:
        parts.append(target)

    nested = getattr(root, "expression", None)
    if isinstance(nested, exp.Expression):
        nested_op = _operation_name_for_root(nested) or "SELECT"
        parts.append(nested_op)
        exclude = {target.lower()} if target else set()
        parts.extend(
            _query_summary_targets_for_expression(
                nested,
                raw_sql=raw_sql,
                max_targets=max_targets,
                exclude=exclude,
            )
        )
    return parts


def _operation_name_for_root(root: exp.Expression) -> str | None:
    if isinstance(root, exp.Select):
        return "SELECT"
    if isinstance(root, exp.Update):
        return "UPDATE"
    if isinstance(root, exp.Delete):
        return "DELETE"
    if isinstance(root, exp.Insert):
        return "INSERT"
    if isinstance(root, exp.Create):
        return "CREATE"
    if isinstance(root, exp.Drop):
        return "DROP"
    key = getattr(root, "key", None)
    if isinstance(key, str) and key:
        return key.replace("_", " ").upper()
    return None


def _query_summary_targets_for_expression(
    root: exp.Expression,
    *,
    raw_sql: str,
    max_targets: int,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    sql_lower = raw_sql.lower()

    tables = extract_table_refs(root)
    formatted: list[tuple[int, str]] = []
    for table in tables:
        key = _format_table_for_summary(table)
        if not key:
            continue
        key_lower = key.lower()
        if key_lower in exclude:
            continue
        pos = _best_effort_table_position(sql_lower, key_lower)
        formatted.append((pos, key))

    formatted.sort(key=lambda item: item[0])

    out: list[str] = []
    seen: set[str] = set()
    for _, key in formatted:
        key_lower = key.lower()
        if key_lower in seen:
            continue
        out.append(key)
        seen.add(key_lower)
        if max_targets > 0 and len(out) >= max_targets:
            break
    return out


def _best_effort_table_position(sql_lower: str, table_key_lower: str) -> int:
    pos = sql_lower.find(table_key_lower)
    if pos != -1:
        return pos
    if "." in table_key_lower:
        _, table = table_key_lower.split(".", 1)
        pos = sql_lower.find(table)
        if pos != -1:
            return pos
    return 10**9


def _format_table_for_summary(node: object) -> str | None:
    if not isinstance(node, exp.Table):
        return None
    schema = node.db
    name = node.name
    if not name:
        return None
    if schema:
        return f"{schema}.{name}"
    return name


def _truncate_query_summary_parts(parts: list[str], *, max_len: int) -> str:
    kept: list[str] = []
    length = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        add_len = len(part) + (1 if kept else 0)
        if length + add_len > max_len:
            break
        kept.append(part)
        length += add_len
    return " ".join(kept)


def semantic_diff_sql_duckdb(
    before_sql: str,
    after_sql: str,
    *,
    schema: SchemaMapping | None = None,
) -> tuple[str, ...]:
    """Return a semantic diff between two SQL strings (DuckDB dialect).

    Returns
    -------
    tuple[str, ...]
        Human-readable diff actions between the two queries.
    """
    before = canonicalize_expression_duckdb(parse_one_duckdb(before_sql), schema=schema)
    after = canonicalize_expression_duckdb(parse_one_duckdb(after_sql), schema=schema)
    actions = semantic_diff(before, after)
    return tuple(str(action) for action in actions)


def extract_table_refs(root: exp.Expression) -> tuple[exp.Table, ...]:
    """Extract physical table references from a parsed AST.

    Notes
    -----
    Uses scope traversal to avoid treating CTE names as physical tables.

    Returns
    -------
    tuple[sqlglot.expressions.Table, ...]
        Physical table nodes referenced by the query.
    """
    tables: list[exp.Table] = []
    for scope in traverse_scope(root):
        tables.extend(source for source in scope.sources.values() if isinstance(source, exp.Table))
    return tuple(tables)


def extract_table_keys_duckdb(sql: str) -> frozenset[str]:
    """Extract referenced physical table keys from a DuckDB SQL string.

    Returns lowercased keys of the form ``schema.table`` when schema-qualified,
    otherwise ``table``.

    Returns
    -------
    frozenset[str]
        Referenced table keys.
    """
    root = parse_one_duckdb(sql)
    out: set[str] = set()
    for table in extract_table_refs(root):
        schema = table.db
        name = table.name
        out.add(f"{schema}.{name}".lower() if schema else name.lower())
    return frozenset(out)


def extract_column_lineage_duckdb(
    sql: str,
    *,
    schema: SchemaMapping | None = None,
) -> dict[str, frozenset[str]]:
    """Extract column-level lineage for a DuckDB SQL string.

    Parameters
    ----------
    sql
        DuckDB SQL string.
    schema
        Optional schema mapping to improve qualification/lineage accuracy.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping of output column name to upstream column keys (table.column).
    """
    root = parse_one_duckdb(sql)
    canonical = canonicalize_expression_duckdb(root, schema=schema)
    scope = build_scope(canonical)
    if scope is None:
        return {}

    output_columns = [name for name in scope.expression.named_selects if name != "*"]
    if not output_columns:
        return {}

    alias_map = _collect_table_aliases(canonical)
    out: dict[str, frozenset[str]] = {}
    for name in output_columns:
        try:
            lineage_node = build_lineage(name, canonical, dialect=DUCKDB_DIALECT, scope=scope)
        except SqlglotError:
            continue
        upstream = _collect_upstream_columns(lineage_node, alias_map)
        if upstream:
            out[name] = frozenset(upstream)
    return out


def _collect_table_aliases(root: exp.Expression) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for table in root.find_all(exp.Table):
        table_key = _table_key_for_table(table)
        alias = table.alias_or_name
        if alias:
            alias_map[alias.lower()] = table_key
        alias_map[table.name.lower()] = table_key
        if table.db:
            alias_map[f"{table.db}.{table.name}".lower()] = table_key
    return alias_map


def _table_key_for_table(table: exp.Table) -> str:
    schema = table.db
    name = table.name
    return f"{schema}.{name}".lower() if schema else name.lower()


def _collect_upstream_columns(node: Node, alias_map: Mapping[str, str]) -> set[str]:
    upstream: set[str] = set()
    for leaf in node.walk():
        if leaf.downstream:
            continue
        if not isinstance(leaf.expression, exp.Table):
            continue
        ref = _resolve_lineage_ref(leaf.name, alias_map)
        if ref is not None:
            upstream.add(ref)
    return upstream


def _resolve_lineage_ref(name: str, alias_map: Mapping[str, str]) -> str | None:
    cleaned = name.replace('"', "").replace("`", "")
    parts = [part for part in cleaned.split(".") if part]
    if len(parts) < _MIN_LINEAGE_PARTS:
        return None
    column = parts[-1]
    table = parts[-2]
    schema = parts[-3] if len(parts) >= _SCHEMA_QUALIFIED_PARTS else None

    table_key = alias_map.get(table.lower())
    if table_key is None and schema is not None:
        table_key = f"{schema}.{table}".lower()
    if table_key is None:
        table_key = table.lower()
    return f"{table_key}.{column}".lower()


def _normalize_schema_mapping(schema: SchemaMapping | None) -> dict[str, dict[str, str]] | None:
    if schema is None:
        return None
    return {table: dict(columns) for table, columns in schema.items()}


def extract_table_keys_from_roots(roots: Iterable[exp.Expression]) -> frozenset[str]:
    """Extract referenced physical table keys from multiple SQLGlot roots.

    Returns
    -------
    frozenset[str]
        Referenced table keys.
    """
    out: set[str] = set()
    for root in roots:
        for table in extract_table_refs(root):
            schema = table.db
            name = table.name
            out.add(f"{schema}.{name}".lower() if schema else name.lower())
    return frozenset(out)
