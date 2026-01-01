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
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

from sqlglot import diff as semantic_diff
from sqlglot import exp, parse, parse_one
from sqlglot.dialects.dialect import DialectType
from sqlglot.dialects.duckdb import DuckDB
from sqlglot.errors import ErrorLevel, ParseError, SqlglotError, UnsupportedError
from sqlglot.lineage import lineage as build_lineage
from sqlglot.optimizer import build_scope, normalize_identifiers, optimize, qualify
from sqlglot.optimizer.scope import traverse_scope

from codeintel.core.constants import DUCKDB_DIALECT
from codeintel.core.schemas.type_mappings import normalize_engine_column_type

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlglot.lineage import Node

    from codeintel.core.schemas.primitives import ColumnType

__all__ = [
    "SELECT_ONLY_DISALLOWED_NODES",
    "AstCapabilityConfig",
    "AstCapabilityError",
    "AstCapabilityIssue",
    "AstCapabilityReport",
    "GeneratorConfig",
    "ParseError",
    "QuerySummaryConfig",
    "canonical_sql_duckdb",
    "canonicalize_expression_duckdb",
    "canonicalize_select_duckdb",
    "capability_envelope_report",
    "ensure_ast_capability",
    "extract_column_lineage_duckdb",
    "extract_column_lineage_from_ast",
    "extract_table_keys_duckdb",
    "extract_table_refs",
    "fingerprint_canonical_sql",
    "fingerprint_expression_duckdb",
    "fingerprint_sql_duckdb",
    "fingerprint_sql_duckdb_safe",
    "normalize_sql_for_hash",
    "parse_one_duckdb",
    "render_sql_duckdb",
    "render_sql_duckdb_safe",
    "schema_mapping_for_table_key",
    "semantic_diff_sql_duckdb",
    "summarize_sql_duckdb",
    "table_expr_from_ref",
]

SchemaMapping = Mapping[str, Mapping[str, str]]
_MIN_LINEAGE_PARTS = 2
_SCHEMA_QUALIFIED_PARTS = 3
_MAX_QUERY_SUMMARY_CHARS = 255
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_HASHED_TARGET_PREFIX = "h:"
_SUSPICIOUS_TARGET_RE = re.compile(r"(?:/|\\\\|://)")

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

LOG = logging.getLogger(__name__)

_MUTATION_DISALLOWED_NODES: tuple[type[exp.Expression], ...] = (
    exp.Alter,
    exp.Analyze,
    exp.Attach,
    exp.Command,
    exp.Copy,
    exp.Create,
    exp.Delete,
    exp.Detach,
    exp.Drop,
    exp.Grant,
    exp.Insert,
    exp.Merge,
    exp.Pragma,
    exp.Refresh,
    exp.Revoke,
    exp.Rollback,
    exp.Commit,
    exp.Set,
    exp.Transaction,
    exp.Update,
    exp.Use,
)
_CAPABILITY_DISALLOWED_NODES: tuple[type[exp.Expression], ...] = (
    *_MUTATION_DISALLOWED_NODES,
    exp.Group,
    exp.Having,
    exp.Distinct,
    exp.Lateral,
    exp.Subquery,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Window,
)
SELECT_ONLY_DISALLOWED_NODES = _MUTATION_DISALLOWED_NODES
_SAFE_DIALECT_UNSUPPORTED_NODES: tuple[type[exp.Expression], ...] = _CAPABILITY_DISALLOWED_NODES


def _unsupported_transform(
    node_type: type[exp.Expression],
) -> Callable[[DuckDB.Generator, exp.Expression], str]:
    def _transform(generator: DuckDB.Generator, _expression: exp.Expression) -> str:
        return cast(
            "str",
            generator.unsupported(f"{node_type.__name__} is not supported in DuckDBSafe"),
        )

    return _transform


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuration for DuckDB SQL generation."""

    pretty: bool | None = None
    identify: str | bool = False
    normalize: bool = False
    pad: int = 2
    indent: int = 2
    normalize_functions: str | bool | None = None
    unsupported_level: ErrorLevel = ErrorLevel.WARN
    max_unsupported: int = 3
    leading_comma: bool = False
    max_text_width: int = 80
    comments: bool = True
    dialect: DialectType | None = None


class _GeneratorConfigParams(TypedDict, total=False):
    pretty: bool | None
    identify: str | bool
    normalize: bool
    pad: int
    indent: int
    normalize_functions: str | bool | None
    unsupported_level: ErrorLevel
    max_unsupported: int
    leading_comma: bool
    max_text_width: int
    comments: bool
    dialect: DialectType | None


class DuckDBSafe(DuckDB):
    """DuckDB dialect that rejects unsupported SQL constructs by default."""

    class Generator(DuckDB.Generator):
        def __init__(
            self,
            *,
            config: GeneratorConfig | None = None,
            **kwargs: Unpack[_GeneratorConfigParams],
        ) -> None:
            if config is None:
                config = GeneratorConfig(**kwargs)
            elif kwargs:
                config = replace(config, **kwargs)
            super().__init__(
                pretty=config.pretty,
                identify=config.identify,
                normalize=config.normalize,
                pad=config.pad,
                indent=config.indent,
                normalize_functions=config.normalize_functions,
                unsupported_level=config.unsupported_level,
                max_unsupported=config.max_unsupported,
                leading_comma=config.leading_comma,
                max_text_width=config.max_text_width,
                comments=config.comments,
                dialect=config.dialect,
            )
            self.TRANSFORMS = {
                **self.TRANSFORMS,
                **{
                    node_type: _unsupported_transform(node_type)
                    for node_type in _SAFE_DIALECT_UNSUPPORTED_NODES
                },
            }


@dataclass(frozen=True, slots=True)
class QuerySummaryConfig:
    """Configuration for db.query.summary generation."""

    max_len: int = _MAX_QUERY_SUMMARY_CHARS
    max_targets: int = 6
    emit_ellipsis: bool = True
    hash_suspicious_targets: bool = True
    hash_target_len: int = 12
    hash_target_min_len: int = 64
    include_subquery_operations: bool = True
    include_multi_statement: bool = True


@dataclass(frozen=True, slots=True)
class _SummaryTokens:
    tokens: tuple[str, ...]
    capped: bool = False


@dataclass(frozen=True, slots=True)
class AstCapabilityIssue:
    """Single capability envelope issue detected in an AST."""

    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class AstCapabilityReport:
    """Capability envelope report for a SQLGlot AST."""

    supported: bool
    issues: tuple[AstCapabilityIssue, ...]
    features: tuple[str, ...]


class AstCapabilityError(ValueError):
    """Raised when an AST violates the capability envelope."""

    def __init__(self, issues: Iterable[AstCapabilityIssue]) -> None:
        issues_tuple = tuple(issues)
        summary = ", ".join(issue.detail for issue in issues_tuple)
        super().__init__(f"Unsupported SQL AST features: {summary}")
        self.issues = issues_tuple


@dataclass(frozen=True, slots=True)
class AstCapabilityConfig:
    """Configuration for AST capability validation."""

    allowed_anonymous_functions: frozenset[str] | None = None
    disallowed_nodes: tuple[type[exp.Expression], ...] | None = None
    allow_aggregates: bool = False
    enforce_safe_sql: bool = True
    log_context: str | None = None


def _capability_issue_for_sql(root: exp.Expression) -> AstCapabilityIssue | None:
    try:
        _ = root.sql(
            dialect=DuckDBSafe,
            unsupported_level=ErrorLevel.RAISE,
            max_unsupported=0,
        )
    except UnsupportedError as exc:
        return AstCapabilityIssue(
            kind="unsupported_sqlglot",
            detail=str(exc),
        )
    return None


def _scan_capability_node(
    node: exp.Expression,
    *,
    allowed_anonymous_functions: frozenset[str] | None,
    disallowed_nodes: tuple[type[exp.Expression], ...],
    allow_aggregates: bool,
) -> tuple[list[AstCapabilityIssue], list[str]]:
    issues: list[AstCapabilityIssue] = []
    features: list[str] = []
    if isinstance(node, disallowed_nodes):
        issues.append(AstCapabilityIssue(kind="unsupported_node", detail=type(node).__name__))
        return issues, features
    _append_agg_capability(
        node, issues=issues, features=features, allow_aggregates=allow_aggregates
    )
    _append_anonymous_capability(
        node,
        issues=issues,
        features=features,
        allowed_anonymous_functions=allowed_anonymous_functions,
    )
    _append_join_capability(node, features=features)
    _append_feature_flag(node, features=features, expr_type=exp.Cast, feature="cast")
    _append_feature_flag(node, features=features, expr_type=exp.Coalesce, feature="coalesce")
    _append_feature_flag(node, features=features, expr_type=exp.Case, feature="case")
    return issues, features


def _append_agg_capability(
    node: exp.Expression,
    *,
    issues: list[AstCapabilityIssue],
    features: list[str],
    allow_aggregates: bool,
) -> None:
    if not isinstance(node, exp.AggFunc):
        return
    name = node.sql_name().lower()
    features.append(f"agg:{name}")
    if not allow_aggregates:
        issues.append(
            AstCapabilityIssue(
                kind="aggregate_function",
                detail=name,
            )
        )


def _append_anonymous_capability(
    node: exp.Expression,
    *,
    issues: list[AstCapabilityIssue],
    features: list[str],
    allowed_anonymous_functions: frozenset[str] | None,
) -> None:
    if not isinstance(node, exp.Anonymous):
        return
    name = (node.name or "").lower()
    if name:
        features.append(f"func:{name}")
    if allowed_anonymous_functions is not None and name not in allowed_anonymous_functions:
        issues.append(
            AstCapabilityIssue(
                kind="unsupported_function",
                detail=name or "<anonymous>",
            )
        )


def _append_join_capability(node: exp.Expression, *, features: list[str]) -> None:
    if not isinstance(node, exp.Join):
        return
    join_kind = (node.args.get("kind") or "inner").lower()
    features.append(f"join:{join_kind}")


def _append_feature_flag(
    node: exp.Expression,
    *,
    features: list[str],
    expr_type: type[exp.Expression],
    feature: str,
) -> None:
    if isinstance(node, expr_type):
        features.append(feature)


def capability_envelope_report(
    root: exp.Expression,
    *,
    allowed_anonymous_functions: frozenset[str] | None = None,
    disallowed_nodes: tuple[type[exp.Expression], ...] | None = None,
    allow_aggregates: bool = False,
    enforce_safe_sql: bool = True,
) -> AstCapabilityReport:
    """Return a capability envelope report for a SQLGlot AST.

    Parameters
    ----------
    root
        SQLGlot expression to inspect.
    allowed_anonymous_functions
        Optional allowlist of anonymous function names (lowercased).
    disallowed_nodes
        Optional tuple of disallowed SQLGlot node types.
    allow_aggregates
        Whether aggregate functions are permitted.
    enforce_safe_sql
        Whether to enforce the DuckDBSafe SQL capability envelope.

    Returns
    -------
    AstCapabilityReport
        Capability envelope report with issues and feature summary.
    """
    issues: list[AstCapabilityIssue] = []
    features: set[str] = set()

    active_disallowed = disallowed_nodes or _CAPABILITY_DISALLOWED_NODES
    for node in root.walk():
        node_issues, node_features = _scan_capability_node(
            node,
            allowed_anonymous_functions=allowed_anonymous_functions,
            disallowed_nodes=active_disallowed,
            allow_aggregates=allow_aggregates,
        )
        issues.extend(node_issues)
        features.update(node_features)

    if enforce_safe_sql:
        sql_issue = _capability_issue_for_sql(root)
        if sql_issue is not None:
            issues.append(sql_issue)

    return AstCapabilityReport(
        supported=not issues,
        issues=tuple(issues),
        features=tuple(sorted(features)),
    )


def log_ast_capability_report(
    report: AstCapabilityReport,
    *,
    context: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a deterministic capability envelope log entry.

    Parameters
    ----------
    report
        Capability envelope report to log.
    context
        Optional context string to include in the log payload.
    logger
        Optional logger override; defaults to module logger.
    """
    active_logger = logger or LOG
    features = ",".join(report.features) if report.features else "-"
    issues = (
        ",".join(f"{issue.kind}:{issue.detail}" for issue in report.issues)
        if report.issues
        else "-"
    )
    active_logger.debug(
        "sqlglot_ast_capability context=%s supported=%s features=%s issues=%s",
        context or "unknown",
        report.supported,
        features,
        issues,
    )


def ensure_ast_capability(
    root: exp.Expression,
    config: AstCapabilityConfig | None = None,
) -> AstCapabilityReport:
    """Validate a SQLGlot AST against the capability envelope.

    Parameters
    ----------
    root
        SQLGlot expression to validate.
    config
        Optional capability validation configuration.

    Returns
    -------
    AstCapabilityReport
        Capability envelope report when validation passes.

    Raises
    ------
    AstCapabilityError
        When the AST contains unsupported features.
    """
    settings = config or AstCapabilityConfig()
    report = capability_envelope_report(
        root,
        allowed_anonymous_functions=settings.allowed_anonymous_functions,
        disallowed_nodes=settings.disallowed_nodes,
        allow_aggregates=settings.allow_aggregates,
        enforce_safe_sql=settings.enforce_safe_sql,
    )
    log_ast_capability_report(report, context=settings.log_context)
    if not report.supported:
        raise AstCapabilityError(report.issues)
    return report


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
    normalized = normalize_identifiers.normalize_identifiers(root.copy(), dialect=DUCKDB_DIALECT)
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


def canonicalize_select_duckdb(
    root: exp.Select,
    *,
    schema: SchemaMapping | None = None,
) -> exp.Select:
    """Canonicalize a SQLGlot Select expression for DuckDB.

    Parameters
    ----------
    root
        SQLGlot Select expression to canonicalize.
    schema
        Optional schema mapping to improve qualification/optimization.

    Returns
    -------
    sqlglot.expressions.Select
        Canonicalized Select expression.

    Raises
    ------
    TypeError
        If the canonicalized expression is not a Select.
    """
    canonical = canonicalize_expression_duckdb(root, schema=schema)
    if not isinstance(canonical, exp.Select):
        msg = "Expected Select expression after canonicalization"
        raise TypeError(msg)
    return canonical


def schema_mapping_for_table_key(
    table_key: str,
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> SchemaMapping | None:
    """Return a SQLGlot schema mapping for a single table key.

    Parameters
    ----------
    table_key
        Table key (schema.table) to associate with column types.
    column_types
        Column type mapping for the table.

    Returns
    -------
    SchemaMapping | None
        Schema mapping for SQLGlot optimization, when column types are available.
    """
    if not column_types:
        return None
    normalized: dict[str, str] = {}
    for column, column_type in column_types.items():
        try:
            normalized_type = normalize_engine_column_type(column_type)
        except ValueError:
            normalized_type = str(column_type).strip()
        if not normalized_type:
            continue
        normalized[column] = normalized_type
    if not normalized:
        return None
    return {table_key: normalized}


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


def render_sql_duckdb_safe(root: exp.Expression) -> str:
    """Render a SQLGlot expression using the DuckDB safe dialect.

    Returns
    -------
    str
        Rendered DuckDB SQL.
    """
    return root.sql(
        dialect=DuckDBSafe,
        unsupported_level=ErrorLevel.RAISE,
        max_unsupported=0,
    )


def table_expr_from_ref(table_ref: str) -> exp.Table:
    """Return a SQLGlot Table expression for a table reference.

    Returns
    -------
    sqlglot.expressions.Table
        SQLGlot table expression for the reference.
    """
    return exp.to_table(table_ref)


def fingerprint_expression_duckdb(
    root: exp.Expression,
    *,
    schema: SchemaMapping | None = None,
) -> str:
    """Return a stable SHA-256 fingerprint for a SQLGlot expression.

    Parameters
    ----------
    root
        SQLGlot expression to canonicalize and hash.
    schema
        Optional schema mapping to improve qualification/optimization.

    Returns
    -------
    str
        Fingerprint of the canonicalized SQL rendering.
    """
    canonical = canonicalize_expression_duckdb(root, schema=schema)
    return fingerprint_canonical_sql(render_sql_duckdb(canonical))


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
    config: QuerySummaryConfig | None = None,
) -> str | None:
    """Generate a low-cardinality summary for a DuckDB SQL string.

    The summary is intended for observability grouping (``db.query.summary``) and
    keeps cardinality low by excluding literals and aliases.

    Parameters
    ----------
    sql
        SQL statement text.
    config
        Query summary configuration overrides.

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

    summary_config = config or QuerySummaryConfig()
    max_len = max(0, summary_config.max_len)

    try:
        roots = _parse_statements_duckdb(
            stripped,
            include_multi_statement=summary_config.include_multi_statement,
        )
    except (ParseError, SqlglotError, TypeError, ValueError):
        return _fallback_query_summary(stripped, config=summary_config)

    if not roots:
        return None

    parts: list[str] = []
    capped = False
    for root in roots:
        statement_sql = root.sql(dialect=DUCKDB_DIALECT)
        summary_parts = _query_summary_parts_from_root(
            root,
            raw_sql=statement_sql,
            config=summary_config,
        )
        if not summary_parts.tokens:
            continue
        if parts:
            parts.append(";")
        parts.extend(summary_parts.tokens)
        capped = capped or summary_parts.capped

    if not parts:
        return None

    return _truncate_query_summary_parts(
        parts,
        max_len=max_len,
        emit_ellipsis=summary_config.emit_ellipsis,
        force_ellipsis=capped,
    )


def _parse_statements_duckdb(
    sql: str,
    *,
    include_multi_statement: bool,
) -> list[exp.Expression]:
    if include_multi_statement:
        statements = parse(sql, read=DUCKDB_DIALECT)
    else:
        statements = [parse_one_duckdb(sql)]
    return [stmt for stmt in statements if isinstance(stmt, exp.Expression)]


def _fallback_query_summary(sql: str, *, config: QuerySummaryConfig) -> str | None:
    tokens = _FALLBACK_TOKEN_RE.findall(sql)
    if not tokens:
        return None
    parts = [tokens[0]]
    if len(tokens) > 1:
        parts.append(tokens[1])
    return _truncate_query_summary_parts(
        parts,
        max_len=max(0, config.max_len),
        emit_ellipsis=config.emit_ellipsis,
        force_ellipsis=False,
    )


def _query_summary_parts_from_root(
    root: exp.Expression,
    *,
    raw_sql: str,
    config: QuerySummaryConfig,
) -> _SummaryTokens:
    if isinstance(root, exp.With) and isinstance(root.this, exp.Expression):
        return _query_summary_parts_from_root(
            root.this,
            raw_sql=raw_sql,
            config=config,
        )

    if isinstance(root, exp.Insert):
        return _query_summary_parts_for_insert(
            root,
            raw_sql=raw_sql,
            config=config,
        )

    if isinstance(root, exp.Create):
        return _query_summary_parts_for_create(
            root,
            raw_sql=raw_sql,
            config=config,
        )

    operation = _operation_name_for_root(root)
    parts: list[str] = [operation] if operation else []
    if config.include_subquery_operations:
        parts.extend(_nested_subquery_operations(root))
    targets = _query_summary_targets_for_expression(
        root,
        raw_sql=raw_sql,
        config=config,
    )
    parts.extend(targets.tokens)
    return _SummaryTokens(tokens=tuple(parts), capped=targets.capped)


def _query_summary_parts_for_insert(
    root: exp.Insert,
    *,
    raw_sql: str,
    config: QuerySummaryConfig,
) -> _SummaryTokens:
    parts: list[str] = ["INSERT"]
    target = _sanitize_summary_target(
        _format_table_for_summary(getattr(root, "this", None)),
        hash_suspicious_targets=config.hash_suspicious_targets,
        hash_target_len=config.hash_target_len,
        hash_target_min_len=config.hash_target_min_len,
    )
    if target:
        parts.append(target)

    capped = False
    nested = getattr(root, "expression", None)
    if isinstance(nested, exp.Expression):
        nested_op = _operation_name_for_root(nested) or "SELECT"
        parts.append(nested_op)
        if config.include_subquery_operations:
            parts.extend(_nested_subquery_operations(nested))
        exclude = {target.lower()} if target else set()
        targets = _query_summary_targets_for_expression(
            nested,
            raw_sql=raw_sql,
            exclude=exclude,
            config=config,
        )
        parts.extend(targets.tokens)
        capped = targets.capped
    return _SummaryTokens(tokens=tuple(parts), capped=capped)


def _query_summary_parts_for_create(
    root: exp.Create,
    *,
    raw_sql: str,
    config: QuerySummaryConfig,
) -> _SummaryTokens:
    parts: list[str] = ["CREATE"]
    target = _sanitize_summary_target(
        _format_table_for_summary(getattr(root, "this", None)),
        hash_suspicious_targets=config.hash_suspicious_targets,
        hash_target_len=config.hash_target_len,
        hash_target_min_len=config.hash_target_min_len,
    )
    if target:
        parts.append(target)

    capped = False
    nested = getattr(root, "expression", None)
    if isinstance(nested, exp.Expression):
        nested_op = _operation_name_for_root(nested) or "SELECT"
        parts.append(nested_op)
        if config.include_subquery_operations:
            parts.extend(_nested_subquery_operations(nested))
        exclude = {target.lower()} if target else set()
        targets = _query_summary_targets_for_expression(
            nested,
            raw_sql=raw_sql,
            exclude=exclude,
            config=config,
        )
        parts.extend(targets.tokens)
        capped = targets.capped
    return _SummaryTokens(tokens=tuple(parts), capped=capped)


def _operation_name_for_root(root: exp.Expression) -> str | None:
    operation: str | None = None
    for expr_type, name in (
        (exp.Select, "SELECT"),
        (exp.Update, "UPDATE"),
        (exp.Delete, "DELETE"),
        (exp.Insert, "INSERT"),
        (exp.Create, "CREATE"),
        (exp.Drop, "DROP"),
    ):
        if isinstance(root, expr_type):
            operation = name
            break
    if operation is not None:
        return operation
    key = getattr(root, "key", None)
    if isinstance(key, str) and key:
        return key.replace("_", " ").upper()
    return None


def _query_summary_targets_for_expression(
    root: exp.Expression,
    *,
    raw_sql: str,
    config: QuerySummaryConfig,
    exclude: set[str] | None = None,
) -> _SummaryTokens:
    exclude = exclude or set()
    sql_lower = raw_sql.lower()
    max_targets = max(0, config.max_targets)

    tables = extract_table_refs(root)
    formatted: list[tuple[int, str]] = []
    for table in tables:
        raw_key = _format_table_for_summary(table)
        if not raw_key:
            continue
        pos = _best_effort_table_position(sql_lower, raw_key.lower())
        key = _sanitize_summary_target(
            raw_key,
            hash_suspicious_targets=config.hash_suspicious_targets,
            hash_target_len=config.hash_target_len,
            hash_target_min_len=config.hash_target_min_len,
        )
        if not key:
            continue
        key_lower = key.lower()
        if key_lower in exclude:
            continue
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
    capped = False
    if max_targets > 0 and len(out) > max_targets:
        out = out[:max_targets]
        capped = True
    return _SummaryTokens(tokens=tuple(out), capped=capped)


def _nested_subquery_operations(root: exp.Expression) -> list[str]:
    operations: list[str] = []
    for subquery in root.find_all(exp.Subquery):
        if _is_cte_subquery(subquery):
            continue
        inner = getattr(subquery, "this", None)
        if isinstance(inner, exp.Expression):
            op = _operation_name_for_root(inner)
            if op:
                operations.append(op)
    return operations


def _is_cte_subquery(node: exp.Subquery) -> bool:
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, exp.CTE):
            return True
        parent = getattr(parent, "parent", None)
    return False


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


def _sanitize_summary_target(
    target: str | None,
    *,
    hash_suspicious_targets: bool,
    hash_target_len: int,
    hash_target_min_len: int,
) -> str | None:
    if not target:
        return None
    if not hash_suspicious_targets:
        return target
    if target.startswith(_HASHED_TARGET_PREFIX):
        return target
    if _SUSPICIOUS_TARGET_RE.search(target) or len(target) >= hash_target_min_len:
        hashed = _short_target_hash(target, length=hash_target_len)
        return f"{_HASHED_TARGET_PREFIX}{hashed}"
    return target


def _short_target_hash(text: str, *, length: int) -> str:
    trimmed = max(4, min(64, length))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:trimmed]


def _truncate_query_summary_parts(
    parts: list[str],
    *,
    max_len: int,
    emit_ellipsis: bool,
    force_ellipsis: bool,
) -> str:
    cleaned_parts = [token.strip() for token in parts if token.strip()]
    kept: list[str] = []
    truncated = False
    for token in cleaned_parts:
        candidate = [*kept, token]
        if _tokens_len(candidate) > max_len:
            truncated = True
            break
        kept.append(token)

    truncated = truncated or force_ellipsis
    if emit_ellipsis and truncated:
        kept = _append_ellipsis(kept, max_len=max_len)
    return " ".join(kept)


def _tokens_len(tokens: list[str]) -> int:
    if not tokens:
        return 0
    return sum(len(token) for token in tokens) + len(tokens) - 1


def _append_ellipsis(tokens: list[str], *, max_len: int) -> list[str]:
    if not tokens:
        return tokens
    candidate = [*tokens, "..."]
    if _tokens_len(candidate) <= max_len:
        return candidate
    trimmed = tokens[:]
    while trimmed and _tokens_len([*trimmed, "..."]) > max_len:
        trimmed.pop()
    if trimmed:
        trimmed = [*trimmed, "..."]
    return trimmed


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
    return extract_column_lineage_from_ast(root, schema=schema)


def extract_column_lineage_from_ast(
    root: exp.Expression,
    *,
    schema: SchemaMapping | None = None,
) -> dict[str, frozenset[str]]:
    """Extract column-level lineage for a SQLGlot AST.

    Parameters
    ----------
    root
        SQLGlot expression to analyze.
    schema
        Optional schema mapping to improve qualification/lineage accuracy.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping of output column name to upstream column keys (table.column).
    """
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
