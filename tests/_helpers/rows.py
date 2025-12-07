"""Factory helpers for commonly used analytics rows."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from codeintel.graphs.catalog import FunctionMeta
from tests._helpers.builders import FunctionMetricsRow, ModuleRow
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO


def function_meta(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    line_span: tuple[int, int] = (1, 1),
) -> FunctionMeta:
    """Build a FunctionMeta with consistent URN formatting.

    Returns
    -------
    FunctionMeta
        Catalog entry with normalized URN and line span.
    """
    repo, commit = snapshot
    start_line, end_line = line_span
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    return FunctionMeta(
        goid=goid,
        urn=urn,
        rel_path=rel_path,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
    )


def module_row(
    *,
    module: str,
    path: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
) -> ModuleRow:
    """Create a ModuleRow for analytics modules.

    Returns
    -------
    ModuleRow
        Row ready for insertion into analytics.modules.
    """
    repo, commit = snapshot
    return ModuleRow(module=module, path=path, repo=repo, commit=commit)


def function_metrics_row(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    metrics: Mapping[str, int | str | bool | datetime | float | None] | None = None,
) -> FunctionMetricsRow:
    """Create a FunctionMetricsRow with sensible defaults and override support.

    Returns
    -------
    FunctionMetricsRow
        Row with defaulted metrics for analytics.function_metrics.
    """
    repo, commit = snapshot
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    defaults: dict[str, object] = {
        "language": "python",
        "kind": "function",
        "start_line": 1,
        "end_line": 1,
        "loc": 2,
        "logical_loc": 2,
        "param_count": 0,
        "positional_params": 0,
        "keyword_only_params": 0,
        "has_varargs": False,
        "has_varkw": False,
        "is_async": False,
        "is_generator": False,
        "return_count": 0,
        "yield_count": 0,
        "raise_count": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 1,
        "stmt_count": 2,
        "decorator_count": 0,
        "has_docstring": False,
        "complexity_bucket": "low",
        "created_at": datetime.now(tz=UTC),
    }
    if metrics:
        defaults.update(metrics)

    def _as_int(key: str) -> int:
        value = defaults.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float | str):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        default_value = defaults[key]
        return default_value if isinstance(default_value, int) else int(default_value)  # type: ignore[arg-type]

    def _as_bool(key: str) -> bool:
        value = defaults.get(key)
        return bool(value) if value is not None else bool(defaults[key])  # type: ignore[arg-type]

    def _as_str(key: str) -> str:
        value = defaults.get(key)
        if value is None:
            return str(defaults[key])
        return value if isinstance(value, str) else str(value)

    created_at_value = defaults.get("created_at")
    created_at = (
        created_at_value if isinstance(created_at_value, datetime) else datetime.now(tz=UTC)
    )
    return FunctionMetricsRow(
        function_goid_h128=goid,
        urn=urn,
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        language=_as_str("language"),
        kind=_as_str("kind"),
        qualname=qualname,
        start_line=_as_int("start_line"),
        end_line=_as_int("end_line"),
        loc=_as_int("loc"),
        logical_loc=_as_int("logical_loc"),
        param_count=_as_int("param_count"),
        positional_params=_as_int("positional_params"),
        keyword_only_params=_as_int("keyword_only_params"),
        has_varargs=_as_bool("has_varargs"),
        has_varkw=_as_bool("has_varkw"),
        is_async=_as_bool("is_async"),
        is_generator=_as_bool("is_generator"),
        return_count=_as_int("return_count"),
        yield_count=_as_int("yield_count"),
        raise_count=_as_int("raise_count"),
        cyclomatic_complexity=_as_int("cyclomatic_complexity"),
        max_nesting_depth=_as_int("max_nesting_depth"),
        stmt_count=_as_int("stmt_count"),
        decorator_count=_as_int("decorator_count"),
        has_docstring=_as_bool("has_docstring"),
        complexity_bucket=_as_str("complexity_bucket"),
        created_at=created_at,
    )


__all__ = ["function_meta", "function_metrics_row", "module_row"]
