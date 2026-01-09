"""SCIP external symbol usage analytics tables."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SCIP_EXTERNAL_SYMBOL_USAGE_TARGET_NAME = "scip_external_symbol_usage"
SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_KEY = "analytics.scip_external_symbol_usage"
SCIP_EXTERNAL_SYMBOL_USAGE_CONTRACT = contract_ref_for_table(
    table_key=SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_KEY,
    target_name=SCIP_EXTERNAL_SYMBOL_USAGE_TARGET_NAME,
    input_name="scip_external_symbol_usage__base",
    required_cols=(),
    clip_column=None,
)


def _normalize_package_value(value: object | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def scip_external_symbol_usage__base(
    env: BuildEnv,
    q__core__scip_external_symbols: InferableTabularInput,
) -> pa.Table:
    """Aggregate external symbol usage by package identity.

    Returns
    -------
    pa.Table
        Reader with aggregated external symbol usage rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    rows = collect_scoped_rows(
        q__core__scip_external_symbols,
        (
            "repo",
            "commit",
            "symbol",
            "package_manager",
            "package_name",
            "package_version",
        ),
        scope=scope,
    )
    if not rows:
        return empty_table_for_table(SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_KEY)
    usage: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        symbol_raw = row.get("symbol")
        if symbol_raw is None:
            continue
        key = (
            _normalize_package_value(row.get("package_manager")),
            _normalize_package_value(row.get("package_name")),
            _normalize_package_value(row.get("package_version")),
        )
        symbols = usage.setdefault(key, set())
        symbols.add(str(symbol_raw))
    if not usage:
        return empty_table_for_table(SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_KEY)
    created_at = datetime.now(tz=UTC)
    output_rows = [
        {
            "repo": env.repo,
            "commit": env.commit,
            "package_manager": package_manager,
            "package_name": package_name,
            "package_version": package_version,
            "symbol_count": len(symbols),
            "created_at": created_at,
        }
        for (package_manager, package_name, package_version), symbols in sorted(usage.items())
    ]
    return finalize_analytics_rows(SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_KEY, output_rows)


_MODULE = sys.modules[__name__]
_SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=SCIP_EXTERNAL_SYMBOL_USAGE_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_SCIP_EXTERNAL_SYMBOL_USAGE_TABLE_TARGET_SPEC)
scip_external_symbol_usage__table = _MODULE.scip_external_symbol_usage__table
scip_external_symbol_usage__table_materializations = (
    _MODULE.scip_external_symbol_usage__table_materializations
)
t__scip_external_symbol_usage = _MODULE.t__scip_external_symbol_usage


__all__ = [
    "scip_external_symbol_usage__base",
    "scip_external_symbol_usage__table",
    "t__scip_external_symbol_usage",
]
