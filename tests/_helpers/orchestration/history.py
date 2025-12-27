"""Orchestration functions for history and timeseries testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.build.schemas import (
    ContractProvider,
    ContractResolutionMode,
    ContractResolutionSettings,
    get_contract_provider,
)
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.gateway import seed_contract_catalog

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs.history_config import SnapshotSpec

_FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
_MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
_FUNCTION_HISTORY_TABLE_KEY = "analytics.function_history"


@dataclass(slots=True)
class _ContractCache:
    contracts_by_table: dict[str, DatasetContract] | None = None


_DEFAULT_CONTRACT_CACHE = _ContractCache()
_DEFAULT_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {}


def _contracts_by_table(
    contract_provider: ContractProvider | None = None,
) -> dict[str, DatasetContract]:
    if contract_provider is None:
        if _DEFAULT_CONTRACT_CACHE.contracts_by_table is None:
            _DEFAULT_CONTRACT_CACHE.contracts_by_table = dict(
                get_contract_provider(
                    settings=ContractResolutionSettings(mode=ContractResolutionMode.FULL)
                ).iter_contracts_by_table_key()
            )
        return _DEFAULT_CONTRACT_CACHE.contracts_by_table
    return dict(contract_provider.iter_contracts_by_table_key())


def contracts_cache_initialized() -> bool:
    """Return True if the default contract cache has been populated.

    Returns
    -------
    bool
        True when the default contract cache has been populated.
    """
    return _DEFAULT_CONTRACT_CACHE.contracts_by_table is not None


def _columns_for_table_key(
    table_key: str,
    *,
    contract_provider: ContractProvider | None = None,
) -> tuple[str, ...]:
    if contract_provider is None and table_key in _DEFAULT_COLUMNS_BY_TABLE:
        return _DEFAULT_COLUMNS_BY_TABLE[table_key]
    contract = _contracts_by_table(contract_provider).get(table_key)
    schema = getattr(contract, "schema", None)
    if schema is None:
        return ()
    columns = tuple(schema.column_names())
    if contract_provider is None:
        _DEFAULT_COLUMNS_BY_TABLE[table_key] = columns
    return columns


def _require_columns(
    table_key: str,
    *,
    contract_provider: ContractProvider | None = None,
) -> tuple[str, ...]:
    columns = _columns_for_table_key(table_key, contract_provider=contract_provider)
    if not columns:
        msg = f"Missing schema columns for {table_key}"
        raise ValueError(msg)
    return columns


def _function_profile_row(
    spec: SnapshotSpec,
    *,
    contract_provider: ContractProvider | None = None,
) -> tuple[object, ...]:
    """Build a function profile row from a snapshot spec.

    Returns
    -------
    tuple[object, ...]
        Row tuple for function_profile table.
    """
    columns = _require_columns(_FUNCTION_PROFILE_TABLE_KEY, contract_provider=contract_provider)
    defaults: dict[str, object | None] = dict.fromkeys(columns, None)
    defaults.update(
        {
            "function_goid_h128": spec.goid,
            "urn": f"goid:{spec.repo}/{spec.rel_path}#{spec.qualname}",
            "repo": spec.repo,
            "commit": spec.commit,
            "rel_path": spec.rel_path,
            "module": spec.module,
            "language": "python",
            "kind": "function",
            "qualname": spec.qualname,
            "start_line": 1,
            "end_line": spec.loc,
            "loc": spec.loc,
            "cyclomatic_complexity": spec.cyclomatic_complexity,
            "coverage_ratio": spec.coverage_ratio,
            "risk_score": spec.risk_score,
            "risk_level": spec.risk_level,
        }
    )
    return tuple(defaults[col] for col in columns)


def _module_profile_row(
    spec: SnapshotSpec,
    *,
    contract_provider: ContractProvider | None = None,
) -> tuple[object, ...]:
    """Build a module profile row from a snapshot spec.

    Returns
    -------
    tuple[object, ...]
        Row tuple for module_profile table.
    """
    columns = _require_columns(_MODULE_PROFILE_TABLE_KEY, contract_provider=contract_provider)
    defaults: dict[str, object | None] = dict.fromkeys(columns, None)
    defaults.update(
        {
            "repo": spec.repo,
            "commit": spec.commit,
            "module": spec.module,
            "path": spec.rel_path,
            "language": "python",
            "file_count": 1,
            "total_loc": spec.loc,
            "module_coverage_ratio": spec.coverage_ratio,
            "max_risk_score": spec.risk_score,
            "avg_risk_score": spec.risk_score,
        }
    )
    return tuple(defaults[col] for col in columns)


def create_snapshot_db(
    base_dir: Path,
    spec: SnapshotSpec,
    *,
    contract_provider: ContractProvider | None = None,
) -> Path:
    """
    Create a minimal snapshot DuckDB with function/module profile rows.

    Parameters
    ----------
    base_dir
        Directory to place the database file.
    spec
        Snapshot specification.
    contract_provider
        Optional contract provider override for deterministic tests.

    Returns
    -------
    Path
        Path to the created DuckDB file.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / f"codeintel-{spec.commit}.duckdb"
    cfg = StorageConfig.for_ingest(db_path)
    gateway = open_gateway(cfg, seed_contract_catalog=seed_contract_catalog)
    con = gateway.con
    apply_all_schemas(con)
    fp_columns = _require_columns(_FUNCTION_PROFILE_TABLE_KEY, contract_provider=contract_provider)
    mp_columns = _require_columns(_MODULE_PROFILE_TABLE_KEY, contract_provider=contract_provider)
    fp_df = pd.DataFrame(
        [_function_profile_row(spec, contract_provider=contract_provider)],
        columns=pd.Index(fp_columns),
    )
    mp_df = pd.DataFrame(
        [_module_profile_row(spec, contract_provider=contract_provider)],
        columns=pd.Index(mp_columns),
    )
    con.register("fp_df", fp_df)
    con.register("mp_df", mp_df)
    con.execute("INSERT INTO analytics.function_profile BY NAME SELECT * FROM fp_df")
    con.execute("INSERT INTO analytics.module_profile BY NAME SELECT * FROM mp_df")
    gateway.close()
    return db_path


def insert_function_history_row(
    gateway: StorageGateway,
    spec: SnapshotSpec,
    *,
    contract_provider: ContractProvider | None = None,
) -> None:
    """Insert a minimal function_history row for validation helpers."""
    con = gateway.con
    fh_columns = _require_columns(_FUNCTION_HISTORY_TABLE_KEY, contract_provider=contract_provider)
    defaults: dict[str, object | None] = dict.fromkeys(fh_columns, None)
    now = datetime.now(tz=UTC)
    defaults.update(
        {
            "repo": spec.repo,
            "commit": spec.commit,
            "function_goid_h128": spec.goid,
            "urn": f"goid:{spec.repo}/{spec.rel_path}#{spec.qualname}",
            "rel_path": spec.rel_path,
            "module": spec.module,
            "qualname": spec.qualname,
            "created_in_commit": spec.commit,
            "created_at": now,
            "last_modified_commit": spec.commit,
            "last_modified_at": now,
            "age_days": 0,
            "commit_count": 1,
            "author_count": 1,
            "lines_added": 3,
            "lines_deleted": 0,
            "churn_score": 0.3,
            "stability_bucket": "new_hot",
            "created_at_row": now,
        }
    )
    fh_df = pd.DataFrame(
        [tuple(defaults[col] for col in fh_columns)],
        columns=pd.Index(fh_columns),
    )
    con.register("fh_df", fh_df)
    con.execute("INSERT INTO analytics.function_history BY NAME SELECT * FROM fh_df")


__all__ = [
    "contracts_cache_initialized",
    "create_snapshot_db",
    "insert_function_history_row",
]
