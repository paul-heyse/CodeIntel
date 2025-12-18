"""Manage DuckDB session lifecycle and bootstrapping.

This module is the canonical owner of DuckDB runtime bootstrapping:

- Connection open (writer + reader).
- Env-driven DuckDB connect configuration.
- Extension policy (INSTALL vs LOAD; read-only safety).
- History database attachment.
- Schema application.
- Secret + init SQL setup.
- Optional attach/export/import helpers.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.storage.gateway.extensions import load_extensions_from_env
from codeintel.storage.schema import apply_all_schemas

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection

_INIT_SQL_ENV = "CODEINTEL_DUCKDB_INIT_SQL"
_SECRETS_ENV = "CODEINTEL_DUCKDB_SECRETS"
_FSSPEC_FILESYSTEMS_ENV = "CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS"

DuckDBConnectConfigValue = bool | float | int | list[str] | str
DuckDBConnectConfig = dict[str, DuckDBConnectConfigValue]

_READ_ONLY_DUCKDB_CONFIG_DEFAULTS: DuckDBConnectConfig = {
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
}


@dataclass(frozen=True, slots=True)
class DuckDBSession:
    """Create and manage DuckDB connections for storage operations.

    Parameters
    ----------
    config
        Storage configuration controlling the database path and bootstrap behaviors.
    duckdb_config
        Optional DuckDB client configuration (threads, memory limit, etc.).
    """

    config: StorageConfig
    duckdb_config: DuckDBConnectConfig | None = None

    def open(self) -> DuckDBConnection:
        """Open a new DuckDB connection for this session.

        Returns
        -------
        DuckDBConnection
            Open DuckDB connection.
        """
        con = _open_connection(
            self.config,
            duckdb_config=self._resolve_duckdb_config(),
        )
        load_extensions_from_env(con, allow_install=not self.config.read_only)
        _attach_history_if_needed(con, self.config)
        _apply_schema(con, self.config)
        _bootstrap_duckdb_secrets_from_env(con)
        _register_fsspec_filesystems_from_env()
        _run_init_sql_from_env(con)
        return con

    def open_reader(self) -> DuckDBConnection:
        """Open a new read-only connection to the same database.

        Returns
        -------
        DuckDBConnection
            Read-only DuckDB connection.
        """
        cfg = replace(
            self.config,
            read_only=True,
            apply_schema=False,
        )
        resolved = self._resolve_duckdb_config()
        readonly_duckdb_config = dict(resolved) if resolved else {}
        readonly_duckdb_config.update(_READ_ONLY_DUCKDB_CONFIG_DEFAULTS)
        con = _open_connection(
            cfg,
            duckdb_config=readonly_duckdb_config or None,
        )
        load_extensions_from_env(con, allow_install=False)
        _attach_history_if_needed(con, cfg)
        _bootstrap_duckdb_secrets_from_env(con)
        _register_fsspec_filesystems_from_env()
        _run_init_sql_from_env(con)
        return con

    def connect(self) -> AbstractContextManager[DuckDBConnection]:
        """Return a context manager that yields an open connection.

        Returns
        -------
        AbstractContextManager[DuckDBConnection]
            Context manager that opens and closes the connection.
        """

        class _ConnCtx:
            def __init__(self, session: DuckDBSession) -> None:
                self._session = session
                self._con: DuckDBConnection | None = None

            def __enter__(self) -> DuckDBConnection:
                self._con = self._session.open()
                return self._con

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                if self._con is not None:
                    self._con.close()

        return _ConnCtx(self)

    def _resolve_duckdb_config(self) -> DuckDBConnectConfig | None:
        env_cfg = _duckdb_connect_config_from_env()
        if self.duckdb_config is None:
            return env_cfg if env_cfg else None
        merged: DuckDBConnectConfig = {**env_cfg, **self.duckdb_config}
        return merged if merged else None

    @staticmethod
    def attach_database(con: DuckDBConnection, *, db_path: Path, alias: str) -> None:
        """Attach a DuckDB database file to an existing connection.

        Parameters
        ----------
        con
            Connection to attach onto.
        db_path
            Path to a DuckDB database file.
        alias
            Alias name used to refer to the attached database.
        """
        escaped_path = str(db_path).replace("'", "''")
        escaped_alias = alias.replace('"', '""')
        con.execute(f"ATTACH DATABASE '{escaped_path}' AS \"{escaped_alias}\"")

    @staticmethod
    def export_database(con: DuckDBConnection, *, directory: Path) -> None:
        """Export the current database to a directory via DuckDB EXPORT DATABASE.

        Parameters
        ----------
        con
            Connection to export from.
        directory
            Directory to write the export into.
        """
        directory.mkdir(parents=True, exist_ok=True)
        escaped_dir = str(directory).replace("'", "''")
        con.execute(f"EXPORT DATABASE '{escaped_dir}'")

    @staticmethod
    def import_database(con: DuckDBConnection, *, directory: Path) -> None:
        """Import a database directory via DuckDB IMPORT DATABASE.

        Parameters
        ----------
        con
            Connection to import into.
        directory
            Directory previously created by EXPORT DATABASE.
        """
        escaped_dir = str(directory).replace("'", "''")
        con.execute(f"IMPORT DATABASE '{escaped_dir}'")


__all__ = ["DuckDBConnectConfig", "DuckDBConnectConfigValue", "DuckDBSession"]


def _duckdb_connect_config_from_env() -> DuckDBConnectConfig:
    config: DuckDBConnectConfig = {}

    threads = os.environ.get("CODEINTEL_DUCKDB_THREADS", "").strip()
    if threads:
        config["threads"] = int(threads)

    memory_limit = os.environ.get("CODEINTEL_DUCKDB_MEMORY_LIMIT", "").strip()
    if memory_limit:
        config["memory_limit"] = memory_limit

    temp_directory = os.environ.get("CODEINTEL_DUCKDB_TEMP_DIRECTORY", "").strip()
    if temp_directory:
        config["temp_directory"] = temp_directory

    autoinstall_known_extensions = os.environ.get(
        "CODEINTEL_DUCKDB_AUTOINSTALL_KNOWN_EXTENSIONS", ""
    ).strip()
    if autoinstall_known_extensions:
        config["autoinstall_known_extensions"] = _parse_bool_or_string(autoinstall_known_extensions)

    autoload_known_extensions = os.environ.get(
        "CODEINTEL_DUCKDB_AUTOLOAD_KNOWN_EXTENSIONS", ""
    ).strip()
    if autoload_known_extensions:
        config["autoload_known_extensions"] = _parse_bool_or_string(autoload_known_extensions)

    enable_external_file_cache = os.environ.get(
        "CODEINTEL_DUCKDB_ENABLE_EXTERNAL_FILE_CACHE", ""
    ).strip()
    if enable_external_file_cache:
        config["enable_external_file_cache"] = _parse_bool_or_string(enable_external_file_cache)

    parquet_metadata_cache = os.environ.get("CODEINTEL_DUCKDB_PARQUET_METADATA_CACHE", "").strip()
    if parquet_metadata_cache:
        config["parquet_metadata_cache"] = _parse_int_or_string(parquet_metadata_cache)

    enable_profiling = os.environ.get("CODEINTEL_DUCKDB_ENABLE_PROFILING", "").strip()
    if enable_profiling:
        config["enable_profiling"] = _parse_bool_or_string(enable_profiling)

    profiling_output = os.environ.get("CODEINTEL_DUCKDB_PROFILING_OUTPUT", "").strip()
    if profiling_output:
        config["profiling_output"] = profiling_output

    return config


def _parse_bool_or_string(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value.strip()


def _parse_int_or_string(value: str) -> int | str:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _open_connection(
    config: StorageConfig,
    *,
    duckdb_config: DuckDBConnectConfig | None = None,
) -> DuckDBConnection:
    """
    Open a DuckDB connection using the provided configuration.

    Parameters
    ----------
    config
        Storage configuration controlling path, schema application, and validation.
    duckdb_config
        Optional DuckDB connection configuration (e.g., threads, memory_limit).

    Returns
    -------
    DuckDBConnection
        Live DuckDB connection.
    """
    if not config.read_only and config.db_path != Path(":memory:"):
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return _open_primary_connection(config, duckdb_config=duckdb_config)


def _open_primary_connection(
    config: StorageConfig,
    *,
    duckdb_config: DuckDBConnectConfig | None = None,
) -> DuckDBConnection:
    cfg = duckdb_config
    if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
        con = duckdb.connect(str(Path(":memory:")))
        db_path_str = str(config.db_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')")
        con.execute("USE main_db")
        con.close()
        if cfg is None:
            return duckdb.connect(str(config.db_path), read_only=False)
        return duckdb.connect(str(config.db_path), read_only=False, config=cfg)

    if cfg is None:
        return duckdb.connect(str(config.db_path), read_only=config.read_only)
    return duckdb.connect(str(config.db_path), read_only=config.read_only, config=cfg)


def _attach_history_if_needed(con: DuckDBConnection, config: StorageConfig) -> None:
    if not config.attach_history:
        return
    if config.history_db_path is None:
        message = "attach_history requires history_db_path"
        raise ValueError(message)
    if not config.history_db_path.exists():
        message = f"History database not found: {config.history_db_path}"
        raise FileNotFoundError(message)
    history_path_str = str(config.history_db_path).replace("'", "''")
    con.execute(f"ATTACH DATABASE '{history_path_str}' AS history")


def _apply_schema(con: DuckDBConnection, config: StorageConfig) -> None:
    if config.apply_schema and not config.read_only:
        apply_all_schemas(con)


def _run_init_sql_from_env(con: DuckDBConnection) -> None:
    """Execute optional initialization SQL configured by environment.

    The environment variable `CODEINTEL_DUCKDB_INIT_SQL` can contain either:
    - a JSON array of SQL statements, or
    - a newline-delimited string of SQL statements.

    Raises
    ------
    TypeError
        If the JSON payload is not an array of strings.
    ValueError
        If the JSON payload cannot be decoded.
    """
    raw = os.environ.get(_INIT_SQL_ENV, "").strip()
    if not raw:
        return

    statements: list[str]
    if raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"Invalid {_INIT_SQL_ENV} JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            msg = f"{_INIT_SQL_ENV} must be a JSON array of strings"
            raise TypeError(msg)
        statements = [item.strip() for item in payload if item.strip()]
    else:
        statements = [line.strip() for line in raw.splitlines() if line.strip()]

    for stmt in statements:
        con.execute(stmt)


def _bootstrap_duckdb_secrets_from_env(con: DuckDBConnection) -> None:
    """Create DuckDB secrets configured by environment.

    The environment variable `CODEINTEL_DUCKDB_SECRETS` contains a JSON array of
    secret specs. Each spec is a mapping with the following keys:

    - `name` (str): Secret name (identifier).
    - `type` (str): DuckDB secret type (e.g., "s3").
    - `persistent` (bool, optional): When true, uses CREATE PERSISTENT SECRET.
    - `config` (object): Key/value pairs passed to DuckDB (e.g., KEY_ID, SECRET, REGION).

    Notes
    -----
    Secret values must never be logged. This helper intentionally does not emit
    the generated SQL string in exception messages.

    Raises
    ------
    TypeError
        If the JSON payload is not an array of objects.
    ValueError
        If the JSON payload cannot be decoded.
    """
    raw = os.environ.get(_SECRETS_ENV, "").strip()
    if not raw:
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Invalid {_SECRETS_ENV} JSON: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(payload, list):
        msg = f"{_SECRETS_ENV} must be a JSON array"
        raise TypeError(msg)

    for item in payload:
        if not isinstance(item, dict):
            msg = f"{_SECRETS_ENV} entries must be JSON objects"
            raise TypeError(msg)
        _create_duckdb_secret(con, item)


def _create_duckdb_secret(con: DuckDBConnection, spec: dict[str, object]) -> None:
    name = _require_secret_str(spec, "name")
    secret_type = _require_secret_str(spec, "type")
    persistent = _require_secret_bool(spec, "persistent", default=False)
    config = _require_secret_config(spec)

    _validate_secret_identifier(name, label="name")
    _validate_secret_identifier(secret_type, label="type")

    config_parts = _build_secret_config_parts(secret_type=secret_type, config=config)
    create_kind = "CREATE PERSISTENT SECRET" if persistent else "CREATE SECRET"
    sql = f"{create_kind} {name} ({', '.join(config_parts)})"
    try:
        con.execute(sql)
    except Exception as exc:
        msg = f"Failed to create DuckDB secret {name!r}"
        raise RuntimeError(msg) from exc


def _duckdb_secret_literal(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    msg = f"Unsupported DuckDB secret literal type: {type(value).__name__}"
    raise TypeError(msg)


def _require_secret_str(spec: dict[str, object], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        msg = f"DuckDB secret spec requires non-empty {key!r}"
        raise TypeError(msg)
    return value


def _require_secret_bool(spec: dict[str, object], key: str, *, default: bool) -> bool:
    value = spec.get(key, default)
    if not isinstance(value, bool):
        msg = f"DuckDB secret spec {key!r} must be a boolean when provided"
        raise TypeError(msg)
    return value


def _require_secret_config(spec: dict[str, object]) -> dict[str, object]:
    value = spec.get("config")
    if not isinstance(value, dict) or not value:
        msg = "DuckDB secret spec requires non-empty 'config' mapping"
        raise TypeError(msg)
    return value


def _validate_secret_identifier(value: str, *, label: str) -> None:
    if not value.replace("_", "").isalnum():
        msg = f"DuckDB secret {label!r} must be alphanumeric/underscore"
        raise ValueError(msg)


def _build_secret_config_parts(*, secret_type: str, config: dict[str, object]) -> list[str]:
    parts: list[str] = [f"TYPE {secret_type}"]
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            msg = "DuckDB secret config keys must be non-empty strings"
            raise TypeError(msg)
        _validate_secret_identifier(key, label="config key")
        parts.append(f"{key} {_duckdb_secret_literal(value)}")
    return parts


def _register_fsspec_filesystems_from_env() -> None:
    """Register fsspec filesystems for DuckDB based on environment.

    The environment variable `CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS` may contain:
    - a JSON array of protocol strings (preferred), or
    - a comma-delimited list of protocol strings.

    Examples
    --------
    - `CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS=["gcs","sftp"]`
    - `CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS=gcs,sftp`

    Raises
    ------
    RuntimeError
        If `fsspec` is not installed but filesystem registration is requested.
    TypeError
        If the JSON payload is not an array of strings.
    ValueError
        If the JSON payload cannot be decoded, or if a protocol is invalid.
    """
    raw = os.environ.get(_FSSPEC_FILESYSTEMS_ENV, "").strip()
    if not raw:
        return

    protocols: list[str]
    if raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"Invalid {_FSSPEC_FILESYSTEMS_ENV} JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            msg = f"{_FSSPEC_FILESYSTEMS_ENV} must be a JSON array of strings"
            raise TypeError(msg)
        protocols = [item.strip() for item in payload if item.strip()]
    else:
        protocols = [item.strip() for item in raw.split(",") if item.strip()]

    if not protocols:
        return

    try:
        fsspec = importlib.import_module("fsspec")
    except ImportError as exc:  # pragma: no cover
        msg = f"{_FSSPEC_FILESYSTEMS_ENV} requires the optional dependency 'fsspec'"
        raise RuntimeError(msg) from exc

    for protocol in protocols:
        if not protocol.replace("_", "").isalnum():
            msg = f"Invalid fsspec protocol in {_FSSPEC_FILESYSTEMS_ENV}: {protocol!r}"
            raise ValueError(msg)
        filesystem = fsspec.filesystem(protocol)
        duckdb.register_filesystem(filesystem)
