"""Serving snapshot helpers aligned with production pointer formats."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import duckdb
import pyarrow.parquet as pq
from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.build.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from codeintel.config.primitives import BuildPaths
from codeintel.core.config.settings import IcebergSettings
from codeintel.core.hamilton import tags as ht
from codeintel.core.hashing import stable_hash
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.snapshot_properties import (
    SnapshotPropertyInputs,
    snapshot_properties_for_write,
)
from codeintel.core.manifests import ServingSnapshotManifest
from codeintel.core.schemas import table_schema_from_json_obj
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema, normalize_column_type
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.iceberg.migration import IcebergAddFilesRequest, add_files_to_iceberg
from codeintel.storage.serving.search_index import build_search_documents_table
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.gateway import seed_contract_catalog
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tests._helpers.harnesses.serving_harness import ServingTargetHarness


@dataclass(frozen=True, slots=True)
class ServingSnapshot:
    """Serving snapshot paths and identity metadata."""

    serve_dir: Path
    snapshot_root: Path
    snapshot_manifest_path: Path
    db_path: Path
    registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path
    pointer_path: Path
    repo: str
    commit: str
    run_id: str
    iceberg_settings: IcebergSettings


@dataclass(frozen=True, slots=True)
class SnapshotArtifacts:
    """Optional artifacts and callbacks for snapshot generation."""

    views: list[dict[str, object]] | None = None
    tables: list[dict[str, object]] | None = None
    db_setup: Callable[[Path], None] | None = None


@dataclass(frozen=True, slots=True)
class _SnapshotPaths:
    serve_dir: Path
    db_path: Path
    registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path
    pointer_path: Path
    snapshot_root: Path
    snapshot_manifest_path: Path


@dataclass(frozen=True, slots=True)
class _IcebergWriteContext:
    run_id: str
    repo: str
    commit: str
    settings: IcebergSettings


@dataclass(frozen=True)
class ServingSnapshotFactory:
    """Factory for producing serving snapshots with production pointer semantics."""

    tmp_path: Path
    serve_dir: Path | None = None
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit

    def demo_snapshot(
        self,
        *,
        run_id: str = "run-1",
        row_count: int = 3,
        pointer_path: Path | None = None,
        publish: bool = False,
    ) -> ServingSnapshot:
        """Create a demo snapshot matching existing fixture behavior.

        Parameters
        ----------
        run_id
            Run identifier to embed in the pointer.
        row_count
            Number of demo rows to insert.
        pointer_path
            Optional pointer path override.
        publish
            When True, publish via the production snapshot publisher.

        Returns
        -------
        ServingSnapshot
            Snapshot paths and identity metadata.
        """
        artifacts = SnapshotArtifacts(
            views=_demo_views(),
            tables=_demo_tables(),
            db_setup=lambda db_path: _write_demo_db(
                db_path,
                row_count=row_count,
                repo=self.repo,
                commit=self.commit,
            ),
        )
        return self.make_snapshot(
            run_id=run_id,
            pointer_path=pointer_path,
            publish=publish,
            artifacts=artifacts,
        )

    def make_snapshot(
        self,
        *,
        run_id: str = "run-1",
        pointer_path: Path | None = None,
        artifacts: SnapshotArtifacts | None = None,
        publish: bool = False,
        use_production_pointer_model: bool = True,
    ) -> ServingSnapshot:
        """Create a serving snapshot with optional custom artifacts.

        Parameters
        ----------
        run_id
            Run identifier to embed in the pointer.
        pointer_path
            Optional pointer path override.
        artifacts
            Optional snapshot artifacts and seed callbacks.
        publish
            When True, publish via the production snapshot publisher.
        use_production_pointer_model
            When True, write pointer JSON via ServingSnapshotPointer.

        Returns
        -------
        ServingSnapshot
            Snapshot paths and identity metadata.
        """
        paths = _resolve_snapshot_paths(
            pointer_path=pointer_path,
            default_serve_dir=self._default_serve_dir(),
        )

        resolved_artifacts = artifacts or SnapshotArtifacts()
        views = resolved_artifacts.views or _demo_views()
        tables = resolved_artifacts.tables or _demo_tables()
        db_setup = resolved_artifacts.db_setup

        if db_setup is not None:
            db_setup(paths.db_path)
        else:
            _write_demo_db(paths.db_path, row_count=3, repo=self.repo, commit=self.commit)

        _write_registry(paths.registry_path, views=views)
        _write_schema_manifest(paths.schema_manifest_path, tables=tables)
        _write_buildspec(paths.buildspec_path, tables=tables)
        iceberg_settings = _iceberg_settings(paths.serve_dir)
        _write_iceberg_tables(
            paths.db_path,
            serve_dir=paths.serve_dir,
            tables=tables,
            context=_IcebergWriteContext(
                run_id=run_id,
                repo=self.repo,
                commit=self.commit,
                settings=iceberg_settings,
            ),
        )

        snapshot = ServingSnapshot(
            serve_dir=paths.serve_dir,
            snapshot_root=paths.snapshot_root,
            snapshot_manifest_path=paths.snapshot_manifest_path,
            db_path=paths.db_path,
            registry_path=paths.registry_path,
            schema_manifest_path=paths.schema_manifest_path,
            buildspec_path=paths.buildspec_path,
            pointer_path=paths.pointer_path,
            repo=self.repo,
            commit=self.commit,
            run_id=run_id,
            iceberg_settings=iceberg_settings,
        )

        if publish:
            _publish_snapshot(snapshot)
            return _snapshot_from_pointer(
                pointer_path=snapshot.pointer_path,
                serve_dir=snapshot.serve_dir,
            )

        published_at = datetime.now(tz=UTC)
        semantic_layer_version = _semantic_version(
            snapshot.registry_path,
            snapshot.schema_manifest_path,
            snapshot.buildspec_path,
        )
        _write_snapshot_manifest(
            snapshot,
            published_at=published_at,
            semantic_layer_version=semantic_layer_version,
        )
        _write_pointer(
            snapshot,
            use_production_pointer_model=use_production_pointer_model,
            published_at=published_at,
            semantic_layer_version=semantic_layer_version,
        )
        return snapshot

    @staticmethod
    def publish_from_harness(
        harness: ServingTargetHarness,
        *,
        run_id: str = "run-1",
        keep_last: int = 2,
    ) -> ServingSnapshot:
        """Publish a serving snapshot via ServingTargetHarness.

        Parameters
        ----------
        harness
            ServingTargetHarness configured with seeded modules.
        run_id
            Run identifier to publish.
        keep_last
            Number of snapshots to retain.

        Returns
        -------
        ServingSnapshot
            Snapshot paths and identity metadata.
        """
        harness.publish_snapshot(run_id=run_id, keep_last=keep_last)
        serve_dir = harness.harness.ctx.build_paths.build_dir / "serving"
        pointer_path = serve_dir / "current.json"
        return _snapshot_from_pointer(pointer_path=pointer_path, serve_dir=serve_dir)

    def _default_serve_dir(self) -> Path:
        return self.serve_dir or (self.tmp_path / "serve")


def _write_demo_db(db_path: Path, *, row_count: int, repo: str, commit: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        ensure_production_schemas(con)
        con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        con.execute(
            "INSERT INTO docs.demo SELECT i, 'label-' || i::VARCHAR FROM range(1, ?) t(i)",
            [row_count + 1],
        )
        con.execute("CREATE OR REPLACE VIEW docs.v_demo AS SELECT id, label FROM docs.demo")
        module_payload = {
            "module": "pkg.mod",
            "path": "pkg/mod.py",
            "repo": repo,
            "commit": commit,
            "language": "python",
            "tags": None,
            "owners": None,
        }
        module_payload["row_hash"] = stable_hash(module_payload)
        con.execute(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners, row_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                module_payload["module"],
                module_payload["path"],
                module_payload["repo"],
                module_payload["commit"],
                module_payload["language"],
                module_payload["tags"],
                module_payload["owners"],
                module_payload["row_hash"],
            ],
        )
        build_search_documents_table(con)
        _ensure_search_documents(con, repo=repo, commit=commit)
    finally:
        con.close()


def _ensure_search_documents(con: duckdb.DuckDBPyConnection, *, repo: str, commit: str) -> None:
    row = con.execute("SELECT COUNT(*) FROM docs.search_documents").fetchone()
    count = int(row[0]) if row is not None and row[0] is not None else 0
    if count > 0:
        return
    con.execute(
        """
        INSERT INTO docs.search_documents (
            doc_id, kind, name, module, rel_path, text, ref_goid_h128, repo, commit
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "module:demo",
            "module",
            "pkg.mod",
            "pkg.mod",
            "pkg/mod.py",
            "pkg.mod pkg/mod.py",
            None,
            repo,
            commit,
        ],
    )


def _write_registry(path: Path, *, views: list[dict[str, object]]) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_semantic_registry(path=path, views=views)


def _write_schema_manifest(path: Path, *, tables: list[dict[str, object]]) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    enriched = [_ensure_table_schema_hash(dict(table)) for table in tables]
    artifacts.write_schema_manifest(
        path=path,
        tables=enriched,
        view_modules=_demo_view_modules(),
    )


def _write_buildspec(path: Path, *, tables: Iterable[dict[str, object]]) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    datasets = []
    for table in tables:
        table_key = str(table.get("table_key", "")).strip()
        if not table_key:
            continue
        schema_hash = _schema_hash_from_table(table)
        if schema_hash is None:
            msg = f"schema_hash is required for buildspec dataset: {table_key}"
            raise ValueError(msg)
        datasets.append({"table_key": table_key, "schema_hash": schema_hash})
    artifacts.write_buildspec(
        path=path,
        datasets=datasets,
    )


def _iceberg_settings(serve_dir: Path) -> IcebergSettings:
    catalog_dir = serve_dir / "iceberg"
    catalog_path = (catalog_dir / "catalog.duckdb").resolve()
    warehouse_path = (catalog_dir / "warehouse").resolve()
    return IcebergSettings(
        read_enabled=True,
        write_enabled=True,
        catalog_type="sql",
        catalog_uri=f"duckdb:///{catalog_path}",
        catalog_warehouse=str(warehouse_path),
    )


def _write_iceberg_tables(
    db_path: Path,
    *,
    serve_dir: Path,
    tables: Iterable[dict[str, object]],
    context: _IcebergWriteContext,
) -> None:
    warehouse = Path(
        context.settings.catalog_warehouse or (serve_dir / "iceberg" / "warehouse")
    )
    warehouse.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for table in tables:
            table_key = str(table.get("table_key", "")).strip()
            if not table_key:
                continue
            try:
                schema_name, table_name = split_table_key(table_key)
            except ValueError:
                continue
            try:
                relation = con.sql(f'SELECT * FROM "{schema_name}"."{table_name}"')
                arrow_table = relation.fetch_arrow_table()
            except duckdb.Error:
                continue
            if arrow_table is None:
                continue
            parquet_path = _parquet_path_for_table(
                warehouse,
                table_key=table_key,
                run_id=context.run_id,
            )
            pq.write_table(arrow_table, parquet_path)
            snapshot_properties = snapshot_properties_for_write(
                SnapshotPropertyInputs(
                    table_key=table_key,
                    repo=context.repo,
                    commit=context.commit,
                    run_id=context.run_id,
                )
            )
            request = IcebergAddFilesRequest(
                table_key=table_key,
                file_paths=(parquet_path,),
                snapshot_properties=snapshot_properties,
            )
            add_files_to_iceberg(request, settings=context.settings)
            _ensure_snapshot_refs(
                iceberg_settings=context.settings,
                table_key=table_key,
                run_id=context.run_id,
                commit=context.commit,
            )
    finally:
        con.close()


def _parquet_path_for_table(warehouse: Path, *, table_key: str, run_id: str) -> Path:
    schema_name, table_name = split_table_key(table_key)
    table_dir = warehouse / schema_name / table_name
    table_dir.mkdir(parents=True, exist_ok=True)
    return table_dir / f"{run_id}.parquet"


def _ensure_snapshot_refs(
    *,
    iceberg_settings: IcebergSettings,
    table_key: str,
    run_id: str,
    commit: str,
) -> None:
    provider = IcebergCatalogProvider(iceberg_settings)
    try:
        table = provider.load_table(table_key)
    except (RuntimeError, ValueError, KeyError, OSError):
        return
    snapshot = table.current_snapshot()
    if snapshot is None:
        return
    try:
        with table.manage_snapshots() as manager:
            if commit:
                _ensure_tag(table, manager, snapshot.snapshot_id, f"commit/{commit}")
            if run_id:
                _ensure_tag(table, manager, snapshot.snapshot_id, f"run/{run_id}")
    except (RuntimeError, ValueError, KeyError, OSError):
        return


def _ensure_tag(table: object, manager: object, snapshot_id: int, ref_name: str) -> None:
    snapshot_by_name = getattr(table, "snapshot_by_name", None)
    if callable(snapshot_by_name) and snapshot_by_name(ref_name) is not None:
        return
    create_tag = getattr(manager, "create_tag", None)
    if callable(create_tag):
        create_tag(snapshot_id, ref_name)


def _write_snapshot_manifest(
    snapshot: ServingSnapshot,
    *,
    published_at: datetime,
    semantic_layer_version: str,
) -> None:
    manifest = ServingSnapshotManifest(
        run_id=snapshot.run_id,
        repo=snapshot.repo,
        commit=snapshot.commit,
        published_at=published_at.isoformat(),
        db_path=str(snapshot.db_path),
        semantic_registry_path=str(snapshot.registry_path),
        schema_manifest_path=str(snapshot.schema_manifest_path),
        buildspec_path=str(snapshot.buildspec_path),
        semantic_layer_version=semantic_layer_version,
    )
    manifest.write_json(snapshot.snapshot_manifest_path)


def _ensure_table_schema_hash(table: dict[str, object]) -> dict[str, object]:
    schema_hash = _schema_hash_from_table(table)
    if schema_hash is None:
        table_key = table.get("table_key")
        msg = f"schema_hash is required for schema manifest table: {table_key}"
        raise ValueError(msg)
    table["schema_hash"] = schema_hash
    return table


def _schema_hash_from_table(table: Mapping[str, object]) -> str | None:
    raw_hash = table.get("schema_hash")
    if isinstance(raw_hash, str) and raw_hash.strip():
        return raw_hash
    return _schema_hash_for_table_entry(table)


def _schema_hash_for_table_entry(table: Mapping[str, object]) -> str | None:
    schema = table.get("schema")
    name = table.get("name")
    if (
        not isinstance(schema, str)
        or not schema.strip()
        or not isinstance(name, str)
        or not name.strip()
    ):
        table_key = table.get("table_key")
        if not isinstance(table_key, str) or not table_key.strip():
            return None
        try:
            schema, name = split_table_key(table_key)
        except ValueError:
            return None
    columns_raw = table.get("columns")
    if not isinstance(columns_raw, list):
        return None
    columns = _columns_from_raw(columns_raw)
    if not columns:
        return None
    table_schema = TableSchema(schema=schema, name=name, columns=columns)
    return compute_schema_hash(table_schema)


def _table_schema_from_entry(table: Mapping[str, object]) -> TableSchema | None:
    try:
        return table_schema_from_json_obj(table)
    except (TypeError, ValueError):
        return None


def _columns_from_raw(columns_raw: list[object]) -> list[Column] | None:
    columns: list[Column] = []
    for col in columns_raw:
        if isinstance(col, Column):
            columns.append(col)
            continue
        if not isinstance(col, dict):
            return None
        col_name = col.get("name")
        col_type = col.get("type")
        if not isinstance(col_name, str) or not isinstance(col_type, str):
            return None
        try:
            normalized_type = normalize_column_type(col_type)
        except ValueError:
            return None
        description = col.get("description")
        description_str = description if isinstance(description, str) else None
        columns.append(
            Column(
                name=col_name,
                type=normalized_type,
                nullable=bool(col.get("nullable", True)),
                description=description_str,
            )
        )
    return columns or None


def _resolve_snapshot_paths(
    *,
    pointer_path: Path | None,
    default_serve_dir: Path,
) -> _SnapshotPaths:
    serve_dir = pointer_path.parent if pointer_path else default_serve_dir
    serve_dir.mkdir(parents=True, exist_ok=True)
    db_path = serve_dir / "codeintel.duckdb"
    registry_path = serve_dir / "semantic_registry.json"
    schema_manifest_path = serve_dir / "schema_manifest.json"
    buildspec_path = serve_dir / "buildspec.json"
    resolved_pointer_path = pointer_path or serve_dir / "current.json"
    snapshot_root = serve_dir
    snapshot_manifest_path = serve_dir / "snapshot_manifest.json"
    return _SnapshotPaths(
        serve_dir=serve_dir,
        db_path=db_path,
        registry_path=registry_path,
        schema_manifest_path=schema_manifest_path,
        buildspec_path=buildspec_path,
        pointer_path=resolved_pointer_path,
        snapshot_root=snapshot_root,
        snapshot_manifest_path=snapshot_manifest_path,
    )


def _write_pointer(
    snapshot: ServingSnapshot,
    *,
    use_production_pointer_model: bool,
    published_at: datetime,
    semantic_layer_version: str,
) -> None:
    if use_production_pointer_model:
        pointer = ServingSnapshotPointer(
            snapshot_root=snapshot.snapshot_root,
            snapshot_manifest_path=snapshot.snapshot_manifest_path,
            db_path=snapshot.db_path,
            semantic_registry_path=snapshot.registry_path,
            schema_manifest_path=snapshot.schema_manifest_path,
            buildspec_path=snapshot.buildspec_path,
            repo=snapshot.repo,
            commit=snapshot.commit,
            run_id=snapshot.run_id,
            published_at=published_at,
            semantic_layer_version=semantic_layer_version,
        )
        snapshot.pointer_path.write_text(pointer.to_json(), encoding="utf-8")
        return
    payload: dict[str, object] = {
        "snapshot_root": str(snapshot.snapshot_root),
        "snapshot_manifest_path": str(snapshot.snapshot_manifest_path),
        "db_path": str(snapshot.db_path),
        "semantic_registry_path": str(snapshot.registry_path),
        "schema_manifest_path": str(snapshot.schema_manifest_path),
        "buildspec_path": str(snapshot.buildspec_path),
        "repo": snapshot.repo,
        "commit": snapshot.commit,
        "run_id": snapshot.run_id,
        "published_at": published_at.isoformat(),
        "semantic_layer_version": semantic_layer_version,
    }
    snapshot.pointer_path.write_text(_json_dump(payload), encoding="utf-8")


def _semantic_version(registry_path: Path, manifest_path: Path, buildspec_path: Path) -> str:
    hasher = hashlib.sha256()
    for path in (registry_path, manifest_path, buildspec_path):
        if path.is_file():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _publish_snapshot(snapshot: ServingSnapshot) -> None:
    config = StorageConfig(
        db_path=snapshot.db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    gateway = open_gateway(config, seed_contract_catalog=seed_contract_catalog)
    try:
        request = PublishServingSnapshotRequest(
            run_id=snapshot.run_id,
            serve_dir=snapshot.serve_dir,
            semantic_registry_path=snapshot.registry_path,
            schema_manifest_path=snapshot.schema_manifest_path,
            buildspec_path=snapshot.buildspec_path,
            keep_last=2,
        )
        publish_serving_snapshot(gateway=gateway, request=request)
    finally:
        gateway.close()


def _snapshot_from_pointer(*, pointer_path: Path, serve_dir: Path) -> ServingSnapshot:
    pointer = ServingSnapshotPointer.load(pointer_path)
    _ = ServingSnapshotManifest.from_path(pointer.snapshot_manifest_path)
    iceberg_settings = _iceberg_settings(serve_dir)
    return ServingSnapshot(
        serve_dir=serve_dir,
        snapshot_root=pointer.snapshot_root,
        snapshot_manifest_path=pointer.snapshot_manifest_path,
        db_path=pointer.db_path,
        registry_path=pointer.semantic_registry_path,
        schema_manifest_path=pointer.schema_manifest_path,
        buildspec_path=pointer.buildspec_path,
        pointer_path=pointer_path,
        repo=pointer.repo,
        commit=pointer.commit,
        run_id=pointer.run_id,
        iceberg_settings=iceberg_settings,
    )


def _demo_views() -> list[dict[str, object]]:
    return [
        {
            "id": "demo.view",
            "kind": "view",
            "table_key": "docs.v_demo",
            "entity": "demo",
            "grain": "per_row",
            "description": "Demo view",
            "primary_key": ["id"],
            "columns": ["id", "label"],
            "joins": [],
            "defaults": {"limit": 200, "order_by": ["id"]},
            "sensitivity": "internal",
        }
    ]


def _demo_view_modules() -> tuple[ModuleType, ...]:
    module = ModuleType("tests.serving_snapshot_factory.demo_views")

    @h_tag(output_kind=ht.OUTPUT_KIND_VIEW, table_key="docs.v_demo")
    def v_demo() -> exp.Expression:
        return parse_one("SELECT id, label FROM docs.demo", read="duckdb")

    v_demo.__module__ = module.__name__
    module.__dict__["v_demo"] = v_demo
    return (module,)


def _demo_tables() -> list[dict[str, object]]:
    return [
        {
            "schema": "docs",
            "name": "demo",
            "table_key": "docs.demo",
            "primary_key": ["id"],
            "indexes": [],
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "label", "type": "VARCHAR", "nullable": True},
            ],
        }
    ]


def _json_dump(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["ServingSnapshot", "ServingSnapshotFactory"]
