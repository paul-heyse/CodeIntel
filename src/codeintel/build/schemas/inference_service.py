"""Schema inference service for native Hamilton outputs."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args, get_origin

import hamilton.driver as h_driver

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.providers import create_default_providers
from codeintel.build.run_context import BuildRunContext
from codeintel.build.schemas.infer_duckdb import infer_table_schema_from_relation
from codeintel.build.schemas.seed_harness import MiniSeedHarness
from codeintel.build.tabular.duckdb_relation import coerce_to_relation
from codeintel.build.tabular.types import TabularInput
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.gateway import open_inference_gateway

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = [
    "HamiltonSchemaProvider",
    "SchemaInferenceService",
    "get_schema_inference_service",
    "infer_schema_for_table_key",
    "infer_table_schemas",
    "inferable_native_table_keys",
]


@dataclass(frozen=True)
class _ComputeInferenceJob:
    target_name: str
    compute_name: str
    exec_name: str
    table_key: str
    qparams: frozenset[str]
    requires_env: bool
    requires_catalog: bool


@dataclass(frozen=True, slots=True)
class _InferenceContext:
    driver: h_driver.Driver
    catalog: DagCatalog


def _looks_inferable_compute(fn: Callable[..., object]) -> bool:
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    if return_annotation is inspect.Signature.empty:
        return False
    return _is_tabular_annotation(return_annotation)


def _is_tabular_annotation(annotation: object) -> bool:
    tabular_types = tuple(get_args(TabularInput))
    if annotation in tabular_types:
        return True
    if isinstance(annotation, type) and issubclass(annotation, tabular_types):
        return True
    origin = get_origin(annotation)
    if origin in {types.UnionType, typing.Union}:
        return any(
            _is_tabular_annotation(arg) for arg in get_args(annotation) if arg is not type(None)
        )
    if isinstance(annotation, str):
        return any(
            token in annotation
            for token in (
                "DuckDBPyRelation",
                "DuckDBRelation",
                "TabularInput",
                "pa.Table",
                "pyarrow.Table",
                "pa.RecordBatchReader",
                "pyarrow.RecordBatchReader",
                "pl.DataFrame",
                "pl.LazyFrame",
                "polars.DataFrame",
                "polars.LazyFrame",
            )
        )
    return any(
        token in str(annotation) for token in ("DuckDBPyRelation", "DuckDBRelation", "TabularInput")
    )


@contextmanager
def _schema_inference_gateway(
    *,
    schema_provider: SchemaProvider,
) -> Iterator[StorageGateway]:
    gateway = open_inference_gateway(schema_provider=schema_provider)
    try:
        yield gateway
    finally:
        gateway.close()


@lru_cache(maxsize=1)
def _schema_inference_providers() -> Providers:
    return create_default_providers(ToolsConfig.default())


def _output_data_node(
    *,
    context: _InferenceContext,
    table_key: str,
) -> str | None:
    output = context.catalog.table_outputs.get(table_key)
    if output is None:
        return None
    saver_node = context.driver.graph.nodes.get(output.saver_node)
    if saver_node is None:
        return None
    tabular_deps = [dep.name for dep in saver_node.dependencies if _is_tabular_annotation(dep.type)]
    if len(tabular_deps) != 1:
        return None
    return tabular_deps[0]


def _resolve_inference_job(
    *,
    context: _InferenceContext,
    table_key: str,
) -> _ComputeInferenceJob:
    output = context.catalog.table_outputs.get(table_key)
    if output is None:
        msg = f"Unknown table_key (no producing target): {table_key}"
        raise KeyError(msg)

    compute_name = _output_data_node(context=context, table_key=table_key)
    if compute_name is None:
        msg = f"Table {table_key} is not inferable from any tabular output node"
        raise ValueError(msg)

    node = context.driver.graph.nodes.get(compute_name)
    if node is None or not node.originating_functions:
        msg = f"Missing compute node for {table_key}: {compute_name}"
        raise ValueError(msg)

    compute_fn_obj = node.originating_functions[0]
    if not isinstance(compute_fn_obj, Callable):
        msg = f"Compute node {compute_name} for {table_key} is not callable"
        raise TypeError(msg)
    compute_fn: Callable[..., object] = compute_fn_obj
    if not _looks_inferable_compute(compute_fn):
        msg = f"Compute node {compute_name} for {table_key} is not inferable"
        raise ValueError(msg)

    qparams, requires_env, requires_catalog = _inference_requirements(
        context=context,
        compute_name=compute_name,
    )
    exec_name = _compute_node_for_inference(context, compute_name=compute_name)
    return _ComputeInferenceJob(
        target_name=output.producer_target,
        compute_name=compute_name,
        exec_name=exec_name,
        table_key=table_key,
        qparams=frozenset(qparams),
        requires_env=requires_env,
        requires_catalog=requires_catalog,
    )


def _compute_node_for_inference(context: _InferenceContext, *, compute_name: str) -> str:
    raw_name = f"{compute_name}_raw"
    return raw_name if raw_name in context.driver.graph.nodes else compute_name


def _inference_requirements(
    *,
    context: _InferenceContext,
    compute_name: str,
) -> tuple[set[str], bool, bool]:
    effective_compute_name = _compute_node_for_inference(context, compute_name=compute_name)
    node = context.driver.graph.nodes.get(effective_compute_name)
    if node is None:
        msg = f"Compute node not found in Hamilton DAG: {effective_compute_name}"
        raise ValueError(msg)

    qparams: set[str] = set()
    requires_env = False
    requires_catalog = False
    visited: set[str] = set()
    stack = list(node.dependencies)

    while stack:
        dep = stack.pop()
        if dep.name in visited:
            continue
        visited.add(dep.name)

        if dep.name == "env":
            requires_env = True
            continue

        if dep.name == "catalog":
            requires_catalog = True
            continue

        if dep.name.startswith("q__"):
            qparams.add(dep.name)
            continue

        if dep.user_defined:
            msg = (
                f"Compute node {compute_name} depends on unsupported input {dep.name}; "
                "schema inference supports only env, catalog, and q__ inputs."
            )
            raise ValueError(msg)

        if dep.name.startswith("t__") and not dep.name.endswith("__compute"):
            msg = (
                f"Compute node {compute_name} depends on target node {dep.name}; "
                "schema inference requires q__-driven compute graphs without target execution."
            )
            raise ValueError(msg)

        tags = dep.tags if isinstance(dep.tags, dict) else {}
        if tags.get("hamilton.data_saver") is True:
            msg = (
                f"Compute node {compute_name} depends on data_saver node {dep.name}; "
                "schema inference requires compute-only graphs."
            )
            raise ValueError(msg)

        stack.extend(dep.dependencies)

    return qparams, requires_env, requires_catalog


def _default_build_settings() -> BuildSettings:
    try:
        engine_version = version("codeintel")
    except PackageNotFoundError:
        engine_version = "unknown"
    return BuildSettings(
        engine_version=engine_version,
        export_audit=ExportAuditSettings(),
    )


def _inference_env(*, gateway: StorageGateway, force_targets: frozenset[str]) -> BuildEnv:
    snapshot = SnapshotRef.from_args(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=Path.cwd(),
    )
    settings = _default_build_settings()
    context = BuildRunContext(
        snapshot=snapshot,
        gateway=gateway,
        paths=BuildPaths.from_repo_root(snapshot.repo_root),
        providers=_schema_inference_providers(),
        config=BuildConfig.empty(),
        settings=settings,
        execution_settings=HamiltonExecutionSettings(),
        execution_options=BuildExecutionOptions(profile="schema_inference"),
        force_targets=force_targets,
    )
    return context.build_env(load_catalogs=False, load_schema_service=False)


def _infer_table_schema_for_compute(
    *,
    context: _InferenceContext,
    declared_provider: SchemaProvider,
    job: _ComputeInferenceJob,
) -> TableSchema:
    with _schema_inference_gateway(schema_provider=declared_provider) as gateway:
        harness = MiniSeedHarness(gateway=gateway, schema_provider=declared_provider)
        overrides: dict[str, object] = dict(harness.build_inputs(set(job.qparams)))
        inputs: dict[str, object] = {}
        if job.requires_env:
            inputs["env"] = _inference_env(
                gateway=gateway,
                force_targets=frozenset({job.target_name}),
            )
        if job.requires_catalog:
            inputs["catalog"] = context.catalog

        out = context.driver.execute([job.exec_name], inputs=inputs, overrides=overrides)
        expr_obj = out[job.exec_name]
        if expr_obj is None:
            msg = f"{job.exec_name} returned None; expected tabular output"
            raise TypeError(msg)
        relation = coerce_to_relation(
            gateway.con,
            cast("TabularInput", expr_obj),
            name_hint=job.table_key,
        )
        return infer_table_schema_from_relation(relation=relation, table_key=job.table_key)


def infer_schema_for_table_key(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    table_key: str,
    declared_provider: SchemaProvider,
) -> TableSchema:
    """Infer schema for a single output table produced by a native compute node.

    Parameters
    ----------
    driver
        Hamilton driver used to execute compute nodes.
    catalog
        DAG catalog defining outputs for each build target.
    table_key
        Output table key to infer (schema.table).
    declared_provider
        Provider used to seed upstream input tables.

    Returns
    -------
    TableSchema
        Inferred schema for the target's compute output.

    Raises
    ------
    KeyError
        If no build target produces the specified table_key.
    TypeError
        If the compute node does not return a tabular output.
    ValueError
        If the table_key is not inferable from any native compute node.
    """
    context = _InferenceContext(driver=driver, catalog=catalog)

    try:
        job = _resolve_inference_job(context=context, table_key=table_key)
        return _infer_table_schema_for_compute(
            context=context,
            declared_provider=declared_provider,
            job=job,
        )
    except KeyError as exc:
        msg = f"Failed to infer schema for {table_key}: {exc}"
        raise KeyError(msg) from exc
    except TypeError as exc:
        msg = f"Failed to infer schema for {table_key}: {exc}"
        raise TypeError(msg) from exc
    except ValueError as exc:
        msg = f"Failed to infer schema for {table_key}: {exc}"
        raise ValueError(msg) from exc


Inferer = Callable[[str], TableSchema]


@dataclass
class HamiltonSchemaProvider(SchemaProvider):
    """SchemaProvider that prefers inferred schemas for selected table keys."""

    declared: SchemaProvider
    inferer: Inferer
    inferable_table_keys: frozenset[str]
    fallback_to_declared_on_error: bool = True
    _cache: dict[str, TableSchema] = field(default_factory=dict)

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return schema for table_key, preferring inference when enabled.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved schema, or None when unknown.
        """
        cached = self._cache.get(table_key)
        if cached is not None:
            return cached

        if table_key in self.inferable_table_keys:
            try:
                inferred = self.inferer(table_key)
            except Exception:
                if not self.fallback_to_declared_on_error:
                    raise
                return self.declared.get_table_schema(table_key)
            self._cache[table_key] = inferred
            return inferred

        return self.declared.get_table_schema(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Resolved schema.

        Raises
        ------
        KeyError
            If table_key is unknown to both inferred and declared providers.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> tuple[TableSchema, ...]:
        """Return all schemas known to this provider.

        Returns
        -------
        tuple[TableSchema, ...]
            Deterministic tuple of schemas keyed by table_key.
        """
        seen = {schema.table_key: schema for schema in self.declared.iter_table_schemas()}
        seen.update(self._cache)
        return tuple(seen[key] for key in sorted(seen))

    def prefill_cache(self, schemas: Mapping[str, TableSchema]) -> None:
        """Prefill the inference cache with known schemas.

        Parameters
        ----------
        schemas
            Mapping of table_key to schema to seed into the cache.
        """
        self._cache.update(schemas)


def inferable_native_table_keys(*, driver: h_driver.Driver, catalog: DagCatalog) -> frozenset[str]:
    """Return output table keys that appear inferable from native compute nodes.

    Parameters
    ----------
    driver
        Hamilton driver used to inspect native compute nodes.
    catalog
        DAG catalog defining outputs for each build target.

    Returns
    -------
    frozenset[str]
        Output table keys inferred from tabular compute nodes, including
        relation-first outputs with no q__ dependencies.
    """
    context = _InferenceContext(driver=driver, catalog=catalog)
    inferable: set[str] = set()
    for table_key in catalog.table_outputs:
        try:
            _resolve_inference_job(context=context, table_key=table_key)
        except (KeyError, ValueError):
            continue
        inferable.add(table_key)
    return frozenset(inferable)


def _build_inference_jobs(
    *,
    context: _InferenceContext,
    table_keys: list[str],
) -> list[_ComputeInferenceJob]:
    return [
        _resolve_inference_job(context=context, table_key=table_key) for table_key in table_keys
    ]


def _union_qparams(jobs: Iterable[_ComputeInferenceJob]) -> frozenset[str]:
    return frozenset({name for job in jobs for name in job.qparams})


def _infer_job_schema(
    *,
    context: _InferenceContext,
    job: _ComputeInferenceJob,
    base_overrides: Mapping[str, object],
    env: BuildEnv,
    con: DuckDBConnection,
) -> TableSchema:
    inputs: dict[str, object] = {}
    overrides = dict(base_overrides)
    if job.requires_env:
        inputs["env"] = env
    if job.requires_catalog:
        inputs["catalog"] = context.catalog

    out = context.driver.execute([job.exec_name], inputs=inputs, overrides=overrides)
    expr_obj = out[job.exec_name]
    if expr_obj is None:
        msg = f"{job.exec_name} returned None; expected tabular output"
        raise TypeError(msg)

    relation = coerce_to_relation(
        con,
        cast("TabularInput", expr_obj),
        name_hint=job.table_key,
    )
    return infer_table_schema_from_relation(relation=relation, table_key=job.table_key)


def infer_table_schemas(
    table_keys: Iterable[str],
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    declared_provider: SchemaProvider,
) -> dict[str, TableSchema]:
    """Infer schemas for multiple output tables in a single session.

    Parameters
    ----------
    table_keys
        Output table keys to infer (schema.table).
    driver
        Hamilton driver used to execute compute nodes.
    catalog
        DAG catalog defining outputs for each build target.
    declared_provider
        Provider used to seed upstream input tables.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table_key to inferred TableSchema.
    """
    unique_keys = sorted(set(table_keys))
    if not unique_keys:
        return {}

    context = _InferenceContext(driver=driver, catalog=catalog)
    jobs = _build_inference_jobs(context=context, table_keys=unique_keys)
    union_qparams = _union_qparams(jobs)

    with _schema_inference_gateway(schema_provider=declared_provider) as gateway:
        harness = MiniSeedHarness(gateway=gateway, schema_provider=declared_provider)
        base_overrides: dict[str, object] = dict(harness.build_inputs(set(union_qparams)))
        env = _inference_env(
            gateway=gateway,
            force_targets=frozenset({job.target_name for job in jobs}),
        )

        inferred: dict[str, TableSchema] = {}
        for job in jobs:
            inferred[job.table_key] = _infer_job_schema(
                context=context,
                job=job,
                base_overrides=base_overrides,
                env=env,
                con=gateway.con,
            )

    return inferred


@dataclass(frozen=True, slots=True)
class SchemaInferenceService:
    """Service for schema inference using native Hamilton compute nodes."""

    driver: h_driver.Driver
    catalog: DagCatalog

    def infer_table_schema(
        self, table_key: str, *, declared_provider: SchemaProvider
    ) -> TableSchema:
        """Infer schema for a single table key.

        Parameters
        ----------
        table_key
            Output table key to infer (schema.table).
        declared_provider
            Provider used to seed upstream input tables.

        Returns
        -------
        TableSchema
            Inferred table schema.
        """
        return infer_schema_for_table_key(
            driver=self.driver,
            catalog=self.catalog,
            table_key=table_key,
            declared_provider=declared_provider,
        )

    def infer_table_schemas(
        self,
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
    ) -> dict[str, TableSchema]:
        """Infer schemas for multiple table keys.

        Parameters
        ----------
        table_keys
            Output table keys to infer (schema.table).
        declared_provider
            Provider used to seed upstream input tables.

        Returns
        -------
        dict[str, TableSchema]
            Mapping of table_key to inferred TableSchema.
        """
        return infer_table_schemas(
            table_keys,
            driver=self.driver,
            catalog=self.catalog,
            declared_provider=declared_provider,
        )

    def inferable_table_keys(self) -> frozenset[str]:
        """Return inferable table keys for the bound DAG catalog.

        Returns
        -------
        frozenset[str]
            Output table keys inferred from native compute nodes.
        """
        return inferable_native_table_keys(driver=self.driver, catalog=self.catalog)


def get_schema_inference_service(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
) -> SchemaInferenceService:
    """Return a runtime-bound SchemaInferenceService.

    Parameters
    ----------
    driver
        Hamilton driver used for inference execution.
    catalog
        DAG catalog defining outputs for each build target.

    Returns
    -------
    SchemaInferenceService
        Service bound to the provided driver and catalog.
    """
    return SchemaInferenceService(driver=driver, catalog=catalog)
