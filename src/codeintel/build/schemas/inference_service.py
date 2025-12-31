"""Schema inference service for native Hamilton outputs."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard, cast, get_args, get_origin

import hamilton.driver as h_driver
import polars as pl
import pyarrow as pa

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.providers import create_default_providers
from codeintel.build.run_context import BuildRunContext
from codeintel.build.schemas.observations import (
    SchemaObservationAccumulator,
    SchemaObservationBundle,
    SchemaObservationInputs,
    merge_table_schema_hints,
    observe_batches,
    schema_hints_from_tag_sets,
    table_schema_from_tag_sets,
)
from codeintel.build.schemas.seed_harness import DatasetSeedHarness, SeedDatasetConfig
from codeintel.build.tabular.types import InferableTabularInput, TabularInput
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
)
from codeintel.core.hamilton import tags as hamilton_tags
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.gateway import open_inference_gateway

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hamilton.node import Node

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.build.schemas.observations import SchemaHints
    from codeintel.build.schemas.seed_harness import SchemaObservationProvider
    from codeintel.core.schemas.arrow_gen import ExtrasPolicy
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "HamiltonSchemaProvider",
    "InferabilityRecord",
    "SchemaInferenceService",
    "SchemaObservationAccumulator",
    "SchemaObservationBundle",
    "SchemaObservationContext",
    "SeedDatasetConfig",
    "get_schema_inference_service",
    "infer_schema_for_table_key",
    "infer_table_schemas",
    "inferability_inventory",
    "inferable_native_table_keys",
    "observe_schema_from_batches",
    "observe_schema_from_reader",
    "table_schema_from_tabular",
]

_SCHEMA_OUTPUT_TAG = "hamilton.internal.schema_output"
_INFERABLE_RUNTIME_TYPES: tuple[type[object], ...] = (
    pa.RecordBatchReader,
    pa.Table,
    pl.DataFrame,
    pl.LazyFrame,
)


@dataclass(frozen=True)
class _ComputeInferenceJob:
    target_name: str
    compute_name: str
    exec_name: str
    table_key: str
    qparams: frozenset[str]
    dataset_refs: tuple[tuple[str, str], ...]
    requires_env: bool
    requires_catalog: bool


@dataclass(frozen=True, slots=True)
class _InferenceContext:
    driver: h_driver.Driver
    catalog: DagCatalog


@dataclass(frozen=True, slots=True)
class _InferenceRequirementUpdate:
    requires_env: bool = False
    requires_catalog: bool = False
    qparam: str | None = None
    dataset_ref: tuple[str, str] | None = None
    skip_children: bool = False


@dataclass(frozen=True, slots=True)
class InferabilityRecord:
    """Describe inferability metadata for a DAG-produced table."""

    table_key: str
    status: Literal["inferable", "non_inferable"]
    target_name: str
    saver_node: str
    sink: str
    compute_node: str | None
    reason: str | None
    qparams: tuple[str, ...] | None
    requires_env: bool | None
    requires_catalog: bool | None


def _looks_inferable_compute(fn: Callable[..., object]) -> bool:
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    if return_annotation is inspect.Signature.empty:
        return False
    return _is_tabular_annotation(return_annotation)


def _is_record_batch_arg(annotation: object) -> bool:
    return annotation is pa.RecordBatch


def _is_batch_iterable_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return False
    if isinstance(origin, type):
        try:
            if not issubclass(origin, Iterable):
                return False
        except TypeError:
            return False
    else:
        return False
    args = get_args(annotation)
    return any(_is_record_batch_arg(arg) for arg in args)


def _is_arrow_table(value: object) -> TypeGuard[pa.Table]:
    return isinstance(value, pa.Table)


def _is_record_batch_reader(value: object) -> TypeGuard[pa.RecordBatchReader]:
    return isinstance(value, pa.RecordBatchReader)


def _is_tabular_annotation(annotation: object) -> bool:
    if annotation in _INFERABLE_RUNTIME_TYPES or annotation in {
        InferableTabularInput,
        TabularInput,
    }:
        return True
    if isinstance(annotation, type) and issubclass(annotation, _INFERABLE_RUNTIME_TYPES):
        return True
    origin = get_origin(annotation)
    if origin in {types.UnionType, typing.Union}:
        return any(
            _is_tabular_annotation(arg) for arg in get_args(annotation) if arg is not type(None)
        )
    if _is_batch_iterable_annotation(annotation):
        return True
    if isinstance(annotation, str):
        return any(
            token in annotation
            for token in (
                "TabularInput",
                "InferableTabularInput",
                "pa.RecordBatchReader",
                "pyarrow.RecordBatchReader",
                "pa.RecordBatch",
                "pyarrow.RecordBatch",
                "RecordBatch",
                "pa.Table",
                "pyarrow.Table",
                "pl.DataFrame",
                "polars.DataFrame",
                "pl.LazyFrame",
                "polars.LazyFrame",
            )
        )
    return any(
        token in str(annotation)
        for token in (
            "TabularInput",
            "InferableTabularInput",
            "pa.RecordBatchReader",
            "pyarrow.RecordBatchReader",
            "pa.RecordBatch",
            "pyarrow.RecordBatch",
            "RecordBatch",
            "pa.Table",
            "pyarrow.Table",
            "pl.DataFrame",
            "polars.DataFrame",
            "pl.LazyFrame",
            "polars.LazyFrame",
        )
    )


def _is_dataset_ref_annotation(annotation: object) -> bool:
    if annotation is DatasetRef:
        return True
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, DatasetRef)
        except TypeError:
            return False
    origin = get_origin(annotation)
    if origin in {types.UnionType, typing.Union}:
        return any(
            _is_dataset_ref_annotation(arg) for arg in get_args(annotation) if arg is not type(None)
        )
    if isinstance(annotation, str):
        return "DatasetRef" in annotation
    return "DatasetRef" in str(annotation)


def _dataset_param_to_table_key(param_name: str) -> str | None:
    if not param_name.startswith("d__"):
        return None
    payload = param_name.removeprefix("d__")
    if not payload:
        return None
    return payload.replace("__", ".")


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


def _seed_harness(
    *,
    declared_provider: SchemaProvider,
    observation_provider: SchemaObservationProvider | None,
    seed_dataset: SeedDatasetConfig | None,
) -> DatasetSeedHarness:
    if seed_dataset is None:
        return DatasetSeedHarness(
            schema_provider=declared_provider,
            observation_provider=observation_provider,
        )
    return DatasetSeedHarness(
        schema_provider=declared_provider,
        observation_provider=observation_provider,
        dataset_root_dir=seed_dataset.dataset_root_dir,
        snapshot_id=seed_dataset.snapshot_id,
        scan_settings=seed_dataset.scan_settings,
    )


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


def _schema_hints_for_table_key(
    *,
    catalog: DagCatalog,
    table_key: str,
) -> SchemaHints | None:
    tag_sets = _schema_tag_sets_for_table(catalog=catalog, table_key=table_key)
    return schema_hints_from_tag_sets(tag_sets)


def _schema_tag_sets_for_table(
    *,
    catalog: DagCatalog,
    table_key: str,
) -> tuple[Mapping[str, object], ...]:
    tag_sets: list[Mapping[str, object]] = []
    output = catalog.table_outputs.get(table_key)
    if output is not None:
        tag_sets.append(output.tags)
        tag_sets.extend(_schema_output_tag_sets(catalog=catalog, saver_node=output.saver_node))
    tag_sets.extend(
        node.tags
        for node in catalog.nodes.values()
        if node.tags.get(hamilton_tags.TAG_TABLE_KEY) == table_key
    )
    return tuple(tag_sets)


def _schema_observation_context(
    *,
    catalog: DagCatalog,
    table_key: str,
    declared_provider: SchemaProvider,
    target_name: str | None = None,
) -> SchemaObservationContext:
    tag_sets = _schema_tag_sets_for_table(catalog=catalog, table_key=table_key)
    declared_schema = declared_provider.get_table_schema(table_key)
    if declared_schema is None:
        declared_schema = table_schema_from_tag_sets(
            table_key=table_key,
            tag_sets=tag_sets,
        )
    schema_hints = schema_hints_from_tag_sets(tag_sets)
    return SchemaObservationContext(
        declared_schema=declared_schema,
        schema_hints=schema_hints,
        target_name=target_name,
    )


def _schema_output_tag_sets(
    *,
    catalog: DagCatalog,
    saver_node: str,
) -> list[Mapping[str, object]]:
    node = catalog.nodes.get(saver_node)
    if node is None:
        return []
    visited: set[str] = set()
    stack = list(node.deps)
    tag_sets: list[Mapping[str, object]] = []
    while stack:
        node_name = stack.pop()
        if node_name in visited:
            continue
        visited.add(node_name)
        candidate = catalog.nodes.get(node_name)
        if candidate is None:
            continue
        if _SCHEMA_OUTPUT_TAG in candidate.tags:
            tag_sets.append(candidate.tags)
        stack.extend(candidate.deps)
    return tag_sets


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

    qparams, requires_env, requires_catalog, dataset_refs = _inference_requirements(
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
        dataset_refs=tuple(sorted(dataset_refs.items())),
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
) -> tuple[set[str], bool, bool, dict[str, str]]:
    effective_compute_name = _compute_node_for_inference(context, compute_name=compute_name)
    node = context.driver.graph.nodes.get(effective_compute_name)
    if node is None:
        msg = f"Compute node not found in Hamilton DAG: {effective_compute_name}"
        raise ValueError(msg)

    qparams: set[str] = set()
    dataset_refs: dict[str, str] = {}
    requires_env = False
    requires_catalog = False
    visited: set[str] = set()
    stack = list(node.dependencies)

    while stack:
        dep = stack.pop()
        if dep.name in visited:
            continue
        visited.add(dep.name)
        update = _inspect_inference_dependency(dep, compute_name=compute_name)
        if update.requires_env:
            requires_env = True
        if update.requires_catalog:
            requires_catalog = True
        if update.qparam is not None:
            qparams.add(update.qparam)
        if update.dataset_ref is not None:
            ref_name, table_key = update.dataset_ref
            dataset_refs[ref_name] = table_key
        if update.skip_children:
            continue
        stack.extend(dep.dependencies)

    return qparams, requires_env, requires_catalog, dataset_refs


def _inspect_inference_dependency(
    dep: Node,
    *,
    compute_name: str,
) -> _InferenceRequirementUpdate:
    if dep.name == "env":
        return _InferenceRequirementUpdate(requires_env=True, skip_children=True)
    if dep.name == "catalog":
        return _InferenceRequirementUpdate(requires_catalog=True, skip_children=True)
    if dep.name.startswith("q__"):
        return _InferenceRequirementUpdate(qparam=dep.name, skip_children=True)
    if _is_dataset_ref_annotation(dep.type):
        table_key = _dataset_param_to_table_key(dep.name)
        if table_key is None:
            msg = f"DatasetRef dependency missing table key: {dep.name}"
            raise ValueError(msg)
        return _InferenceRequirementUpdate(
            dataset_ref=(dep.name, table_key),
            skip_children=True,
        )
    _validate_inference_dependency(dep, compute_name=compute_name)
    return _InferenceRequirementUpdate()


def _validate_inference_dependency(dep: Node, *, compute_name: str) -> None:
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
    if dep.user_defined and not _is_tabular_annotation(dep.type):
        msg = (
            f"Compute node {compute_name} depends on non-tabular input {dep.name}; "
            "schema inference requires tabular compute dependencies."
        )
        raise ValueError(msg)


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
    observation_context: SchemaObservationContext | None,
    seed_dataset: SeedDatasetConfig | None,
) -> TableSchema:
    with _schema_inference_gateway(schema_provider=declared_provider) as gateway:
        harness = _seed_harness(
            declared_provider=declared_provider,
            observation_provider=gateway.schemas,
            seed_dataset=seed_dataset,
        )
        overrides: dict[str, object] = dict(harness.build_inputs(set(job.qparams)))
        inputs: dict[str, object] = {}
        env: BuildEnv | None = None
        if job.requires_env or job.dataset_refs:
            env = _inference_env(
                gateway=gateway,
                force_targets=frozenset({job.target_name}),
            )
            inputs["env"] = env
        if job.requires_catalog:
            inputs["catalog"] = context.catalog
        if job.dataset_refs:
            if env is None:
                msg = "DatasetRef inference requires BuildEnv inputs"
                raise ValueError(msg)
            overrides.update(_dataset_ref_overrides(job=job, env=env))

        out = context.driver.execute([job.exec_name], inputs=inputs, overrides=overrides)
        expr_obj = out[job.exec_name]
        if expr_obj is None:
            msg = f"{job.exec_name} returned None; expected tabular output"
            raise TypeError(msg)
        return _table_schema_from_tabular(
            cast("InferableTabularInput", expr_obj),
            table_key=job.table_key,
            observation_context=observation_context,
        )


def infer_schema_for_table_key(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    table_key: str,
    declared_provider: SchemaProvider,
    seed_dataset: SeedDatasetConfig | None = None,
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
    seed_dataset
        Optional dataset-backed seed configuration for q__ inputs.

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
        observation_context = _schema_observation_context(
            catalog=catalog,
            table_key=table_key,
            declared_provider=declared_provider,
            target_name=job.target_name,
        )
        inferred = _infer_table_schema_for_compute(
            context=context,
            declared_provider=declared_provider,
            job=job,
            observation_context=observation_context,
            seed_dataset=seed_dataset,
        )
        declared_schema = observation_context.declared_schema
        schema_hints = observation_context.schema_hints
        return merge_table_schema_hints(
            inferred,
            declared_schema,
            schema_hints=schema_hints,
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
_INFERENCE_ERRORS = (KeyError, TypeError, ValueError, RuntimeError)


@dataclass(frozen=True, slots=True)
class SchemaObservationContext:
    """Optional metadata used when generating schema observations."""

    declared_schema: TableSchema | None = None
    schema_hints: SchemaHints | None = None
    repo: str | None = None
    commit: str | None = None
    target_name: str | None = None
    extras_policy: ExtrasPolicy | None = None
    drift_history: Sequence[Mapping[str, object] | None] | None = None


@dataclass
class HamiltonSchemaProvider(SchemaProvider):
    """SchemaProvider that prefers inferred schemas for selected table keys."""

    declared: SchemaProvider
    inferer: Inferer
    inferable_table_keys: frozenset[str]
    fallback_to_declared_on_error: bool = False
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
            except _INFERENCE_ERRORS:
                if not self.fallback_to_declared_on_error:
                    return None
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


def inferability_inventory(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
) -> tuple[InferabilityRecord, ...]:
    """Return inferability diagnostics for DAG-produced table outputs.

    Parameters
    ----------
    driver
        Hamilton driver used to inspect compute nodes.
    catalog
        DAG catalog defining outputs for each build target.

    Returns
    -------
    tuple[InferabilityRecord, ...]
        Diagnostic records keyed by table output.
    """
    context = _InferenceContext(driver=driver, catalog=catalog)
    records: list[InferabilityRecord] = []
    for table_key, output in sorted(catalog.table_outputs.items()):
        compute_node = _output_data_node(context=context, table_key=table_key)
        status: Literal["inferable", "non_inferable"] = "inferable"
        reason: str | None = None
        qparams: tuple[str, ...] | None = None
        requires_env: bool | None = None
        requires_catalog: bool | None = None
        try:
            job = _resolve_inference_job(context=context, table_key=table_key)
            compute_node = job.compute_name
            qparams = tuple(sorted(job.qparams))
            requires_env = job.requires_env
            requires_catalog = job.requires_catalog
        except (KeyError, TypeError, ValueError) as exc:
            status = "non_inferable"
            reason = str(exc)
        records.append(
            InferabilityRecord(
                table_key=table_key,
                status=status,
                target_name=output.producer_target,
                saver_node=output.saver_node,
                sink=output.sink,
                compute_node=compute_node,
                reason=reason,
                qparams=qparams,
                requires_env=requires_env,
                requires_catalog=requires_catalog,
            )
        )
    return tuple(records)


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
    observation_context: SchemaObservationContext | None,
) -> TableSchema:
    inputs: dict[str, object] = {}
    overrides = dict(base_overrides)
    if job.requires_env:
        inputs["env"] = env
    if job.requires_catalog:
        inputs["catalog"] = context.catalog
    if job.dataset_refs:
        overrides.update(_dataset_ref_overrides(job=job, env=env))

    out = context.driver.execute([job.exec_name], inputs=inputs, overrides=overrides)
    expr_obj = out[job.exec_name]
    if expr_obj is None:
        msg = f"{job.exec_name} returned None; expected tabular output"
        raise TypeError(msg)

    return _table_schema_from_tabular(
        cast("InferableTabularInput", expr_obj),
        table_key=job.table_key,
        observation_context=observation_context,
    )


def infer_table_schemas(
    table_keys: Iterable[str],
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    declared_provider: SchemaProvider,
    seed_dataset: SeedDatasetConfig | None = None,
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
    seed_dataset
        Optional dataset-backed seed configuration for q__ inputs.

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
        harness = _seed_harness(
            declared_provider=declared_provider,
            observation_provider=gateway.schemas,
            seed_dataset=seed_dataset,
        )
        base_overrides: dict[str, object] = dict(harness.build_inputs(set(union_qparams)))
        env = _inference_env(
            gateway=gateway,
            force_targets=frozenset({job.target_name for job in jobs}),
        )

        inferred: dict[str, TableSchema] = {}
        for job in jobs:
            observation_context = _schema_observation_context(
                catalog=catalog,
                table_key=job.table_key,
                declared_provider=declared_provider,
                target_name=job.target_name,
            )
            inferred_schema = _infer_job_schema(
                context=context,
                job=job,
                base_overrides=base_overrides,
                env=env,
                observation_context=observation_context,
            )
            inferred[job.table_key] = merge_table_schema_hints(
                inferred_schema,
                observation_context.declared_schema,
                schema_hints=observation_context.schema_hints,
            )

    return inferred


def _dataset_ref_overrides(
    *,
    job: _ComputeInferenceJob,
    env: BuildEnv,
) -> dict[str, DatasetRef]:
    return {
        param_name: DatasetRef(
            table_key=table_key,
            repo=env.repo,
            commit=env.commit,
            source_target=job.target_name,
        )
        for param_name, table_key in job.dataset_refs
    }


def _normalize_record_batch_iterable(
    batches: Iterable[object],
) -> tuple[pa.Schema, Iterable[pa.RecordBatch]]:
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration as exc:
        msg = "Record batch iterable yielded no batches."
        raise ValueError(msg) from exc
    if not isinstance(first, pa.RecordBatch):
        msg = f"Expected RecordBatch iterable, got {type(first)}."
        raise TypeError(msg)
    first_batch = cast("pa.RecordBatch", first)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        yield first_batch
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                msg = f"Expected RecordBatch iterable, got {type(batch)}."
                raise TypeError(msg)
            yield batch

    return first_batch.schema, _iter_batches()


def _table_schema_from_tabular(
    obj: InferableTabularInput,
    *,
    table_key: str,
    observation_context: SchemaObservationContext | None = None,
) -> TableSchema:
    if _is_arrow_table(obj):
        table_schema = table_schema_from_arrow_schema(arrow_schema=obj.schema, table_key=table_key)
    elif _is_record_batch_reader(obj):
        if observation_context is None:
            table_schema = table_schema_from_arrow_schema(
                arrow_schema=obj.schema,
                table_key=table_key,
            )
        else:
            bundle = observe_schema_from_reader(
                obj,
                table_key=table_key,
                context=observation_context,
            )
            table_schema = bundle.table_schema
    elif isinstance(obj, pl.DataFrame):
        table_schema = table_schema_from_polars_dataframe(frame=obj, table_key=table_key)
    elif isinstance(obj, pl.LazyFrame):
        table_schema = table_schema_from_polars_lazyframe(frame=obj, table_key=table_key)
    elif isinstance(obj, Iterable):
        schema, normalized = _normalize_record_batch_iterable(obj)
        if observation_context is None:
            table_schema = table_schema_from_arrow_schema(arrow_schema=schema, table_key=table_key)
        else:
            bundle = observe_schema_from_batches(
                batches=normalized,
                schema=schema,
                table_key=table_key,
                context=observation_context,
            )
            table_schema = bundle.table_schema
    else:
        msg = f"Unsupported tabular output for schema inference: {type(obj)}"
        raise TypeError(msg)
    return table_schema


def table_schema_from_tabular(obj: InferableTabularInput, *, table_key: str) -> TableSchema:
    """Return a TableSchema derived from a tabular output.

    Parameters
    ----------
    obj
        Tabular output to inspect.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the tabular output.
    """
    return _table_schema_from_tabular(obj, table_key=table_key)


def observe_schema_from_reader(
    reader: pa.RecordBatchReader,
    *,
    table_key: str,
    context: SchemaObservationContext | None = None,
) -> SchemaObservationBundle:
    """Observe a schema from a streaming RecordBatchReader.

    Parameters
    ----------
    reader
        RecordBatchReader to observe.
    table_key
        Fully qualified table key (schema.table).
    context
        Optional observation context containing declared schema and provenance.

    Returns
    -------
    SchemaObservationBundle
        Bundle containing observation, registry, and schema version records.
    """
    resolved = context or SchemaObservationContext()
    accumulator = SchemaObservationAccumulator(
        table_key=table_key,
        declared_schema=resolved.declared_schema,
        schema_hints=resolved.schema_hints,
    )
    observe_batches(reader, accumulator=accumulator)
    inputs = SchemaObservationInputs(
        repo=resolved.repo,
        commit=resolved.commit,
        target_name=resolved.target_name,
        extras_policy=resolved.extras_policy,
        drift_history=resolved.drift_history,
    )
    return accumulator.finalize(arrow_schema=reader.schema, inputs=inputs)


def observe_schema_from_batches(
    *,
    batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    table_key: str,
    context: SchemaObservationContext | None = None,
) -> SchemaObservationBundle:
    """Observe a schema from an iterator of record batches.

    Parameters
    ----------
    batches
        Record batch iterable to observe.
    schema
        Arrow schema describing the batch stream.
    table_key
        Fully qualified table key (schema.table).
    context
        Optional observation context containing declared schema and provenance.

    Returns
    -------
    SchemaObservationBundle
        Bundle containing observation, registry, and schema version records.
    """
    resolved = context or SchemaObservationContext()
    accumulator = SchemaObservationAccumulator(
        table_key=table_key,
        declared_schema=resolved.declared_schema,
        schema_hints=resolved.schema_hints,
    )
    observe_batches(batches, accumulator=accumulator)
    inputs = SchemaObservationInputs(
        repo=resolved.repo,
        commit=resolved.commit,
        target_name=resolved.target_name,
        extras_policy=resolved.extras_policy,
        drift_history=resolved.drift_history,
    )
    return accumulator.finalize(arrow_schema=schema, inputs=inputs)


@dataclass(frozen=True, slots=True)
class SchemaInferenceService:
    """Service for schema inference using native Hamilton compute nodes."""

    driver: h_driver.Driver
    catalog: DagCatalog
    seed_dataset: SeedDatasetConfig | None = None

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
            seed_dataset=self.seed_dataset,
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
            seed_dataset=self.seed_dataset,
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
    seed_dataset: SeedDatasetConfig | None = None,
) -> SchemaInferenceService:
    """Return a runtime-bound SchemaInferenceService.

    Parameters
    ----------
    driver
        Hamilton driver used for inference execution.
    catalog
        DAG catalog defining outputs for each build target.
    seed_dataset
        Optional dataset-backed seed configuration for q__ inputs.

    Returns
    -------
    SchemaInferenceService
        Service bound to the provided driver and catalog.
    """
    return SchemaInferenceService(
        driver=driver,
        catalog=catalog,
        seed_dataset=seed_dataset,
    )
