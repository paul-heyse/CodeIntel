"""Schema inference service for native Hamilton outputs."""

from __future__ import annotations

import inspect
import logging
import time
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, TypeGuard, cast, get_args, get_origin

import hamilton.driver as h_driver
import polars as pl
import pyarrow as pa
from polars.exceptions import PolarsError

try:
    from dulwich.repo import Repo as _DulwichRepo
except ImportError:
    _DulwichRepo = None

from codeintel.build.assets.emitter import RunArtifactSpec, record_run_artifact
from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.build_log import record_build_event
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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput, TabularInput
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.columnar.polars_utils import resolve_query_opt_flags
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
)
from codeintel.core.duckdb_types import DuckDBConnection, DuckDBRelation
from codeintel.core.env import get_bool
from codeintel.core.execution.ids import new_run_id
from codeintel.core.hamilton import tags as hamilton_tags
from codeintel.core.manifests import (
    InferencePlanDatasetRef,
    InferencePlanLoaderOverride,
    InferencePlanManifest,
    InferencePlanSeedDataset,
    InferencePlanSettings,
    write_manifest_json,
)
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hamilton.node import Node

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.build.schemas.observations import SchemaHints
    from codeintel.build.schemas.seed_harness import SchemaObservationProvider
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.arrow_gen import ExtrasPolicy

LOG = logging.getLogger(__name__)

__all__ = [
    "HamiltonSchemaProvider",
    "InferabilityRecord",
    "SchemaInferenceInputs",
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
_INFERENCE_PLAN_ARTIFACT_NAME = "inference_plan_manifest"
_INFERABLE_RUNTIME_TYPES: tuple[type[object], ...] = (
    pa.RecordBatchReader,
    pa.Table,
    pl.DataFrame,
    pl.LazyFrame,
)


class InferenceGatewayFactory(Protocol):
    """Protocol for creating inference gateways without storage coupling."""

    def __call__(self, *, schema_provider: SchemaProvider) -> BuildGateway: ...


class InferenceGatewayError(RuntimeError):
    """Base error for inference gateway stubs."""


class InferenceGatewayAttributeError(InferenceGatewayError):
    """Raised when accessing missing gateway attributes."""

    def __init__(self, name: str) -> None:
        msg = f"Inference gateway does not provide attribute {name}."
        super().__init__(msg)


class InferenceGatewayConnectionError(InferenceGatewayError):
    """Raised when a DuckDB connection is requested from the inference gateway."""

    def __init__(self) -> None:
        super().__init__("Inference gateway does not expose a DuckDB connection.")


class InferenceGatewayExecuteError(InferenceGatewayError):
    """Raised when SQL execution is requested from the inference gateway."""

    def __init__(self) -> None:
        super().__init__("Inference gateway does not execute SQL.")


class InferenceGatewayRegisterError(InferenceGatewayError):
    """Raised when registration is requested from the inference gateway."""

    def __init__(self) -> None:
        super().__init__("Inference gateway does not support registration.")


class InferenceGatewayUnregisterError(InferenceGatewayError):
    """Raised when unregistration is requested from the inference gateway."""

    def __init__(self) -> None:
        super().__init__("Inference gateway does not support unregistration.")


class InferenceGatewayRelationError(InferenceGatewayError):
    """Raised when table relation resolution is requested from the inference gateway."""

    def __init__(self) -> None:
        super().__init__("Inference gateway does not resolve relations.")


class _NullAccess:
    def __getattr__(self, name: str) -> NoReturn:
        raise InferenceGatewayAttributeError(name)


@dataclass(frozen=True, slots=True)
class _NullGatewayConfig:
    read_only: bool = True
    db_path: Path = Path(":memory:")


@dataclass(frozen=True, slots=True)
class _NullDatasetRegistry:
    by_name: dict[str, object] = field(default_factory=dict)
    jsonl_datasets: dict[str, str] = field(default_factory=dict)
    parquet_datasets: dict[str, str] = field(default_factory=dict)
    dataset_root_dir: Path | None = None

    def with_dataset_root(self, dataset_root_dir: Path | None) -> _NullDatasetRegistry:
        return _NullDatasetRegistry(
            by_name=self.by_name,
            jsonl_datasets=self.jsonl_datasets,
            parquet_datasets=self.parquet_datasets,
            dataset_root_dir=dataset_root_dir,
        )


class _NullSchemaObservationProvider:
    @staticmethod
    def load_latest_schema_observation(*, table_key: str) -> object | None:
        _ = table_key
        return None


@dataclass(frozen=True, slots=True)
class _NullGateway:
    config: _NullGatewayConfig = field(default_factory=_NullGatewayConfig)
    datasets: _NullDatasetRegistry = field(default_factory=_NullDatasetRegistry)
    schemas: _NullSchemaObservationProvider = field(default_factory=_NullSchemaObservationProvider)
    assets: _NullAccess = field(default_factory=_NullAccess)
    build: _NullAccess = field(default_factory=_NullAccess)
    exports: _NullAccess = field(default_factory=_NullAccess)
    policy: _NullAccess = field(default_factory=_NullAccess)
    runs: _NullAccess = field(default_factory=_NullAccess)

    @staticmethod
    def close() -> None:
        return None

    @property
    def con(self) -> DuckDBConnection:
        raise InferenceGatewayConnectionError

    @staticmethod
    def execute(
        sql: str,
        params: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBConnection:
        _ = sql
        _ = params
        raise InferenceGatewayExecuteError

    @staticmethod
    def register(name: str, obj: object) -> None:
        _ = name
        _ = obj
        raise InferenceGatewayRegisterError

    @staticmethod
    def unregister(name: str) -> None:
        _ = name
        raise InferenceGatewayUnregisterError

    @staticmethod
    def relation_from_table_key(table_key: str) -> DuckDBRelation:
        _ = table_key
        raise InferenceGatewayRelationError


@dataclass(frozen=True)
class _ComputeInferenceJob:
    target_name: str
    compute_name: str
    exec_name: str
    table_key: str
    qparams: frozenset[str]
    dataset_refs: tuple[tuple[str, str], ...]
    loader_nodes: tuple[tuple[str, str], ...]
    requires_env: bool
    requires_catalog: bool


@dataclass(frozen=True, slots=True)
class _InferenceContext:
    driver: h_driver.Driver
    catalog: DagCatalog


@dataclass(frozen=True, slots=True)
class SchemaInferenceInputs:
    """Input bundle for schema inference helpers."""

    driver: h_driver.Driver
    catalog: DagCatalog
    declared_provider: SchemaProvider
    seed_dataset: SeedDatasetConfig | None = None
    gateway_factory: InferenceGatewayFactory | None = None


@dataclass(frozen=True, slots=True)
class _InferenceComputeInputs:
    context: _InferenceContext
    job: _ComputeInferenceJob
    observation_context: SchemaObservationContext | None
    schema_inputs: SchemaInferenceInputs
    env: BuildEnv | None = None


@dataclass(frozen=True, slots=True)
class _InferenceJobRunInputs:
    context: _InferenceContext
    job: _ComputeInferenceJob
    base_overrides: Mapping[str, object]
    harness: DatasetSeedHarness
    env: BuildEnv
    observation_context: SchemaObservationContext | None


@dataclass(frozen=True, slots=True)
class _InferenceRequirementUpdate:
    requires_env: bool = False
    requires_catalog: bool = False
    qparam: str | None = None
    dataset_ref: tuple[str, str] | None = None
    loader_node: tuple[str, str] | None = None
    skip_children: bool = False


@dataclass(slots=True)
class _InferenceRequirementState:
    qparams: set[str] = field(default_factory=set)
    dataset_refs: dict[str, str] = field(default_factory=dict)
    loader_nodes: dict[str, str] = field(default_factory=dict)
    requires_env: bool = False
    requires_catalog: bool = False
    visited: set[str] = field(default_factory=set)

    def apply(self, update: _InferenceRequirementUpdate) -> None:
        if update.requires_env:
            self.requires_env = True
        if update.requires_catalog:
            self.requires_catalog = True
        if update.qparam is not None:
            self.qparams.add(update.qparam)
        if update.dataset_ref is not None:
            ref_name, table_key = update.dataset_ref
            self.dataset_refs[ref_name] = table_key
        if update.loader_node is not None:
            node_name, table_key = update.loader_node
            self.loader_nodes[node_name] = table_key


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
    gateway_factory: InferenceGatewayFactory | None,
) -> Iterator[BuildGateway]:
    if gateway_factory is None:
        gateway = cast("BuildGateway", _NullGateway())
    else:
        gateway = gateway_factory(schema_provider=schema_provider)
    try:
        yield gateway
    finally:
        close = getattr(gateway, "close", None)
        if callable(close):
            close()


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


def _saver_node_for_table(
    *,
    context: _InferenceContext,
    table_key: str,
) -> Node | None:
    output = context.catalog.table_outputs.get(table_key)
    if output is None:
        return None
    return context.driver.graph.nodes.get(output.saver_node)


def _data_node_from_tags(saver_node: Node) -> str | None:
    saver_tags = getattr(saver_node, "tags", None)
    if not isinstance(saver_tags, dict):
        return None
    data_node = saver_tags.get("ci.data_node")
    if isinstance(data_node, str) and data_node:
        return data_node
    return None


def _data_node_from_tagged_deps(saver_node: Node, *, table_key: str) -> str | None:
    tagged_deps: list[str] = []
    dataset_tagged_deps: list[str] = []
    for dep in saver_node.dependencies:
        tags = dep.tags if isinstance(dep.tags, dict) else {}
        if tags.get(hamilton_tags.TAG_TABLE_KEY) != table_key:
            continue
        tagged_deps.append(dep.name)
        if tags.get(hamilton_tags.TAG_NODE_TYPE) == hamilton_tags.NODE_TYPE_DATASET:
            dataset_tagged_deps.append(dep.name)
    if len(dataset_tagged_deps) == 1:
        return dataset_tagged_deps[0]
    if len(tagged_deps) == 1:
        return tagged_deps[0]
    return None


def _data_node_from_tabular_deps(saver_node: Node) -> str | None:
    tabular_deps = [dep.name for dep in saver_node.dependencies if _is_tabular_annotation(dep.type)]
    if len(tabular_deps) != 1:
        return None
    return tabular_deps[0]


def _output_data_node(
    *,
    context: _InferenceContext,
    table_key: str,
) -> str | None:
    saver_node = _saver_node_for_table(context=context, table_key=table_key)
    if saver_node is None:
        return None
    return (
        _data_node_from_tags(saver_node)
        or _data_node_from_tagged_deps(saver_node, table_key=table_key)
        or _data_node_from_tabular_deps(saver_node)
    )


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

    qparams, requires_env, requires_catalog, dataset_refs, loader_nodes = _inference_requirements(
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
        loader_nodes=tuple(sorted(loader_nodes.items())),
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
) -> tuple[set[str], bool, bool, dict[str, str], dict[str, str]]:
    effective_compute_name = _compute_node_for_inference(context, compute_name=compute_name)
    node = context.driver.graph.nodes.get(effective_compute_name)
    if node is None:
        msg = f"Compute node not found in Hamilton DAG: {effective_compute_name}"
        raise ValueError(msg)

    state = _InferenceRequirementState()
    stack: list[tuple[Node, int]] = [(dep, 0) for dep in node.dependencies]

    while stack:
        dep, depth = stack.pop()
        if dep.name in state.visited:
            continue
        state.visited.add(dep.name)
        update = _inspect_inference_dependency(dep, compute_name=compute_name, depth=depth)
        state.apply(update)
        if update.skip_children:
            continue
        stack.extend((child, depth + 1) for child in dep.dependencies)

    return (
        state.qparams,
        state.requires_env,
        state.requires_catalog,
        state.dataset_refs,
        state.loader_nodes,
    )


def _special_dependency_update(dep: Node) -> _InferenceRequirementUpdate | None:
    if dep.name == "env":
        return _InferenceRequirementUpdate(requires_env=True, skip_children=True)
    if dep.name == "catalog":
        return _InferenceRequirementUpdate(requires_catalog=True, skip_children=True)
    return None


def _loader_dependency_update(dep: Node) -> _InferenceRequirementUpdate | None:
    tags = dep.tags if isinstance(dep.tags, dict) else {}
    if tags.get(hamilton_tags.TAG_NODE_TYPE) != hamilton_tags.NODE_TYPE_LOADER_QUERY:
        return None
    table_key = tags.get(hamilton_tags.TAG_TABLE_KEY)
    if not isinstance(table_key, str) or not table_key:
        return None
    return _InferenceRequirementUpdate(loader_node=(dep.name, table_key), skip_children=True)


def _qparam_dependency_update(dep: Node) -> _InferenceRequirementUpdate | None:
    if dep.name.startswith("q__"):
        return _InferenceRequirementUpdate(qparam=dep.name, skip_children=True)
    return None


def _dataset_ref_dependency_update(
    dep: Node,
    *,
    depth: int,
) -> _InferenceRequirementUpdate | None:
    if not _is_dataset_ref_annotation(dep.type):
        return None
    if depth > 0:
        return _InferenceRequirementUpdate(skip_children=True)
    table_key = _dataset_param_to_table_key(dep.name)
    if table_key is None:
        msg = f"DatasetRef dependency missing table key: {dep.name}"
        raise ValueError(msg)
    return _InferenceRequirementUpdate(dataset_ref=(dep.name, table_key), skip_children=True)


def _inspect_inference_dependency(
    dep: Node,
    *,
    compute_name: str,
    depth: int,
) -> _InferenceRequirementUpdate:
    for resolver in (
        _special_dependency_update,
        _loader_dependency_update,
        _qparam_dependency_update,
    ):
        update = resolver(dep)
        if update is not None:
            return update
    update = _dataset_ref_dependency_update(dep, depth=depth)
    if update is not None:
        return update
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
        if dep.dependencies:
            return
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
    polars_profile = bool(get_bool("CODEINTEL_BUILD_POLARS_PROFILE", default=False) or False)
    polars_inspect = bool(get_bool("CODEINTEL_BUILD_POLARS_INSPECT", default=False) or False)
    return BuildSettings(
        engine_version=engine_version,
        export_audit=ExportAuditSettings(),
        polars_profile=polars_profile,
        polars_inspect=polars_inspect,
    )


def _inference_env(*, gateway: BuildGateway, force_targets: frozenset[str]) -> BuildEnv:
    snapshot = _dulwich_snapshot()
    if snapshot is None:
        msg = "Schema inference requires a valid dulwich snapshot (repo root + HEAD commit)."
        raise RuntimeError(msg)
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
    return context.build_env()


def _dulwich_snapshot() -> SnapshotRef | None:
    if _DulwichRepo is None:
        return None
    try:
        repo = _DulwichRepo.discover(Path.cwd())
    except (OSError, ValueError):
        return None
    repo_root = Path(repo.path).resolve()
    head = repo.head()
    if isinstance(head, bytes):
        commit = head.decode("ascii", errors="ignore").strip()
    else:
        commit = str(head).strip()
    if not commit:
        return None
    repo_name = repo_root.name or "repo"
    return SnapshotRef.from_args(
        repo=repo_name,
        commit=commit,
        repo_root=repo_root,
    )


def _infer_table_schema_for_compute(inputs: _InferenceComputeInputs) -> TableSchema:
    schema_inputs = inputs.schema_inputs
    with _schema_inference_gateway(
        schema_provider=schema_inputs.declared_provider,
        gateway_factory=schema_inputs.gateway_factory,
    ) as gateway:
        harness = _seed_harness(
            declared_provider=schema_inputs.declared_provider,
            observation_provider=gateway.schemas,
            seed_dataset=schema_inputs.seed_dataset,
        )
        overrides: dict[str, object] = dict(harness.build_inputs(set(inputs.job.qparams)))
        overrides.update(_loader_overrides(job=inputs.job, harness=harness))
        exec_inputs: dict[str, object] = {}
        env = inputs.env
        if env is None and (inputs.job.requires_env or inputs.job.dataset_refs):
            env = _inference_env(
                gateway=gateway,
                force_targets=frozenset({inputs.job.target_name}),
            )
        if env is not None:
            exec_inputs["env"] = env
        if inputs.job.requires_catalog:
            exec_inputs["catalog"] = inputs.context.catalog
        if inputs.job.dataset_refs:
            if env is None:
                msg = "DatasetRef inference requires BuildEnv inputs"
                raise ValueError(msg)
            overrides.update(_dataset_ref_overrides(job=inputs.job, env=env))

        out = inputs.context.driver.execute(
            [inputs.job.exec_name],
            inputs=exec_inputs,
            overrides=overrides,
        )
        expr_obj = out[inputs.job.exec_name]
        if expr_obj is None:
            msg = f"{inputs.job.exec_name} returned None; expected tabular output"
            raise TypeError(msg)
        return _table_schema_from_tabular(
            cast("InferableTabularInput", expr_obj),
            table_key=inputs.job.table_key,
            observation_context=inputs.observation_context,
        )


def infer_schema_for_table_key(
    table_key: str,
    *,
    schema_inputs: SchemaInferenceInputs,
    env: BuildEnv | None = None,
) -> TableSchema:
    """Infer schema for a single output table produced by a native compute node.

    Parameters
    ----------
    table_key
        Output table key to infer (schema.table).
    schema_inputs
        Shared inference inputs (driver, catalog, provider, optional seed dataset).
    env
        Optional BuildEnv to use for inference execution.

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
    context = _InferenceContext(
        driver=schema_inputs.driver,
        catalog=schema_inputs.catalog,
    )

    try:
        job = _resolve_inference_job(context=context, table_key=table_key)
        observation_context = _schema_observation_context(
            catalog=schema_inputs.catalog,
            table_key=table_key,
            declared_provider=schema_inputs.declared_provider,
            target_name=job.target_name,
        )
        inferred = _infer_table_schema_for_compute(
            _InferenceComputeInputs(
                context=context,
                job=job,
                observation_context=observation_context,
                schema_inputs=schema_inputs,
                env=env,
            )
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
_INFERENCE_JOB_ERRORS = (*_INFERENCE_ERRORS, OSError, PolarsError, pa.ArrowInvalid)


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


@dataclass(frozen=True, slots=True)
class _InferenceSession:
    env: BuildEnv
    context: _InferenceContext
    jobs: tuple[_ComputeInferenceJob, ...]
    union_qparams: frozenset[str]
    harness: DatasetSeedHarness
    base_overrides: dict[str, object]
    run_id: str
    table_keys: tuple[str, ...]


def _prepare_inference_session(
    *,
    table_keys: tuple[str, ...],
    schema_inputs: SchemaInferenceInputs,
    env: BuildEnv | None,
    gateway: BuildGateway,
) -> _InferenceSession:
    context = _InferenceContext(
        driver=schema_inputs.driver,
        catalog=schema_inputs.catalog,
    )
    jobs = _build_inference_jobs(context=context, table_keys=table_keys)
    union_qparams = _union_qparams(jobs)
    harness = _seed_harness(
        declared_provider=schema_inputs.declared_provider,
        observation_provider=gateway.schemas,
        seed_dataset=schema_inputs.seed_dataset,
    )
    base_overrides: dict[str, object] = dict(harness.build_inputs(set(union_qparams)))
    resolved_env = env
    if resolved_env is None:
        resolved_env = _inference_env(
            gateway=gateway,
            force_targets=frozenset({job.target_name for job in jobs}),
        )
    return _InferenceSession(
        env=resolved_env,
        context=context,
        jobs=jobs,
        union_qparams=union_qparams,
        harness=harness,
        base_overrides=base_overrides,
        run_id=_inference_run_id(resolved_env),
        table_keys=table_keys,
    )


def _emit_inference_plan_for_session(
    *,
    session: _InferenceSession,
    seed_dataset: SeedDatasetConfig | None,
) -> None:
    try:
        manifest_path, _ = _emit_inference_plan_manifest(
            env=session.env,
            jobs=session.jobs,
            union_qparams=session.union_qparams,
            seed_dataset=seed_dataset,
            run_id=session.run_id,
        )
        record_run_artifact(
            env=session.env,
            run_id=session.run_id,
            spec=RunArtifactSpec(
                artifact_name=_INFERENCE_PLAN_ARTIFACT_NAME,
                artifact_type="json",
                path=manifest_path,
                meta={
                    "table_keys_count": len(session.table_keys),
                    "qparams_count": len(session.union_qparams),
                },
            ),
        )
        LOG.info(
            "build.inference.plan.emit run_id=%s repo=%s commit=%s table_keys_count=%d "
            "qparams_count=%d manifest_path=%s",
            session.run_id,
            session.env.repo,
            session.env.commit,
            len(session.table_keys),
            len(session.union_qparams),
            manifest_path,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        LOG.exception(
            "build.inference.plan.fail run_id=%s repo=%s commit=%s",
            session.run_id,
            session.env.repo,
            session.env.commit,
        )
        raise


def _run_inference_jobs(
    *,
    session: _InferenceSession,
    schema_inputs: SchemaInferenceInputs,
) -> dict[str, TableSchema]:
    inferred: dict[str, TableSchema] = {}
    for job in session.jobs:
        job_started = time.perf_counter()
        qparams_count = len(job.qparams)
        loader_overrides_count = len(job.loader_nodes)
        LOG.info(
            "build.inference.job.start run_id=%s repo=%s commit=%s table_key=%s target=%s "
            "qparams_count=%d loader_overrides_count=%d",
            session.run_id,
            session.env.repo,
            session.env.commit,
            job.table_key,
            job.target_name,
            qparams_count,
            loader_overrides_count,
        )
        record_build_event(
            "build.inference.job.start",
            table_key=job.table_key,
            target=job.target_name,
            qparams_count=qparams_count,
            loader_overrides_count=loader_overrides_count,
        )
        observation_context = _schema_observation_context(
            catalog=schema_inputs.catalog,
            table_key=job.table_key,
            declared_provider=schema_inputs.declared_provider,
            target_name=job.target_name,
        )
        try:
            inferred_schema = _infer_job_schema(
                _InferenceJobRunInputs(
                    context=session.context,
                    job=job,
                    base_overrides=session.base_overrides,
                    harness=session.harness,
                    env=session.env,
                    observation_context=observation_context,
                )
            )
        except _INFERENCE_JOB_ERRORS as exc:
            record_build_event(
                "build.inference.job.fail",
                table_key=job.table_key,
                target=job.target_name,
                exception_type=type(exc).__name__,
                error=str(exc),
            )
            LOG.exception(
                "build.inference.job.fail run_id=%s repo=%s commit=%s table_key=%s target=%s",
                session.run_id,
                session.env.repo,
                session.env.commit,
                job.table_key,
                job.target_name,
            )
            raise
        duration_ms = (time.perf_counter() - job_started) * 1000
        record_build_event(
            "build.inference.job.ok",
            table_key=job.table_key,
            target=job.target_name,
            duration_ms=duration_ms,
        )
        LOG.info(
            "build.inference.job.ok run_id=%s repo=%s commit=%s table_key=%s target=%s "
            "duration_ms=%.1f",
            session.run_id,
            session.env.repo,
            session.env.commit,
            job.table_key,
            job.target_name,
            duration_ms,
        )
        inferred[job.table_key] = merge_table_schema_hints(
            inferred_schema,
            observation_context.declared_schema,
            schema_hints=observation_context.schema_hints,
        )

    return inferred


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
    table_keys: Sequence[str],
) -> tuple[_ComputeInferenceJob, ...]:
    return tuple(
        _resolve_inference_job(context=context, table_key=table_key) for table_key in table_keys
    )


def _union_qparams(jobs: Iterable[_ComputeInferenceJob]) -> frozenset[str]:
    return frozenset({name for job in jobs for name in job.qparams})


def _lazyframe_for_diagnostics(value: InferableTabularInput) -> pl.LazyFrame | None:
    if isinstance(value, pl.LazyFrame):
        return value
    if isinstance(value, pl.DataFrame):
        return value.lazy()
    if isinstance(value, pa.Table):
        return tabular_to_lazyframe(value)
    return None


def _polars_plan_kwargs(
    func: object,
    *,
    streaming: bool,
    query_opt_flags: object | None,
) -> dict[str, object]:
    try:
        signature = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, object] = {}
    if "engine" in signature.parameters and streaming:
        kwargs["engine"] = "streaming"
    elif "streaming" in signature.parameters:
        kwargs["streaming"] = streaming
    if query_opt_flags is not None:
        if "optimization_flags" in signature.parameters:
            kwargs["optimization_flags"] = query_opt_flags
        elif "query_opt_flags" in signature.parameters:
            kwargs["query_opt_flags"] = query_opt_flags
        elif "optimizations" in signature.parameters:
            kwargs["optimizations"] = query_opt_flags
    return kwargs


def _safe_polars_explain(
    frame: pl.LazyFrame,
    *,
    streaming: bool,
    query_opt_flags: object | None,
) -> str | None:
    explain_fn = getattr(frame, "explain", None)
    if not callable(explain_fn):
        return None
    try:
        kwargs = _polars_plan_kwargs(
            explain_fn,
            streaming=streaming,
            query_opt_flags=query_opt_flags,
        )
        result = explain_fn(**kwargs)
    except (PolarsError, TypeError, ValueError):
        return None
    return result if isinstance(result, str) else None


def _safe_polars_profile(
    frame: pl.LazyFrame,
    *,
    streaming: bool,
    query_opt_flags: object | None,
) -> str | None:
    profile_fn = getattr(frame, "profile", None)
    if not callable(profile_fn):
        return None
    try:
        kwargs = _polars_plan_kwargs(
            profile_fn,
            streaming=streaming,
            query_opt_flags=query_opt_flags,
        )
        result = profile_fn(**kwargs)
    except (PolarsError, TypeError, ValueError):
        return None
    to_string = getattr(result, "to_string", None)
    if callable(to_string):
        try:
            rendered = to_string()
        except (TypeError, ValueError):
            return None
        return rendered if isinstance(rendered, str) else str(rendered)
    if result is None:
        return None
    return str(result)


def _log_inference_diagnostics(
    value: InferableTabularInput,
    *,
    table_key: str,
    settings: BuildSettings,
) -> None:
    if not settings.polars_inspect and not settings.polars_profile:
        return
    frame = _lazyframe_for_diagnostics(value)
    if frame is None:
        return
    query_opt_flags = resolve_query_opt_flags(settings.polars_query_opt_flags)
    if settings.polars_inspect:
        explain = _safe_polars_explain(
            frame,
            streaming=settings.polars_streaming,
            query_opt_flags=query_opt_flags,
        )
        if explain:
            LOG.debug("polars_inference_explain table=%s plan=%s", table_key, explain)
    if settings.polars_profile:
        profile_repr = _safe_polars_profile(
            frame,
            streaming=settings.polars_streaming,
            query_opt_flags=query_opt_flags,
        )
        if profile_repr:
            LOG.info("polars_inference_profile table=%s profile=%s", table_key, profile_repr)


def _infer_job_schema(inputs: _InferenceJobRunInputs) -> TableSchema:
    exec_inputs: dict[str, object] = {}
    overrides = dict(inputs.base_overrides)
    overrides.update(_loader_overrides(job=inputs.job, harness=inputs.harness))
    if inputs.job.requires_env:
        exec_inputs["env"] = inputs.env
    if inputs.job.requires_catalog:
        exec_inputs["catalog"] = inputs.context.catalog
    if inputs.job.dataset_refs:
        overrides.update(_dataset_ref_overrides(job=inputs.job, env=inputs.env))

    out = inputs.context.driver.execute(
        [inputs.job.exec_name],
        inputs=exec_inputs,
        overrides=overrides,
    )
    expr_obj = out[inputs.job.exec_name]
    if expr_obj is None:
        msg = f"{inputs.job.exec_name} returned None; expected tabular output"
        raise TypeError(msg)

    _log_inference_diagnostics(
        cast("InferableTabularInput", expr_obj),
        table_key=inputs.job.table_key,
        settings=inputs.env.settings,
    )
    return _table_schema_from_tabular(
        cast("InferableTabularInput", expr_obj),
        table_key=inputs.job.table_key,
        observation_context=inputs.observation_context,
    )


def _inference_run_id(env: BuildEnv) -> str:
    run = env.run_context
    if run is not None:
        return run.run_id
    return new_run_id("schema")


def _inference_manifest_path(env: BuildEnv, *, run_id: str) -> Path:
    return env.paths.build_dir / "schema" / f"inference_plan_{run_id}.json"


def _seed_manifest(seed: SeedDatasetConfig | None) -> InferencePlanSeedDataset | None:
    if seed is None:
        return None
    settings = seed.scan_settings
    dataset_root = str(seed.dataset_root_dir) if seed.dataset_root_dir is not None else None
    return InferencePlanSeedDataset(
        dataset_root_dir=dataset_root,
        snapshot_id=seed.snapshot_id,
        scan_mode=settings.mode,
        sample_rows=settings.sample_rows,
        batch_size=settings.batch_size,
        fragment_readahead=settings.fragment_readahead,
    )


def _emit_inference_plan_manifest(
    *,
    env: BuildEnv,
    jobs: Sequence[_ComputeInferenceJob],
    union_qparams: frozenset[str],
    seed_dataset: SeedDatasetConfig | None,
    run_id: str | None = None,
) -> tuple[Path, str]:
    resolved_run_id = run_id or _inference_run_id(env)
    loader_entries = sorted(
        {(node, table_key) for job in jobs for node, table_key in job.loader_nodes},
        key=lambda item: (item[0], item[1]),
    )
    dataset_refs = sorted(
        {(param, table_key) for job in jobs for param, table_key in job.dataset_refs},
        key=lambda item: (item[0], item[1]),
    )
    manifest = InferencePlanManifest(
        manifest_version=1,
        run_id=resolved_run_id,
        repo=env.repo,
        commit=env.commit,
        repo_root=str(env.snapshot.repo_root),
        generated_at=datetime.now(UTC).isoformat(),
        table_keys=tuple(sorted({job.table_key for job in jobs})),
        targets=tuple(sorted({job.target_name for job in jobs})),
        qparams=tuple(sorted(union_qparams)),
        loader_overrides=tuple(
            InferencePlanLoaderOverride(node=node, table_key=table_key)
            for node, table_key in loader_entries
        ),
        dataset_refs=tuple(
            InferencePlanDatasetRef(param=param, table_key=table_key)
            for param, table_key in dataset_refs
        ),
        seed_dataset=_seed_manifest(seed_dataset),
        settings=InferencePlanSettings(
            engine_version=env.settings.engine_version,
            polars_profile=env.settings.polars_profile,
            polars_inspect=env.settings.polars_inspect,
            polars_query_opt_flags=env.settings.polars_query_opt_flags,
            polars_streaming=env.settings.polars_streaming,
            polars_streaming_fallback=env.settings.polars_streaming_fallback,
        ),
    )
    path = _inference_manifest_path(env, run_id=resolved_run_id)
    write_manifest_json(path, manifest)
    return path, resolved_run_id


def infer_table_schemas(
    table_keys: Iterable[str],
    *,
    schema_inputs: SchemaInferenceInputs,
    env: BuildEnv | None = None,
) -> dict[str, TableSchema]:
    """Infer schemas for multiple output tables in a single session.

    Parameters
    ----------
    table_keys
        Output table keys to infer (schema.table).
    schema_inputs
        Shared inference inputs (driver, catalog, provider, optional seed dataset).
    env
        Optional BuildEnv to use for inference execution.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table_key to inferred TableSchema.

    """
    unique_keys = tuple(sorted(set(table_keys)))
    if not unique_keys:
        return {}

    with _schema_inference_gateway(
        schema_provider=schema_inputs.declared_provider,
        gateway_factory=schema_inputs.gateway_factory,
    ) as gateway:
        session = _prepare_inference_session(
            table_keys=unique_keys,
            schema_inputs=schema_inputs,
            env=env,
            gateway=gateway,
        )
        _emit_inference_plan_for_session(
            session=session,
            seed_dataset=schema_inputs.seed_dataset,
        )
        return _run_inference_jobs(session=session, schema_inputs=schema_inputs)


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


def _loader_overrides(
    *,
    job: _ComputeInferenceJob,
    harness: DatasetSeedHarness,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for node_name, table_key in job.loader_nodes:
        reader = harness.seed_table(table_key)
        overrides[node_name] = tabular_to_lazyframe(reader)
    return overrides


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
    env: BuildEnv | None = None
    seed_dataset: SeedDatasetConfig | None = None
    gateway_factory: InferenceGatewayFactory | None = None

    def infer_table_schema(
        self,
        table_key: str,
        *,
        declared_provider: SchemaProvider,
        env: BuildEnv | None = None,
    ) -> TableSchema:
        """Infer schema for a single table key.

        Parameters
        ----------
        table_key
            Output table key to infer (schema.table).
        declared_provider
            Provider used to seed upstream input tables.
        env
            Optional BuildEnv to use for inference execution.

        Returns
        -------
        TableSchema
            Inferred table schema.
        """
        return infer_schema_for_table_key(
            table_key=table_key,
            schema_inputs=SchemaInferenceInputs(
                driver=self.driver,
                catalog=self.catalog,
                declared_provider=declared_provider,
                seed_dataset=self.seed_dataset,
                gateway_factory=self.gateway_factory,
            ),
            env=env or self.env,
        )

    def infer_table_schemas(
        self,
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
        env: BuildEnv | None = None,
    ) -> dict[str, TableSchema]:
        """Infer schemas for multiple table keys.

        Parameters
        ----------
        table_keys
            Output table keys to infer (schema.table).
        declared_provider
            Provider used to seed upstream input tables.
        env
            Optional BuildEnv to use for inference execution.

        Returns
        -------
        dict[str, TableSchema]
            Mapping of table_key to inferred TableSchema.
        """
        return infer_table_schemas(
            table_keys,
            schema_inputs=SchemaInferenceInputs(
                driver=self.driver,
                catalog=self.catalog,
                declared_provider=declared_provider,
                seed_dataset=self.seed_dataset,
                gateway_factory=self.gateway_factory,
            ),
            env=env or self.env,
        )

    def inferable_table_keys(self) -> frozenset[str]:
        """Return inferable table keys for the bound DAG catalog.

        Returns
        -------
        frozenset[str]
            Output table keys inferred from native compute nodes.
        """
        return inferable_native_table_keys(driver=self.driver, catalog=self.catalog)

    def output_data_node(self, table_key: str) -> str | None:
        """Return the output data node name for a table key when available.

        Returns
        -------
        str | None
            Output data node name if resolved.
        """
        context = _InferenceContext(driver=self.driver, catalog=self.catalog)
        return _output_data_node(context=context, table_key=table_key)


def get_schema_inference_service(
    *,
    driver: h_driver.Driver,
    catalog: DagCatalog,
    env: BuildEnv | None = None,
    seed_dataset: SeedDatasetConfig | None = None,
    gateway_factory: InferenceGatewayFactory | None = None,
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
    env
        Optional BuildEnv to use for inference execution.
    gateway_factory
        Optional factory for building an inference gateway with schema access.

    Returns
    -------
    SchemaInferenceService
        Service bound to the provided driver and catalog.
    """
    return SchemaInferenceService(
        driver=driver,
        catalog=catalog,
        env=env,
        seed_dataset=seed_dataset,
        gateway_factory=gateway_factory,
    )
