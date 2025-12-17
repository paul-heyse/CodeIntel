"""Hamilton-backed schema provider for inferred schemas (Phase 2).

This module provides:
- an inference routine for a single output table_key produced by a native compute node
- a composite SchemaProvider that prefers inferred schemas for selected table keys

Inference strategy (Strategy B)
------------------------------
1. Create an ephemeral in-memory DuckDB database.
2. Seed empty upstream tables required by the compute function (q__ inputs).
3. Execute the target's compute node to obtain an Ibis table expression.
4. Compile the expression to SQL and run DuckDB ``DESCRIBE`` to recover types.
5. Convert to a ``TableSchema``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import ibis.expr.types as ir

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import compute_node
from codeintel.build.providers import Providers, create_default_providers
from codeintel.build.schemas.infer_duckdb import infer_table_schema_from_ibis
from codeintel.build.schemas.seed_harness import MiniSeedHarness
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.gateway.ephemeral import ephemeral_gateway
from codeintel.storage.gateway.protocol import StorageGateway

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import TargetGraph
    from codeintel.storage.gateway.protocol import DuckDBConnection


@dataclass(frozen=True)
class _ComputeInferenceJob:
    target_name: str
    compute_name: str
    table_key: str
    qparams: frozenset[str]
    requires_env: bool
    requires_graph: bool


def _looks_inferable_compute(fn: Callable[..., object]) -> bool:
    """Return True if callable looks like a q__-driven Ibis compute function.

    Parameters
    ----------
    fn
        Callable to inspect.

    Returns
    -------
    bool
        True when the callable resembles an Ibis-native compute node.
    """
    sig = inspect.signature(fn)
    qparams = [name for name in sig.parameters if name.startswith("q__")]
    if not qparams:
        return False
    return_annotation = sig.return_annotation
    if return_annotation is inspect.Signature.empty:
        return False
    return "ir.Table" in str(return_annotation)


@lru_cache(maxsize=1)
def _runtime_auto() -> HamiltonRuntime:
    return build_driver(enable_cache=False)


@lru_cache(maxsize=1)
def _schema_inference_providers() -> Providers:
    return create_default_providers(ToolsConfig.default())


def _producers_by_table_key(graph: TargetGraph) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for target in graph.all_targets:
        for table_key in target.contract.table_keys:
            mapping.setdefault(table_key, []).append(target.name)
    return mapping


def _inferable_candidates_for_table_key(
    *,
    table_key: str,
    producers: list[str],
) -> list[tuple[str, Callable[..., object]]]:
    candidates: list[tuple[str, Callable[..., object]]] = []
    runtime = _runtime_auto()

    for target_name in sorted(producers):
        compute_name = compute_node(target_name)
        node = runtime.dr.graph.nodes.get(compute_name)
        if node is None:
            continue
        if not node.originating_functions:
            continue
        compute_fn_obj = node.originating_functions[0]
        if not isinstance(compute_fn_obj, Callable):
            continue
        compute_fn: Callable[..., object] = compute_fn_obj
        if not _looks_inferable_compute(compute_fn):
            continue
        try:
            _inference_requirements(runtime=runtime, compute_name=compute_name)
        except ValueError:
            continue
        candidates.append((target_name, compute_fn))

    if not candidates:
        msg = f"Table {table_key} is not inferable from any native compute target"
        raise ValueError(msg)

    return candidates


def _compute_node_for_inference(runtime: HamiltonRuntime, *, compute_name: str) -> str:
    """Return the best node name to use for schema inference execution.

    Hamilton decorators like ``@check_output_custom`` wrap the original compute
    node with validator plumbing and create an internal ``*_raw`` node that
    represents the underlying compute graph. For schema inference we want the
    compute-only graph, not the validator wrapper.

    Parameters
    ----------
    runtime
        Hamilton runtime containing the resolved FunctionGraph.
    compute_name
        Canonical compute node name (e.g., ``t__risk_factors__compute``).

    Returns
    -------
    str
        Node name to execute/traverse for inference. This is either
        ``{compute_name}_raw`` when present, or ``compute_name`` otherwise.
    """
    raw_name = f"{compute_name}_raw"
    return raw_name if raw_name in runtime.dr.graph.nodes else compute_name


def _inference_requirements(
    *,
    runtime: HamiltonRuntime,
    compute_name: str,
) -> tuple[set[str], bool, bool]:
    """Return q__ inputs and env requirement for executing a compute node.

    This is derived from the *actual* Hamilton dependency graph for the compute
    node (including nodes injected by @pipe_input), rather than relying solely
    on the compute function signature.

    Parameters
    ----------
    runtime
        Hamilton runtime containing the resolved FunctionGraph.
    compute_name
        Name of the compute node to analyze (e.g., ``t__risk_factors__compute``).

    Returns
    -------
    tuple[set[str], bool, bool]
        Tuple of (q__ node names required, whether an ``env`` input is required,
        whether a ``graph`` input is required).

    Raises
    ------
    ValueError
        If the compute node depends on target/materialize nodes or other inputs
        that cannot be satisfied by seeding q__ inputs and env.
    """
    effective_compute_name = _compute_node_for_inference(runtime, compute_name=compute_name)
    node = runtime.dr.graph.nodes.get(effective_compute_name)
    if node is None:
        msg = f"Compute node not found in Hamilton DAG: {effective_compute_name}"
        raise ValueError(msg)

    qparams: set[str] = set()
    requires_env = False
    requires_graph = False
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

        if dep.name == "graph":
            requires_graph = True
            continue

        if dep.name.startswith("q__"):
            qparams.add(dep.name)
            continue  # q__ inputs are injected directly; stop traversal.

        if dep.user_defined:
            msg = (
                f"Compute node {compute_name} depends on unsupported input {dep.name}; "
                "schema inference supports only env, graph, and q__ inputs."
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

    return qparams, requires_env, requires_graph


def _inference_env(*, gateway: StorageGateway, force_targets: frozenset[str]) -> BuildEnv:
    snapshot = SnapshotRef.from_args(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=Path.cwd(),
    )
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=BuildPaths.from_repo_root(snapshot.repo_root),
        providers=_schema_inference_providers(),
        config=BuildConfig.empty(),
        profile="schema_inference",
        force_targets=force_targets,
    )


def _infer_table_schema_for_compute(
    *,
    runtime: HamiltonRuntime,
    declared_provider: SchemaProvider,
    job: _ComputeInferenceJob,
) -> TableSchema:
    with ephemeral_gateway(schema_provider=declared_provider) as gateway:
        harness = MiniSeedHarness(gateway=gateway, schema_provider=declared_provider)
        inputs: dict[str, object] = dict(harness.build_inputs(set(job.qparams)))
        if job.requires_env:
            inputs["env"] = _inference_env(
                gateway=cast(StorageGateway, gateway),
                force_targets=frozenset({job.target_name}),
            )
        if job.requires_graph:
            inputs["graph"] = runtime.graph

        out = runtime.dr.execute([job.compute_name], inputs=inputs)
        expr_obj = out[job.compute_name]
        if not isinstance(expr_obj, ir.Table):
            msg = f"{job.compute_name} returned {type(expr_obj)}; expected ibis Table"
            raise TypeError(msg)

        return infer_table_schema_from_ibis(expr=expr_obj, con=gateway.con, table_key=job.table_key)


def infer_schema_for_table_key(
    *,
    table_key: str,
    declared_provider: SchemaProvider,
) -> TableSchema:
    """Infer schema for a single output table produced by a native compute node.

    Parameters
    ----------
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
        If the compute node does not return an Ibis table expression.
    ValueError
        If the table_key is not inferable from any native compute node.
    """
    runtime = _runtime_auto()
    graph = runtime.graph

    producers = _producers_by_table_key(graph).get(table_key)
    if not producers:
        msg = f"Unknown table_key (no producing target): {table_key}"
        raise KeyError(msg)

    try:
        candidates = _inferable_candidates_for_table_key(table_key=table_key, producers=producers)
        target_name, compute_fn = candidates[0]
        compute_name = compute_node(target_name)

        _ = compute_fn
        qparams, requires_env, requires_graph = _inference_requirements(
            runtime=runtime,
            compute_name=compute_name,
        )
        job = _ComputeInferenceJob(
            target_name=target_name,
            compute_name=compute_name,
            table_key=table_key,
            qparams=frozenset(qparams),
            requires_env=requires_env,
            requires_graph=requires_graph,
        )
        return _infer_table_schema_for_compute(
            runtime=runtime,
            declared_provider=declared_provider,
            job=job,
        )
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
        seen: dict[str, TableSchema] = {}
        for schema in self.declared.iter_table_schemas():
            seen[schema.table_key] = schema
        for table_key, inferred in self._cache.items():
            seen[table_key] = inferred
        return tuple(seen[key] for key in sorted(seen))

    def prefill_cache(self, schemas: Mapping[str, TableSchema]) -> None:
        """Prefill the inference cache with known schemas.

        Parameters
        ----------
        schemas
            Mapping of table_key to schema to seed into the cache.
        """
        self._cache.update(schemas)


def inferable_native_table_keys(*, graph: TargetGraph) -> frozenset[str]:
    """Return output table keys that appear inferable from native compute nodes.

    Parameters
    ----------
    graph
        Target graph defining outputs for each build target.

    Returns
    -------
    frozenset[str]
        Output table keys from targets that have q__-driven Ibis compute nodes.
    """
    runtime = _runtime_auto()
    inferable: set[str] = set()
    for target in graph.all_targets:
        compute_name = compute_node(target.name)
        node = runtime.dr.graph.nodes.get(compute_name)
        if node is None or not node.originating_functions:
            continue
        fn_obj = node.originating_functions[0]
        if not callable(fn_obj) or not _looks_inferable_compute(fn_obj):
            continue
        try:
            _inference_requirements(runtime=runtime, compute_name=compute_name)
        except ValueError:
            continue
        inferable.update(target.contract.table_keys)
    return frozenset(inferable)


def _build_inference_jobs(
    *,
    runtime: HamiltonRuntime,
    table_keys: list[str],
) -> list[_ComputeInferenceJob]:
    graph = runtime.graph
    producers_by_key = _producers_by_table_key(graph)
    jobs: list[_ComputeInferenceJob] = []
    for table_key in table_keys:
        producers = producers_by_key.get(table_key)
        if not producers:
            msg = f"Unknown table_key (no producing target): {table_key}"
            raise KeyError(msg)

        candidates = _inferable_candidates_for_table_key(table_key=table_key, producers=producers)
        if not candidates:
            msg = f"No inferable native compute candidates for table_key: {table_key}"
            raise ValueError(msg)

        target_name, compute_fn = candidates[0]
        compute_name = compute_node(target_name)
        _ = compute_fn
        qparams, requires_env, requires_graph = _inference_requirements(
            runtime=runtime,
            compute_name=compute_name,
        )
        jobs.append(
            _ComputeInferenceJob(
                target_name=target_name,
                compute_name=compute_name,
                table_key=table_key,
                qparams=frozenset(qparams),
                requires_env=requires_env,
                requires_graph=requires_graph,
            )
        )
    return jobs


def _union_qparams(jobs: Iterable[_ComputeInferenceJob]) -> frozenset[str]:
    return frozenset({name for job in jobs for name in job.qparams})


def _infer_job_schema(
    *,
    runtime: HamiltonRuntime,
    job: _ComputeInferenceJob,
    base_inputs: Mapping[str, object],
    env: BuildEnv,
    con: DuckDBConnection,
) -> TableSchema:
    inputs = dict(base_inputs)
    if job.requires_env:
        inputs["env"] = env
    if job.requires_graph:
        inputs["graph"] = runtime.graph

    out = runtime.dr.execute([job.compute_name], inputs=inputs)
    expr_obj = out[job.compute_name]
    if not isinstance(expr_obj, ir.Table):
        msg = f"{job.compute_name} returned {type(expr_obj)}; expected ibis Table"
        raise TypeError(msg)

    return infer_table_schema_from_ibis(
        expr=expr_obj,
        con=con,
        table_key=job.table_key,
    )


def infer_table_schemas(
    table_keys: Iterable[str],
    *,
    declared_provider: SchemaProvider,
) -> dict[str, TableSchema]:
    """Infer schemas for multiple output tables in a single ephemeral DuckDB session.

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
    unique_keys = sorted(set(table_keys))
    if not unique_keys:
        return {}

    runtime = _runtime_auto()
    jobs = _build_inference_jobs(runtime=runtime, table_keys=unique_keys)
    union_qparams = _union_qparams(jobs)

    with ephemeral_gateway(schema_provider=declared_provider) as gateway:
        harness = MiniSeedHarness(gateway=gateway, schema_provider=declared_provider)
        base_inputs: dict[str, object] = dict(harness.build_inputs(set(union_qparams)))
        env = _inference_env(
            gateway=cast(StorageGateway, gateway),
            force_targets=frozenset({job.target_name for job in jobs}),
        )

        inferred: dict[str, TableSchema] = {}
        for job in jobs:
            inferred[job.table_key] = _infer_job_schema(
                runtime=runtime,
                job=job,
                base_inputs=base_inputs,
                env=env,
                con=gateway.con,
            )

    return inferred


__all__ = [
    "HamiltonSchemaProvider",
    "infer_schema_for_table_key",
    "infer_table_schemas",
    "inferable_native_table_keys",
]
