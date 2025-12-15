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

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import ibis.expr.types as ir

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.naming import compute_node
from codeintel.build.schemas.infer_duckdb import infer_table_schema_from_ibis
from codeintel.build.schemas.seed_harness import MiniSeedHarness
from codeintel.build.unified_registry import get_unified_registry
from codeintel.config.primitives import SnapshotRef
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.gateway.ephemeral import ephemeral_gateway

if TYPE_CHECKING:
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import TargetGraph


@dataclass(frozen=True)
class SchemaInferenceEnv:
    """Minimal env object for schema compilation execution.

    Native compute nodes commonly reference ``env.snapshot`` and occasionally
    ``env.gateway``. This object provides those attributes without requiring
    a full BuildEnv instance.
    """

    gateway: object
    snapshot: SnapshotRef


@dataclass(frozen=True)
class _ComputeInferenceJob:
    compute_name: str
    table_key: str
    qparams: frozenset[str]
    requires_env: bool


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
    return build_driver(mode="auto")


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
    registry = get_unified_registry()
    candidates: list[tuple[str, Callable[..., object]]] = []

    for target_name in sorted(producers):
        module_path = registry.get_native_module(target_name)
        if module_path is None:
            continue
        module = importlib.import_module(module_path)
        compute_name = compute_node(target_name)
        compute_fn_obj = getattr(module, compute_name, None)
        if not isinstance(compute_fn_obj, Callable):
            continue
        compute_fn: Callable[..., object] = compute_fn_obj
        if not _looks_inferable_compute(compute_fn):
            continue
        candidates.append((target_name, compute_fn))

    if not candidates:
        msg = f"Table {table_key} is not inferable from any native compute target"
        raise ValueError(msg)

    return candidates


def _inference_requirements(
    *,
    target_name: str,
    compute_fn: Callable[..., object],
) -> tuple[set[str], bool]:
    sig = inspect.signature(compute_fn)
    qparams = {name for name in sig.parameters if name.startswith("q__")}
    requires_env = "env" in sig.parameters

    extra_params = {name for name in sig.parameters if name not in qparams and name != "env"}
    if extra_params:
        msg = (
            f"Target {target_name} compute has unsupported params for inference: "
            f"{sorted(extra_params)}"
        )
        raise ValueError(msg)

    return qparams, requires_env


def _inference_env(*, gateway: object) -> SchemaInferenceEnv:
    snapshot = SnapshotRef.from_args(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=Path.cwd(),
    )
    return SchemaInferenceEnv(gateway=gateway, snapshot=snapshot)


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
            inputs["env"] = _inference_env(gateway=gateway)

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

        qparams, requires_env = _inference_requirements(target_name=target_name, compute_fn=compute_fn)
        job = _ComputeInferenceJob(
            compute_name=compute_name,
            table_key=table_key,
            qparams=frozenset(qparams),
            requires_env=requires_env,
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
        for table_key in self.inferable_table_keys:
            inferred = self.get_table_schema(table_key)
            if inferred is not None:
                seen[table_key] = inferred
        return tuple(seen[key] for key in sorted(seen))


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
    registry = get_unified_registry()

    inferable: set[str] = set()
    for target in graph.all_targets:
        module_path = registry.get_native_module(target.name)
        if module_path is None:
            continue
        module = importlib.import_module(module_path)
        compute_name = compute_node(target.name)
        fn = getattr(module, compute_name, None)
        if fn is None or not callable(fn):
            continue
        if not _looks_inferable_compute(fn):
            continue
        inferable.update(target.contract.table_keys)
    return frozenset(inferable)


__all__ = [
    "HamiltonSchemaProvider",
    "infer_schema_for_table_key",
    "inferable_native_table_keys",
]
