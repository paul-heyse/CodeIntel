"""Generate Hamilton support nodes from build target metadata.

This module generates **support nodes** that are derived mechanically from the
`TargetGraph` and target contracts:

- Dataset nodes (`d__*`): extract `DatasetRef` from a producing target record.
- Loader nodes (`q__*`, `df__*`): load datasets as Ibis tables or pandas DataFrames.
- Artifact nodes (`a__*`): expose artifact references from a producing target record.

Optionally, it can also generate **stub target nodes** (`t__*`) that fail with a
clear error when no native implementation is present. In normal operation, the
native Hamilton modules provide the real `t__*` nodes and override these stubs.
"""

from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.ibis_adapter import load_dataset_df, load_dataset_ibis
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataframe_node,
    dataset_node,
    query_node,
    target_node,
)
from codeintel.build.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_DATASET,
    NODE_TYPE_LOADER_DATAFRAME,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
)
from codeintel.build.target_catalog import load_target_specs
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.targets import OutputTarget

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupportGenerationOptions:
    """Options for support-node module generation."""

    include_target_stubs: bool = True
    include_dataset_nodes: bool = True
    include_loader_nodes: bool = True
    include_artifact_nodes: bool = True
    include_targets: frozenset[str] | None = None
    exclude_targets: frozenset[str] | None = None


@dataclass
class _SupportModuleCache:
    module: ModuleType | None = None
    config_key: tuple[bool, bool, bool, bool, frozenset[str], frozenset[str]] | None = None


_MODULE_CACHE = _SupportModuleCache()


class _SupportModule(Protocol):
    """Protocol for support modules that receive generated mapping attributes."""

    TARGET_TO_NODE: dict[str, str]
    DATASET_TO_NODE: dict[str, str]
    QUERY_TO_NODE: dict[str, str]
    DATAFRAME_TO_NODE: dict[str, str]
    ARTIFACT_TO_NODE: dict[str, str]


class _SignaturedCallable(Protocol):
    """Protocol for callables with mutable signature/annotation metadata."""

    __signature__: inspect.Signature
    __annotations__: dict[str, object]


class _NamedCallable(Protocol):
    """Protocol for callables with mutable naming metadata."""

    __name__: str
    __module__: str


@dataclass
class _SupportNodeMappings:
    """Collect node-name mappings for a generated support module."""

    target_to_node: dict[str, str] = field(default_factory=dict)
    dataset_to_node: dict[str, str] = field(default_factory=dict)
    query_to_node: dict[str, str] = field(default_factory=dict)
    dataframe_to_node: dict[str, str] = field(default_factory=dict)
    artifact_to_node: dict[str, str] = field(default_factory=dict)

    def attach_to(self, module: ModuleType) -> None:
        """Attach mapping dicts to the support module."""
        mod = cast("_SupportModule", module)
        mod.TARGET_TO_NODE = self.target_to_node
        mod.DATASET_TO_NODE = self.dataset_to_node
        mod.QUERY_TO_NODE = self.query_to_node
        mod.DATAFRAME_TO_NODE = self.dataframe_to_node
        mod.ARTIFACT_TO_NODE = self.artifact_to_node


def _set_signature[T](fn: Callable[..., T], signature: inspect.Signature) -> Callable[..., T]:
    """Attach an inspect.Signature to a callable for Hamilton compatibility.

    Parameters
    ----------
    fn
        Function object to annotate.
    signature
        Signature that Hamilton should use for dependency resolution.

    Returns
    -------
    Callable[..., T]
        The input function with signature metadata applied.
    """
    meta = cast("_SignaturedCallable", fn)
    meta.__signature__ = signature

    annotations: dict[str, object] = dict(getattr(fn, "__annotations__", {}))
    for name, param in signature.parameters.items():
        if param.annotation is inspect.Signature.empty:
            continue
        annotations[name] = param.annotation
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    meta.__annotations__ = annotations
    return fn


def _attach_node(module: ModuleType, *, node_name: str, fn: object) -> None:
    """Attach a callable to a module under a stable node name."""
    named = cast("_NamedCallable", fn)
    named.__name__ = node_name
    named.__module__ = module.__name__
    setattr(module, node_name, fn)


def _create_stub_target_node_function(
    target: OutputTarget,
    dep_node_names: list[str],
    *,
    domain: str,
) -> Callable[..., TargetRunRecord]:
    """Create a stub `t__*` node for a target.

    The stub exists to keep the DAG complete and produce a deterministic
    error when a target lacks a native implementation. Native modules should
    override this node by defining a real `t__<target>` function.

    Returns
    -------
    Callable[..., TargetRunRecord]
        Stub node callable returning a failed TargetRunRecord.
    """
    target_name = target.name

    def node_fn(
        env: BuildEnv,
        graph: TargetGraph,
        **dependencies: TargetRunRecord,
    ) -> TargetRunRecord:
        _ = env, graph
        failed_upstream = [rec.target for rec in dependencies.values() if rec.status == "failed"]
        if failed_upstream:
            return TargetRunRecord(
                target=target_name,
                plugin_name=f"template:{target_name}",
                status="failed",
                input_hash=None,
                error=f"Upstream target(s) failed: {', '.join(failed_upstream)}",
            )
        return TargetRunRecord(
            target=target_name,
            plugin_name=f"template:{target_name}",
            status="failed",
            input_hash=None,
            error=f"Missing native implementation for target '{target_name}'.",
        )

    params: list[inspect.Parameter] = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            "graph",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetGraph,
        ),
        *[
            inspect.Parameter(
                dep_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=TargetRunRecord,
            )
            for dep_name in dep_node_names
        ],
    ]

    _set_signature(node_fn, inspect.Signature(params, return_annotation=TargetRunRecord))
    node_fn.__name__ = target_node(target_name)
    node_fn.__doc__ = f"Stub target node for {target_name}. Native implementation missing."

    return tag(domain=domain, target=target_name, node_type=NODE_TYPE_MATERIALIZE)(node_fn)


def _create_dataset_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., DatasetRef]:
    """Create a `d__*` dataset node extracting DatasetRef from TargetRunRecord.

    Returns
    -------
    Callable[..., DatasetRef]
        Dataset node callable returning a DatasetRef.
    """
    d_name = dataset_node(table_key)
    t_name = target_node(target_name)

    def dataset_fn(**kwargs: object) -> DatasetRef:
        rec = kwargs.get(t_name)
        if rec is None:
            msg = f"Missing dependency {t_name} for dataset node {d_name}"
            raise ValueError(msg)
        if not isinstance(rec, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(rec)}"
            raise TypeError(msg)
        ds = rec.get_dataset(table_key)
        if ds is None:
            msg = f"Missing DatasetRef for {table_key} from {target_name}"
            raise ValueError(msg)
        if isinstance(ds, DatasetRef):
            return ds
        return DatasetRef(
            table_key=ds.table_key,
            repo=ds.repo,
            commit=ds.commit,
            row_count=ds.row_count,
            source_target=target_name,
        )

    params = [
        inspect.Parameter(
            t_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetRunRecord,
        ),
    ]
    _set_signature(dataset_fn, inspect.Signature(params, return_annotation=DatasetRef))
    dataset_fn.__name__ = d_name
    dataset_fn.__doc__ = f"Extract {table_key} dataset from {target_name} target."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_DATASET)(dataset_fn)


def _create_query_node_function(
    *,
    table_key: str,
) -> Callable[..., ir.Table]:
    """Create a `q__*` node that loads an Ibis expression for a DatasetRef.

    Returns
    -------
    Callable[..., ir.Table]
        Query node callable returning an Ibis table expression.
    """
    q_name = query_node(table_key)
    d_name = dataset_node(table_key)

    def query_fn(env: BuildEnv, **kwargs: object) -> ir.Table:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_ibis(gateway=env.gateway, ref=ds_ref)

    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            d_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=DatasetRef,
        ),
    ]
    _set_signature(query_fn, inspect.Signature(params, return_annotation=ir.Table))
    query_fn.__name__ = q_name
    query_fn.__doc__ = f"Load {table_key} as Ibis expression."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_LOADER_QUERY)(query_fn)


def _create_dataframe_node_function(
    *,
    table_key: str,
) -> Callable[..., pd.DataFrame]:
    """Create a `df__*` node that loads a pandas DataFrame for a DatasetRef.

    Returns
    -------
    Callable[..., pd.DataFrame]
        DataFrame loader node callable returning a pandas DataFrame.
    """
    df_name = dataframe_node(table_key)
    d_name = dataset_node(table_key)

    def dataframe_fn(env: BuildEnv, **kwargs: object) -> pd.DataFrame:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_df(gateway=env.gateway, ref=ds_ref)

    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            d_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=DatasetRef,
        ),
    ]
    _set_signature(dataframe_fn, inspect.Signature(params, return_annotation=pd.DataFrame))
    dataframe_fn.__name__ = df_name
    dataframe_fn.__doc__ = f"Load {table_key} as pandas DataFrame."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_LOADER_DATAFRAME)(
        dataframe_fn
    )


def _create_artifact_node_function(
    *,
    artifact_name: str,
    target_name: str,
) -> Callable[..., ArtifactRef]:
    """Create an `a__*` node extracting artifact references from TargetRunRecord.

    Returns
    -------
    Callable[..., ArtifactRef]
        Artifact node callable returning an ArtifactRef.
    """
    a_name = artifact_node(artifact_name)
    t_name = target_node(target_name)

    def artifact_fn(env: BuildEnv, **kwargs: object) -> ArtifactRef:
        record = kwargs.get(t_name)
        if not isinstance(record, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(record)}"
            raise TypeError(msg)
        for art in record.artifacts:
            if art.name == artifact_name:
                if isinstance(art, ArtifactRef):
                    return art
                return ArtifactRef(
                    name=art.name,
                    artifact_type=art.artifact_type,
                    repo=art.repo,
                    commit=art.commit,
                    path=art.path,
                )
        return ArtifactRef(
            name=artifact_name,
            artifact_type="unknown",
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            path=None,
        )

    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            t_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetRunRecord,
        ),
    ]
    _set_signature(artifact_fn, inspect.Signature(params, return_annotation=ArtifactRef))
    artifact_fn.__name__ = a_name
    artifact_fn.__doc__ = f"Access {artifact_name} artifact from {target_name} target."

    domain = artifact_name.split(".", 1)[0] if "." in artifact_name else "main"
    return tag(domain=domain, artifact=artifact_name, node_type=NODE_TYPE_ARTIFACT)(artifact_fn)


def _build_contract_graph() -> TargetGraph:
    """Build a TargetGraph from the canonical target catalog.

    Returns
    -------
    TargetGraph
        Graph containing all registered build targets.
    """
    graph = TargetGraph()
    for target in load_target_specs():
        graph.register(target)
    return graph


def _new_support_module() -> ModuleType:
    """Create a fresh support module instance.

    Returns
    -------
    ModuleType
        New module instance registered in sys.modules.
    """
    module = ModuleType("codeintel.build.hamilton.nodes.support")
    module.__doc__ = "Auto-generated Hamilton support nodes (datasets/loaders/artifacts)."
    sys.modules[module.__name__] = module
    return module


def _include_target(*, target_name: str, include: frozenset[str], exclude: frozenset[str]) -> bool:
    """Return True when a target should be included for generation.

    Parameters
    ----------
    target_name
        Target name to evaluate.
    include
        Target names explicitly included.
    exclude
        Target names explicitly excluded.

    Returns
    -------
    bool
        True when the target is included and not excluded.
    """
    return target_name in include and target_name not in exclude


def _populate_for_target(
    module: ModuleType,
    *,
    target: OutputTarget,
    options: SupportGenerationOptions,
    mappings: _SupportNodeMappings,
) -> None:
    """Attach all enabled support nodes for a target to the module."""
    if options.include_target_stubs:
        dep_node_names = [target_node(dep) for dep in target.dependencies]
        t_name = target_node(target.name)
        mappings.target_to_node[target.name] = t_name
        _attach_node(
            module,
            node_name=t_name,
            fn=_create_stub_target_node_function(target, dep_node_names, domain=target.module),
        )

    if options.include_dataset_nodes:
        for table_key in target.contract.table_keys:
            d_name = dataset_node(table_key)
            mappings.dataset_to_node[table_key] = d_name
            _attach_node(
                module,
                node_name=d_name,
                fn=_create_dataset_node_function(table_key=table_key, target_name=target.name),
            )

            if options.include_loader_nodes:
                q_name = query_node(table_key)
                mappings.query_to_node[table_key] = q_name
                _attach_node(
                    module,
                    node_name=q_name,
                    fn=_create_query_node_function(table_key=table_key),
                )

                df_name = dataframe_node(table_key)
                mappings.dataframe_to_node[table_key] = df_name
                _attach_node(
                    module,
                    node_name=df_name,
                    fn=_create_dataframe_node_function(table_key=table_key),
                )

    if options.include_artifact_nodes:
        for artifact_name in target.contract.artifact_names:
            a_name = artifact_node(artifact_name)
            mappings.artifact_to_node[artifact_name] = a_name
            _attach_node(
                module,
                node_name=a_name,
                fn=_create_artifact_node_function(
                    artifact_name=artifact_name,
                    target_name=target.name,
                ),
            )


def build_support_module(*, options: SupportGenerationOptions | None = None) -> ModuleType:
    """Build a Python module containing Hamilton support nodes.

    Parameters
    ----------
    options
        Optional generation options.

    Returns
    -------
    ModuleType
        Module populated with support-node callables.
    """
    resolved = options or SupportGenerationOptions()

    graph = _build_contract_graph()
    include = resolved.include_targets or frozenset(t.name for t in graph.all_targets)
    exclude = resolved.exclude_targets or frozenset()

    module = _new_support_module()
    mappings = _SupportNodeMappings()

    for target in graph.all_targets:
        if not _include_target(target_name=target.name, include=include, exclude=exclude):
            continue
        _populate_for_target(module, target=target, options=resolved, mappings=mappings)

    mappings.attach_to(module)

    return module


def _cache_key(
    options: SupportGenerationOptions,
) -> tuple[bool, bool, bool, bool, frozenset[str], frozenset[str]]:
    return (
        options.include_target_stubs,
        options.include_dataset_nodes,
        options.include_loader_nodes,
        options.include_artifact_nodes,
        options.include_targets or frozenset(),
        options.exclude_targets or frozenset(),
    )


def get_support_module(*, options: SupportGenerationOptions | None = None) -> ModuleType:
    """Return a cached support-node module, creating it on first call.

    Returns
    -------
    ModuleType
        Cached module containing generated support nodes.
    """
    resolved = options or SupportGenerationOptions()
    key = _cache_key(resolved)
    if _MODULE_CACHE.module is None or _MODULE_CACHE.config_key != key:
        _MODULE_CACHE.module = build_support_module(options=resolved)
        _MODULE_CACHE.config_key = key
    if _MODULE_CACHE.module is None:
        _MODULE_CACHE.module = build_support_module(options=resolved)
        _MODULE_CACHE.config_key = key
    return _MODULE_CACHE.module


def clear_support_module_cache() -> None:
    """Clear the cached support module (useful for tests)."""
    if _MODULE_CACHE.module is not None:
        sys.modules.pop(_MODULE_CACHE.module.__name__, None)
    _MODULE_CACHE.module = None
    _MODULE_CACHE.config_key = None


__all__ = [
    "SupportGenerationOptions",
    "build_support_module",
    "clear_support_module_cache",
    "get_support_module",
]
