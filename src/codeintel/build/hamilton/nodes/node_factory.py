"""Dynamic node generation from TargetGraph metadata.

This module generates Hamilton nodes programmatically from the target
graph, enabling automatic coverage of all targets without manual
node definitions.

Design Principles
-----------------
1. Nodes are created dynamically with proper Hamilton signatures.
2. Dependencies are derived from the TargetGraph.
3. Generated nodes reuse shared _run_target execution logic.
4. Dataset nodes (d__*) are generated for all contract tables.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import tag

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import (
    DatasetRef,
    refs_from_target_result,
    refs_to_tuple,
)
from codeintel.build.hamilton.io.ibis_adapter import load_dataset_df, load_dataset_ibis
from codeintel.build.hamilton.manifest_hook import (
    ManifestSaveRequest,
    SkipCheckRequest,
    TargetRunRecord,
    compute_target_input_hash,
    compute_target_options_hash,
    save_manifest,
    should_skip,
)
from codeintel.build.hamilton.metadata_bridge import from_target
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
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.registry import get_target_graph
from codeintel.build.targets import TargetGraph
from codeintel.build.unified_registry import get_unified_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.plugin import TargetPluginProtocol
    from codeintel.build.targets import OutputTarget

__all__ = [
    "GenerationOptions",
    "build_target_module",
    "clear_generated_module_cache",
    "get_generated_module",
]

log = logging.getLogger(__name__)


@dataclass
class _GeneratedModuleCache:
    module: ModuleType | None = None
    config_key: tuple[bool, bool, bool, bool, frozenset[str], frozenset[str]] | None = None


_MODULE_CACHE = _GeneratedModuleCache()


def _set_signature[T](fn: Callable[..., T], signature: inspect.Signature) -> Callable[..., T]:
    """Attach an inspect.Signature to a callable for Hamilton compatibility.

    Returns
    -------
    Callable[..., T]
        The input function with signature metadata applied.
    """
    fn_any = cast("Any", fn)
    fn_any.__signature__ = signature

    annotations: dict[str, object] = dict(getattr(fn, "__annotations__", {}))
    for name, param in signature.parameters.items():
        if param.annotation is inspect.Signature.empty:
            continue
        annotations[name] = param.annotation
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    fn_any.__annotations__ = annotations
    return fn


@dataclass(frozen=True)
class GenerationOptions:
    """Configuration for generated Hamilton node modules."""

    include_target_nodes: bool = True
    include_dataset_nodes: bool = True
    include_loader_nodes: bool = True
    include_artifact_nodes: bool = True
    include_targets: set[str] | None = None
    exclude_targets: set[str] | None = None


@dataclass(frozen=True)
class _PluginExecContext:
    """Bundled context for plugin execution."""

    plugin: TargetPluginProtocol
    plugin_name: str
    input_hash: str
    options_hash: str | None


def _run_target(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    upstream: tuple[TargetRunRecord, ...],
) -> TargetRunRecord:
    """Execute a target plugin and return the run record.

    This is the core execution wrapper for generated Hamilton nodes.
    It handles skip checks, upstream failure gating, plugin execution,
    and manifest persistence.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and providers.
    graph
        Target graph for looking up target metadata.
    target_name
        Name of the target to execute.
    upstream
        Records from upstream dependencies.

    Returns
    -------
    TargetRunRecord
        Execution record with status, timing, and dataset refs.
    """
    # Check for upstream failures
    failed_upstream = [rec.target for rec in upstream if rec.status == "failed"]
    if failed_upstream:
        return TargetRunRecord(
            target=target_name,
            plugin_name="skipped",
            status="failed",
            input_hash=None,
            options_hash=None,
            error=f"Upstream target(s) failed: {', '.join(failed_upstream)}",
        )

    target = graph.get(target_name)
    plugin = get_unified_registry().instantiate_plugin(target_name)
    meta = from_target(target)
    plugin_name = meta.name  # Canonical name like "analytics.function_metrics"

    # Compute hashes
    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        manifests=env.manifest_index,
    )
    options_hash = compute_target_options_hash(None)  # No raw parameters available here

    # Check for skip
    skip_request = SkipCheckRequest(
        gateway=env.gateway,
        target=target_name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
        manifest_index=env.manifest_index,
    )

    if target_name not in env.force_targets and should_skip(skip_request):
        return TargetRunRecord(
            target=target_name,
            plugin_name=plugin_name,
            status="skipped",
            input_hash=input_hash,
            options_hash=options_hash,
        )

    exec_ctx = _PluginExecContext(
        plugin=plugin,
        plugin_name=plugin_name,
        input_hash=input_hash,
        options_hash=options_hash,
    )
    return _execute_plugin(env, target, exec_ctx)


def _execute_plugin(
    env: BuildEnv,
    target: OutputTarget,
    exec_ctx: _PluginExecContext,
) -> TargetRunRecord:
    """Execute the plugin and return a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and providers.
    target
        Target to execute.
    exec_ctx
        Bundled plugin and hash context.

    Returns
    -------
    TargetRunRecord
        Execution record with status, timing, and dataset refs.
    """
    resources = ContextResources(
        gateway=env.gateway,
        providers=env.providers,
    )
    ctx = TargetExecutionContext(
        target=target,
        resources=resources,
        parameters=EMPTY_PARAMETERS,
        snapshot=env.snapshot,
        paths=env.paths,
    )

    start_time = time.perf_counter()
    try:
        raw_result = exec_ctx.plugin.execute(ctx)
        # Handle async plugins
        if asyncio.iscoroutine(raw_result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(raw_result)
        else:
            result = raw_result
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        log.exception("Plugin execution failed for %s", target.name)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return TargetRunRecord(
            target=target.name,
            plugin_name=exec_ctx.plugin_name,
            status="failed",
            input_hash=exec_ctx.input_hash,
            options_hash=exec_ctx.options_hash,
            duration_ms=duration_ms,
            error=str(exc),
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    row_counts = dict(result.row_counts) if result.row_counts else {}
    table_keys = target.table_keys
    datasets_dict = refs_from_target_result(
        target_name=target.name,
        table_keys=table_keys,
        row_counts=row_counts,
        snapshot=env.snapshot,
    )
    datasets = refs_to_tuple(datasets_dict)

    # Save manifest
    save_request = ManifestSaveRequest(
        target=target.name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        plugin=exec_ctx.plugin_name,
        duration_ms=duration_ms,
        input_hash=exec_ctx.input_hash,
        row_count=sum(row_counts.values()) if row_counts else None,
        options_hash=exec_ctx.options_hash,
    )
    save_manifest(gateway=env.gateway, request=save_request)

    return TargetRunRecord(
        target=target.name,
        plugin_name=exec_ctx.plugin_name,
        status="succeeded",
        input_hash=exec_ctx.input_hash,
        options_hash=exec_ctx.options_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
        datasets=datasets,
    )


def _create_node_function(
    target: OutputTarget,
    dep_node_names: list[str],
    domain: str,
) -> Callable[..., TargetRunRecord]:
    """Create a Hamilton node function for a target.

    Parameters
    ----------
    target
        Output target definition.
    dep_node_names
        Hamilton node names of dependencies.
    domain
        Domain for tagging (ingestion, graphs, analytics).

    Returns
    -------
    Callable[..., TargetRunRecord]
        Node function with correct signature for Hamilton.
    """
    target_name = target.name

    def node_fn(
        env: BuildEnv,
        graph: TargetGraph,
        **dependencies: TargetRunRecord,
    ) -> TargetRunRecord:
        upstream = tuple(rec for rec in dependencies.values() if isinstance(rec, TargetRunRecord))
        return _run_target(
            env=env,
            graph=graph,
            target_name=target_name,
            upstream=upstream,
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
    ]

    params.extend(
        [
            inspect.Parameter(
                dep_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=TargetRunRecord,
            )
            for dep_name in dep_node_names
        ]
    )

    _set_signature(node_fn, inspect.Signature(params))
    node_fn.__name__ = target_node(target_name)
    node_fn.__doc__ = f"Execute the {target_name} target ({domain})."

    return tag(domain=domain, target=target_name, node_type=NODE_TYPE_MATERIALIZE)(node_fn)


def _create_dataset_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., DatasetRef]:
    """Create a Hamilton node function for a dataset extraction.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    target_name
        Target that produces this dataset.

    Returns
    -------
    Callable[..., DatasetRef]
        Node function that extracts DatasetRef from TargetRunRecord.
    """
    d_name = dataset_node(table_key)
    t_name = target_node(target_name)

    def dataset_fn(**kwargs: object) -> DatasetRef:
        rec = kwargs.get(t_name)
        if not isinstance(rec, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(rec)}"
            raise TypeError(msg)
        ds = rec.get_dataset(table_key)
        if ds is None:
            msg = f"Missing DatasetRef for {table_key} from {target_name}"
            raise ValueError(msg)
        return ds

    params = [
        inspect.Parameter(
            t_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetRunRecord,
        ),
    ]

    _set_signature(
        dataset_fn,
        inspect.Signature(params, return_annotation=DatasetRef),
    )
    dataset_fn.__name__ = d_name
    dataset_fn.__doc__ = f"Extract {table_key} dataset from {target_name} target."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_DATASET)(dataset_fn)


def _create_query_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for Ibis query loading.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    target_name
        Target that produces this dataset.

    Returns
    -------
    Callable[..., ir.Table]
        Node function that loads an Ibis table expression.
    """
    q_name = query_node(table_key)
    d_name = dataset_node(table_key)

    def query_fn(env: BuildEnv, **kwargs: object) -> object:
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
    query_fn.__doc__ = f"Load {table_key} as Ibis expression from {target_name} target."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_LOADER_QUERY)(query_fn)


def _create_dataframe_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for pandas DataFrame loading.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    target_name
        Target that produces this dataset.

    Returns
    -------
    Callable[..., pd.DataFrame]
        Node function that loads a pandas DataFrame.
    """
    df_name = dataframe_node(table_key)
    d_name = dataset_node(table_key)

    def dataframe_fn(env: BuildEnv, **kwargs: object) -> object:
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
    dataframe_fn.__doc__ = f"Load {table_key} as pandas DataFrame from {target_name} target."

    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table_key=table_key, node_type=NODE_TYPE_LOADER_DATAFRAME)(dataframe_fn)


def _create_artifact_node_function(
    *,
    artifact_name: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for artifact access.

    Artifact nodes provide access to non-tabular build artifacts like
    FAISS indexes, model weights, or SCIP indexes.

    Parameters
    ----------
    artifact_name
        Artifact identifier (e.g., "faiss_index", "scip_index").
    target_name
        Target that produces this artifact.

    Returns
    -------
    Callable[..., ArtifactRef]
        Node function that provides artifact reference.
    """
    a_name = artifact_node(artifact_name)
    t_name = target_node(target_name)

    def artifact_fn(env: BuildEnv, **kwargs: object) -> ArtifactRef:
        run_record = kwargs.get(t_name)
        if not isinstance(run_record, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(run_record)}"
            raise TypeError(msg)

        for art in run_record.artifacts:
            if art.name == artifact_name:
                return art

        return ArtifactRef(
            name=artifact_name,
            artifact_type="unknown",
            repo=env.repo,
            commit=env.commit,
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

    _set_signature(artifact_fn, inspect.Signature(params))
    artifact_fn.__name__ = a_name
    artifact_fn.__doc__ = f"Access {artifact_name} artifact from {target_name} target."

    domain = artifact_name.split(".", 1)[0] if "." in artifact_name else "main"
    return tag(domain=domain, artifact=artifact_name, node_type=NODE_TYPE_ARTIFACT)(artifact_fn)


@dataclass(frozen=True)
class _GeneratedMappings:
    target_to_node: dict[str, str]
    dataset_to_node: dict[str, str]
    query_to_node: dict[str, str]
    dataframe_to_node: dict[str, str]
    artifact_to_node: dict[str, str]


def _attach_node(module: ModuleType, *, node_name: str, fn: object) -> None:
    fn_any = cast("Any", fn)
    fn_any.__name__ = node_name
    fn_any.__module__ = module.__name__
    setattr(module, node_name, fn_any)


def _generate_nodes_for_target(
    module: ModuleType,
    target: OutputTarget,
    meta_domain: str,
    options: GenerationOptions,
    mappings: _GeneratedMappings,
) -> None:
    # Generate target node only if enabled
    if options.include_target_nodes:
        dep_node_names = [target_node(dep) for dep in target.dependencies]
        node_fn = _create_node_function(
            target=target,
            dep_node_names=dep_node_names,
            domain=meta_domain,
        )
        node_name = target_node(target.name)
        _attach_node(module, node_name=node_name, fn=node_fn)
        mappings.target_to_node[target.name] = node_name

    if options.include_dataset_nodes:
        table_keys = target.contract.table_keys
        for table_key in table_keys:
            d_name = dataset_node(table_key)
            _attach_node(
                module,
                node_name=d_name,
                fn=_create_dataset_node_function(table_key=table_key, target_name=target.name),
            )
            mappings.dataset_to_node[table_key] = d_name

            if options.include_loader_nodes:
                q_name = query_node(table_key)
                _attach_node(
                    module,
                    node_name=q_name,
                    fn=_create_query_node_function(table_key=table_key, target_name=target.name),
                )
                mappings.query_to_node[table_key] = q_name

                df_name = dataframe_node(table_key)
                _attach_node(
                    module,
                    node_name=df_name,
                    fn=_create_dataframe_node_function(table_key=table_key, target_name=target.name),
                )
                mappings.dataframe_to_node[table_key] = df_name

    if options.include_artifact_nodes:
        art_names = target.contract.artifact_names if target.contract else ()
        for art_name in art_names:
            a_node_name = artifact_node(art_name)
            _attach_node(
                module,
                node_name=a_node_name,
                fn=_create_artifact_node_function(
                    artifact_name=art_name,
                    target_name=target.name,
                ),
            )
            mappings.artifact_to_node[art_name] = a_node_name


def _attach_mappings(module: ModuleType, mappings: _GeneratedMappings) -> None:
    module_any = cast("Any", module)
    module_any.TARGET_TO_NODE = mappings.target_to_node
    module_any.DATASET_TO_NODE = mappings.dataset_to_node
    module_any.QUERY_TO_NODE = mappings.query_to_node
    module_any.DATAFRAME_TO_NODE = mappings.dataframe_to_node
    module_any.ARTIFACT_TO_NODE = mappings.artifact_to_node


def build_target_module(
    options: GenerationOptions | None = None,
) -> ModuleType:
    """Generate a module containing Hamilton nodes for all targets.

    Parameters
    ----------
    options
        GenerationOptions controlling which targets to include and which helper
        nodes (datasets, loaders, artifacts) to emit. If omitted, all targets
        are included with dataset, loader, and artifact nodes enabled.

    Returns
    -------
    ModuleType
        Module populated with Hamilton node functions and mapping dictionaries.
    """
    resolved = options or GenerationOptions()
    graph = get_target_graph()
    all_target_names = {t.name for t in graph.all_targets}
    include = resolved.include_targets or all_target_names
    exclude = resolved.exclude_targets or set()

    module = ModuleType("codeintel.build.hamilton.nodes.generated")
    module.__doc__ = "Auto-generated Hamilton nodes from TargetGraph."
    sys.modules[module.__name__] = module

    mappings = _GeneratedMappings(
        target_to_node={},
        dataset_to_node={},
        query_to_node={},
        dataframe_to_node={},
        artifact_to_node={},
    )

    for target in graph.all_targets:
        if target.name not in include or target.name in exclude:
            continue
        meta = from_target(target)
        _generate_nodes_for_target(
            module=module,
            target=target,
            meta_domain=meta.domain,
            options=resolved,
            mappings=mappings,
        )

    _attach_mappings(module, mappings)

    log.debug(
        "build_target_module: generated %d target, %d dataset, %d query, %d df, %d artifact",
        len(mappings.target_to_node),
        len(mappings.dataset_to_node),
        len(mappings.query_to_node),
        len(mappings.dataframe_to_node),
        len(mappings.artifact_to_node),
    )

    return module


def _cache_key(
    options: GenerationOptions,
) -> tuple[bool, bool, bool, bool, frozenset[str], frozenset[str]]:
    return (
        options.include_target_nodes,
        options.include_dataset_nodes,
        options.include_loader_nodes,
        options.include_artifact_nodes,
        frozenset(options.include_targets or ()),
        frozenset(options.exclude_targets or ()),
    )


def _should_use_cache(key: tuple[bool, bool, bool, bool, frozenset[str], frozenset[str]]) -> bool:
    return _MODULE_CACHE.module is not None and _MODULE_CACHE.config_key == key


def get_generated_module(options: GenerationOptions | None = None) -> ModuleType:
    """Get or create the generated nodes module.

    Returns a cached module instance, creating it on first call.
    This avoids regenerating nodes for each driver build.

    Parameters
    ----------
    options
        Optional GenerationOptions controlling which targets and helper nodes are generated.
        Passing different options produces distinct cached modules.

    Returns
    -------
    ModuleType
        Cached generated module.

    Examples
    --------
    >>> module = get_generated_module()
    >>> hasattr(module, "TARGET_TO_NODE")
    True
    """
    resolved = options or GenerationOptions()
    key = _cache_key(resolved)
    if not _should_use_cache(key):
        _MODULE_CACHE.module = build_target_module(options=resolved)
        _MODULE_CACHE.config_key = key
    if _MODULE_CACHE.module is None:
        _MODULE_CACHE.module = build_target_module(options=resolved)
        _MODULE_CACHE.config_key = key
    return _MODULE_CACHE.module


def clear_generated_module_cache() -> None:
    """Clear the cached generated module.

    Useful for testing or when the TargetGraph has changed.
    """
    if _MODULE_CACHE.module is not None:
        sys.modules.pop(_MODULE_CACHE.module.__name__, None)
    _MODULE_CACHE.module = None
    _MODULE_CACHE.config_key = None
