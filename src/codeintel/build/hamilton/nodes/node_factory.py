"""Dynamic node generation from TargetGraph metadata.

This module generates Hamilton nodes programmatically from the target
graph, enabling automatic coverage of all targets without manual
node definitions.

Design Principles
-----------------
1. Nodes are created dynamically with proper Hamilton signatures.
2. Dependencies are derived from the TargetGraph.
3. Generated nodes reuse _run_target from targets_phase0.py.
4. The generated module can replace explicit Phase 0 nodes.
5. Dataset nodes (d__*) are generated for all contract tables.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from types import ModuleType

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.metadata_bridge import from_target
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataframe_node,
    dataset_node,
    query_node,
    target_node,
)
from codeintel.build.hamilton.nodes.targets_phase0 import _run_target
from codeintel.build.registry import get_target_graph
from codeintel.build.targets import OutputTarget, TargetGraph

__all__ = [
    "build_target_module",
    "clear_generated_module_cache",
    "get_generated_module",
]

log = logging.getLogger(__name__)


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
        # Extract upstream records for failure gating
        upstream = tuple(rec for rec in dependencies.values() if isinstance(rec, TargetRunRecord))
        return _run_target(
            env=env,
            graph=graph,
            target_name=target_name,
            upstream=upstream,
        )

    # Build signature that Hamilton can inspect
    # Types are imported at runtime so Hamilton can resolve them
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

    # Add dependency parameters
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

    node_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    node_fn.__name__ = target_node(target_name)
    node_fn.__doc__ = f"Execute the {target_name} target ({domain})."

    # Apply tag decorator for Hamilton observability
    return tag(domain=domain, target=target_name)(node_fn)


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

    # Build signature with single parameter for the target node
    params = [
        inspect.Parameter(
            t_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetRunRecord,
        ),
    ]

    dataset_fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        params,
        return_annotation=DatasetRef,
    )
    dataset_fn.__name__ = d_name
    dataset_fn.__doc__ = f"Extract {table_key} dataset from {target_name} target."

    # Apply tag decorator for Hamilton observability
    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table=table_key)(dataset_fn)


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
    from codeintel.build.hamilton.io.ibis_adapter import load_dataset_ibis

    q_name = query_node(table_key)
    d_name = dataset_node(table_key)

    def query_fn(env: BuildEnv, **kwargs: object) -> object:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_ibis(gateway=env.gateway, ref=ds_ref)

    # Build signature with env and dataset ref parameters
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

    query_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    query_fn.__name__ = q_name
    query_fn.__doc__ = f"Load {table_key} as Ibis expression from {target_name} target."

    # Apply tag decorator for Hamilton observability
    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table=table_key, node_type="query")(query_fn)


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
    from codeintel.build.hamilton.io.ibis_adapter import load_dataset_df

    df_name = dataframe_node(table_key)
    d_name = dataset_node(table_key)

    def dataframe_fn(env: BuildEnv, **kwargs: object) -> object:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_df(gateway=env.gateway, ref=ds_ref)

    # Build signature with env and dataset ref parameters
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

    dataframe_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    dataframe_fn.__name__ = df_name
    dataframe_fn.__doc__ = f"Load {table_key} as pandas DataFrame from {target_name} target."

    # Apply tag decorator for Hamilton observability
    domain = table_key.split(".", 1)[0] if "." in table_key else "main"
    return tag(domain=domain, table=table_key, node_type="dataframe")(dataframe_fn)


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
    from codeintel.build.hamilton.io.artifact_ref import ArtifactRef

    a_name = artifact_node(artifact_name)
    t_name = target_node(target_name)

    def artifact_fn(env: BuildEnv, **kwargs: object) -> ArtifactRef:
        run_record = kwargs.get(t_name)
        if not isinstance(run_record, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(run_record)}"
            raise TypeError(msg)

        # Look up artifact from the run record
        for art in run_record.artifacts:
            if art.name == artifact_name:
                return art

        # Return a placeholder if artifact not found (for skipped targets)
        return ArtifactRef(
            name=artifact_name,
            artifact_type="unknown",
            repo=env.repo,
            commit=env.commit,
        )

    # Build signature with env and target node parameters
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

    artifact_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    artifact_fn.__name__ = a_name
    artifact_fn.__doc__ = f"Access {artifact_name} artifact from {target_name} target."

    # Apply tag decorator for Hamilton observability
    domain = artifact_name.split(".", 1)[0] if "." in artifact_name else "main"
    return tag(domain=domain, artifact=artifact_name, node_type="artifact")(artifact_fn)


def build_target_module(
    *,
    include_targets: set[str] | None = None,
    exclude_targets: set[str] | None = None,
    include_dataset_nodes: bool = True,
    include_loader_nodes: bool = True,
    include_artifact_nodes: bool = True,
) -> ModuleType:
    """Generate a module containing Hamilton nodes for all targets.

    Creates Python functions dynamically from the TargetGraph, with
    proper signatures and dependencies for Hamilton execution.

    Parameters
    ----------
    include_targets
        If provided, only generate nodes for these targets.
    exclude_targets
        If provided, exclude these targets from generation.
    include_dataset_nodes
        If True (default), also generate d__* dataset nodes for
        all tables in target contracts.
    include_loader_nodes
        If True (default), also generate q__* and df__* loader nodes
        for all tables in target contracts.
    include_artifact_nodes
        If True (default), also generate a__* artifact nodes for
        all artifacts in target contracts.

    Returns
    -------
    ModuleType
        Module containing generated node functions.

    Examples
    --------
    >>> module = build_target_module(exclude_targets={"export_jsonl"})
    >>> "t__function_metrics" in dir(module)
    True
    """
    graph = get_target_graph()
    all_target_names = {t.name for t in graph.all_targets}
    include = include_targets or all_target_names
    exclude = exclude_targets or set()

    # Create module
    module = ModuleType("codeintel.build.hamilton.nodes.generated")
    module.__doc__ = "Auto-generated Hamilton nodes from TargetGraph."

    # Track generated node names for mappings
    target_to_node: dict[str, str] = {}
    dataset_to_node: dict[str, str] = {}
    query_to_node: dict[str, str] = {}
    dataframe_to_node: dict[str, str] = {}
    artifact_to_node: dict[str, str] = {}

    for target in graph.all_targets:
        if target.name not in include or target.name in exclude:
            continue

        # Get metadata for domain
        meta = from_target(target)

        # Map dependencies to node names
        dep_node_names = [target_node(dep) for dep in target.dependencies]

        # Create and register target node function
        node_fn = _create_node_function(
            target=target,
            dep_node_names=dep_node_names,
            domain=meta.domain,
        )

        node_name = target_node(target.name)
        setattr(module, node_name, node_fn)
        target_to_node[target.name] = node_name

        # Generate dataset nodes for all contract tables
        if include_dataset_nodes:
            table_keys = target.contract.table_keys or target.table_keys
            for table_key in table_keys:
                dataset_fn = _create_dataset_node_function(
                    table_key=table_key,
                    target_name=target.name,
                )
                d_name = dataset_node(table_key)
                setattr(module, d_name, dataset_fn)
                dataset_to_node[table_key] = d_name

                # Generate q__* and df__* loader nodes
                if include_loader_nodes:
                    # Query node (returns Ibis expression)
                    q_fn = _create_query_node_function(
                        table_key=table_key,
                        target_name=target.name,
                    )
                    q_name = query_node(table_key)
                    setattr(module, q_name, q_fn)
                    query_to_node[table_key] = q_name

                    # DataFrame node (returns pandas DataFrame)
                    df_fn = _create_dataframe_node_function(
                        table_key=table_key,
                        target_name=target.name,
                    )
                    df_name = dataframe_node(table_key)
                    setattr(module, df_name, df_fn)
                    dataframe_to_node[table_key] = df_name

        # Generate artifact nodes for targets with artifact contracts
        if include_artifact_nodes:
            art_names = target.contract.artifact_names if target.contract else ()
            for art_name in art_names:
                a_fn = _create_artifact_node_function(
                    artifact_name=art_name,
                    target_name=target.name,
                )
                a_node_name = artifact_node(art_name)
                setattr(module, a_node_name, a_fn)
                artifact_to_node[art_name] = a_node_name

    # Attach mappings for executor lookups
    module.TARGET_TO_NODE = target_to_node  # type: ignore[attr-defined]
    module.DATASET_TO_NODE = dataset_to_node  # type: ignore[attr-defined]
    module.QUERY_TO_NODE = query_to_node  # type: ignore[attr-defined]
    module.DATAFRAME_TO_NODE = dataframe_to_node  # type: ignore[attr-defined]
    module.ARTIFACT_TO_NODE = artifact_to_node  # type: ignore[attr-defined]

    log.debug(
        "build_target_module: generated %d target, %d dataset, %d query, %d df, %d artifact",
        len(target_to_node),
        len(dataset_to_node),
        len(query_to_node),
        len(dataframe_to_node),
        len(artifact_to_node),
    )

    return module


# Cache for the generated module
_generated_module: ModuleType | None = None


def get_generated_module() -> ModuleType:
    """Get or create the generated nodes module.

    Returns a cached module instance, creating it on first call.
    This avoids regenerating nodes for each driver build.

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
    global _generated_module
    if _generated_module is None:
        _generated_module = build_target_module()
    return _generated_module


def clear_generated_module_cache() -> None:
    """Clear the cached generated module.

    Useful for testing or when the TargetGraph has changed.
    """
    global _generated_module
    _generated_module = None
