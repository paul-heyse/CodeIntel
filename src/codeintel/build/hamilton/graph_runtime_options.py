"""Build-side configuration loader for GraphRuntimeOptions.

Graph runtime options are used by analytics targets that need access to graph engines
(call graph, import graph, CFG/DFG views, etc.). These options should be loaded from
``env.config.parameters_for(target_name)`` to avoid plan/execution drift.
"""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.env import BuildEnv
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.runtime import GraphRuntimeOptions


def _parse_graph_kind(value: object) -> GraphKind:
    if isinstance(value, GraphKind):
        return value
    if isinstance(value, int):
        return GraphKind(value)
    if isinstance(value, str):
        tokens = [t.strip() for t in value.split(",") if t.strip()]
        if not tokens:
            message = "graphs value must not be empty"
            raise ValueError(message)
        kind = GraphKind.NONE
        for token in tokens:
            name = token.upper()
            try:
                kind |= GraphKind[name]
            except KeyError as exc:
                message = f"Unknown GraphKind: {token!r}"
                raise ValueError(message) from exc
        return kind
    if isinstance(value, list):
        kind = GraphKind.NONE
        for item in value:
            kind |= _parse_graph_kind(item)
        return kind
    message = f"Unsupported graphs value type: {type(value)}"
    raise TypeError(message)


def _parse_graph_backend(value: object) -> GraphBackendConfig:
    if isinstance(value, GraphBackendConfig):
        return value
    if isinstance(value, dict):
        try:
            backend = GraphBackendConfig(**value)
        except TypeError as exc:
            message = f"Invalid backend mapping for GraphBackendConfig: {value!r}"
            raise TypeError(message) from exc
        return backend
    message = f"Unsupported backend value type: {type(value)}"
    raise TypeError(message)


def _parse_graph_features(value: object) -> GraphFeatureFlags:
    if isinstance(value, GraphFeatureFlags):
        value.validate()
        return value
    if isinstance(value, dict):
        try:
            flags = GraphFeatureFlags(**value)
        except TypeError as exc:
            message = f"Invalid features mapping for GraphFeatureFlags: {value!r}"
            raise TypeError(message) from exc
        flags.validate()
        return flags
    message = f"Unsupported features value type: {type(value)}"
    raise TypeError(message)


def load_graph_runtime_options(
    env: BuildEnv,
    *,
    target_name: str,
) -> GraphRuntimeOptions:
    """Load GraphRuntimeOptions from BuildEnv configuration.

    Parameters
    ----------
    env
        Build environment providing snapshot and configuration.
    target_name
        Target name whose configuration section should be loaded.

    Returns
    -------
    GraphRuntimeOptions
        Runtime options with snapshot/feature/backend defaults normalized.
    """
    params = env.config.parameters_for(target_name).as_dict()
    graphs = GraphKind.ALL
    eager = False
    validate = False
    cache_key: str | None = None
    graph_cache_dir: Path | None = None
    backend: GraphBackendConfig | None = None
    features = GraphFeatureFlags()

    raw_graphs = params.get("graphs")
    if raw_graphs is not None:
        graphs = _parse_graph_kind(raw_graphs)

    raw_eager = params.get("eager")
    if raw_eager is not None:
        if not isinstance(raw_eager, bool):
            message = f"Expected eager to be bool, got {type(raw_eager)}"
            raise TypeError(message)
        eager = raw_eager

    raw_validate = params.get("validate")
    if raw_validate is not None:
        if not isinstance(raw_validate, bool):
            message = f"Expected validate to be bool, got {type(raw_validate)}"
            raise TypeError(message)
        validate = raw_validate

    raw_cache_key = params.get("cache_key")
    if raw_cache_key is not None:
        if not isinstance(raw_cache_key, str):
            message = f"Expected cache_key to be str, got {type(raw_cache_key)}"
            raise TypeError(message)
        cache_key = raw_cache_key

    raw_graph_cache_dir = params.get("graph_cache_dir")
    if raw_graph_cache_dir is not None:
        if isinstance(raw_graph_cache_dir, str):
            graph_cache_dir = Path(raw_graph_cache_dir)
        elif isinstance(raw_graph_cache_dir, Path):
            graph_cache_dir = raw_graph_cache_dir
        else:
            message = f"Expected graph_cache_dir to be str|Path, got {type(raw_graph_cache_dir)}"
            raise TypeError(message)

    raw_backend = params.get("backend")
    if raw_backend is not None:
        backend = _parse_graph_backend(raw_backend)

    raw_features = params.get("features")
    if raw_features is not None:
        features = _parse_graph_features(raw_features)

    return GraphRuntimeOptions(
        snapshot=env.snapshot,
        backend=backend,
        graphs=graphs,
        eager=eager,
        validate=validate,
        cache_key=cache_key,
        engine=None,
        graph_cache_dir=graph_cache_dir,
        features=features,
    )


__all__ = [
    "load_graph_runtime_options",
]
