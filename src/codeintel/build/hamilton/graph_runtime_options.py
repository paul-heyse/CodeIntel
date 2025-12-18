"""Build-side configuration loader for GraphRuntimeOptions.

Graph runtime options are used by analytics targets that need access to graph engines
(call graph, import graph, CFG/DFG views, etc.). These options should be loaded from
``env.config.parameters_for(target_name)`` to avoid plan/execution drift.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from codeintel.build.hamilton.env import BuildEnv
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.runtime import GraphRuntimeOptions


def _parse_with_default[T](
    raw: object | None,
    *,
    default: T,
    parser: Callable[[object], T],
) -> T:
    if raw is None:
        return default
    return parser(raw)


def _parse_optional[T](raw: object | None, *, parser: Callable[[object], T]) -> T | None:
    if raw is None:
        return None
    return parser(raw)


def _parse_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    message = f"Expected {key} to be bool, got {type(value)}"
    raise TypeError(message)


def _parse_str(value: object, *, key: str) -> str:
    if isinstance(value, str):
        return value
    message = f"Expected {key} to be str, got {type(value)}"
    raise TypeError(message)


def _parse_eager(value: object) -> bool:
    return _parse_bool(value, key="eager")


def _parse_validate(value: object) -> bool:
    return _parse_bool(value, key="validate")


def _parse_cache_key(value: object) -> str:
    return _parse_str(value, key="cache_key")


def _parse_graph_cache_dir(value: object) -> Path:
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, Path):
        return value
    message = f"Expected graph_cache_dir to be str|Path, got {type(value)}"
    raise TypeError(message)


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
    graphs = _parse_with_default(
        params.get("graphs"),
        default=GraphKind.ALL,
        parser=_parse_graph_kind,
    )
    eager = _parse_with_default(
        params.get("eager"),
        default=False,
        parser=_parse_eager,
    )
    validate = _parse_with_default(
        params.get("validate"),
        default=False,
        parser=_parse_validate,
    )
    cache_key = _parse_optional(params.get("cache_key"), parser=_parse_cache_key)
    graph_cache_dir = _parse_optional(params.get("graph_cache_dir"), parser=_parse_graph_cache_dir)
    backend = _parse_optional(params.get("backend"), parser=_parse_graph_backend)
    features = _parse_with_default(
        params.get("features"),
        default=GraphFeatureFlags(),
        parser=_parse_graph_features,
    )

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
