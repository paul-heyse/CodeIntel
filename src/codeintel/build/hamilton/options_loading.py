"""Canonical target options loading for native build targets.

The build planner computes an options hash from ``env.config.parameters_for(target_name)``.
To avoid plan/execution drift, native targets should construct their options objects from the
same parameter mapping, rather than instantiating options dataclasses directly.
"""

from __future__ import annotations

from dataclasses import Field, is_dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard, cast, runtime_checkable

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.parameters import TargetParameters

if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
class _FromParameters[OptionsT: object](Protocol):
    @classmethod
    def from_parameters(cls, params: Mapping[str, object], /) -> OptionsT: ...


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]


class _ScopePathsDataclass(_DataclassInstance, Protocol):
    scope_paths: list[str] | None


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return is_dataclass(value) and not isinstance(value, type)


def load_target_options[OptionsT: object](
    env: BuildEnv,
    *,
    target_name: str,
    options_type: type[OptionsT],
) -> OptionsT:
    """Load a target options object from BuildEnv configuration.

    Parameters
    ----------
    env
        Build environment providing config access.
    target_name
        Target name whose configuration section should be loaded.
    options_type
        Options class to instantiate.

    Returns
    -------
    OptionsT
        Instantiated options object.

    Raises
    ------
    TypeError
        If options cannot be constructed.
    """
    params: TargetParameters = env.config.parameters_for(target_name)
    if len(params) == 0:
        return cast("OptionsT", _apply_global_scope(env, options_type()))
    mapping = cast("Mapping[str, object]", params)

    from_params = getattr(options_type, "from_parameters", None)
    if callable(from_params):
        options = cast("_FromParameters[OptionsT]", options_type).from_parameters(mapping)
        return cast("OptionsT", _apply_global_scope(env, options))

    if not is_dataclass(options_type):
        msg = f"Options type must be a dataclass or implement from_parameters(): {options_type}"
        raise TypeError(msg)

    try:
        options = options_type(**dict(mapping))
        return cast("OptionsT", _apply_global_scope(env, options))
    except TypeError as exc:
        msg = f"Failed to construct {options_type} for {target_name} from params={dict(mapping)!r}"
        raise TypeError(msg) from exc


def _apply_global_scope(env: BuildEnv, options: object) -> object:
    scope_paths = _resolve_global_scope_paths(env)
    if scope_paths is None:
        return options
    if not _is_dataclass_instance(options):
        return options
    if "scope_paths" not in options.__dataclass_fields__:
        return options
    current = getattr(options, "scope_paths", None)
    if current is not None:
        return options
    return replace(cast("_ScopePathsDataclass", options), scope_paths=list(scope_paths))


def _resolve_global_scope_paths(env: BuildEnv) -> tuple[str, ...] | None:
    raw = env.config.get("scope.scope_paths")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        msg = "scope.scope_paths must be a list of strings"
        raise TypeError(msg)
    normalized = tuple(value.strip() for value in raw if value.strip())
    if not normalized:
        return None
    return normalized


__all__ = [
    "load_target_options",
]
