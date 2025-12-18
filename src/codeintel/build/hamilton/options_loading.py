"""Canonical target options loading for native build targets.

The build planner computes an options hash from ``env.config.parameters_for(target_name)``.
To avoid plan/execution drift, native targets should construct their options objects from the
same parameter mapping, rather than instantiating options dataclasses directly.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, cast, runtime_checkable

from codeintel.build.hamilton.env import BuildEnv

if TYPE_CHECKING:
    from collections.abc import Mapping

OptionsT = TypeVar("OptionsT", bound=object)


@runtime_checkable
class _FromParameters(Protocol[OptionsT]):
    @classmethod
    def from_parameters(cls, params: Mapping[str, object], /) -> OptionsT: ...


def load_target_options(env: BuildEnv, *, target_name: str, options_type: type[OptionsT]) -> OptionsT:
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
        If configuration parameters are not a mapping, or if options cannot be constructed.
    """
    params_raw = env.config.parameters_for(target_name)
    if not params_raw:
        return options_type()
    if not isinstance(params_raw, dict):
        msg = (
            f"Expected env.config.parameters_for({target_name!r}) to return dict[str, object], "
            f"got {type(params_raw)}"
        )
        raise TypeError(msg)

    params = cast("Mapping[str, object]", params_raw)

    from_params = getattr(options_type, "from_parameters", None)
    if callable(from_params):
        return cast("_FromParameters[OptionsT]", options_type).from_parameters(params)

    if not is_dataclass(options_type):
        msg = f"Options type must be a dataclass or implement from_parameters(): {options_type}"
        raise TypeError(msg)

    try:
        return options_type(**dict(params))
    except TypeError as exc:
        msg = f"Failed to construct {options_type} for {target_name} from params={dict(params)!r}"
        raise TypeError(msg) from exc


__all__ = [
    "load_target_options",
]
