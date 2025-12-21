"""Native OutputTarget spec loader used for Hamilton graph construction."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.hamilton.native.discovery import load_native_modules
from codeintel.build.targets import OutputTarget

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType


def _iter_module_target_specs(module: ModuleType) -> Iterable[OutputTarget]:
    specs_obj = getattr(module, "TARGET_SPECS", None)
    if specs_obj is None:
        return ()

    if isinstance(specs_obj, (tuple, list)):
        specs: list[OutputTarget] = []
        for item in specs_obj:
            if not isinstance(item, OutputTarget):
                msg = (
                    f"{module.__name__}.TARGET_SPECS contains non-OutputTarget element: "
                    f"{type(item)}"
                )
                raise TypeError(msg)
            specs.append(item)
        return tuple(specs)

    msg = (
        f"{module.__name__}.TARGET_SPECS must be a tuple/list of OutputTarget, got "
        f"{type(specs_obj)}"
    )
    raise TypeError(msg)


def _validate_specs(specs: Iterable[OutputTarget]) -> tuple[OutputTarget, ...]:
    by_name: dict[str, OutputTarget] = {}
    for target in specs:
        if target.name in by_name:
            msg = f"Duplicate target spec name: {target.name}"
            raise ValueError(msg)
        if target.dependencies:
            msg = (
                "Target specs must not declare dependencies; Hamilton is the single source of "
                f"truth. Found dependencies for {target.name}: {target.dependencies!r}"
            )
            raise ValueError(msg)
        by_name[target.name] = target

    return tuple(by_name[name] for name in sorted(by_name))


@lru_cache(maxsize=1)
def load_native_target_specs() -> tuple[OutputTarget, ...]:
    """Load OutputTarget specs from native Hamilton modules.

    Returns
    -------
    tuple[OutputTarget, ...]
        Validated output target specs in deterministic name order.
    """
    specs: list[OutputTarget] = []
    for module in load_native_modules():
        specs.extend(_iter_module_target_specs(module))
    return _validate_specs(specs)


__all__ = [
    "load_native_target_specs",
]
