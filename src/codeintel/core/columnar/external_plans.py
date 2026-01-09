"""Registration helpers for optional external plan engines."""

from __future__ import annotations

from collections.abc import Callable

_register_substrait_plan_runner: Callable[[], None] | None
_register_datafusion_plan_runner: Callable[[], None] | None
_register_rustworkx_plan_runner: Callable[[], None] | None

try:
    from codeintel.build.tabular.substrait_ops import (
        register_substrait_plan_runner as _register_substrait_plan_runner,
    )
except ImportError:
    _register_substrait_plan_runner = None

try:
    from codeintel.build.tabular.datafusion_ops import (
        register_datafusion_plan_runner as _register_datafusion_plan_runner,
    )
except ImportError:
    _register_datafusion_plan_runner = None

try:
    from codeintel.build.graphs.external_plan import (
        register_rustworkx_plan_runner as _register_rustworkx_plan_runner,
    )
except ImportError:
    _register_rustworkx_plan_runner = None


def register_default_external_plan_runners() -> None:
    """Register optional external plan runners when available."""
    if _register_substrait_plan_runner is not None:
        _register_substrait_plan_runner()
    if _register_datafusion_plan_runner is not None:
        _register_datafusion_plan_runner()
    if _register_rustworkx_plan_runner is not None:
        _register_rustworkx_plan_runner()


__all__ = ["register_default_external_plan_runners"]
