"""Registration helpers for optional external plan engines."""

from __future__ import annotations


def register_default_external_plan_runners() -> None:
    """Register optional external plan runners when available."""
    _register_substrait_runner()
    _register_datafusion_runner()


def _register_substrait_runner() -> None:
    try:
        from codeintel.build.tabular.substrait_ops import register_substrait_plan_runner
    except ImportError:
        return
    register_substrait_plan_runner()


def _register_datafusion_runner() -> None:
    try:
        from codeintel.build.tabular.datafusion_ops import register_datafusion_plan_runner
    except ImportError:
        return
    register_datafusion_plan_runner()


__all__ = ["register_default_external_plan_runners"]
