"""Guardrails for semantic serving execution paths."""

from __future__ import annotations

import logging

LOG = logging.getLogger("codeintel.serving.guardrails")


def warn_eager_materialization(*, engine: str, context: str) -> None:
    """Log a warning when an eager materialization path is used.

    Parameters
    ----------
    engine
        Engine name responsible for eager materialization.
    context
        Additional context identifier for the eager materialization site.
    """
    LOG.warning(
        "Eager materialization in serving path",
        extra={"engine": engine, "context": context},
    )


__all__ = ["warn_eager_materialization"]
