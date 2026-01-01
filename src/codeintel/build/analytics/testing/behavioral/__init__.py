"""Behavioral tagging and importance scoring.

This subpackage provides:

- ``tags``: Behavioral tag inference and LLM integration
- ``importance``: Flakiness and importance scoring
"""

from __future__ import annotations

from codeintel.build.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.build.analytics.testing.behavioral.tags import (
    BehaviorRowHooks,
    build_behavior_rows,
    build_test_ast_index,
    infer_behavior_tags,
    load_behavioral_context,
)

__all__ = [
    "BehaviorRowHooks",
    "build_behavior_rows",
    "build_test_ast_index",
    "compute_flakiness_score",
    "compute_importance_score",
    "infer_behavior_tags",
    "load_behavioral_context",
]
