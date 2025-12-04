"""Behavioral tagging - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.behavioral.tags``.

Note: This module provides a wrapper for build_behavior_rows that uses
this module's namespace for ensure_schema. This supports legacy test
patterns that override ensure_schema at the module level.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from codeintel.analytics.ast_features.patterns import DEFAULT_PATTERNS
from codeintel.analytics.testing.behavioral.tags import (
    BehaviorRowHooks,
    SpanConfig,
    SpanState,
    _build_behavior_row,
    build_test_ast_index,
    infer_behavior_tags,
    load_behavioral_context,
)
from codeintel.analytics.testing.coverage.inputs import load_test_records
from codeintel.analytics.testing.profiles.types import (
    BehavioralContext,
    BehavioralLLMRequest,
    BehavioralLLMRunner,
    TestAstInfo,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema as _ensure_schema

# Module-level reference that can be overridden by tests
ensure_schema = _ensure_schema


def build_behavior_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    *,
    llm_runner: BehavioralLLMRunner | None = None,
    hooks: BehaviorRowHooks | None = None,
) -> list[tuple[object, ...]]:
    """Build behavioral coverage rows for insertion.

    This wrapper uses module-level ensure_schema to support legacy test
    patterns that override it at the module level.

    Returns
    -------
    list[tuple[object, ...]]
        Rows aligned with ``analytics.behavioral_coverage`` column order.
    """
    # Access this module's namespace to get potentially overridden function
    this_module = sys.modules[__name__]
    ensure_fn = this_module.ensure_schema

    con = gateway.con
    ensure_fn(con, "analytics.behavioral_coverage")

    load_tests_fn = hooks.load_tests if hooks is not None else None
    if load_tests_fn is None:
        load_tests_fn = load_test_records
    tests = load_tests_fn(con, cfg)
    if not tests:
        return []

    ast_builder = hooks.build_ast if hooks is not None else None
    if ast_builder is None:
        ast_builder = build_test_ast_index
    ast_info = ast_builder(cfg.repo_root, tests, DEFAULT_PATTERNS)

    profile_loader = hooks.load_profile_ctx if hooks is not None else None
    if profile_loader is None:
        profile_loader = load_behavioral_context
    profile_ctx = profile_loader(con, cfg)

    behavior_ctx = BehavioralContext(
        cfg=cfg,
        ast_info=ast_info,
        profile_ctx=profile_ctx,
        now=datetime.now(tz=UTC),
        llm_runner=llm_runner,
    )

    row_fn = hooks.row_builder if hooks is not None else None
    if row_fn is None:
        row_fn = _build_behavior_row
    return [row_fn(test, behavior_ctx) for test in tests]


__all__ = [
    "BehaviorRowHooks",
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMRunner",
    "SpanConfig",
    "SpanState",
    "TestAstInfo",
    "TestRecord",
    "build_behavior_rows",
    "build_test_ast_index",
    "ensure_schema",
    "infer_behavior_tags",
    "load_behavioral_context",
    "load_test_records",
]
