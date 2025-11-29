"""Behavioral tagging helpers for test analytics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import cast

from codeintel.analytics.tests_profiles.legacy import legacy
from codeintel.analytics.tests_profiles.types import (
    BehavioralLLMRunner,
    IoFlags,
    TestAstInfo,
)
from codeintel.config import BehavioralCoverageStepConfig
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema


def build_behavior_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    *,
    llm_runner: BehavioralLLMRunner | None = None,
) -> list[tuple[object, ...]]:
    """
    Build behavioral coverage rows for insertion.

    Returns
    -------
    list[tuple[object, ...]]
        Rows aligned with ``analytics.behavioral_coverage`` column order.
    """
    con = gateway.con
    ensure_schema(con, "analytics.behavioral_coverage")
    tests = legacy.load_test_records_public(con, cfg.repo, cfg.commit)
    if not tests:
        return []

    ast_info = legacy.build_test_ast_index(
        cfg.repo_root,
        tests,
        legacy.DEFAULT_IO_SPEC,
        legacy.CONCURRENCY_LIBS,
    )
    profile_ctx = legacy.load_test_profile_context(con, cfg.repo, cfg.commit)
    behavior_ctx = legacy.BehavioralContext(
        cfg=cfg,
        ast_info=ast_info,
        profile_ctx=profile_ctx,
        now=datetime.now(tz=UTC),
        llm_runner=llm_runner,
    )
    return [legacy.build_behavior_row(test, behavior_ctx) for test in tests]


def infer_behavior_tags(
    *,
    name: str,
    markers: Iterable[str],
    io_flags: IoFlags,
    ast_info: TestAstInfo,
) -> list[str]:
    """
    Infer behavior tags from name, markers, IO flags, and AST info.

    Returns
    -------
    list[str]
        Sorted list of inferred behavior tags.
    """
    return legacy.infer_behavior_tags(
        name=name,
        markers=markers,
        io_flags=io_flags,
        ast_info=ast_info,
    )


def load_behavioral_context(
    con: DuckDBConnection,
    cfg: BehavioralCoverageStepConfig,
) -> Mapping[str, dict[str, object]]:
    """
    Load behavioral profile context from analytics.test_profile.

    Returns
    -------
    Mapping[str, dict[str, object]]
        Context keyed by ``test_id``.
    """
    return cast(
        "Mapping[str, dict[str, object]]",
        legacy.load_test_profile_context(con, cfg.repo, cfg.commit),
    )
