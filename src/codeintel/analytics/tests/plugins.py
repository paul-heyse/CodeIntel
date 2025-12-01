"""Analytics plugins for test profiles and behavioral coverage."""

from __future__ import annotations

from typing import cast

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.tests.profiles import build_behavioral_coverage, build_test_profile
from codeintel.analytics.tests_profiles.types import BehavioralLLMRunner
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig, TestProfileStepConfig


def _test_profile_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from generic context to build_test_profile.

    Returns
    -------
    dict[str, int]
        Row count summary for test_profile.

    Raises
    ------
    ValueError
        If the test profile config is missing from the execution context.
    """
    if ctx.test_profile_cfg is None:
        message = "TestProfileStepConfig is required in AnalyticsExecutionContext.test_profile_cfg"
        raise ValueError(message)

    cfg: TestProfileStepConfig = ctx.test_profile_cfg
    gateway = ctx.gateway
    build_test_profile(gateway, cfg)
    con = gateway.con
    row = con.execute(
        """
        SELECT COUNT(*) FROM analytics.test_profile
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    row_count = int(row[0]) if row else 0
    return {"profile_rows": row_count}


TEST_PROFILE_PLUGIN = AnalyticsPlugin(
    name="tests.profile",
    description="Build per-test profiles with coverage and subsystem context.",
    stage="test",
    enabled_by_default=True,
    run=_test_profile_run,
    severity="fatal",
    depends_on=(),
    provides=("analytics.test_profile",),
    requires=("core.goids", "coverage.test_edges"),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=60_000,
        requires_gpu=False,
        priority=20,
    ),
    version_hash=None,
    row_count_tables=("analytics.test_profile",),
)


def _behavioral_coverage_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from generic context to build_behavioral_coverage.

    Returns
    -------
    dict[str, int]
        Row count summary for behavioral coverage.

    Raises
    ------
    ValueError
        If the behavioral coverage config is missing.
    TypeError
        If the provided llm runner is not callable.
    """
    if ctx.behavioral_cfg is None:
        message = (
            "BehavioralCoverageStepConfig is required in AnalyticsExecutionContext.behavioral_cfg"
        )
        raise ValueError(message)

    cfg: BehavioralCoverageStepConfig = ctx.behavioral_cfg
    gateway = ctx.gateway

    llm_runner_raw = ctx.extra.get("behavioral_llm_runner")
    llm_runner: BehavioralLLMRunner | None = None
    if llm_runner_raw is not None:
        if not callable(llm_runner_raw):
            message = "behavioral_llm_runner in ctx.extra must be callable or None"
            raise TypeError(message)
        llm_runner = cast("BehavioralLLMRunner", llm_runner_raw)

    if llm_runner_raw is not None and llm_runner is None:
        message = "behavioral_llm_runner in ctx.extra must be callable or None"
        raise TypeError(message)

    build_behavioral_coverage(
        gateway,
        cfg,
        llm_runner=llm_runner,
    )

    con = gateway.con
    row = con.execute(
        """
        SELECT COUNT(*) FROM analytics.behavioral_coverage
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    row_count = int(row[0]) if row else 0
    return {"behavior_rows": row_count}


BEHAVIORAL_COVERAGE_PLUGIN = AnalyticsPlugin(
    name="tests.behavioral_coverage",
    description="Assign heuristic behavior tags to tests (unit/integration/etc.).",
    stage="test",
    enabled_by_default=True,
    run=_behavioral_coverage_run,
    severity="fatal",
    depends_on=(),
    provides=("analytics.behavioral_coverage",),
    requires=("analytics.test_profile",),
    options_model=None,
    options_default={"enable_llm": False, "llm_model": None},
    resource_hints=ResourceHints(
        max_runtime_ms=120_000,
        requires_gpu=False,
        priority=30,
    ),
    version_hash=None,
    row_count_tables=("analytics.behavioral_coverage",),
)

register_analytics_plugin(TEST_PROFILE_PLUGIN)
register_analytics_plugin(BEHAVIORAL_COVERAGE_PLUGIN)


__all__ = [
    "BEHAVIORAL_COVERAGE_PLUGIN",
    "TEST_PROFILE_PLUGIN",
]
