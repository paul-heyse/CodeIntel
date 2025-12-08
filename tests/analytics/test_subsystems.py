"""Subsystem inference tests covering clustering and risk aggregation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from codeintel.analytics.subsystems import build_subsystems
from codeintel.config import ConfigBuilder, SnapshotInit
from tests._helpers import TestContext
from tests._helpers.builders import (
    ConfigValueRow,
    ImportGraphEdgeRow,
    RiskFactorRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.rows import function_metrics_row, module_row

# Test constants
EXPECTED_SUBSYSTEMS = 2
EXPECTED_MEMBERSHIPS = 3
TARGET_CLUSTER_SIZE = 2
EXPECTED_HIGH_RISK_COUNT = 1

# GOID constants for subsystem test functions
GOID_API_HANDLER = 10
GOID_CORE_SERVICE = 11


def _seed_clustering_data(ctx: TestContext) -> None:
    """Seed modules, edges, and risk data to drive clustering.

    This creates a specific dataset designed to test the clustering algorithm:
    - Two tightly-coupled modules (pkg.api, pkg.core) that should cluster together
    - One isolated module (pkg.misc) that should be in its own cluster
    - Risk factors that result in one high-risk function

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    # Seed modules
    insert_rows(
        ctx.gateway,
        [
            module_row(
                module="pkg.api",
                path="pkg/api.py",
                snapshot=(ctx.repo, ctx.commit),
            ),
            module_row(
                module="pkg.core",
                path="pkg/core.py",
                snapshot=(ctx.repo, ctx.commit),
            ),
            module_row(
                module="pkg.misc",
                path="pkg/misc.py",
                snapshot=(ctx.repo, ctx.commit),
            ),
        ],
    )

    # Seed bidirectional import edges to create tight coupling
    insert_rows(
        ctx.gateway,
        [
            ImportGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                src_module="pkg.api",
                dst_module="pkg.core",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
            ImportGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                src_module="pkg.core",
                dst_module="pkg.api",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
        ],
    )

    # Seed symbol use edge
    insert_rows(
        ctx.gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym_core",
                def_path="pkg/core.py",
                use_path="pkg/api.py",
                same_file=False,
                same_module=False,
            )
        ],
    )

    # Seed config value referencing both coupled modules
    insert_rows(
        ctx.gateway,
        [
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=["pkg.api", "pkg.core"],
                reference_count=2,
            )
        ],
    )

    # Seed function metrics and risk factors
    now = datetime.now(tz=UTC)
    _seed_function_metrics(ctx, now)
    _seed_risk_factors(ctx, now)


def _seed_function_metrics(ctx: TestContext, now: datetime) -> None:
    """Seed function metrics for the test functions.

    Parameters
    ----------
    ctx
        Test context with gateway.
    now
        Timestamp for created_at fields.
    """
    insert_rows(
        ctx.gateway,
        [
            function_metrics_row(
                goid=GOID_API_HANDLER,
                rel_path="pkg/api.py",
                qualname="pkg.api.handler",
                snapshot=(ctx.repo, ctx.commit),
                metrics={
                    "language": "python",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "loc": 4,
                    "logical_loc": 3,
                    "param_count": 1,
                    "positional_params": 1,
                    "has_docstring": True,
                    "created_at": now,
                },
            ),
            function_metrics_row(
                goid=GOID_CORE_SERVICE,
                rel_path="pkg/core.py",
                qualname="pkg.core.service",
                snapshot=(ctx.repo, ctx.commit),
                metrics={
                    "language": "python",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "loc": 4,
                    "logical_loc": 3,
                    "param_count": 1,
                    "positional_params": 1,
                    "has_docstring": True,
                    "created_at": now,
                },
            ),
        ],
    )


def _seed_risk_factors(ctx: TestContext, now: datetime) -> None:
    """Seed risk factors with one high-risk function.

    Parameters
    ----------
    ctx
        Test context with gateway.
    now
        Timestamp for created_at fields.
    """
    insert_rows(
        ctx.gateway,
        [
            RiskFactorRow(
                function_goid_h128=GOID_API_HANDLER,
                urn=f"goid:{ctx.repo}#python:function:pkg.api.handler",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="pkg/api.py",
                language="python",
                kind="function",
                qualname="pkg.api.handler",
                loc=4,
                logical_loc=3,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=4,
                covered_lines=2,
                coverage_ratio=0.5,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="all_passing",
                risk_score=0.2,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
            RiskFactorRow(
                function_goid_h128=GOID_CORE_SERVICE,
                urn=f"goid:{ctx.repo}#python:function:pkg.core.service",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="pkg/core.py",
                language="python",
                kind="function",
                qualname="pkg.core.service",
                loc=4,
                logical_loc=3,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=4,
                covered_lines=2,
                coverage_ratio=0.5,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="all_passing",
                risk_score=0.8,
                risk_level="high",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
        ],
    )


def test_subsystems_cluster_and_risk_aggregation(test_ctx: TestContext) -> None:
    """Cluster modules and aggregate risk across subsystems.

    Verifies that:
    - Tightly-coupled modules (pkg.api, pkg.core) cluster together
    - The cluster with high-risk functions is marked as high-risk
    - All modules are assigned to subsystems
    """
    _seed_clustering_data(test_ctx)

    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(
            repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
        ),
    ).subsystems(
        max_subsystems=2,
        min_modules=1,
    )
    build_subsystems(test_ctx.gateway, cfg)

    # Verify subsystem count
    subsystems = test_ctx.query(
        """
        SELECT subsystem_id, modules_json, risk_level, high_risk_function_count
        FROM analytics.subsystems
        """
    )
    if len(subsystems) != EXPECTED_SUBSYSTEMS:
        pytest.fail(f"Expected {EXPECTED_SUBSYSTEMS} subsystems, found {len(subsystems)}")

    # Find the larger cluster and verify its properties
    by_size: dict[int, tuple[str, str, int]] = {}
    for row in subsystems:
        modules_json = str(row.modules_json)
        risk_level = str(row.risk_level)
        high_risk_count_raw = row.high_risk_function_count
        high_risk_count = int(str(high_risk_count_raw)) if high_risk_count_raw is not None else 0
        modules_list: list[str] = json.loads(modules_json)
        by_size[len(modules_list)] = (modules_json, risk_level, high_risk_count)
    large_modules, large_risk, high_count = by_size[TARGET_CLUSTER_SIZE]
    if "pkg.api" not in large_modules or "pkg.core" not in large_modules:
        pytest.fail(f"Subsystem missing expected modules: {large_modules}")
    if large_risk != "high":
        pytest.fail(f"Expected high risk for core cluster, got {large_risk}")
    if high_count != EXPECTED_HIGH_RISK_COUNT:
        pytest.fail(f"Expected one high-risk function, got {high_count}")

    # Verify all modules are assigned
    memberships = test_ctx.query("SELECT subsystem_id, module FROM analytics.subsystem_modules")
    if len(memberships) != EXPECTED_MEMBERSHIPS:
        pytest.fail(f"Expected {EXPECTED_MEMBERSHIPS} memberships, got {len(memberships)}")
    members = {str(row.module) for row in memberships}
    if members != {"pkg.api", "pkg.core", "pkg.misc"}:
        pytest.fail(f"Unexpected subsystem membership: {members}")
