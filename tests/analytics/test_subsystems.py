"""Subsystem inference tests covering clustering and risk aggregation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.subsystems import build_subsystems
from codeintel.config import ConfigBuilder
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import (
    ConfigValueRow,
    FunctionMetricsRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RiskFactorRow,
    SymbolUseEdgeRow,
    insert_config_values,
    insert_function_metrics,
    insert_import_graph_edges,
    insert_modules,
    insert_risk_factors,
    insert_symbol_use_edges,
)

REPO = "demo/repo"
COMMIT = "abc123"
EXPECTED_SUBSYSTEMS = 2
EXPECTED_MEMBERSHIPS = 3
TARGET_CLUSTER_SIZE = 2
EXPECTED_HIGH_RISK_COUNT = 1


def _seed_modules(gateway: StorageGateway) -> None:
    """Insert sample modules, edges, and risk data to drive clustering."""
    insert_modules(
        gateway,
        [
            ModuleRow(
                module="pkg.api",
                path="pkg/api.py",
                repo=REPO,
                commit=COMMIT,
                tags='["api"]',
            ),
            ModuleRow(
                module="pkg.core",
                path="pkg/core.py",
                repo=REPO,
                commit=COMMIT,
                tags='["api"]',
            ),
            ModuleRow(
                module="pkg.misc",
                path="pkg/misc.py",
                repo=REPO,
                commit=COMMIT,
            ),
        ],
    )
    insert_import_graph_edges(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=REPO,
                commit=COMMIT,
                src_module="pkg.api",
                dst_module="pkg.core",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
            ImportGraphEdgeRow(
                repo=REPO,
                commit=COMMIT,
                src_module="pkg.core",
                dst_module="pkg.api",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
        ],
    )
    insert_symbol_use_edges(
        gateway,
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
    insert_config_values(
        gateway,
        [
            ConfigValueRow(
                repo=REPO,
                commit=COMMIT,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=["pkg.api", "pkg.core"],
                reference_count=2,
            )
        ],
    )
    now = datetime.now(tz=UTC)
    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=10,
                urn="goid:demo/repo#python:function:pkg.api.handler",
                repo=REPO,
                commit=COMMIT,
                rel_path="pkg/api.py",
                language="python",
                kind="function",
                qualname="pkg.api.handler",
                start_line=1,
                end_line=2,
                loc=4,
                logical_loc=3,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=2,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
            FunctionMetricsRow(
                function_goid_h128=11,
                urn="goid:demo/repo#python:function:pkg.core.service",
                repo=REPO,
                commit=COMMIT,
                rel_path="pkg/core.py",
                language="python",
                kind="function",
                qualname="pkg.core.service",
                start_line=1,
                end_line=2,
                loc=4,
                logical_loc=3,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=2,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
        ],
    )
    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=10,
                urn="goid:demo/repo#python:function:pkg.api.handler",
                repo=REPO,
                commit=COMMIT,
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
                function_goid_h128=11,
                urn="goid:demo/repo#python:function:pkg.core.service",
                repo=REPO,
                commit=COMMIT,
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


def test_subsystems_cluster_and_risk_aggregation(fresh_gateway: StorageGateway) -> None:
    """Clusters modules and aggregates risk across subsystems."""
    gateway = fresh_gateway
    con = gateway.con
    _seed_modules(gateway)

    cfg = ConfigBuilder.from_snapshot(repo=REPO, commit=COMMIT, repo_root=Path()).subsystems(
        max_subsystems=2,
        min_modules=1,
    )
    build_subsystems(gateway, cfg)

    subsystems = con.execute(
        """
        SELECT subsystem_id, modules_json, risk_level, high_risk_function_count
        FROM analytics.subsystems
        """
    ).fetchall()
    if len(subsystems) != EXPECTED_SUBSYSTEMS:
        pytest.fail(f"Expected {EXPECTED_SUBSYSTEMS} subsystems, found {len(subsystems)}")

    by_size = {len(json.loads(mods)): (mods, risk, high) for _, mods, risk, high in subsystems}
    large_modules, large_risk, high_count = by_size[TARGET_CLUSTER_SIZE]
    if "pkg.api" not in large_modules or "pkg.core" not in large_modules:
        pytest.fail(f"Subsystem missing expected modules: {large_modules}")
    if large_risk != "high":
        pytest.fail(f"Expected high risk for core cluster, got {large_risk}")
    if high_count != EXPECTED_HIGH_RISK_COUNT:
        pytest.fail(f"Expected one high-risk function, got {high_count}")

    memberships = con.execute(
        "SELECT subsystem_id, module FROM analytics.subsystem_modules"
    ).fetchall()
    if len(memberships) != EXPECTED_MEMBERSHIPS:
        pytest.fail(f"Expected {EXPECTED_MEMBERSHIPS} memberships, got {len(memberships)}")
    members = {row[1] for row in memberships}
    if members != {"pkg.api", "pkg.core", "pkg.misc"}:
        pytest.fail(f"Unexpected subsystem membership: {members}")
