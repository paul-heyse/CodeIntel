"""Risk factors should use catalog module map when core.modules is empty."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import GraphBackendConfig, ToolsConfig
from codeintel.core.config import (
    ExecutionConfig,
    PathsConfig,
    ScanProfilesConfig,
    SnapshotConfig,
)
from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.function_catalog_service import FunctionCatalogService
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
from codeintel.orchestration.steps import PipelineContext, RiskFactorsStep
from codeintel.storage.gateway import StorageConfig, open_gateway


def test_risk_factors_uses_catalog_modules_when_core_modules_empty(tmp_path: Path) -> None:
    """Risk factors should populate rows using catalog module map when core.modules is empty."""
    db_path = tmp_path / "db.duckdb"
    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=True, validate_schema=True)
    )
    con = gateway.con

    # Insert minimal function_metrics row required by risk factors.
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, loc, logical_loc, param_count, positional_params,
            keyword_only_params, has_varargs, has_varkw, is_async, is_generator,
            return_count, yield_count, raise_count, cyclomatic_complexity,
            max_nesting_depth, stmt_count, decorator_count, has_docstring,
            complexity_bucket, created_at
        ) VALUES (
            1, 'urn:fn', 'r', 'c', 'm.py', 'python', 'function', 'pkg.m.fn',
            1, 2, 2, 2, 0, 0, 0, FALSE, FALSE, FALSE, FALSE, 0, 0, 0, 1, 0, 1, 0, TRUE, 'low', NOW()
        )
        """
    )

    # Provide coverage_functions so risk score math has concrete values.
    con.execute(
        """
        INSERT INTO analytics.coverage_functions (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, executable_lines, covered_lines, coverage_ratio, tested,
            untested_reason, created_at
        ) VALUES (
            1, 'urn:fn', 'r', 'c', 'm.py', 'python', 'function', 'pkg.m.fn',
            1, 2, 2, 2, 1.0, TRUE, 'tested', NOW()
        )
        """
    )

    # Catalog supplies module mapping in lieu of core.modules rows.
    catalog = FunctionCatalog(functions=(), module_by_path={"m.py": "pkg.m"})
    provider = FunctionCatalogService(catalog)
    snapshot = SnapshotConfig(repo_root=tmp_path, repo_slug="r", commit="c")
    profiles = ScanProfilesConfig(
        code=default_code_profile(tmp_path),
        config=default_config_profile(tmp_path),
    )
    execution = ExecutionConfig.for_default_pipeline(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.model_validate({}),
        profiles=profiles,
        graph_backend=GraphBackendConfig(),
    )
    paths = PathsConfig(snapshot=snapshot, execution=execution)
    ctx = PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
        function_catalog=provider,
    )

    RiskFactorsStep().run(ctx)

    row = con.execute(
        "SELECT rel_path, risk_level FROM analytics.goid_risk_factors WHERE repo = 'r' AND commit = 'c'"
    ).fetchone()
    if row is None:
        pytest.fail("risk_factors row missing")
    rel_path, _risk_level = row
    if rel_path != "m.py":
        message = f"Unexpected rel_path: {rel_path}"
        pytest.fail(message)
