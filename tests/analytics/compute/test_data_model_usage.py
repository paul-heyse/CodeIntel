"""Tests for data model usage classification."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.data_models import compute_data_model_usage
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config import ConfigBuilder
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import GatewayFactory


def _function_ast(code: str, *, goid: int, rel_path: str, qualname: str) -> FunctionAst:
    module = ast.parse(code)
    func = next(
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return FunctionAst(
        goid=goid,
        rel_path=rel_path,
        qualname=qualname,
        start_line=func.lineno,
        end_line=getattr(func, "end_lineno", func.lineno),
        node=func,
        lines=code.splitlines(),
    )


@pytest.fixture
def gateway() -> Iterator[StorageGateway]:
    """Provide a gateway with schema applied but validation disabled.

    Yields
    ------
    StorageGateway
        Gateway configured for analytics classification tests.
    """
    gw = GatewayFactory().without_validation().without_views().open()
    try:
        yield gw
    finally:
        gw.close()


def test_compute_data_model_usage_records_multiple_kinds(gateway: StorageGateway) -> None:
    """Classify model interactions across create/update/serialize/delete operations."""
    cfg = ConfigBuilder.from_snapshot(
        repo="demo/repo", commit="abc123", repo_root=Path.cwd()
    ).data_model_usage(max_examples_per_usage=2)

    con = gateway.con
    con.execute(
        """
        INSERT INTO analytics.data_models (
            repo, commit, model_id, goid_h128, model_name, module, rel_path,
            model_kind, base_classes_json, doc_short, doc_long, created_at
        ) VALUES (
            ?, ?, 'model-1', 1, 'User', 'app.models', 'app/models.py',
            'pydantic', '[]', 'User doc', NULL, NOW()
        )
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        INSERT INTO analytics.function_types (
            function_goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, total_params, annotated_params,
            unannotated_params, param_typed_ratio, has_return_annotation,
            return_type, return_type_source, type_comment, param_types,
            fully_typed, partial_typed, untyped, typedness_bucket,
            typedness_source, created_at
        ) VALUES (
            10, 'urn:func', ?, ?, 'app/models.py', 'python', 'function',
            'app.models.process_user', 1, 20, 1, 1, 0, 1.0, TRUE,
            'User', 'annotation', NULL, '{"user": "User"}',
            TRUE, FALSE, FALSE, 'typed', 'annotations', NOW()
        )
        """,
        [cfg.repo, cfg.commit],
    )

    function_ast = _function_ast(
        """
def process_user(user: User) -> dict[str, str]:
    created = User(name="abc")
    user.save()
    payload = dict(user)
    del user
    return payload
        """,
        goid=10,
        rel_path="app/models.py",
        qualname="app.models.process_user",
    )
    module_map = {"app/models.py": "app.models"}
    ast_by_goid = {10: function_ast}

    compute_data_model_usage(
        gateway,
        cfg,
        module_map=module_map,
        ast_by_goid=ast_by_goid,
        missing_goids=set(),
    )

    row = con.execute(
        """
        SELECT model_id, usage_kinds_json, evidence_json, context_json
        FROM analytics.data_model_usage
        WHERE function_goid_h128 = 10
        """
    ).fetchone()

    assert row is not None
    _, usages_raw, evidence_raw, context_raw = row
    usages = json.loads(usages_raw)
    assert set(usages) >= {"create", "update", "serialize", "delete"}
    evidence = json.loads(evidence_raw)
    assert evidence["create"]
    assert json.loads(context_raw)["module"] == "app.models"
