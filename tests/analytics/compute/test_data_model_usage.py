"""Tests for data model usage classification."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.data_models import compute_data_model_usage
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config import SnapshotInit
from tests._helpers import TestContext, create_test_context
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.config_factory import data_model_usage_cfg


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
def data_model_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a test context with schema ready for data model usage.

    Yields
    ------
    TestContext
        Context with database schema prepared for data model usage tests.
    """
    ctx = create_test_context(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


def test_compute_data_model_usage_records_multiple_kinds(data_model_ctx: TestContext) -> None:
    """Classify model interactions across create/update/serialize/delete operations."""
    cfg = data_model_usage_cfg(
        SnapshotInit(
            repo=data_model_ctx.repo,
            commit=data_model_ctx.commit,
            repo_root=data_model_ctx.repo_root,
        ),
        max_examples_per_usage=2,
    )

    con = data_model_ctx.gateway.con
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
        [data_model_ctx.repo, data_model_ctx.commit],
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
        data_model_ctx.gateway,
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

    if row is None:
        pytest.fail("Expected data_model_usage row for function goid")
    _, usages_raw, evidence_raw, context_raw = row
    usages = json.loads(usages_raw)
    expect_true(
        set(usages) >= {"create", "update", "serialize", "delete"},
        message="usage kinds captured",
    )
    evidence = json.loads(evidence_raw)
    expect_true(evidence["create"], message="create evidence")
    expect_equal(json.loads(context_raw)["module"], "app.models", label="module context")
