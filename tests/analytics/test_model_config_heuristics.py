"""Thin-slice tests to lock in heuristic analytics for models and config flows."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.graphs import compute_config_data_flow
from codeintel.config import ConfigBuilder
from codeintel.storage.data_models import NormalizedDataModel, fetch_models_normalized
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from tests._helpers.builders import (
    CallGraphNodeRow,
    FunctionTypesRow,
    GoidRow,
    ModuleRow,
    insert_call_graph_nodes,
    insert_function_types,
    insert_goids,
    insert_modules,
)
from tests._helpers.gateway import open_ingestion_gateway

REPO = "test/repo"
COMMIT = "deadbeef"


@contextmanager
def _gateway_with_schema() -> Iterator[StorageGateway]:
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        yield gateway
    finally:
        gateway.close()


def _write_fixture(repo_root: Path, rel_path: str, content: str) -> Path:
    abs_path = repo_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    return abs_path


def _goid_rows_for_defs(rel_path: str, source: str, start: int) -> list[GoidRow]:
    tree = ast.parse(source)
    rows: list[GoidRow] = []
    counter = start
    module = rel_path.replace("/", ".").removesuffix(".py")
    created_at = datetime.now(tz=UTC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{module}.{node.name}"
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            rows.append(
                GoidRow(
                    goid_h128=counter,
                    urn=f"goid:{REPO}/{rel_path}#{qualname}",
                    repo=REPO,
                    commit=COMMIT,
                    rel_path=rel_path,
                    kind=kind,
                    qualname=qualname,
                    start_line=int(node.lineno),
                    end_line=int(getattr(node, "end_lineno", node.lineno)),
                    created_at=created_at,
                )
            )
            counter += 1
    return rows


def _seed_modules(gateway: StorageGateway, modules: Iterable[ModuleRow]) -> None:
    insert_modules(gateway, modules)


def _seed_goids(gateway: StorageGateway, goids: Iterable[GoidRow]) -> None:
    insert_goids(gateway, goids)


def _seed_call_graph_nodes(gateway: StorageGateway, goids: Iterable[GoidRow]) -> None:
    gateway.con.execute("DELETE FROM graph.call_graph_nodes")
    node_rows = [
        CallGraphNodeRow(
            goid_h128=row.goid_h128,
            language="python",
            kind=row.kind,
            arity=0,
            is_public=True,
            rel_path=row.rel_path,
        )
        for row in goids
    ]
    insert_call_graph_nodes(gateway, node_rows)


def _seed_function_types(gateway: StorageGateway, goids: Iterable[GoidRow]) -> None:
    now = datetime.now(tz=UTC)
    param_type_map: dict[str, dict[str, str]] = {
        "create_user": {"session": "Session", "name": "str"},
        "fetch_user": {"session": "Session"},
        "serialize_post": {"post": "Post"},
        "serialize_payload": {"payload": "UserPayload"},
        "config_checks": {"settings": "dict[str, object]"},
    }
    return_type_map: dict[str, str] = {
        "create_user": "User",
        "fetch_user": "User | None",
        "serialize_post": "dict[str, object]",
        "serialize_payload": "dict[str, object]",
        "config_checks": "bool",
    }
    rows: list[FunctionTypesRow] = []
    for row in goids:
        if row.kind != "function":
            continue
        leaf = row.qualname.split(".")[-1]
        params = param_type_map.get(leaf, {})
        total_params = len(params)
        rows.append(
            FunctionTypesRow(
                function_goid_h128=row.goid_h128,
                urn=row.urn,
                repo=row.repo,
                commit=row.commit,
                rel_path=row.rel_path,
                language="python",
                kind="function",
                qualname=row.qualname,
                start_line=row.start_line,
                end_line=row.end_line,
                total_params=total_params,
                annotated_params=total_params,
                unannotated_params=0,
                param_typed_ratio=1.0,
                has_return_annotation=True,
                return_type=return_type_map.get(leaf, ""),
                return_type_source="annotation",
                type_comment=None,
                param_types_json=json.dumps(params),
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="analysis",
                created_at=now,
            )
        )
    insert_function_types(gateway, rows)


def _seed_config_values(con: DuckDBConnection, rel_path: str) -> None:
    con.execute(
        "DELETE FROM analytics.config_values WHERE repo = ? AND commit = ?",
        [REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO analytics.config_values (
            repo, commit, config_path, format, key, reference_paths, reference_modules, reference_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            REPO,
            COMMIT,
            "config.yaml",
            "yaml",
            "feature.flag",
            json.dumps([rel_path]),
            json.dumps(["pkg.config_usage"]),
            2,
        ),
    )


def _seed_entrypoints(con: DuckDBConnection, handler_goid: int) -> None:
    con.execute(
        """
        INSERT INTO analytics.entrypoints (
            repo, commit, entrypoint_id, kind, handler_goid_h128, handler_urn,
            handler_rel_path, handler_module, handler_qualname, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            REPO,
            COMMIT,
            "ep1",
            "api",
            handler_goid,
            f"urn:{handler_goid}",
            "tests/fixtures/heuristics/config_usage.py",
            "tests.fixtures.heuristics.config_usage",
            "config_checks",
            datetime.now(tz=UTC),
        ),
    )


def _prepare_repo(
    repo_root: Path,
) -> tuple[list[GoidRow], list[ModuleRow], dict[str, int], str]:
    fixture_root = Path(__file__).parents[1] / "fixtures" / "heuristics"
    fixture_specs = (
        ("models_sqlalchemy.py", "tests/fixtures/heuristics/models_sqlalchemy.py"),
        ("models_pydantic.py", "tests/fixtures/heuristics/models_pydantic.py"),
        ("service_usage.py", "tests/fixtures/heuristics/service_usage.py"),
        ("config_usage.py", "tests/fixtures/heuristics/config_usage.py"),
    )

    goid_rows: list[GoidRow] = []
    module_rows: list[ModuleRow] = []
    goid_index: dict[str, int] = {}
    start = 1

    for filename, rel_path in fixture_specs:
        content = (fixture_root / filename).read_text(encoding="utf-8")
        dest = _write_fixture(repo_root, rel_path, content)
        rel_module = dest.relative_to(repo_root).as_posix().replace("/", ".").removesuffix(".py")
        rel_path_str = dest.relative_to(repo_root).as_posix()
        module_rows.append(
            ModuleRow(
                module=rel_module,
                path=rel_path_str,
                repo=REPO,
                commit=COMMIT,
            )
        )
        rows = _goid_rows_for_defs(rel_path_str, content, start)
        goid_rows.extend(rows)
        for row in rows:
            qualname_leaf = row.qualname.split(".")[-1]
            if qualname_leaf in {"create_user", "fetch_user", "serialize_payload", "config_checks"}:
                goid_index[qualname_leaf] = row.goid_h128
        start += len(rows)

    return goid_rows, module_rows, goid_index, "tests/fixtures/heuristics/config_usage.py"


def _assert_models(gateway: StorageGateway) -> dict[str, str]:
    models: list[NormalizedDataModel] = fetch_models_normalized(
        gateway,
        REPO,
        COMMIT,
    )
    if not any(model.model_name == "User" and model.model_kind == "orm_model" for model in models):
        pytest.fail("Expected User ORM model in analytics.data_models")
    if not any(
        model.model_name == "UserPayload" and model.model_kind == "pydantic_model"
        for model in models
    ):
        pytest.fail("Expected UserPayload Pydantic model in analytics.data_models")
    post_model = next(model for model in models if model.model_name == "Post")
    if not any(rel.target_model_name == "User" for rel in post_model.relationships):
        pytest.fail("Expected Post relationship targeting User")
    user_payload = next(model for model in models if model.model_name == "UserPayload")
    if not any(
        field.name == "name" and field.source == "pydantic_field" for field in user_payload.fields
    ):
        pytest.fail("Expected UserPayload.name field via normalized view")
    return {model.model_name: model.model_id for model in models}


def _assert_field_relationship_tables(con: DuckDBConnection, model_ids: dict[str, str]) -> None:
    fields_rows = con.execute(
        """
        SELECT field_name, field_type, source
        FROM analytics.data_model_fields
        WHERE repo = ? AND commit = ? AND model_id = ?
        """,
        [REPO, COMMIT, model_ids["UserPayload"]],
    ).fetchall()
    if not any(field == "name" and source == "pydantic_field" for field, _, source in fields_rows):
        pytest.fail("Expected UserPayload.name field in data_model_fields")

    relationships_rows = con.execute(
        """
        SELECT target_model_name, relationship_kind, multiplicity
        FROM analytics.data_model_relationships
        WHERE repo = ? AND commit = ? AND source_model_id = ?
        """,
        [REPO, COMMIT, model_ids["Post"]],
    ).fetchall()
    if not any(target == "User" for target, _, _ in relationships_rows):
        pytest.fail("Expected Post->User relationship in data_model_relationships")


def _assert_model_usage(con: DuckDBConnection, goid_index: dict[str, int]) -> None:
    usage_rows = con.execute(
        """
        SELECT model_id, function_goid_h128, usage_kinds_json
        FROM analytics.data_model_usage
        WHERE repo = ? AND commit = ?
        """,
        [REPO, COMMIT],
    ).fetchall()
    usage_by_func = {row[1]: json.loads(row[2]) for row in usage_rows}
    if "create" not in usage_by_func.get(goid_index["create_user"], []):
        pytest.fail("Expected create usage for create_user")
    if "read" not in usage_by_func.get(goid_index["fetch_user"], []):
        pytest.fail("Expected read usage for fetch_user")
    if "serialize" not in usage_by_func.get(goid_index["serialize_payload"], []):
        pytest.fail("Expected serialize usage for serialize_payload")


def _assert_config_usage(con: DuckDBConnection) -> None:
    config_rows = con.execute(
        """
        SELECT usage_kind, evidence_json
        FROM analytics.config_data_flow
        WHERE repo = ? AND commit = ? AND config_key = 'feature.flag'
        """,
        [REPO, COMMIT],
    ).fetchall()
    kinds = {row[0] for row in config_rows}
    if {"read", "write", "conditional_branch"} - kinds:
        pytest.fail(f"Missing config usage kinds in config_data_flow: {kinds}")


def test_data_models_and_usage_and_config_flow(tmp_path: Path) -> None:
    """Validate heuristics pipeline across SQLAlchemy, Pydantic, usage, and config fixtures."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    goid_rows, module_rows, goid_index, config_rel_path = _prepare_repo(repo_root)

    with _gateway_with_schema() as gateway:
        con = gateway.con
        _seed_modules(gateway, module_rows)
        _seed_goids(gateway, goid_rows)
        _seed_call_graph_nodes(gateway, [row for row in goid_rows if row.kind == "function"])
        _seed_function_types(gateway, goid_rows)
        _seed_config_values(con, config_rel_path)
        _seed_entrypoints(con, goid_index["config_checks"])

        builder = ConfigBuilder.from_snapshot(
            repo=REPO,
            commit=COMMIT,
            repo_root=repo_root,
        )
        compute_data_models(gateway, builder.data_models())
        compute_data_model_usage(
            gateway=gateway,
            cfg=builder.data_model_usage(),
        )
        compute_config_data_flow(
            gateway=gateway,
            cfg=builder.config_data_flow(),
        )

        model_ids = _assert_models(gateway)
        _assert_field_relationship_tables(con, model_ids)
        _assert_model_usage(con, goid_index)
        _assert_config_usage(con)
