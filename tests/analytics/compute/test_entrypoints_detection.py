"""Entrypoint detection and evidence tests (combined suite)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    EntryPointCandidate,
    ImportContext,
    detect_entrypoints,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    assert_evidence_location,
    assert_evidence_snippet_contains,
    assert_evidence_urn,
    assert_mapping_list,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds.entrypoints import (
    ENTRYPOINTS_CLI_END,
    ENTRYPOINTS_CLI_START,
    ENTRYPOINTS_HELLO_END,
    ENTRYPOINTS_HELLO_START,
    ENTRYPOINTS_MOD_FQN,
    ENTRYPOINTS_MOD_PATH,
    ENTRYPOINTS_PACK,
    write_entrypoints_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests._helpers.context import TestContext

REL_PATH = ENTRYPOINTS_MOD_PATH
MODULE = ENTRYPOINTS_MOD_FQN


@pytest.fixture
def entrypoints_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide seeded context with canonical entrypoints source.

    Yields
    ------
    Iterator[TestContext]
        Seeded test context with entrypoints pack and source files.
    """
    ctx = TestScenario.minimal().with_seeds(ENTRYPOINTS_PACK).build(tmp_path)
    source = write_entrypoints_source(ctx.repo_root)
    ctx.extra["entrypoints_source"] = source
    try:
        yield ctx
    finally:
        ctx.close()


def test_detector_settings_defaults_and_disable() -> None:
    """DetectorSettings toggles default and selective frameworks."""
    defaults = DetectorSettings()
    expect_true(defaults.detect_fastapi)
    expect_true(defaults.detect_flask)
    expect_true(defaults.detect_click)
    expect_true(defaults.detect_typer)
    expect_true(defaults.detect_cron)
    expect_true(defaults.detect_django)
    expect_true(defaults.detect_celery)
    expect_true(defaults.detect_airflow)
    expect_true(defaults.detect_generic_routes)

    disabled = DetectorSettings(detect_fastapi=False, detect_flask=False)
    expect_false(disabled.detect_fastapi)
    expect_false(disabled.detect_flask)
    assert_frozen(defaults, "detect_fastapi", new_value=False)


def test_import_context_immutability() -> None:
    """ImportContext supports expected fields and is frozen."""
    ctx = ImportContext(
        alias_to_lib={"app": "fastapi.FastAPI"},
        fastapi_targets={"app"},
        flask_targets=set(),
        flask_blueprints=set(),
        typer_targets=set(),
        click_groups=set(),
        django_url_helpers=set(),
        celery_apps=set(),
    )
    expect_in("app", ctx.fastapi_targets)
    expect_equal(ctx.alias_to_lib["app"], "fastapi.FastAPI")
    assert_frozen(ctx, "alias_to_lib", {"new": "value"})


def test_entrypoint_candidate_core_metadata() -> None:
    """EntryPointCandidate stores required and optional fields."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="get_users",
        lineno=10,
        end_lineno=15,
    )
    expect_equal(candidate.kind, "http_handler")
    expect_equal(candidate.framework, "fastapi")
    expect_equal(candidate.rel_path, REL_PATH)
    expect_equal(candidate.module, MODULE)
    expect_equal(candidate.qualname, "get_users")
    expect_equal(candidate.lineno, 10)
    expect_equal(candidate.end_lineno, 15)

    http = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="create_user",
        lineno=20,
        end_lineno=30,
        http_method="POST",
        route_path="/users",
        status_codes=[201, 400],
    )
    expect_equal(http.http_method, "POST")
    expect_equal(http.route_path, "/users")
    expect_equal(http.status_codes, [201, 400])

    cli = EntryPointCandidate(
        kind="cli_command",
        framework="click",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="cli_main",
        lineno=1,
        end_lineno=10,
        command_name="main",
    )
    expect_equal(cli.command_name, "main")
    expect_equal(cli.kind, "cli_command")

    job = EntryPointCandidate(
        kind="background_job",
        framework="celery",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="process_task",
        lineno=1,
        end_lineno=10,
        schedule="0 * * * *",
        trigger="cron",
    )
    expect_equal(job.schedule, "0 * * * *")
    expect_equal(job.trigger, "cron")

    extra = EntryPointCandidate(
        kind="http_handler",
        framework="custom",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
        extra={"custom_field": "value", "count": 42},
    )
    extra_data = expect_is_not_none(extra.extra)
    expect_equal(extra_data["custom_field"], "value")
    assert_frozen(candidate, "kind", "cli_command")


def test_entrypoint_candidate_optionals_and_schema() -> None:
    """EntryPointCandidate handles optional fields and schemas."""
    base = EntryPointCandidate(
        kind="http_handler",
        framework=None,
        rel_path=REL_PATH,
        module=MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )
    expect_true(base.framework is None)
    expect_true(base.http_method is None)
    expect_true(base.route_path is None)
    expect_true(base.status_codes is None)
    expect_true(base.auth_required is None)
    expect_true(base.command_name is None)
    expect_true(base.schedule is None)
    expect_true(base.trigger is None)
    expect_true(base.extra is None)

    evidence_payload: list[dict[str, object]] = [{"type": "decorator", "line": 10}]
    evidence_candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
        evidence=evidence_payload,
    )
    expect_equal(evidence_candidate.evidence, evidence_payload)

    schema = {
        "name": {"type": "str", "required": True},
        "count": {"type": "int", "default": 1},
    }
    with_schema = EntryPointCandidate(
        kind="cli_command",
        framework="click",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="cmd",
        lineno=1,
        end_lineno=5,
        arguments_schema=schema,
    )
    expect_is_not_none(with_schema.arguments_schema)
    expect_in("name", with_schema.arguments_schema or {})


@pytest.mark.parametrize(
    "kind",
    ["http_handler", "cli_command", "background_job", "scheduled_job", "websocket"],
)
def test_entrypoint_candidate_kinds(kind: str) -> None:
    """Test various entrypoint kinds."""
    candidate = EntryPointCandidate(
        kind=kind,
        framework="generic",
        rel_path=REL_PATH,
        module=MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )
    expect_equal(candidate.kind, kind)


@pytest.mark.parametrize(
    "framework",
    ["fastapi", "flask", "click", "typer", "celery", "django", "airflow", None],
)
def test_entrypoint_candidate_frameworks(framework: str | None) -> None:
    """Test various framework values."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework=framework,
        rel_path=REL_PATH,
        module=MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )
    expect_equal(candidate.framework, framework)


def test_detect_entrypoints_basic_behaviors() -> None:
    """Detect entrypoints handles empty, syntax errors, and basic flows."""
    settings = DetectorSettings()
    expect_equal(
        detect_entrypoints("", rel_path=REL_PATH, module=MODULE, settings=settings),
        [],
    )
    expect_equal(
        detect_entrypoints("def broken(", rel_path=REL_PATH, module=MODULE, settings=settings),
        [],
    )

    source = """
def helper_function(x):
    return x + 1

class MyClass:
    def method(self):
        pass
"""
    candidates = detect_entrypoints(source, rel_path=REL_PATH, module=MODULE, settings=settings)
    expect_equal(candidates, [])

    result = detect_entrypoints("x = 1", rel_path=REL_PATH, module=MODULE, settings=settings)
    expect_is_instance(result, list)


def test_detect_entrypoints_with_seeded_pack(entrypoints_ctx: TestContext) -> None:
    """Detect entrypoints and evidence from the canonical seeded module."""
    source_raw = entrypoints_ctx.extra["entrypoints_source"]
    source = str(source_raw)
    candidates = detect_entrypoints(
        source,
        rel_path=REL_PATH,
        module=MODULE,
        settings=DetectorSettings(),
    )
    expect_length(candidates, 2)

    hello = next(candidate for candidate in candidates if candidate.qualname.endswith("hello"))
    expect_equal(hello.lineno, ENTRYPOINTS_HELLO_START)
    expect_equal(hello.end_lineno, ENTRYPOINTS_HELLO_END)
    assert_evidence_location(hello.evidence[0], path=REL_PATH)
    assert_evidence_snippet_contains(hello.evidence[0], "app.get")

    cli = next(candidate for candidate in candidates if candidate.qualname.endswith("cli_main"))
    expect_equal(cli.lineno, ENTRYPOINTS_CLI_START)
    expect_equal(cli.end_lineno, ENTRYPOINTS_CLI_END)
    assert_evidence_location(cli.evidence[0], path=REL_PATH)
    assert_evidence_snippet_contains(cli.evidence[0], "cli.command")

    urn_row = expect_is_not_none(
        entrypoints_ctx.gateway.con.execute(
            "SELECT urn FROM core.goids WHERE goid_h128 = ?",
            [ENTRYPOINTS_PACK.hello_goid],
        ).fetchone()
    )
    assert_evidence_urn({"urn": urn_row[0]}, ENTRYPOINTS_PACK.hello_urn)


def test_detects_fastapi_and_flask_routes() -> None:
    """Detect HTTP entrypoints for FastAPI and Flask-style decorators."""
    source = """
from fastapi import FastAPI
from flask import Flask

app = FastAPI()
bp = Flask(__name__)

@app.get("/items", status_code=201)
def list_items() -> str:
    return "ok"

@bp.route("/hello", methods=["POST"])
def say_hello() -> str:
    return "hello"
"""
    candidates = detect_entrypoints(
        source,
        rel_path=REL_PATH,
        module=MODULE,
        settings=DetectorSettings(),
    )
    frameworks = {candidate.framework for candidate in candidates}
    expect_true({"fastapi", "flask"} <= frameworks)

    fastapi = next(candidate for candidate in candidates if candidate.framework == "fastapi")
    expect_equal(fastapi.http_method, "GET")
    expect_equal(fastapi.route_path, "/items")
    expect_equal(fastapi.status_codes, [201])

    flask = next(candidate for candidate in candidates if candidate.framework == "flask")
    expect_equal(flask.http_method, "POST")
    expect_equal(flask.route_path, "/hello")
    expect_equal(flask.auth_required, None)


def test_detects_click_typer_and_generic_routes() -> None:
    """Detect CLI entrypoints and generic route decorators."""
    source = """
import click
import typer

cli = click.Group()
app = typer.Typer()

@cli.command()
@click.option("--count", type=int, required=True, default=2)
def main(count: int) -> None:
    return None

@app.command()
def greet(name: str, excited: bool = False) -> None:
    return None

@cli.route("/generic")
def generic() -> None:
    return None
"""
    candidates = detect_entrypoints(
        source,
        rel_path="cli.py",
        module="cli",
        settings=DetectorSettings(),
    )
    click_entry = next(candidate for candidate in candidates if candidate.framework == "click")
    expect_equal(click_entry.kind, "cli")
    expect_equal(click_entry.command_name, "main")
    expect_is_not_none(click_entry.arguments_schema)
    options = assert_mapping_list(click_entry.arguments_schema, "options")
    expect_equal(options[0]["flags"], ["--count"])

    typer_entry = next(candidate for candidate in candidates if candidate.framework == "typer")
    expect_is_not_none(typer_entry.arguments_schema)
    params = assert_mapping_list(typer_entry.arguments_schema, "params")
    param_names = [str(param["name"]) for param in params]
    expect_equal(param_names, ["name", "excited"])

    generic_route = next(candidate for candidate in candidates if candidate.framework == "generic")
    expect_equal(generic_route.route_path, "/generic")


def test_detects_celery_airflow_cron_and_django() -> None:
    """Detect task, DAG, cron, and Django URL patterns."""
    source = """
from celery import Celery
from airflow import task, dag
from django.urls import path

app = Celery("demo")

@app.task(name="demo.task", queue="high")
def celery_task() -> None:
    return None

@task("air_task")
def airflow_task() -> None:
    return None

@dag("demo_dag")
def airflow_dag() -> None:
    return None

class Scheduler:
    def scheduled_job(self, cron: str) -> None:
        return None

scheduler = Scheduler()

@scheduler.scheduled_job("0 0 * * *")
def cron_job() -> None:
    return None

urlpatterns = [
    path("home/", airflow_task),
]
"""
    candidates = detect_entrypoints(
        source,
        rel_path="tasks.py",
        module="tasks",
        settings=DetectorSettings(),
    )

    celery_entry = next(candidate for candidate in candidates if candidate.framework == "celery")
    expect_equal(celery_entry.command_name, "demo.task")
    expect_equal(celery_entry.extra, {"queue": "high"})

    airflow_task_entry = next(
        candidate
        for candidate in candidates
        if candidate.framework == "airflow" and candidate.kind == "task"
    )
    expect_equal(airflow_task_entry.command_name, "air_task")
    airflow_dag_entry = next(
        candidate
        for candidate in candidates
        if candidate.framework == "airflow" and candidate.kind == "dag"
    )
    expect_equal(airflow_dag_entry.command_name, "demo_dag")

    cron_entry = next(candidate for candidate in candidates if candidate.kind == "cron")
    expect_equal(cron_entry.schedule, "0 0 * * *")

    django_entry = next(candidate for candidate in candidates if candidate.framework == "django")
    expect_equal(django_entry.route_path, "home/")
    expect_true(django_entry.qualname.endswith("airflow_task"))


def test_detect_entrypoints_emits_snippet_evidence() -> None:
    """detect_entrypoints should emit decorator evidence with snippets."""
    source = textwrap.dedent(
        """\
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/ping")
        def ping() -> str:
            return "ok"
        """
    )
    candidates = detect_entrypoints(
        source,
        rel_path=REL_PATH,
        module=MODULE,
        settings=DetectorSettings(),
    )
    if not candidates:
        pytest.fail("Expected a FastAPI entrypoint candidate")
    evidence = candidates[0].evidence
    if not evidence:
        pytest.fail("Entrypoint candidate did not include evidence")
    sample = evidence[0]
    assert_evidence_location(sample, path=REL_PATH, lineno=5)
    assert_evidence_snippet_contains(sample, "@app.get")
    details = sample.get("details")
    expect_true(isinstance(details, dict))
