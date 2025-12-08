"""Entrypoint detection tests for analytics compute layer."""

from __future__ import annotations

from codeintel.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    detect_entrypoints,
)
from tests._helpers.assertions import assert_mapping_list
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


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
        rel_path="api.py",
        module="api",
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
