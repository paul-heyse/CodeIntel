"""Entrypoint detection tests for analytics compute layer."""

from __future__ import annotations

from codeintel.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    detect_entrypoints,
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
    assert {"fastapi", "flask"} <= frameworks

    fastapi = next(candidate for candidate in candidates if candidate.framework == "fastapi")
    assert fastapi.http_method == "GET"
    assert fastapi.route_path == "/items"
    assert fastapi.status_codes == [201]

    flask = next(candidate for candidate in candidates if candidate.framework == "flask")
    assert flask.http_method == "POST"
    assert flask.route_path == "/hello"
    assert flask.auth_required is None


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
    click_entry = next(
        candidate for candidate in candidates if candidate.framework == "click"
    )
    assert click_entry.kind == "cli"
    assert click_entry.command_name == "main"
    assert click_entry.arguments_schema is not None
    assert click_entry.arguments_schema["options"][0]["flags"] == ["--count"]

    typer_entry = next(candidate for candidate in candidates if candidate.framework == "typer")
    assert typer_entry.arguments_schema is not None
    param_names = [
        param["name"] for param in typer_entry.arguments_schema["params"]  # type: ignore[index]
    ]
    assert param_names == ["name", "excited"]

    generic_route = next(candidate for candidate in candidates if candidate.framework == "generic")
    assert generic_route.route_path == "/generic"


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
    assert celery_entry.command_name == "demo.task"
    assert celery_entry.extra == {"queue": "high"}

    airflow_task_entry = next(
        candidate
        for candidate in candidates
        if candidate.framework == "airflow" and candidate.kind == "task"
    )
    assert airflow_task_entry.command_name == "air_task"
    airflow_dag_entry = next(
        candidate
        for candidate in candidates
        if candidate.framework == "airflow" and candidate.kind == "dag"
    )
    assert airflow_dag_entry.command_name == "demo_dag"

    cron_entry = next(candidate for candidate in candidates if candidate.kind == "cron")
    assert cron_entry.schedule == "0 0 * * *"

    django_entry = next(candidate for candidate in candidates if candidate.framework == "django")
    assert django_entry.route_path == "home/"
    assert django_entry.qualname.endswith("airflow_task")
