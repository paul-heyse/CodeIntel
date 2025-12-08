"""Tests for entrypoint detection from source modules.

This module tests:
- Detection configuration dataclasses
- EntryPointCandidate dataclass
- ImportContext dataclass
- Basic detection behavior
"""

from __future__ import annotations

import pytest

from codeintel.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    EntryPointCandidate,
    ImportContext,
    detect_entrypoints,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

TEST_REL_PATH = "app/routes.py"
TEST_MODULE = "app.routes"
TEST_LINENO = 10
TEST_END_LINENO = 15


def test_detector_settings_defaults() -> None:
    """DetectorSettings has all frameworks enabled by default."""
    settings = DetectorSettings()

    expect_true(settings.detect_fastapi)
    expect_true(settings.detect_flask)
    expect_true(settings.detect_click)
    expect_true(settings.detect_typer)
    expect_true(settings.detect_cron)
    expect_true(settings.detect_django)
    expect_true(settings.detect_celery)
    expect_true(settings.detect_airflow)
    expect_true(settings.detect_generic_routes)


def test_detector_settings_disabled() -> None:
    """DetectorSettings can disable specific frameworks."""
    settings = DetectorSettings(
        detect_fastapi=False,
        detect_flask=False,
    )

    expect_false(settings.detect_fastapi)
    expect_false(settings.detect_flask)


def test_detector_settings_immutable() -> None:
    """DetectorSettings is frozen/immutable."""
    settings = DetectorSettings()
    assert_frozen(settings, "detect_fastapi", new_value=False)


def test_entrypoint_candidate_creation() -> None:
    """Create an EntryPointCandidate with required fields."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="get_users",
        lineno=TEST_LINENO,
        end_lineno=TEST_END_LINENO,
    )

    expect_equal(candidate.kind, "http_handler")
    expect_equal(candidate.framework, "fastapi")
    expect_equal(candidate.rel_path, TEST_REL_PATH)
    expect_equal(candidate.module, TEST_MODULE)
    expect_equal(candidate.qualname, "get_users")
    expect_equal(candidate.lineno, TEST_LINENO)
    expect_equal(candidate.end_lineno, TEST_END_LINENO)


def test_entrypoint_candidate_http_metadata() -> None:
    """EntryPointCandidate can store HTTP metadata."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="create_user",
        lineno=20,
        end_lineno=30,
        http_method="POST",
        route_path="/users",
        status_codes=[201, 400],
    )

    expect_equal(candidate.http_method, "POST")
    expect_equal(candidate.route_path, "/users")
    expect_equal(candidate.status_codes, [201, 400])


def test_entrypoint_candidate_cli_metadata() -> None:
    """EntryPointCandidate can store CLI metadata."""
    candidate = EntryPointCandidate(
        kind="cli_command",
        framework="click",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="cli_main",
        lineno=1,
        end_lineno=10,
        command_name="main",
    )

    expect_equal(candidate.command_name, "main")
    expect_equal(candidate.kind, "cli_command")


def test_entrypoint_candidate_job_metadata() -> None:
    """EntryPointCandidate can store job/task metadata."""
    candidate = EntryPointCandidate(
        kind="background_job",
        framework="celery",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="process_task",
        lineno=1,
        end_lineno=10,
        schedule="0 * * * *",
        trigger="cron",
    )

    expect_equal(candidate.schedule, "0 * * * *")
    expect_equal(candidate.trigger, "cron")


def test_entrypoint_candidate_extra_metadata() -> None:
    """EntryPointCandidate can store extra metadata."""
    extra: dict[str, object] = {"custom_field": "value", "count": 42}
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="custom",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
        extra=extra,
    )

    expect_is_not_none(candidate.extra)
    if candidate.extra is None:
        pytest.fail("extra metadata should be present")

    expect_equal(candidate.extra["custom_field"], "value")


def test_entrypoint_candidate_immutable() -> None:
    """EntryPointCandidate is frozen/immutable."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )

    assert_frozen(candidate, "kind", "cli_command")


def test_entrypoint_candidate_optional_fields() -> None:
    """EntryPointCandidate optional fields default to None."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework=None,
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )

    expect_is_none(candidate.http_method)
    expect_is_none(candidate.route_path)
    expect_is_none(candidate.status_codes)
    expect_is_none(candidate.auth_required)
    expect_is_none(candidate.command_name)
    expect_is_none(candidate.schedule)
    expect_is_none(candidate.trigger)
    expect_is_none(candidate.extra)


def test_entrypoint_candidate_evidence_list() -> None:
    """EntryPointCandidate has evidence list."""
    evidence: list[dict[str, object]] = [{"type": "decorator", "line": 10}]
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
        evidence=evidence,
    )

    expect_equal(candidate.evidence, evidence)


def test_detect_entrypoints_empty_source() -> None:
    """Detect entrypoints returns empty list for empty source."""
    settings = DetectorSettings()
    source = ""

    candidates = detect_entrypoints(
        source,
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        settings=settings,
    )

    expect_equal(candidates, [])


def test_detect_entrypoints_syntax_error() -> None:
    """Detect entrypoints returns empty list for syntax errors."""
    settings = DetectorSettings()
    source = "def broken("  # Syntax error

    candidates = detect_entrypoints(
        source,
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        settings=settings,
    )

    expect_equal(candidates, [])


def test_detect_entrypoints_no_frameworks() -> None:
    """Detect entrypoints returns empty for regular code."""
    settings = DetectorSettings()
    source = """
def helper_function(x):
    return x + 1

class MyClass:
    def method(self):
        pass
"""

    candidates = detect_entrypoints(
        source,
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        settings=settings,
    )

    expect_equal(candidates, [])


def test_detect_entrypoints_returns_list() -> None:
    """Detect entrypoints always returns a list."""
    settings = DetectorSettings()
    source = "x = 1"

    result = detect_entrypoints(
        source,
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        settings=settings,
    )

    expect_is_instance(result, list)


def test_import_context_creation() -> None:
    """ImportContext can be created with all fields."""
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


def test_import_context_immutable() -> None:
    """ImportContext is frozen/immutable."""
    ctx = ImportContext(
        alias_to_lib={},
        fastapi_targets=set(),
        flask_targets=set(),
        flask_blueprints=set(),
        typer_targets=set(),
        click_groups=set(),
        django_url_helpers=set(),
        celery_apps=set(),
    )

    assert_frozen(ctx, "alias_to_lib", {"new": "value"})


def test_detector_settings_partial_disable() -> None:
    """DetectorSettings can disable individual frameworks."""
    settings = DetectorSettings(
        detect_fastapi=False,
        detect_flask=True,
        detect_click=False,
        detect_typer=True,
    )

    expect_false(settings.detect_fastapi)
    expect_true(settings.detect_flask)
    expect_false(settings.detect_click)
    expect_true(settings.detect_typer)


def test_entrypoint_candidate_with_arguments_schema() -> None:
    """EntryPointCandidate can store arguments schema."""
    schema = {
        "name": {"type": "str", "required": True},
        "count": {"type": "int", "default": 1},
    }
    candidate = EntryPointCandidate(
        kind="cli_command",
        framework="click",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="cmd",
        lineno=1,
        end_lineno=5,
        arguments_schema=schema,
    )

    expect_is_not_none(candidate.arguments_schema)
    if candidate.arguments_schema is None:
        pytest.fail("arguments schema should be present")

    expect_in("name", candidate.arguments_schema)


def test_entrypoint_candidate_auth_required() -> None:
    """EntryPointCandidate can specify auth_required."""
    candidate = EntryPointCandidate(
        kind="http_handler",
        framework="fastapi",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="protected",
        lineno=1,
        end_lineno=5,
        auth_required=True,
    )

    expect_true(candidate.auth_required)


@pytest.mark.parametrize(
    "kind",
    ["http_handler", "cli_command", "background_job", "scheduled_job", "websocket"],
)
def test_entrypoint_candidate_kinds(kind: str) -> None:
    """Test various entrypoint kinds."""
    candidate = EntryPointCandidate(
        kind=kind,
        framework="generic",
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
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
        rel_path=TEST_REL_PATH,
        module=TEST_MODULE,
        qualname="handler",
        lineno=1,
        end_lineno=5,
    )

    expect_equal(candidate.framework, framework)
