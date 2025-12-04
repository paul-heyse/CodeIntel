"""Tests for entrypoint detection from source modules.

This module tests:
- Detection configuration dataclasses
- EntryPointCandidate dataclass
- ImportContext dataclass
- Basic detection behavior
"""

from __future__ import annotations

import pytest

from codeintel.analytics.entrypoint_detectors import (
    DetectorSettings,
    EntryPointCandidate,
    ImportContext,
    detect_entrypoints,
)

TEST_REL_PATH = "app/routes.py"
TEST_MODULE = "app.routes"
TEST_LINENO = 10
TEST_END_LINENO = 15


def test_detector_settings_defaults() -> None:
    """DetectorSettings has all frameworks enabled by default."""
    settings = DetectorSettings()

    assert settings.detect_fastapi is True
    assert settings.detect_flask is True
    assert settings.detect_click is True
    assert settings.detect_typer is True
    assert settings.detect_cron is True
    assert settings.detect_django is True
    assert settings.detect_celery is True
    assert settings.detect_airflow is True
    assert settings.detect_generic_routes is True


def test_detector_settings_disabled() -> None:
    """DetectorSettings can disable specific frameworks."""
    settings = DetectorSettings(
        detect_fastapi=False,
        detect_flask=False,
    )

    assert settings.detect_fastapi is False
    assert settings.detect_flask is False


def test_detector_settings_immutable() -> None:
    """DetectorSettings is frozen/immutable."""
    settings = DetectorSettings()
    with pytest.raises(AttributeError):
        settings.detect_fastapi = False  # type: ignore[misc]


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

    assert candidate.kind == "http_handler"
    assert candidate.framework == "fastapi"
    assert candidate.rel_path == TEST_REL_PATH
    assert candidate.module == TEST_MODULE
    assert candidate.qualname == "get_users"
    assert candidate.lineno == TEST_LINENO
    assert candidate.end_lineno == TEST_END_LINENO


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

    assert candidate.http_method == "POST"
    assert candidate.route_path == "/users"
    assert candidate.status_codes == [201, 400]


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

    assert candidate.command_name == "main"
    assert candidate.kind == "cli_command"


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

    assert candidate.schedule == "0 * * * *"
    assert candidate.trigger == "cron"


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

    assert candidate.extra is not None
    assert candidate.extra["custom_field"] == "value"


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

    with pytest.raises(AttributeError):
        candidate.kind = "cli_command"  # type: ignore[misc]


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

    assert candidate.http_method is None
    assert candidate.route_path is None
    assert candidate.status_codes is None
    assert candidate.auth_required is None
    assert candidate.command_name is None
    assert candidate.schedule is None
    assert candidate.trigger is None
    assert candidate.extra is None


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

    assert candidate.evidence == evidence


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

    assert candidates == []


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

    assert candidates == []


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

    assert candidates == []


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

    assert isinstance(result, list)


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

    assert "app" in ctx.fastapi_targets
    assert ctx.alias_to_lib["app"] == "fastapi.FastAPI"


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

    with pytest.raises(AttributeError):
        ctx.alias_to_lib = {"new": "value"}  # type: ignore[misc]


def test_detector_settings_partial_disable() -> None:
    """DetectorSettings can disable individual frameworks."""
    settings = DetectorSettings(
        detect_fastapi=False,
        detect_flask=True,
        detect_click=False,
        detect_typer=True,
    )

    assert settings.detect_fastapi is False
    assert settings.detect_flask is True
    assert settings.detect_click is False
    assert settings.detect_typer is True


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

    assert candidate.arguments_schema is not None
    assert "name" in candidate.arguments_schema


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

    assert candidate.auth_required is True


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

    assert candidate.kind == kind


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

    assert candidate.framework == framework
