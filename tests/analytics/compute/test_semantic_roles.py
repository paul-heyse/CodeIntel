"""Semantic role classification tests."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.compute.semantic_roles.classification import (
    HELPER_LOC_THRESHOLD,
    ROLE_THRESHOLD,
    SERVICE_FAN_IN_THRESHOLD,
    SERVICE_FAN_OUT_THRESHOLD,
    ModuleRecord,
    RoleAccumulator,
    classify_function_role,
    classify_modules,
    decorator_names,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)
from tests._helpers.fixtures.rows import FunctionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.compute.semantic_roles.classification import (
        FunctionContext,
    )

DEFAULT_LOC = 50
BUMP_VALUE_0_2 = 0.2
BUMP_VALUE_0_3 = 0.3
BUMP_VALUE_0_5 = 0.5
BUMP_VALUE_0_4 = 0.4
BUMP_VALUE_0_6 = 0.6
BUMP_VALUE_0_8 = 0.8
BUMP_VALUE_0_7 = 0.7
BUMP_VALUE_1_5 = 1.5
EXPECTED_SOURCES_2 = 2
EXPECTED_ROWS_1 = 1
EXPECTED_ROWS_3 = 3
EXPECTED_DECORATORS_2 = 2
CONFIDENCE_CAP = 1.0
CONFIDENCE_ZERO = 0.0
LARGE_LOC = 200


def _make_context(
    builder: FunctionContextBuilder | None = None, **overrides: object
) -> FunctionContext:
    base = builder or FunctionContextBuilder()
    kwargs: dict[str, object] = {}
    for key, value in overrides.items():
        if key in {"decorators", "module_tags"} and isinstance(value, tuple):
            kwargs[key] = list(value)
        elif key == "graph" and isinstance(value, dict):
            kwargs[key] = {str(k): int(v) for k, v in value.items()}
        else:
            kwargs[key] = value
    updated = replace(base, **kwargs)
    return updated.build()


def _get_now() -> datetime:
    return datetime.now(tz=UTC)


def test_context_name_property() -> None:
    """Extract function name from qualname."""
    context = _make_context(qualname="module.submodule.my_function")
    expect_equal(context.name, "my_function")


def test_context_name_simple() -> None:
    """Simple qualname without dots."""
    context = _make_context(qualname="simple_function")
    expect_equal(context.name, "simple_function")


def test_context_rel_path_lower() -> None:
    """Lower-case relative path."""
    context = _make_context(rel_path="Src/API/Routes.py")
    expect_equal(context.rel_path_lower, "src/api/routes.py")


def test_context_module_lower() -> None:
    """Lower-case module name."""
    context = _make_context(module_name="Module.SubModule")
    expect_equal(context.module_lower, "module.submodule")


def test_context_module_lower_none() -> None:
    """Handle None module name."""
    context = _make_context(module_name=None)
    expect_true(not context.module_lower)


def test_context_tag_strings() -> None:
    """Normalize tag strings."""
    context = _make_context(module_tags=["API", "Service", None])
    tags = context.tag_strings
    expect_in("api", tags)
    expect_in("service", tags)


def test_accumulator_bump_single() -> None:
    """Bump single role score."""
    acc = RoleAccumulator()
    acc.bump("test", BUMP_VALUE_0_5, "unit test")
    expect_equal(acc.scores["test"], BUMP_VALUE_0_5)
    expect_in("unit test", acc.sources["test"])


def test_accumulator_bump_accumulates() -> None:
    """Multiple bumps accumulate."""
    acc = RoleAccumulator()
    acc.bump("api_handler", BUMP_VALUE_0_3, "reason1")
    acc.bump("api_handler", BUMP_VALUE_0_2, "reason2")
    expect_equal(acc.scores["api_handler"], BUMP_VALUE_0_5)
    expect_length(acc.sources["api_handler"], EXPECTED_SOURCES_2)


def test_accumulator_bump_with_framework() -> None:
    """Framework hint is stored."""
    acc = RoleAccumulator()
    acc.bump("api_handler", BUMP_VALUE_0_7, "decorator", framework_hint="fastapi")
    expect_equal(acc.frameworks.get("api_handler"), "fastapi")


def test_accumulator_framework_first_wins() -> None:
    """First framework hint is kept."""
    acc = RoleAccumulator()
    acc.bump("cli_command", BUMP_VALUE_0_5, "reason1", framework_hint="click")
    acc.bump("cli_command", BUMP_VALUE_0_3, "reason2", framework_hint="typer")
    expect_equal(acc.frameworks.get("cli_command"), "click")


def test_accumulator_finalize_empty() -> None:
    """Empty accumulator returns 'other'."""
    acc = RoleAccumulator()
    role, confidence, _, _ = acc.finalize()
    expect_equal(role, "other")
    expect_equal(confidence, CONFIDENCE_ZERO)


def test_accumulator_finalize_below_threshold() -> None:
    """Below threshold returns 'other'."""
    acc = RoleAccumulator()
    acc.bump("test", ROLE_THRESHOLD - 0.1, "weak signal")
    role, _, _, _ = acc.finalize()
    expect_equal(role, "other")


def test_accumulator_finalize_above_threshold() -> None:
    """Above threshold returns winning role."""
    acc = RoleAccumulator()
    acc.bump("test", ROLE_THRESHOLD + 0.1, "strong signal")
    role, confidence, _, _ = acc.finalize()
    expect_equal(role, "test")
    expect_true(confidence >= ROLE_THRESHOLD)


def test_accumulator_finalize_picks_highest() -> None:
    """Finalize picks highest scoring role."""
    acc = RoleAccumulator()
    acc.bump("test", BUMP_VALUE_0_3, "test signal")
    acc.bump("api_handler", BUMP_VALUE_0_7, "api signal")
    role, _, _, _ = acc.finalize()
    expect_equal(role, "api_handler")


def test_accumulator_finalize_caps_confidence() -> None:
    """Confidence is capped at 1.0."""
    acc = RoleAccumulator()
    acc.bump("test", BUMP_VALUE_1_5, "very strong")
    _, confidence, _, _ = acc.finalize()
    expect_true(confidence <= CONFIDENCE_CAP)


def test_classify_test_function() -> None:
    """Classify test function from path and name."""
    context = _make_context(
        rel_path="tests/test_module.py",
        qualname="tests.test_module.test_feature",
    )
    role, confidence, _, _ = classify_function_role(context)
    expect_equal(role, "test")
    expect_true(confidence >= ROLE_THRESHOLD)


def test_classify_test_prefix() -> None:
    """Classify by test_ prefix in name."""
    context = _make_context(
        rel_path="src/module.py",
        qualname="module.test_something",
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "test")


def test_classify_pytest_fixture() -> None:
    """Classify pytest fixtures as test_helper."""
    context = _make_context(
        rel_path="tests/conftest.py",
        qualname="conftest.my_fixture",
        decorators=["pytest.fixture"],
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "test_helper")


def test_classify_api_handler() -> None:
    """Classify API handler from decorators and path."""
    context = _make_context(
        rel_path="src/api/routes.py",
        qualname="api.routes.get_users",
        decorators=["router.get('/users')"],
    )
    role, _, framework, _ = classify_function_role(context)
    expect_equal(role, "api_handler")
    expect_equal(framework, "fastapi")


def test_classify_flask_route() -> None:
    """Classify Flask route handler."""
    context = _make_context(
        rel_path="src/views.py",
        qualname="views.index",
        decorators=["app.route('/')"],
    )
    role, _, framework, _ = classify_function_role(context)
    expect_equal(role, "api_handler")
    expect_equal(framework, "flask")


def test_classify_cli_click() -> None:
    """Classify Click CLI command."""
    context = _make_context(
        rel_path="src/cli/main.py",
        qualname="cli.main.main",
        decorators=["click.command()"],
    )
    role, _, framework, _ = classify_function_role(context)
    expect_equal(role, "cli_command")
    expect_equal(framework, "click")


def test_classify_cli_typer() -> None:
    """Classify Typer CLI command."""
    context = _make_context(
        rel_path="src/main.py",
        qualname="main.run",
        decorators=["typer.command()"],
    )
    role, _, framework, _ = classify_function_role(context)
    expect_equal(role, "cli_command")
    expect_equal(framework, "typer")


def test_classify_cli_path() -> None:
    """Classify CLI from path patterns."""
    context = _make_context(
        rel_path="src/commands/deploy.py",
        qualname="commands.deploy.run_deploy",
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "cli_command")


def test_classify_repository() -> None:
    """Classify repository from effects and path."""
    context = _make_context(
        rel_path="src/repositories/user_repo.py",
        qualname="repositories.user_repo.get_user_by_id",
        effects={"touches_db": True},
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "repository")


def test_classify_service() -> None:
    """Classify service from path and graph metrics."""
    context = _make_context(
        rel_path="src/services/user_service.py",
        qualname="services.user_service.create_user",
        graph={
            "call_fan_in": SERVICE_FAN_IN_THRESHOLD + 1,
            "call_fan_out": SERVICE_FAN_OUT_THRESHOLD + 1,
        },
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "service")


def test_classify_validator() -> None:
    """Classify validator from name and contracts."""
    context = _make_context(
        rel_path="src/validators/input.py",
        qualname="validators.input.validate_email",
        contracts={"raises": [{"exception": "ValueError", "condition": "invalid format"}]},
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "validator")


@pytest.mark.parametrize("prefix", ["validate", "check", "ensure", "assert"])
def test_classify_validator_by_name(prefix: str) -> None:
    """Classify validator by name prefix."""
    context = _make_context(
        rel_path="src/validation.py",
        qualname=f"validation.{prefix}_input",
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "validator")


def test_classify_config_loader() -> None:
    """Classify config loader from path and effects."""
    context = _make_context(
        rel_path="src/config/settings.py",
        qualname="config.settings.load_config",
        effects={"uses_io": True},
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "config_loader")


def test_classify_helper() -> None:
    """Classify small pure functions as helpers."""
    context = _make_context(
        rel_path="src/utils/helpers.py",
        qualname="utils.helpers.format_name",
        loc=HELPER_LOC_THRESHOLD - 5,
        effects={},
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "helper")


def test_classify_module_tags() -> None:
    """Module tags contribute to classification."""
    context = _make_context(
        rel_path="src/api/handlers.py",
        qualname="api.handlers.get_items",
        module_tags=["api"],
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "api_handler")


def test_classify_returns_signals() -> None:
    """Classification returns source signals."""
    context = _make_context(
        rel_path="tests/test_module.py",
        qualname="tests.test_module.test_feature",
    )
    _, _, _, source = classify_function_role(context)
    signals = source.get("signals", [])
    expect_true(bool(signals))


def test_classify_other() -> None:
    """Ambiguous functions classified as 'other'."""
    context = _make_context(
        rel_path="src/misc.py",
        qualname="misc.do_something",
        loc=LARGE_LOC,
    )
    role, _, _, _ = classify_function_role(context)
    expect_equal(role, "other")


def test_modules_classify_empty() -> None:
    """Handle empty module metadata."""
    rows = classify_modules(
        module_meta={},
        roles_by_module={},
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_true(not rows)


def test_modules_classify_from_tags() -> None:
    """Classify module from tags plus function roles."""
    module_meta = {
        "api.routes": ModuleRecord(path="src/api/routes.py", tags=["api"]),
    }

    roles_by_module = {
        "api.routes": [("api_handler", 0.5)],
    }
    rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_length(rows, EXPECTED_ROWS_1)

    expect_equal(rows[0][3], "api_handler")


def test_modules_classify_from_functions() -> None:
    """Classify module from function roles."""
    module_meta = {
        "services.user": ModuleRecord(path="src/services/user.py", tags=[]),
    }
    roles_by_module = {
        "services.user": [
            ("service", BUMP_VALUE_0_8),
            ("service", BUMP_VALUE_0_6),
            ("helper", BUMP_VALUE_0_4),
        ],
    }
    rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_equal(rows[0][3], "service")


def test_modules_classify_ignores_other() -> None:
    """Module classification ignores 'other' function roles."""
    module_meta = {
        "misc": ModuleRecord(path="src/misc.py", tags=[]),
    }
    roles_by_module = {
        "misc": [
            ("other", 0.0),
            ("other", 0.0),
        ],
    }
    rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_equal(rows[0][3], "other")


def test_modules_classify_below_threshold() -> None:
    """Module classified as 'other' if below threshold."""
    module_meta = {
        "weak": ModuleRecord(path="src/weak.py", tags=[]),
    }
    roles_by_module = {
        "weak": [
            ("service", ROLE_THRESHOLD - 0.2),
        ],
    }
    rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_equal(rows[0][3], "other")


def test_modules_classify_multiple() -> None:
    """Classify multiple modules with tags and function roles."""
    module_meta = {
        "api.routes": ModuleRecord(path="src/api/routes.py", tags=["api"]),
        "services.user": ModuleRecord(path="src/services/user.py", tags=["service"]),
        "cli.main": ModuleRecord(path="src/cli/main.py", tags=["cli"]),
    }

    roles_by_module = {
        "api.routes": [("api_handler", BUMP_VALUE_0_6)],
        "services.user": [("service", BUMP_VALUE_0_5)],
        "cli.main": [("cli_command", BUMP_VALUE_0_5)],
    }
    rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo="demo/repo",
        commit="abc123",
        now=_get_now(),
    )
    expect_length(rows, EXPECTED_ROWS_3)
    roles = {row[2]: row[3] for row in rows}
    expect_equal(roles.get("api.routes"), "api_handler")
    expect_equal(roles.get("services.user"), "service")
    expect_equal(roles.get("cli.main"), "cli_command")


def test_decorator_names_empty() -> None:
    """Handle empty decorator list."""
    result = decorator_names([])
    expect_equal(result, [])


def test_decorator_names_simple() -> None:
    """Extract simple decorator name."""
    dec = ast.Name(id="property")
    result = decorator_names([dec])
    expect_equal(result, ["property"])


def test_decorator_names_attribute() -> None:
    """Extract attribute decorator name."""
    dec = ast.Attribute(
        value=ast.Name(id="pytest"),
        attr="fixture",
    )
    result = decorator_names([dec])
    expect_in("fixture", result[0])


def test_decorator_names_call() -> None:
    """Extract call decorator name."""
    dec = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="router"),
            attr="get",
        ),
        args=[ast.Constant(value="/users")],
        keywords=[],
    )
    result = decorator_names([dec])
    expect_true(bool(result))


def test_decorator_names_multiple() -> None:
    """Extract multiple decorator names."""
    decorators: list[ast.expr] = [
        ast.Name(id="staticmethod"),
        ast.Name(id="property"),
    ]
    result = decorator_names(decorators)
    expect_length(result, EXPECTED_DECORATORS_2)


def test_constants_role_threshold_positive() -> None:
    """Role threshold is positive."""
    expect_true(ROLE_THRESHOLD > 0)


def test_constants_service_fan_thresholds_positive() -> None:
    """Service fan thresholds are positive."""
    expect_true(SERVICE_FAN_IN_THRESHOLD > 0)
    expect_true(SERVICE_FAN_OUT_THRESHOLD > 0)


def test_constants_helper_loc_threshold_positive() -> None:
    """Helper LOC threshold is positive."""
    expect_true(HELPER_LOC_THRESHOLD > 0)
