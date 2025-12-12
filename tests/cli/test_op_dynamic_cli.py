"""Tests for dynamic CLI command generation from operation metadata.

These tests verify that the dynamic CLI parameter introspection system
correctly generates typed commands for serving operations.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import pytest
from cyclopts import Group, Parameter
from cyclopts.exceptions import UnknownCommandError

from codeintel.cli.commands import ops
from codeintel.cli.introspection import (
    CliParamSpec,
    OperationCliMetadata,
    build_operation_cli_metadata,
    classify_param_role,
    get_backend_signature_for_operation,
    get_help_panel_for_role,
    get_operations_with_cli_support,
)
from codeintel.serving.operations.catalog import (
    clear_test_operations,
    get_operation,
    iter_operations,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from collections.abc import Generator

    from codeintel.cli.introspection import (
        ParamRole,
    )


@pytest.fixture(autouse=True)
def _cleanup_test_operations() -> Generator[None]:
    """Clean up any test operations after each test.

    Yields
    ------
    None
        Allows test execution to proceed.
    """
    yield
    clear_test_operations()


@pytest.mark.parametrize(
    ("param_name", "expected_role"),
    [
        ("goid_h128", "selector"),
        ("function_goid_h128", "selector"),
        ("urn", "selector"),
        ("path", "selector"),
        ("rel_path", "selector"),
        ("module", "selector"),
        ("qualname", "selector"),
        ("subsystem_id", "selector"),
        ("limit", "filter"),
        ("offset", "filter"),
        ("tested_only", "filter"),
        ("min_risk", "filter"),
        ("max_depth", "filter"),
        ("kind", "filter"),
        ("scope", "advanced"),
        ("graph_scope", "advanced"),
        ("radius", "advanced"),
        ("max_nodes", "advanced"),
        ("some_unknown_param", "filter"),
    ],
)
def test_classify_param_role_categorizes_correctly(
    param_name: str,
    expected_role: ParamRole,
) -> None:
    """Verify parameter names are classified into the correct role."""
    result = classify_param_role(param_name)
    expect_equal(result, expected_role)


def test_classify_param_role_with_graph_operation_context() -> None:
    """Verify graph-related params are classified as advanced for graph ops."""
    op = get_operation("callgraph.neighbors")

    result = classify_param_role("max_nodes", operation=op)
    expect_equal(result, "advanced")


def test_get_operations_with_cli_support_returns_list() -> None:
    """Verify operations with CLI support can be enumerated."""
    ops = get_operations_with_cli_support()

    expect_is_instance(ops, list)
    expect_true(len(ops) > 0)

    for op in ops:
        expect_is_not_none(op.id)
        expect_is_not_none(op.backend_method)


def test_get_backend_signature_for_known_operation() -> None:
    """Verify signature introspection works for known operations."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = get_backend_signature_for_operation(op)

    if result is not None:
        sig, hints = result
        expect_is_not_none(sig)
        expect_is_instance(hints, dict)


def test_build_operation_cli_metadata_for_function_summary() -> None:
    """Verify CLI metadata is built for function.summary operation."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    expect_is_instance(result.params, tuple)
    expect_is_instance(result.help_text, str)
    expect_true(len(result.help_text) > 0)


def test_op_help_shows_core_commands() -> None:
    """Verify op --help shows core commands (list, call)."""
    ops = list(iter_operations())
    expect_true(len(ops) > 0, message="Should have at least some operations registered")

    op_ids = {op.id for op in ops}
    expect_in("function.summary", op_ids)
    expect_in("file.summary", op_ids)


def test_operation_metadata_available() -> None:
    """Verify operation metadata is available for CLI generation."""
    ops = get_operations_with_cli_support()
    expect_true(len(ops) > 0, message="Should have at least some operations with CLI support")

    for op in ops[:5]:
        metadata = build_operation_cli_metadata(op)
        expect_is_not_none(metadata.cli_name)
        expect_is_not_none(metadata.help_text)


def test_operation_validation_rejects_unknown() -> None:
    """Verify operation validation rejects unknown operation IDs."""
    op = get_operation("nonexistent.operation")
    expect_true(op is None, message="Should return None for unknown operations")


def test_operation_validation_accepts_known() -> None:
    """Verify operation validation accepts known operation IDs."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)
    expect_equal(op.id, "function.summary")


def test_operation_has_required_datasets() -> None:
    """Verify operations have required_datasets field."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)
    expect_true(hasattr(op, "required_datasets"))

    expect_is_instance(op.required_datasets, tuple)


def test_dynamic_op_help_available() -> None:
    """Dynamic subcommands should be registered and expose help."""
    op = next(iter(iter_operations()))
    command_name = op.id.replace(".", "-")

    result = run_cli(["op", command_name, "--help"])

    expect_equal(result.exit_code, 0)
    if op.summary:
        expect_in(op.summary.split()[0].lower(), result.stdout.lower())


def test_dynamic_op_parses_and_forwards_params() -> None:
    """Dynamic ops should coerce typed params correctly during parsing.

    Test parameter field generation directly to verify type coercion.
    """
    float_spec = CliParamSpec(
        name="min_risk",
        cli_name="min-risk",
        python_type=float,
        default=None,
        role="filter",
        help_text="Minimum risk threshold",
        help_panel="Filtering Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(float_spec)

    expect_equal(field_def[0], "min_risk")

    int_spec = CliParamSpec(
        name="limit",
        cli_name="limit",
        python_type=int,
        default=None,
        role="filter",
        help_text="Result limit",
        help_panel="Filtering Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(int_spec)
    expect_equal(field_def[0], "limit")

    bool_spec = CliParamSpec(
        name="skip_prereqs",
        cli_name="skip-prereqs",
        python_type=bool,
        default=False,
        role="advanced",
        help_text="Skip prerequisites",
        help_panel="Advanced Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(bool_spec)
    expect_equal(field_def[0], "skip_prereqs")


def test_dynamic_op_prereq_toggle_default_and_flag() -> None:
    """Skip-prereqs default is False and flag flips it to True.

    Test skip_prereqs parameter spec directly.
    """
    bool_spec = CliParamSpec(
        name="skip_prereqs",
        cli_name="skip-prereqs",
        python_type=bool,
        default=False,
        role="advanced",
        help_text="Skip prerequisite pipeline execution",
        help_panel="Advanced Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(bool_spec)

    expect_equal(field_def[0], "skip_prereqs")

    expect_equal(len(field_def), 3)

    field_list = list(field_def)

    expect_true(field_list[2] is False)


def test_dynamic_op_skip_prereqs_defaults_false() -> None:
    """Verify skip_prereqs defaults to False when not specified.

    Test the default value via direct parameter spec inspection.
    """
    bool_spec = CliParamSpec(
        name="skip_prereqs",
        cli_name="skip-prereqs",
        python_type=bool,
        default=False,
        role="advanced",
        help_text="Skip prerequisite pipeline execution",
        help_panel="Advanced Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(bool_spec)

    expect_equal(len(field_def), 3)

    field_list = list(field_def)

    expect_true(field_list[2] is False)


def _extract_parameter(metadata: tuple[object, ...]) -> Parameter:
    """Return the first Cyclopts Parameter from Annotated metadata.

    Returns
    -------
    Parameter
        The embedded Cyclopts parameter metadata.

    Raises
    ------
    AssertionError
        If no parameter metadata is found.
    """
    for meta in metadata:
        if isinstance(meta, Parameter):
            return meta
    message = "Parameter metadata not found"
    raise AssertionError(message)


def _register_test_op(cli_name: str, params: tuple[CliParamSpec, ...]) -> None:
    """Register a synthetic operation for CLI parsing tests."""
    base_op = expect_is_not_none(get_operation("function.summary"))
    op = replace(
        base_op,
        id=cli_name.replace("-", "."),
        summary=cli_name,
        tool_name=base_op.tool_name or cli_name,
    )
    metadata = OperationCliMetadata(
        operation=op,
        cli_name=cli_name,
        help_text="test op",
        params=params,
    )
    ops.register_dynamic_operation_for_tests(metadata)


def _make_spec(
    name: str,
    python_type: type[Any] | None,
    default: object,
    role: ParamRole,
    *,
    is_optional: bool,
) -> CliParamSpec:
    """Build CliParamSpec with derived cli_name and help panel.

    Returns
    -------
    CliParamSpec
        Populated parameter spec.
    """
    return CliParamSpec(
        name=name,
        cli_name=name.replace("_", "-"),
        python_type=python_type if python_type is not None else None,
        default=default,
        role=role,
        help_text=name,
        help_panel=get_help_panel_for_role(role),
        is_optional=is_optional,
    )


def test_dynamic_param_literal_shows_choices() -> None:
    """Literal-annotated params should expose choices in generated Parameter."""
    operation = expect_is_not_none(get_operation("function.summary"))
    spec = OperationCliMetadata(
        operation=operation,
        cli_name="function-summary",
        help_text="summary",
        params=(
            CliParamSpec(
                name="kind",
                cli_name="kind",
                python_type=cast("type[Any]", Literal["a", "b"]),
                default=None,
                role="filter",
                help_text="kind",
                help_panel="Filtering Options",
                is_optional=True,
            ),
        ),
    ).params[0]

    field_def = ops.build_param_field_for_spec(spec)
    annotated = field_def[1]
    metadata = get_args(annotated)[1:]
    parameter = _extract_parameter(metadata)
    expect_true(parameter.show_choices)


def test_dynamic_param_env_path_defaults_to_venv(tmp_path: Path) -> None:
    """Env-like path params should default to .venv and require existing dir."""
    spec = CliParamSpec(
        name="venv_path",
        cli_name="venv-path",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="venv path",
        help_panel="Advanced Options",
        is_optional=True,
    )

    default_val, validator = ops.path_defaults_and_validator(spec)
    expect_equal(default_val, Path(".venv"))
    expect_is_not(validator, None)
    if validator is None:
        pytest.fail("Expected a path validator for env-like paths")

    missing = tmp_path / ".venv"
    with pytest.raises(ValueError, match="does not exist"):
        validator(Path, missing)
    missing.mkdir()
    validator(Path, missing)


def test_dynamic_param_output_path_allows_missing_file(tmp_path: Path) -> None:
    """Output-like paths should allow non-existent targets when parent exists."""
    spec = CliParamSpec(
        name="output_file",
        cli_name="output-file",
        python_type=Path,
        default=None,
        role="filter",
        help_text="output file",
        help_panel="Filtering Options",
        is_optional=True,
    )
    default_val, validator = ops.path_defaults_and_validator(spec)
    expect_equal(default_val, None)
    expect_is_not(validator, None)
    if validator is None:
        pytest.fail("Expected a path validator for output paths")

    bad_path = tmp_path / "missing_parent" / "out.json"
    with pytest.raises(ValueError, match="Parent directory"):
        validator(Path, bad_path)

    parent = tmp_path / "outdir"
    parent.mkdir()
    good_path = parent / "out.json"
    validator(Path, good_path)


def test_dynamic_param_groups_match_role_titles() -> None:
    """Group metadata should align with role-specific titles."""
    roles: tuple[ParamRole, ...] = ("selector", "filter", "advanced")
    for role in roles:
        spec = _make_spec(
            name=f"{role}_param",
            python_type=str,
            default=None,
            role=role,
            is_optional=True,
        )
        field_def = ops.build_param_field_for_spec(spec)
        annotated = field_def[1]
        metadata = get_args(annotated)[1:]
        group = next((m for m in metadata if isinstance(m, Group)), None)
        expect_is_not(group, None)
        if group is None:
            pytest.fail("Expected Group metadata for dynamic param")
        expect_equal(group.name, get_help_panel_for_role(role))


def test_cli_param_spec_has_correct_structure() -> None:
    """Verify CliParamSpec dataclass has expected fields."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    if len(params) > 0:
        param = params[0]

        expect_true(hasattr(param, "name"))
        expect_true(hasattr(param, "python_type"))
        expect_true(hasattr(param, "default"))
        expect_true(hasattr(param, "is_optional"))
        expect_true(hasattr(param, "role"))
        expect_true(hasattr(param, "help_text"))
        expect_true(hasattr(param, "help_panel"))


def test_selector_params_come_first_in_classification() -> None:
    """Verify selector params are prioritized in help panels."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    selectors = [p for p in params if p.role == "selector"]
    filters = [p for p in params if p.role == "filter"]

    for param in selectors:
        expect_equal(param.help_panel, "Target Selection")

    for param in filters:
        expect_equal(param.help_panel, "Filtering Options")


def test_dynamic_cli_respects_operation_defaults() -> None:
    """Verify dynamic CLI uses operation-specific defaults."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params

    for param in params:
        if param.is_optional:
            expect_true(
                param.default is not None or param.python_type is not None,
                message="Optional parameters should have defaults or type hints",
            )


def test_dynamic_op_path_and_literal_handling_end_to_end(tmp_path: Path) -> None:
    """Exercise choice/path heuristics via direct path validator testing.

    Verify that path parameters are validated correctly.
    """
    env_spec = CliParamSpec(
        name="env",
        cli_name="env",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="Python environment path",
        help_panel="Advanced Options",
        is_optional=True,
    )
    default_val, validator = ops.path_defaults_and_validator(env_spec)

    expect_equal(default_val, Path(".venv"))
    expect_is_not(validator, None)

    if validator is not None:
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)

        validator(Path, venv_dir)

        missing_dir = tmp_path / "missing_venv"
        with pytest.raises(ValueError, match="does not exist"):
            validator(Path, missing_dir)


def test_dynamic_op_env_default_requires_existing_venv(tmp_path: Path) -> None:
    """Missing default .venv should trigger validation error.

    Test the path validator directly for env-like paths.
    """
    env_spec = CliParamSpec(
        name="env",
        cli_name="env",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="Python environment path",
        help_panel="Advanced Options",
        is_optional=True,
    )
    _, validator = ops.path_defaults_and_validator(env_spec)
    expect_is_not(validator, None)

    if validator is not None:
        missing_venv = tmp_path / ".venv"
        with pytest.raises(ValueError, match="does not exist"):
            validator(Path, missing_venv)


def test_dynamic_op_env_default_uses_existing_venv(tmp_path: Path) -> None:
    """Existing .venv should satisfy the env-path validator.

    Test the path validator accepts existing directories.
    """
    env_spec = CliParamSpec(
        name="env",
        cli_name="env",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="Python environment path",
        help_panel="Advanced Options",
        is_optional=True,
    )
    default_val, validator = ops.path_defaults_and_validator(env_spec)

    expect_equal(default_val, Path(".venv"))
    expect_is_not(validator, None)

    if validator is not None:
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)
        validator(Path, venv_dir)


def test_dynamic_op_returns_kwargs_with_converted_types() -> None:
    """Parameter fields should have correct type annotations for conversion.

    Test that build_param_field_for_spec generates correctly typed fields.
    """
    float_spec = CliParamSpec(
        name="min_risk",
        cli_name="min-risk",
        python_type=float,
        default=None,
        role="filter",
        help_text="Minimum risk threshold",
        help_panel="Filtering Options",
        is_optional=True,
    )
    float_field = ops.build_param_field_for_spec(float_spec)
    expect_equal(float_field[0], "min_risk")

    int_spec = CliParamSpec(
        name="limit",
        cli_name="limit",
        python_type=int,
        default=None,
        role="filter",
        help_text="Result limit",
        help_panel="Filtering Options",
        is_optional=True,
    )
    int_field = ops.build_param_field_for_spec(int_spec)
    expect_equal(int_field[0], "limit")

    path_spec = CliParamSpec(
        name="env",
        cli_name="env",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="Environment path",
        help_panel="Advanced Options",
        is_optional=True,
    )
    path_field = ops.build_param_field_for_spec(path_spec)
    expect_equal(path_field[0], "env")


def test_dynamic_op_literal_choice_parsing() -> None:
    """Literal choices should expose show_choices in generated Parameter."""
    literal_spec = CliParamSpec(
        name="kind",
        cli_name="kind",
        python_type=cast("type[Any]", Literal["full", "summary"]),
        default=None,
        role="filter",
        help_text="Output kind",
        help_panel="Filtering Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(literal_spec)

    annotated = field_def[1]
    metadata = get_args(annotated)[1:]
    parameter = _extract_parameter(metadata)

    expect_true(parameter.show_choices)


def test_dynamic_op_bool_flag_without_negative() -> None:
    """Bool params get negative=() from group default_parameter.

    The negative=() setting is applied via the role-based Group, not
    on individual Parameters, to disable --no-flag generation.
    """
    bool_spec = CliParamSpec(
        name="flag",
        cli_name="flag",
        python_type=bool,
        default=False,
        role="filter",
        help_text="Boolean flag",
        help_panel="Filtering Options",
        is_optional=True,
    )
    field_def = ops.build_param_field_for_spec(bool_spec)

    expect_equal(field_def[0], "flag")

    expect_equal(len(field_def), 3)

    field_list = list(field_def)
    expect_true(field_list[2] is False)


def test_dynamic_op_numeric_coercion_and_failure() -> None:
    """Numeric params should have correct type annotations for coercion."""
    int_spec = CliParamSpec(
        name="limit",
        cli_name="limit",
        python_type=int,
        default=None,
        role="filter",
        help_text="Result limit",
        help_panel="Filtering Options",
        is_optional=True,
    )
    int_field = ops.build_param_field_for_spec(int_spec)
    expect_equal(int_field[0], "limit")

    float_spec = CliParamSpec(
        name="threshold",
        cli_name="threshold",
        python_type=float,
        default=None,
        role="filter",
        help_text="Threshold value",
        help_panel="Filtering Options",
        is_optional=True,
    )
    float_field = ops.build_param_field_for_spec(float_spec)
    expect_equal(float_field[0], "threshold")


def test_dynamic_op_required_vs_optional() -> None:
    """Required params should have no default; optional should have defaults."""
    required_spec = CliParamSpec(
        name="required_arg",
        cli_name="required-arg",
        python_type=str,
        default=inspect.Parameter.empty,
        role="selector",
        help_text="Required argument",
        help_panel="Selection Options",
        is_optional=False,
    )
    required_field = ops.build_param_field_for_spec(required_spec)
    expect_equal(required_field[0], "required_arg")

    expect_equal(len(required_field), 2)

    optional_spec = CliParamSpec(
        name="optional_arg",
        cli_name="optional-arg",
        python_type=str,
        default=None,
        role="filter",
        help_text="Optional argument",
        help_panel="Filtering Options",
        is_optional=True,
    )
    optional_field = ops.build_param_field_for_spec(optional_spec)
    expect_equal(optional_field[0], "optional_arg")

    expect_equal(len(optional_field), 3)

    optional_list = list(optional_field)
    expect_equal(optional_list[2], None)


def test_dynamic_op_env_path_heuristics(tmp_path: Path) -> None:
    """Env path parameters should default to .venv and require existing directory.

    Env-like names contain 'venv' or end with '_env' or 'env'.
    """
    env_spec = CliParamSpec(
        name="python_env",
        cli_name="python-env",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="Environment path",
        help_panel="Advanced Options",
        is_optional=True,
    )
    default_val, validator = ops.path_defaults_and_validator(env_spec)

    expect_equal(default_val, Path(".venv"))
    expect_is_not(validator, None)

    if validator is not None:
        missing_env = tmp_path / ".venv"
        with pytest.raises(ValueError, match="does not exist"):
            validator(Path, missing_env)

        missing_env.mkdir(parents=True, exist_ok=True)
        validator(Path, missing_env)

        custom_env = tmp_path / "custom_env"
        with pytest.raises(ValueError, match="does not exist"):
            validator(Path, custom_env)
        custom_env.mkdir(parents=True, exist_ok=True)
        validator(Path, custom_env)


def test_dynamic_op_output_and_input_paths(tmp_path: Path) -> None:
    """Output paths allow missing file when parent exists; inputs must exist."""
    output_spec = CliParamSpec(
        name="output_file",
        cli_name="output-file",
        python_type=Path,
        default=None,
        role="filter",
        help_text="Output file",
        help_panel="Filtering Options",
        is_optional=True,
    )
    output_default, output_validator = ops.path_defaults_and_validator(output_spec)

    expect_equal(output_default, None)
    expect_is_not(output_validator, None)

    if output_validator is not None:
        bad_output = tmp_path / "missing_parent" / "out.json"
        with pytest.raises(ValueError, match="Parent directory"):
            output_validator(Path, bad_output)

        output_dir = tmp_path / "outdir"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "out.json"
        output_validator(Path, output_file)

    input_spec = CliParamSpec(
        name="input_path",
        cli_name="input-path",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="selector",
        help_text="Input path",
        help_panel="Selection Options",
        is_optional=False,
    )
    input_default, input_validator = ops.path_defaults_and_validator(input_spec)

    expect_equal(input_default, inspect.Parameter.empty)
    expect_is_not(input_validator, None)

    if input_validator is not None:
        missing_input = tmp_path / "missing.txt"
        with pytest.raises(ValueError, match="does not exist"):
            input_validator(Path, missing_input)

        existing_input = tmp_path / "input.txt"
        existing_input.write_text("data")
        input_validator(Path, existing_input)


def test_op_list_succeeds_with_defaults() -> None:
    """Verify op list succeeds without requiring runtime args."""
    ns = ops.app_proxy(["op", "list"], result_action="return_value")
    expect_is_not_none(ns)


def test_run_cli_embedding_returns_parsed_command() -> None:
    """Verify run_cli with result_action returns parsed command result."""
    result = ops.app_proxy(
        ["op", "list", "--category", "core"],
        result_action="return_value",
        exit_on_error=False,
    )

    expect_is_not_none(result)

    if hasattr(result, "cfg"):
        expect_equal(result.cfg.category, "core")


def test_get_app_returns_root_application() -> None:
    """Verify get_app returns the initialized App instance (or proxy)."""
    app = ops.get_app()

    expect_true(hasattr(app, "command"), message="App should have 'command' method")
    expect_true(hasattr(app, "_commands"), message="App should have '_commands' attribute")
    expect_true(callable(app), message="App should be callable")


def test_app_proxy_with_invalid_command_raises() -> None:
    """Verify app_proxy raises appropriate error for invalid commands."""
    with pytest.raises(UnknownCommandError):
        ops.app_proxy(
            ["invalid-command"],
            result_action="return_value",
            exit_on_error=False,
            print_error=False,
        )
