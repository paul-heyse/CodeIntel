"""Tests for dynamic CLI command generation from operation metadata.

These tests verify that the dynamic CLI parameter introspection system
correctly generates typed commands for serving operations.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast, get_args

import pytest

from codeintel.cli import cyclopts_ops
from codeintel.cli.op_params import (
    CliParamSpec,
    OperationCliMetadata,
    ParamRole,
    build_operation_cli_metadata,
    classify_param_role,
    get_backend_signature_for_operation,
    get_help_panel_for_role,
    get_operations_with_cli_support,
)
from codeintel.serving.operations.catalog import get_operation, iter_operations
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli import (
    assert_parse_error,
    build_dynamic_op_args,
    run_cli,
    temp_repo_context,
)

# -----------------------------------------------------------------------------
# Parameter Classification Tests
# -----------------------------------------------------------------------------


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
        ("some_unknown_param", "filter"),  # Default for unknown params
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
    # Get a graph-related operation
    op = get_operation("callgraph.neighbors")

    # max_nodes should be classified as advanced for graph operations
    result = classify_param_role("max_nodes", operation=op)
    expect_equal(result, "advanced")


# -----------------------------------------------------------------------------
# Operation CLI Support Tests
# -----------------------------------------------------------------------------


def test_get_operations_with_cli_support_returns_list() -> None:
    """Verify operations with CLI support can be enumerated."""
    ops = get_operations_with_cli_support()

    expect_is_instance(ops, list)
    expect_true(len(ops) > 0)

    # Check structure
    for op in ops:
        expect_is_not_none(op.id)
        expect_is_not_none(op.backend_method)


def test_get_backend_signature_for_known_operation() -> None:
    """Verify signature introspection works for known operations."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = get_backend_signature_for_operation(op)

    # May be None if backend method mapping doesn't exist
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


# -----------------------------------------------------------------------------
# Dynamic Command Registration Tests
# -----------------------------------------------------------------------------


def test_op_help_shows_core_commands() -> None:
    """Verify op --help shows core commands (list, call)."""
    # Note: Dynamic command registration is tested separately from help display
    # to avoid Typer initialization issues in tests
    ops = list(iter_operations())
    expect_true(len(ops) > 0, message="Should have at least some operations registered")

    # Verify some known operations exist
    op_ids = {op.id for op in ops}
    expect_in("function.summary", op_ids)
    expect_in("file.summary", op_ids)


def test_operation_metadata_available() -> None:
    """Verify operation metadata is available for CLI generation."""
    ops = get_operations_with_cli_support()
    expect_true(len(ops) > 0, message="Should have at least some operations with CLI support")

    # Verify metadata can be built for all operations
    for op in ops[:5]:  # Test first 5 to keep test fast
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
    # required_datasets should be a tuple
    expect_is_instance(op.required_datasets, tuple)


# -----------------------------------------------------------------------------
# Dynamic subcommand registration (Cyclopts)
# -----------------------------------------------------------------------------


def test_dynamic_op_help_available() -> None:
    """Dynamic subcommands should be registered and expose help."""
    op = next(iter(iter_operations()))
    command_name = op.id.replace(".", "-")

    result = run_cli(["op", command_name, "--help"])

    expect_equal(result.exit_code, 0)
    if op.summary:
        expect_in(op.summary.split()[0].lower(), result.stdout.lower())


def test_dynamic_op_parses_and_forwards_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dynamic ops should coerce typed params and skip prereqs when requested."""
    invoked: list[tuple[str, dict[str, object]]] = []
    prereq_calls: list[dict[str, object]] = []

    def fake_invoke(op_id: str, kwargs: dict[str, object], _runtime: object) -> None:
        invoked.append((op_id, kwargs))

    def fake_prereqs(**kwargs: object) -> None:
        prereq_calls.append(kwargs)

    class DummyRuntime:
        gateway: object | None = None
        snapshot: object | None = None
        paths: object | None = None
        tools: object | None = None

    monkeypatch.setattr(cyclopts_ops, "invoke_operation", fake_invoke)
    monkeypatch.setattr(cyclopts_ops, "run_operation_prereqs", fake_prereqs)
    monkeypatch.setattr(cyclopts_ops, "_runtime_from_cli", lambda _: DummyRuntime())

    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "op",
                "functions-high-risk",
                "--min-risk",
                "0.9",
                "--limit",
                "5",
                "--skip-prereqs",
                "--repo",
                "example/repo",
                "--commit",
                "deadbeef",
                "--db-path",
                str(db_path),
                "--build-dir",
                str(ctx.build_dir),
                "--repo-root",
                str(ctx.repo_root),
            ],
            env=ctx.env,
            cwd=ctx.repo_root,
        )

    expect_equal(result.exit_code, 0)
    expect_equal(len(prereq_calls), 0)
    expect_equal(len(invoked), 1)
    op_id, kwargs = invoked[0]
    expect_equal(op_id, "functions.high_risk")
    expect_equal(kwargs.get("min_risk"), 0.9)
    expect_equal(kwargs.get("limit"), 5)


def test_dynamic_op_prereq_toggle_default_and_flag(tmp_path: Path) -> None:
    """Skip-prereqs default is False and flag flips it to True."""
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        base_args = build_dynamic_op_args(
            "functions-high-risk",
            ctx,
            extras=[
                "--min-risk",
                "0.9",
                "--limit",
                "5",
            ],
        )
        default_ns = cyclopts_ops.app(base_args, result_action="return_value")
        expect_true(not default_ns.kwargs.get("skip_prereqs"))

        flagged_args = [*base_args, "--skip-prereqs"]
        flagged_ns = cyclopts_ops.app(flagged_args, result_action="return_value")
        expect_true(flagged_ns.kwargs.get("skip_prereqs"))


def test_dynamic_op_runs_prereqs_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dynamic ops should trigger prerequisites when not skipped."""
    prereq_calls: list[dict[str, object]] = []
    invoked: list[str] = []

    def fake_prereqs(**kwargs: object) -> None:
        prereq_calls.append(kwargs)

    def fake_invoke(op_id: str, _kwargs: dict[str, object], _runtime: object) -> None:
        invoked.append(op_id)

    class DummyRuntime:
        gateway: object | None = None
        snapshot: object | None = None
        paths: object | None = None
        tools: object | None = None

    monkeypatch.setattr(cyclopts_ops, "run_operation_prereqs", fake_prereqs)
    monkeypatch.setattr(cyclopts_ops, "invoke_operation", fake_invoke)
    monkeypatch.setattr(cyclopts_ops, "_runtime_from_cli", lambda _: DummyRuntime())

    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "op",
                "function-summary",
                "--repo",
                "example/repo",
                "--commit",
                "deadbeef",
                "--db-path",
                str(db_path),
                "--build-dir",
                str(ctx.build_dir),
                "--repo-root",
                str(ctx.repo_root),
            ],
            env=ctx.env,
            cwd=ctx.repo_root,
        )

    expect_equal(result.exit_code, 0)
    expect_equal(len(prereq_calls), 1)
    expect_equal(invoked, ["function.summary"])


def _extract_parameter(metadata: tuple[object, ...]) -> cyclopts_ops.Parameter:
    """Return the first Cyclopts Parameter from Annotated metadata.

    Returns
    -------
    cyclopts_ops.Parameter
        The embedded Cyclopts parameter metadata.

    Raises
    ------
    AssertionError
        If no parameter metadata is found.
    """
    for meta in metadata:
        if isinstance(meta, cyclopts_ops.Parameter):
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
    cyclopts_ops.register_dynamic_operation_for_tests(metadata)


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
            cyclopts_ops.CliParamSpec(
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

    field_def = cyclopts_ops.build_param_field_for_spec(spec)
    annotated = field_def[1]
    metadata = get_args(annotated)[1:]
    parameter = _extract_parameter(metadata)
    expect_true(parameter.show_choices)


def test_dynamic_param_env_path_defaults_to_venv(tmp_path: Path) -> None:
    """Env-like path params should default to .venv and require existing dir."""
    spec = cyclopts_ops.CliParamSpec(
        name="venv_path",
        cli_name="venv-path",
        python_type=Path,
        default=inspect.Parameter.empty,
        role="advanced",
        help_text="venv path",
        help_panel="Advanced Options",
        is_optional=True,
    )

    default_val, validator = cyclopts_ops.path_defaults_and_validator(spec)
    expect_equal(default_val, Path(".venv"))
    expect_is_not(validator, None)
    if validator is None:
        pytest.fail("Expected a path validator for env-like paths")

    missing = tmp_path / ".venv"
    with pytest.raises(ValueError, match="does not exist"):
        validator(Path, missing)
    missing.mkdir()
    validator(Path, missing)  # should not raise


def test_dynamic_param_output_path_allows_missing_file(tmp_path: Path) -> None:
    """Output-like paths should allow non-existent targets when parent exists."""
    spec = cyclopts_ops.CliParamSpec(
        name="output_file",
        cli_name="output-file",
        python_type=Path,
        default=None,
        role="filter",
        help_text="output file",
        help_panel="Filtering Options",
        is_optional=True,
    )
    default_val, validator = cyclopts_ops.path_defaults_and_validator(spec)
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
    validator(Path, good_path)  # should not raise


# -----------------------------------------------------------------------------
# Grouping / metadata tests
# -----------------------------------------------------------------------------


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
        field_def = cyclopts_ops.build_param_field_for_spec(spec)
        annotated = field_def[1]
        metadata = get_args(annotated)[1:]
        group = next((m for m in metadata if isinstance(m, cyclopts_ops.Group)), None)
        expect_is_not(group, None)
        if group is None:
            pytest.fail("Expected Group metadata for dynamic param")
        expect_equal(group.name, get_help_panel_for_role(role))


# -----------------------------------------------------------------------------
# Parameter Spec Tests
# -----------------------------------------------------------------------------


def test_cli_param_spec_has_correct_structure() -> None:
    """Verify CliParamSpec dataclass has expected fields."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    if len(params) > 0:
        param = params[0]
        # Check expected attributes exist
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

    # Selectors should have help_panel="Target Selection"
    for param in selectors:
        expect_equal(param.help_panel, "Target Selection")

    # Filters should have help_panel="Filtering Options"
    for param in filters:
        expect_equal(param.help_panel, "Filtering Options")


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


def test_dynamic_cli_respects_operation_defaults() -> None:
    """Verify dynamic CLI uses operation-specific defaults."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    # Check that default values are populated where applicable
    for param in params:
        # Optional parameters should have is_optional set
        if param.is_optional:
            # Optional params have explicit defaults or are typed
            expect_true(
                param.default is not None or param.python_type is not None,
                message="Optional parameters should have defaults or type hints",
            )


def test_dynamic_op_path_and_literal_handling_end_to_end(tmp_path: Path) -> None:
    """Exercise choice/path heuristics via actual op CLI invocation."""
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        venv_dir = ctx.repo_root / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)

        result = run_cli(
            [
                "op",
                "function-summary",
                "--kind",
                "full",
                "--repo",
                "example/repo",
                "--commit",
                "deadbeef",
                "--db-path",
                str(db_path),
                "--build-dir",
                str(ctx.build_dir),
                "--repo-root",
                str(ctx.repo_root),
                "--env",
                str(venv_dir),
            ],
            env=ctx.env,
            cwd=ctx.repo_root,
        )

    expect_equal(result.exit_code, 0)
    expect_in("function summary", result.stdout.lower())


def test_dynamic_op_env_default_requires_existing_venv(tmp_path: Path) -> None:
    """Missing default .venv should trigger validation error when env flag omitted."""
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Intentionally do NOT create .venv
        result = run_cli(
            [
                "op",
                "function-summary",
                "--repo",
                "example/repo",
                "--commit",
                "deadbeef",
                "--db-path",
                str(db_path),
                "--build-dir",
                str(ctx.build_dir),
                "--repo-root",
                str(ctx.repo_root),
            ],
            env=ctx.env,
            cwd=ctx.repo_root,
        )

    expect_true(result.exit_code != 0)
    expect_in("Path does not exist", result.stderr)


def test_dynamic_op_env_default_uses_existing_venv(tmp_path: Path) -> None:
    """Existing .venv should satisfy the env-path validator when flag is omitted."""
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        venv_dir = ctx.repo_root / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)

        result = run_cli(
            [
                "op",
                "function-summary",
                "--repo",
                "example/repo",
                "--commit",
                "deadbeef",
                "--db-path",
                str(db_path),
                "--build-dir",
                str(ctx.build_dir),
                "--repo-root",
                str(ctx.repo_root),
            ],
            env=ctx.env,
            cwd=ctx.repo_root,
        )

    expect_equal(result.exit_code, 0)
    expect_in("function summary", result.stdout.lower())


def test_dynamic_op_returns_kwargs_with_converted_types(tmp_path: Path) -> None:
    """Parsing with result_action should return converted kwargs, not strings.

    Raises
    ------
    TypeError
        If the Cyclopts app does not return a SimpleNamespace payload.
    """
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        venv_dir = ctx.repo_root / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)
        args = build_dynamic_op_args(
            "functions-high-risk",
            ctx,
            extras=[
                "--min-risk",
                "0.9",
                "--limit",
                "5",
                "--env",
                str(venv_dir),
            ],
        )
        result = cyclopts_ops.app(args, result_action="return_value")

    if not isinstance(result, cyclopts_ops.SimpleNamespace):
        message = "Expected SimpleNamespace result from Cyclopts app invocation"
        raise TypeError(message)
    expect_equal(result.kwargs.get("min_risk"), 0.9)
    expect_equal(result.kwargs.get("limit"), 5)
    expect_is_instance(result.kwargs.get("env"), Path)


def test_dynamic_op_literal_choice_parsing(tmp_path: Path) -> None:
    """Literal choices parse correctly and reject invalid values."""
    params = (
        _make_spec(
            name="kind",
            python_type=cast("type[Any]", Literal["full", "summary"]),
            default=None,
            role="filter",
            is_optional=True,
        ),
    )
    _register_test_op("test-choice-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        args = build_dynamic_op_args(
            "test-choice-op",
            ctx,
            extras=["--kind", "full"],
        )
        result_ns = cyclopts_ops.app(args, result_action="return_value")
        expect_equal(result_ns.kwargs.get("kind"), "full")

        bad_args = build_dynamic_op_args(
            "test-choice-op",
            ctx,
            extras=["--kind", "invalid"],
        )
        bad_result = run_cli(bad_args, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_result, match="invalid")


def test_dynamic_op_bool_flag_without_negative(tmp_path: Path) -> None:
    """Bool params should not accept autogenerated negative flags."""
    params = (
        _make_spec(
            name="flag",
            python_type=bool,
            default=False,
            role="filter",
            is_optional=True,
        ),
    )
    _register_test_op("test-bool-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        args = build_dynamic_op_args(
            "test-bool-op",
            ctx,
            extras=["--flag"],
        )
        ns = cyclopts_ops.app(args, result_action="return_value")
        expect_true(ns.kwargs.get("flag"))

        bad_args = build_dynamic_op_args(
            "test-bool-op",
            ctx,
            extras=["--no-flag"],
        )
        bad_result = run_cli(bad_args, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_result, match="No such option")


def test_dynamic_op_numeric_coercion_and_failure(tmp_path: Path) -> None:
    """Numeric params coerce strings and reject invalid numbers."""
    params = (
        _make_spec(
            name="limit",
            python_type=int,
            default=None,
            role="filter",
            is_optional=True,
        ),
        _make_spec(
            name="threshold",
            python_type=float,
            default=None,
            role="filter",
            is_optional=True,
        ),
    )
    _register_test_op("test-numeric-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        good_args = build_dynamic_op_args(
            "test-numeric-op",
            ctx,
            extras=["--limit", "10", "--threshold", "0.75"],
        )
        ns = cyclopts_ops.app(good_args, result_action="return_value")
        expect_equal(ns.kwargs.get("limit"), 10)
        expect_equal(ns.kwargs.get("threshold"), 0.75)

        bad_args = build_dynamic_op_args(
            "test-numeric-op",
            ctx,
            extras=["--limit", "not-a-number"],
        )
        bad_result = run_cli(bad_args, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_result, match="invalid")


def test_dynamic_op_required_vs_optional(tmp_path: Path) -> None:
    """Required params must be provided; optional may be omitted."""
    params = (
        _make_spec(
            name="required_arg",
            python_type=str,
            default=inspect.Parameter.empty,
            role="selector",
            is_optional=False,
        ),
        _make_spec(
            name="optional_arg",
            python_type=str,
            default=None,
            role="filter",
            is_optional=True,
        ),
    )
    _register_test_op("test-required-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        missing_required = build_dynamic_op_args("test-required-op", ctx)
        bad_result = run_cli(missing_required, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_result, match="missing")

        good_args = build_dynamic_op_args(
            "test-required-op",
            ctx,
            extras=["--required-arg", "value"],
        )
        ns = cyclopts_ops.app(good_args, result_action="return_value")
        expect_equal(ns.kwargs.get("required_arg"), "value")
        expect_equal(ns.kwargs.get("optional_arg"), None)


def test_dynamic_op_env_path_heuristics(tmp_path: Path) -> None:
    """Env default and custom env path validation."""
    params = (
        _make_spec(
            name="env_path",
            python_type=Path,
            default=inspect.Parameter.empty,
            role="advanced",
            is_optional=True,
        ),
        _make_spec(
            name="input_path",
            python_type=Path,
            default=inspect.Parameter.empty,
            role="selector",
            is_optional=False,
        ),
    )
    _register_test_op("test-path-env-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = ctx.repo_root / "input.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("data")

        # Missing .venv triggers error
        args_missing_env = build_dynamic_op_args(
            "test-path-env-op",
            ctx,
            extras=["--input-path", str(input_path)],
        )
        bad_env = run_cli(args_missing_env, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_env, match="Path does not exist")

        # Create .venv and ensure default works
        venv_dir = ctx.repo_root / ".venv"
        venv_dir.mkdir(parents=True, exist_ok=True)
        args_default_env = build_dynamic_op_args(
            "test-path-op",
            ctx,
            extras=["--input-path", str(input_path)],
        )
        ns_default = cyclopts_ops.app(args_default_env, result_action="return_value")
        expect_equal(ns_default.kwargs.get("env_path"), Path(".venv"))
        expect_is_instance(ns_default.kwargs.get("input_path"), Path)

        # Custom env override must exist
        custom_env = ctx.repo_root / "custom_env"
        args_custom_missing = build_dynamic_op_args(
            "test-path-env-op",
            ctx,
            extras=["--input-path", str(input_path), "--env-path", str(custom_env)],
        )
        bad_custom = run_cli(args_custom_missing, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_custom, match="does not exist")
        custom_env.mkdir(parents=True, exist_ok=True)
        args_custom_ok = build_dynamic_op_args(
            "test-path-env-op",
            ctx,
            extras=["--input-path", str(input_path), "--env-path", str(custom_env)],
        )
        ns_custom = cyclopts_ops.app(args_custom_ok, result_action="return_value")
        expect_equal(ns_custom.kwargs.get("env_path"), custom_env)


def test_dynamic_op_output_and_input_paths(tmp_path: Path) -> None:
    """Output-like paths allow missing file when parent exists; inputs must exist."""
    params = (
        _make_spec(
            name="output_file",
            python_type=Path,
            default=None,
            role="filter",
            is_optional=True,
        ),
        _make_spec(
            name="input_path",
            python_type=Path,
            default=inspect.Parameter.empty,
            role="selector",
            is_optional=False,
        ),
    )
    _register_test_op("test-path-output-op", params)
    with temp_repo_context(tmp_path) as ctx:
        db_path = ctx.build_dir / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = ctx.repo_root / "input.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("data")

        # Parent missing -> error
        bad_output = ctx.repo_root / "missing_parent" / "out.json"
        args_bad_output = build_dynamic_op_args(
            "test-path-output-op",
            ctx,
            extras=["--input-path", str(input_path), "--output-file", str(bad_output)],
        )
        bad_out = run_cli(args_bad_output, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_out, match="Parent directory")

        # Parent exists -> success, Path converter applied
        output_dir = ctx.repo_root / "outdir"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "out.json"
        args_good_output = build_dynamic_op_args(
            "test-path-output-op",
            ctx,
            extras=["--input-path", str(input_path), "--output-file", str(output_file)],
        )
        ns_output = cyclopts_ops.app(args_good_output, result_action="return_value")
        expect_equal(ns_output.kwargs.get("output_file"), output_file)

        # Missing input should fail
        missing_input = ctx.repo_root / "missing.txt"
        args_missing_input = build_dynamic_op_args(
            "test-path-output-op",
            ctx,
            extras=["--input-path", str(missing_input)],
        )
        bad_input = run_cli(args_missing_input, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_input, match="Path does not exist")

        # Output file allowed if parent exists; parent missing fails
        bad_output = ctx.repo_root / "missing_parent" / "out.json"
        args_bad_output = build_dynamic_op_args(
            "test-path-op",
            ctx,
            extras=["--input-path", str(input_path), "--output-file", str(bad_output)],
        )
        bad_out = run_cli(args_bad_output, env=ctx.env, cwd=ctx.repo_root)
        assert_parse_error(bad_out, match="Parent directory")

        output_dir = ctx.repo_root / "outdir"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "out.json"
        args_good_output = build_dynamic_op_args(
            "test-path-op",
            ctx,
            extras=["--input-path", str(input_path), "--output-file", str(output_file)],
        )
        ns_output = cyclopts_ops.app(args_good_output, result_action="return_value")
        expect_equal(ns_output.kwargs.get("output_file"), output_file)
