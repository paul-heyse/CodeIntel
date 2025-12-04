"""Comprehensive tests for ingestion step modules.

This module tests config_ingest, tests_ingest, and typing_ingest steps
with full coverage of parsing, flattening, and computation logic.

Note: Tests import private functions (prefixed with _) to ensure full coverage
of internal parsing and computation logic. This is intentional for testing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from codeintel.ingestion.compute import config_ingest, typing_ingest
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
from codeintel.ingestion.compute.typing_ingest import AnnotationInfo
from tests._helpers.fakes import FakeIngestStorage

# Test constants
EXPECTED_PARAMS_RATIO_HALF = 0.5
EXPECTED_RETURNS_RATIO_THREE_QUARTERS = 0.75
EXPECTED_UNTYPED_DEFS_TWO = 2


# =============================================================================
# config_ingest.flatten_dict Tests
# =============================================================================


def test_flatten_dict_simple() -> None:
    """Flatten a simple dict."""
    data = {"a": 1, "b": 2}
    result = config_ingest.flatten_dict(data)
    assert result == [("a", 1), ("b", 2)]


def test_flatten_dict_nested() -> None:
    """Flatten a nested dict."""
    data = {"outer": {"inner": "value"}}
    result = config_ingest.flatten_dict(data)
    assert result == [("outer.inner", "value")]


def test_flatten_dict_deeply_nested() -> None:
    """Flatten a deeply nested dict."""
    data = {"a": {"b": {"c": {"d": "deep"}}}}
    result = config_ingest.flatten_dict(data)
    assert result == [("a.b.c.d", "deep")]


def test_flatten_dict_empty() -> None:
    """Flatten an empty dict."""
    result = config_ingest.flatten_dict({})
    assert result == []


def test_flatten_dict_with_none() -> None:
    """Flatten dict with None values."""
    data = {"key": None}
    result = config_ingest.flatten_dict(data)
    assert result == [("key", None)]


EXPECTED_MIXED_DICT_ITEMS = 4


def test_flatten_dict_with_mixed_types() -> None:
    """Flatten dict with mixed value types."""
    data = {"int_key": 42, "str_key": "value", "bool_key": True, "float_key": 3.14}
    result = config_ingest.flatten_dict(data)
    assert len(result) == EXPECTED_MIXED_DICT_ITEMS
    assert ("int_key", 42) in result
    assert ("str_key", "value") in result
    assert ("bool_key", True) in result
    assert ("float_key", 3.14) in result


# =============================================================================
# config_ingest.flatten_list_items Tests
# =============================================================================


def test_flatten_list_items_simple() -> None:
    """Flatten a list of simple values."""
    result = config_ingest.flatten_list_items([1, 2, 3], "items", ".")
    expected = [("items[0]", 1), ("items[1]", 2), ("items[2]", 3)]
    assert result == expected


def test_flatten_list_items_with_dicts() -> None:
    """Flatten a list containing dicts."""
    items = [{"name": "a"}, {"name": "b"}]
    result = config_ingest.flatten_list_items(items, "items", ".")
    expected = [("items[0].name", "a"), ("items[1].name", "b")]
    assert result == expected


def test_flatten_list_items_empty() -> None:
    """Flatten an empty list."""
    result = config_ingest.flatten_list_items([], "items", ".")
    assert result == []


def test_flatten_list_items_with_strings() -> None:
    """Flatten a list of strings."""
    result = config_ingest.flatten_list_items(["a", "b", "c"], "names", ".")
    expected = [("names[0]", "a"), ("names[1]", "b"), ("names[2]", "c")]
    assert result == expected


# =============================================================================
# Config Parsing Tests
# =============================================================================


def test_parse_toml_valid() -> None:
    """Parse valid TOML content."""
    content = """
[section]
key = "value"
number = 42
"""
    result = config_ingest.parse_toml(content)
    assert result is not None
    assert ("section.key", "value") in result
    assert ("section.number", 42) in result


def test_parse_toml_invalid() -> None:
    """Parse invalid TOML returns None."""
    result = config_ingest.parse_toml("invalid [ toml content")
    assert result is None


def test_parse_toml_with_nested_sections() -> None:
    """Parse TOML with nested sections."""
    content = """
[tool.pytest]
testpaths = "tests"

[tool.mypy]
strict = true
"""
    result = config_ingest.parse_toml(content)
    assert result is not None
    assert ("tool.pytest.testpaths", "tests") in result
    assert ("tool.mypy.strict", True) in result


def test_parse_yaml_valid() -> None:
    """Parse valid YAML content."""
    content = """
section:
  key: value
  number: 42
"""
    result = config_ingest.parse_yaml(content)
    assert result is not None
    assert ("section.key", "value") in result
    assert ("section.number", 42) in result


def test_parse_yaml_invalid() -> None:
    """Parse invalid YAML returns None."""
    result = config_ingest.parse_yaml("invalid:\n  :\n  bad yaml")
    assert result is None


def test_parse_yaml_non_dict() -> None:
    """Parse YAML that's not a dict returns None."""
    result = config_ingest.parse_yaml("- item1\n- item2")
    assert result is None


def test_parse_yaml_empty() -> None:
    """Parse empty YAML returns None."""
    result = config_ingest.parse_yaml("")
    assert result is None


def test_parse_json_valid() -> None:
    """Parse valid JSON content."""
    content = '{"key": "value", "nested": {"inner": 42}}'
    result = config_ingest.parse_json(content)
    assert result is not None
    assert ("key", "value") in result
    assert ("nested.inner", 42) in result


def test_parse_json_invalid() -> None:
    """Parse invalid JSON returns None."""
    result = config_ingest.parse_json("not valid json")
    assert result is None


def test_parse_json_non_dict() -> None:
    """Parse JSON that's not a dict returns None."""
    result = config_ingest.parse_json("[1, 2, 3]")
    assert result is None


def test_parse_ini_valid() -> None:
    """Parse valid INI content."""
    content = """
[section]
key = value
number = 42
"""
    result = config_ingest.parse_ini(content)
    assert result is not None
    assert ("section.key", "value") in result
    assert ("section.number", "42") in result  # INI values are strings


def test_parse_ini_invalid() -> None:
    """Parse invalid INI returns None."""
    # MissingSectionHeaderError
    result = config_ingest.parse_ini("key=value\nno section header")
    assert result is None


def test_parse_ini_multiple_sections() -> None:
    """Parse INI with multiple sections."""
    content = """
[section1]
key1 = value1

[section2]
key2 = value2
"""
    result = config_ingest.parse_ini(content)
    assert result is not None
    assert ("section1.key1", "value1") in result
    assert ("section2.key2", "value2") in result


def test_parse_config_file_by_extension() -> None:
    """Parse config file based on extension."""
    toml_result = config_ingest.parse_config_file(Path("config.toml"), 'key = "value"')
    assert toml_result is not None

    yaml_result = config_ingest.parse_config_file(Path("config.yaml"), "key: value")
    assert yaml_result is not None

    yml_result = config_ingest.parse_config_file(Path("config.yml"), "key: value")
    assert yml_result is not None

    json_result = config_ingest.parse_config_file(Path("config.json"), '{"key": "value"}')
    assert json_result is not None

    ini_result = config_ingest.parse_config_file(Path("config.ini"), "[section]\nkey=value")
    assert ini_result is not None

    cfg_result = config_ingest.parse_config_file(Path("setup.cfg"), "[section]\nkey=value")
    assert cfg_result is not None


def test_parse_config_file_unknown_extension() -> None:
    """Unknown extension returns None."""
    result = config_ingest.parse_config_file(Path("config.xyz"), "content")
    assert result is None


def test_parse_config_file_case_insensitive_extension() -> None:
    """Extension detection should be case insensitive."""
    result = config_ingest.parse_config_file(Path("config.TOML"), 'key = "value"')
    assert result is not None


# =============================================================================
# Typing/Annotation Info Tests
# =============================================================================


def test_compute_annotation_info_fully_typed() -> None:
    """Compute annotation info for fully typed function."""
    source = """
def foo(x: int, y: str) -> bool:
    return True
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    assert info.params_ratio == 1.0
    assert info.returns_ratio == 1.0
    assert info.untyped_defs == 0


def test_compute_annotation_info_partially_typed() -> None:
    """Compute annotation info for partially typed function."""
    source = """
def foo(x: int, y):
    return True
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    assert info.params_ratio == EXPECTED_PARAMS_RATIO_HALF  # 1 of 2 params annotated
    assert info.returns_ratio == 0.0  # No return annotation
    assert info.untyped_defs == 1


def test_compute_annotation_info_untyped() -> None:
    """Compute annotation info for untyped function."""
    source = """
def foo(x, y):
    return x + y
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    assert info.params_ratio == 0.0
    assert info.returns_ratio == 0.0
    assert info.untyped_defs == 1


def test_compute_annotation_info_excludes_self_cls() -> None:
    """Compute annotation info should exclude self and cls."""
    source = """
class Foo:
    def method(self, x: int) -> None:
        pass

    @classmethod
    def clsmethod(cls, y: str) -> None:
        pass
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    # self and cls should be excluded, so both params should be typed
    assert info.params_ratio == 1.0
    assert info.returns_ratio == 1.0
    assert info.untyped_defs == 0


def test_compute_annotation_info_async_function() -> None:
    """Compute annotation info for async function."""
    source = """
async def async_foo(x: int) -> str:
    return str(x)
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    assert info.params_ratio == 1.0
    assert info.returns_ratio == 1.0


def test_compute_annotation_info_handles_empty_module() -> None:
    """Compute annotation info handles an empty module AST gracefully.

    Note: Since the public API takes AST and syntax errors are caught
    internally during ast.parse(), we test that the function handles
    an empty/trivial AST without error.
    """
    # Create a minimal empty module AST
    empty_module = ast.Module(body=[], type_ignores=[])
    info = typing_ingest.compute_annotation_info(empty_module)

    # Empty module should return info with default ratios
    assert info is not None
    assert info.params_ratio == 1.0  # No params means 100% annotated
    assert info.returns_ratio == 1.0  # No functions means 100% return typed
    assert info.untyped_defs == 0


def test_compute_annotation_info_no_functions() -> None:
    """Compute annotation info for code with no functions."""
    source = """
x = 1
y = 2
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    # No params, no functions, ratios should be 1.0 (no violations)
    assert info.params_ratio == 1.0
    assert info.returns_ratio == 1.0
    assert info.untyped_defs == 0


def test_compute_annotation_info_multiple_functions() -> None:
    """Compute annotation info for multiple functions."""
    source = """
def typed_func(x: int) -> str:
    return str(x)

def untyped_func(x):
    return x
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    # 1 typed param, 1 untyped param
    assert info.params_ratio == EXPECTED_PARAMS_RATIO_HALF
    # 1 typed return, 1 untyped
    assert info.returns_ratio == EXPECTED_PARAMS_RATIO_HALF
    assert info.untyped_defs == 1


def test_compute_annotation_info_with_decorators() -> None:
    """Compute annotation info for decorated functions."""
    source = """
@decorator
def decorated(x: int) -> str:
    return str(x)
"""
    tree = ast.parse(source)
    info = typing_ingest.compute_annotation_info(tree)

    assert info is not None
    assert info.params_ratio == 1.0
    assert info.returns_ratio == 1.0


def test_collect_function_params_posonlyargs() -> None:
    """Collect function params should include posonlyargs."""
    source = """
def foo(a, /, b, *, c):
    pass
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    params = typing_ingest.collect_function_params(func)

    # Should include a (posonly), b (regular), c (kwonly)
    param_names = [p.arg for p in params]
    assert "a" in param_names
    assert "b" in param_names
    assert "c" in param_names


def test_collect_function_params_varargs() -> None:
    """Collect function params should handle *args and **kwargs."""
    source = """
def foo(a, *args, b, **kwargs):
    pass
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    params = typing_ingest.collect_function_params(func)

    # Should include regular args and kwonlyargs
    param_names = [p.arg for p in params]
    assert "a" in param_names
    assert "b" in param_names


def test_is_fully_typed_true() -> None:
    """is_fully_typed should return True for fully typed function."""
    source = """
def foo(x: int, y: str) -> bool:
    return True
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    result = typing_ingest.is_fully_typed(func)

    assert result is True


def test_is_fully_typed_false_missing_annotation() -> None:
    """is_fully_typed should return False when param annotation missing."""
    source = """
def foo(x: int, y) -> bool:
    return True
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    result = typing_ingest.is_fully_typed(func)

    assert result is False


def test_is_fully_typed_false_no_return() -> None:
    """is_fully_typed should return False when return annotation missing."""
    source = """
def foo(x: int):
    return x
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    result = typing_ingest.is_fully_typed(func)

    assert result is False


def test_is_fully_typed_ignores_self_cls() -> None:
    """is_fully_typed should ignore self and cls params."""
    source = """
class Foo:
    def method(self, x: int) -> bool:
        return True
"""
    tree = ast.parse(source)
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef)
    func = class_def.body[0]
    assert isinstance(func, ast.FunctionDef)

    result = typing_ingest.is_fully_typed(func)

    assert result is True


def test_is_fully_typed_empty_params() -> None:
    """is_fully_typed should return True for function with no params."""
    source = """
def foo() -> bool:
    return True
"""
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    result = typing_ingest.is_fully_typed(func)

    assert result is True


def test_annotation_info_dataclass() -> None:
    """AnnotationInfo should be a valid dataclass."""
    info = AnnotationInfo(
        params_ratio=EXPECTED_PARAMS_RATIO_HALF,
        returns_ratio=EXPECTED_RETURNS_RATIO_THREE_QUARTERS,
        untyped_defs=EXPECTED_UNTYPED_DEFS_TWO,
    )

    assert info.params_ratio == EXPECTED_PARAMS_RATIO_HALF
    assert info.returns_ratio == EXPECTED_RETURNS_RATIO_THREE_QUARTERS
    assert info.untyped_defs == EXPECTED_UNTYPED_DEFS_TWO


# =============================================================================
# TestsIngestStep Tests
# =============================================================================


# MockStoragePort replaced with FakeIngestStorage from tests._helpers.fakes


def test_tests_ingest_step_success(tmp_path: Path) -> None:
    """TestsIngestStep should parse and persist test results."""
    report_path = tmp_path / "pytest_report.json"
    report_data = {
        "tests": [
            {"nodeid": "tests/test_mod.py::test_a", "outcome": "passed", "duration": 0.1},
            {
                "nodeid": "tests/test_mod.py::test_b",
                "outcome": "failed",
                "duration": 0.2,
                "longrepr": "AssertionError",
            },
        ],
        "summary": {
            "passed": 1,
            "failed": 1,
            "skipped": 0,
            "duration": 0.3,
        },
    }
    report_path.write_text(json.dumps(report_data))

    storage = FakeIngestStorage()
    step = TestsIngestStep(storage)
    result = step.execute([], repo="test/repo", commit="abc123", json_report_path=report_path)

    expected_rows = 3  # 2 test rows + 1 summary row
    assert result.rows_written == expected_rows
    # FakeIngestStorage stores data by table_key, check operations count
    assert len(storage.data) == EXPECTED_UNTYPED_DEFS_TWO


def test_tests_ingest_step_missing_report(tmp_path: Path) -> None:
    """TestsIngestStep should skip when report is missing."""
    report_path = tmp_path / "nonexistent.json"

    storage = FakeIngestStorage()
    step = TestsIngestStep(storage)
    result = step.execute([], repo="test/repo", commit="abc123", json_report_path=report_path)

    assert result.skipped is True
    assert result.skip_reason is not None
    assert "not found" in result.skip_reason.lower()
    assert result.rows_written == 0


def test_tests_ingest_step_invalid_json(tmp_path: Path) -> None:
    """TestsIngestStep should fail on invalid JSON."""
    report_path = tmp_path / "invalid.json"
    report_path.write_text("not valid json {{{")

    storage = FakeIngestStorage()
    step = TestsIngestStep(storage)
    result = step.execute([], repo="test/repo", commit="abc123", json_report_path=report_path)

    # On invalid JSON, errors list will have the failure message
    assert len(result.errors) >= 1
    assert result.rows_written == 0


def test_tests_ingest_step_empty_tests(tmp_path: Path) -> None:
    """TestsIngestStep should handle empty test list."""
    report_path = tmp_path / "empty.json"
    report_data = {"tests": [], "summary": {}}
    report_path.write_text(json.dumps(report_data))

    storage = FakeIngestStorage()
    step = TestsIngestStep(storage)
    result = step.execute([], repo="test/repo", commit="abc123", json_report_path=report_path)

    # Only summary row should be written
    assert result.rows_written == 1


def test_tests_ingest_step_long_longrepr_truncated(tmp_path: Path) -> None:
    """TestsIngestStep should truncate long longrepr to 1000 chars."""
    report_path = tmp_path / "long.json"
    long_repr = "x" * 2000  # Longer than 1000
    report_data = {
        "tests": [{"nodeid": "t::a", "outcome": "failed", "longrepr": long_repr}],
        "summary": {},
    }
    report_path.write_text(json.dumps(report_data))

    storage = FakeIngestStorage()
    step = TestsIngestStep(storage)
    step.execute([], repo="test/repo", commit="abc123", json_report_path=report_path)

    # Get the test result rows from FakeIngestStorage
    test_rows = storage.data.get("core.test_results", [])
    assert len(test_rows) == 1
    first_row = test_rows[0]
    # The longrepr is at index 6 - cast to str to check length
    longrepr = str(first_row[6]) if first_row[6] is not None else ""
    expected_len = 1000
    assert len(longrepr) == expected_len
