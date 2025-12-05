"""Tests for codeintel.analytics.compute.dependencies.detection module.

Testing Charter Compliance:
- Pure function tests with realistic AST sources
- No monkeypatching or test-only code paths
- Tests actual production code paths for dependency detection
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
    DependencyModePattern,
    LibraryPattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)
from codeintel.analytics.compute.dependencies.detection import (
    DependencyCall,
    DependencyCallVisitor,
    build_alias_map,
    build_alias_maps,
    group_calls_by_library,
)
from tests._helpers import assert_frozen

# =============================================================================
# Test Constants
# =============================================================================

EXPECTED_SEVERITY_COUNT = 5
EXPECTED_REQUESTS_CALLS = 2
EXPECTED_GROUPED_LIBRARIES = 2
DEFAULT_CRITICALITY = 2.0
HIGH_CRITICALITY = 3.0
HIGH_RISK_SCORE = 9.0
LINE_NUMBER_START = 10
LINE_NUMBER_END = 12

# =============================================================================
# Test Data
# =============================================================================

REQUESTS_PATTERN = LibraryPattern(
    library="requests",
    service_name="HTTP Client",
    category="http",
    matchers=[
        DependencyModePattern(modes=["read"], method="get"),
        DependencyModePattern(modes=["write"], method="post"),
        DependencyModePattern(modes=["write"], method="put"),
        DependencyModePattern(modes=["delete"], method="delete"),
    ],
    severity="medium",
    criticality=2.0,
)

SQLALCHEMY_PATTERN = LibraryPattern(
    library="sqlalchemy",
    service_name="Database ORM",
    category="database",
    matchers=[
        DependencyModePattern(modes=["query"], method="execute"),
        DependencyModePattern(modes=["write"], method_prefix="insert"),
        DependencyModePattern(modes=["write"], method_prefix="update"),
        DependencyModePattern(modes=["delete"], method_prefix="delete"),
    ],
    severity="high",
    criticality=3.0,
)


# =============================================================================
# Test Classification Module
# =============================================================================


class TestSeverityScore:
    """Tests for severity_score function."""

    @staticmethod
    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            ("critical", 4.0),
            ("high", 3.0),
            ("medium", 2.0),
            ("low", 1.0),
            ("info", 0.5),
        ],
    )
    def test_known_severities(severity: str, expected: float) -> None:
        """Verify known severities return correct scores."""
        assert severity_score(severity) == expected

    @staticmethod
    def test_returns_none_for_none() -> None:
        """Verify None input returns None."""
        assert severity_score(None) is None

    @staticmethod
    def test_returns_none_for_unknown() -> None:
        """Verify unknown severity returns None."""
        assert severity_score("unknown") is None
        assert severity_score("CRITICAL") is None  # Case-sensitive

    @staticmethod
    def test_severity_scores_constant() -> None:
        """Verify SEVERITY_SCORES constant is accessible."""
        assert len(SEVERITY_SCORES) == EXPECTED_SEVERITY_COUNT
        assert "critical" in SEVERITY_SCORES


class TestRiskScore:
    """Tests for risk_score function."""

    @staticmethod
    @pytest.mark.parametrize(
        ("severity", "criticality", "expected"),
        [
            ("high", 3.0, 9.0),
            ("medium", 2.0, 4.0),
            ("low", 1.0, 1.0),
            ("critical", 5.0, 20.0),
        ],
    )
    def test_computes_product(severity: str, criticality: float, expected: float) -> None:
        """Verify risk score is severity_score * criticality."""
        assert risk_score(severity, criticality) == expected

    @staticmethod
    def test_returns_none_for_none_severity() -> None:
        """Verify None severity returns None."""
        assert risk_score(None, 3.0) is None

    @staticmethod
    def test_returns_none_for_none_criticality() -> None:
        """Verify None criticality returns None."""
        assert risk_score("high", None) is None

    @staticmethod
    def test_returns_none_for_unknown_severity() -> None:
        """Verify unknown severity returns None."""
        assert risk_score("unknown", 3.0) is None


class TestRiskLevel:
    """Tests for risk_level function."""

    @staticmethod
    @pytest.mark.parametrize(
        ("modes", "count", "expected"),
        [
            ({"write"}, 1, "high"),
            ({"admin"}, 1, "high"),
            ({"write", "query"}, 5, "high"),
            ({"read"}, CALLSITE_MEDIUM_THRESHOLD, "medium"),
            ({"read"}, CALLSITE_MEDIUM_THRESHOLD + 5, "medium"),
            ({"read"}, 1, "low"),
            ({"query"}, 5, "low"),
        ],
    )
    def test_risk_levels(modes: set[str], count: int, expected: str) -> None:
        """Verify risk level determination."""
        assert risk_level(modes, count) == expected

    @staticmethod
    def test_write_takes_precedence() -> None:
        """Verify write mode returns high even with many callsites."""
        assert risk_level({"write"}, 100) == "high"

    @staticmethod
    def test_empty_modes_low() -> None:
        """Verify empty modes with low count returns low."""
        assert risk_level(set(), 1) == "low"


class TestDependencyModePattern:
    """Tests for DependencyModePattern matching."""

    @staticmethod
    def test_matches_exact_method() -> None:
        """Verify exact method matching."""
        pattern = DependencyModePattern(modes=["read"], method="get")
        assert pattern.matches("get", "requests.get(url)")
        assert not pattern.matches("post", "requests.post(url)")

    @staticmethod
    def test_matches_method_prefix() -> None:
        """Verify method prefix matching."""
        pattern = DependencyModePattern(modes=["write"], method_prefix="insert")
        assert pattern.matches("insert_one", "db.insert_one(doc)")
        assert pattern.matches("insert_many", "db.insert_many(docs)")
        assert not pattern.matches("update_one", "db.update_one(doc)")

    @staticmethod
    def test_matches_target_substring() -> None:
        """Verify target substring matching."""
        pattern = DependencyModePattern(modes=["query"], match="SELECT")
        assert pattern.matches("execute", "conn.execute('SELECT * FROM users')")
        assert not pattern.matches("execute", "conn.execute('INSERT INTO users')")

    @staticmethod
    def test_no_match_with_none_method() -> None:
        """Verify None method doesn't cause errors."""
        pattern = DependencyModePattern(modes=["read"], method="get")
        assert not pattern.matches(None, "some_target")

    @staticmethod
    def test_multiple_match_conditions() -> None:
        """Verify pattern with multiple conditions."""
        pattern = DependencyModePattern(
            modes=["admin"],
            method="admin_execute",
            match="DROP TABLE",
        )
        # Method match
        assert pattern.matches("admin_execute", "db.admin_execute()")
        # Target match
        assert pattern.matches("execute", "db.execute('DROP TABLE users')")


class TestClassifyModes:
    """Tests for classify_modes function."""

    @staticmethod
    def test_matches_single_mode() -> None:
        """Verify single mode classification."""
        modes, matched = classify_modes(REQUESTS_PATTERN, "get", "requests.get(url)")
        assert modes == ["read"]
        assert matched is not None
        assert matched.method == "get"

    @staticmethod
    def test_returns_unknown_for_no_match() -> None:
        """Verify unknown mode when no matchers match."""
        modes, matched = classify_modes(REQUESTS_PATTERN, "head", "requests.head(url)")
        assert modes == ["unknown"]
        assert matched is None

    @staticmethod
    def test_deduplicates_modes() -> None:
        """Verify duplicate modes are removed."""
        pattern = LibraryPattern(
            library="test",
            service_name=None,
            category=None,
            matchers=[
                DependencyModePattern(modes=["read"], method="get"),
                DependencyModePattern(modes=["read"], match="GET"),
            ],
        )
        modes, _ = classify_modes(pattern, "get", "api.get('GET /users')")
        assert modes == ["read"]  # Deduplicated

    @staticmethod
    def test_multiple_modes_sorted() -> None:
        """Verify multiple modes are sorted."""
        pattern = LibraryPattern(
            library="test",
            service_name=None,
            category=None,
            matchers=[
                DependencyModePattern(modes=["write", "admin"], method="execute"),
            ],
        )
        modes, _ = classify_modes(pattern, "execute", "db.execute()")
        assert modes == ["admin", "write"]  # Sorted


# =============================================================================
# Test Detection Module
# =============================================================================


class TestBuildAliasMap:
    """Tests for build_alias_map function."""

    @staticmethod
    def test_simple_import() -> None:
        """Verify simple import statement."""
        source = "import requests"
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        assert alias_map == {"requests": "requests"}

    @staticmethod
    def test_aliased_import() -> None:
        """Verify import with alias."""
        source = "import pandas as pd"
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        assert alias_map == {"pd": "pandas"}

    @staticmethod
    def test_from_import() -> None:
        """Verify from ... import statement."""
        source = "from sqlalchemy import create_engine"
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        assert alias_map == {"create_engine": "sqlalchemy"}

    @staticmethod
    def test_from_import_with_alias() -> None:
        """Verify from ... import with alias."""
        source = "from requests import Session as S"
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        assert alias_map == {"S": "requests"}

    @staticmethod
    def test_dotted_import() -> None:
        """Verify dotted import uses root module."""
        source = "import sqlalchemy.orm"
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        assert alias_map == {"sqlalchemy.orm": "sqlalchemy"}

    @staticmethod
    def test_multiple_imports() -> None:
        """Verify multiple imports in one source."""
        source = """
import requests
import pandas as pd
from sqlalchemy import create_engine
from os.path import join
"""
        tree = ast.parse(source)
        alias_map = build_alias_map(tree)
        expected = {
            "requests": "requests",
            "pd": "pandas",
            "create_engine": "sqlalchemy",
            "join": "os",
        }
        assert alias_map == expected

    @staticmethod
    def test_empty_source() -> None:
        """Verify empty source returns empty map."""
        tree = ast.parse("")
        alias_map = build_alias_map(tree)
        assert alias_map == {}


class TestDependencyCallVisitor:
    """Tests for DependencyCallVisitor."""

    @staticmethod
    def test_detects_simple_call() -> None:
        """Verify detection of simple dependency call."""
        source = """
import requests
requests.get("http://example.com")
"""
        tree = ast.parse(source)
        lines = source.splitlines()
        alias_map = {"requests": "requests"}
        patterns = {"requests": REQUESTS_PATTERN}

        visitor = DependencyCallVisitor(
            alias_map=alias_map,
            patterns=patterns,
            rel_path="test.py",
            lines=lines,
        )
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        call = visitor.calls[0]
        assert call.library == "requests"
        assert call.modes == ["read"]
        assert call.lineno is not None

    @staticmethod
    def test_detects_multiple_calls() -> None:
        """Verify detection of multiple dependency calls."""
        source = """
import requests
requests.get("http://example.com/users")
requests.post("http://example.com/users", data={})
"""
        tree = ast.parse(source)
        lines = source.splitlines()
        alias_map = {"requests": "requests"}
        patterns = {"requests": REQUESTS_PATTERN}

        visitor = DependencyCallVisitor(
            alias_map=alias_map,
            patterns=patterns,
            rel_path="test.py",
            lines=lines,
        )
        visitor.visit(tree)

        assert len(visitor.calls) == EXPECTED_REQUESTS_CALLS
        modes = [c.modes for c in visitor.calls]
        assert ["read"] in modes
        assert ["write"] in modes

    @staticmethod
    def test_ignores_non_dependency_calls() -> None:
        """Verify non-dependency calls are ignored."""
        source = """
import requests
print("Hello")
len([1, 2, 3])
requests.get("http://example.com")
"""
        tree = ast.parse(source)
        lines = source.splitlines()
        alias_map = {"requests": "requests"}
        patterns = {"requests": REQUESTS_PATTERN}

        visitor = DependencyCallVisitor(
            alias_map=alias_map,
            patterns=patterns,
            rel_path="test.py",
            lines=lines,
        )
        visitor.visit(tree)

        # Only the requests.get call should be captured
        assert len(visitor.calls) == 1
        assert visitor.calls[0].library == "requests"

    @staticmethod
    def test_ignores_unknown_libraries() -> None:
        """Verify calls to unknown libraries are ignored."""
        source = """
import unknown_lib
unknown_lib.do_something()
"""
        tree = ast.parse(source)
        lines = source.splitlines()
        alias_map = {"unknown_lib": "unknown_lib"}
        patterns = {"requests": REQUESTS_PATTERN}  # Only requests pattern

        visitor = DependencyCallVisitor(
            alias_map=alias_map,
            patterns=patterns,
            rel_path="test.py",
            lines=lines,
        )
        visitor.visit(tree)

        assert len(visitor.calls) == 0

    @staticmethod
    def test_captures_snippet() -> None:
        """Verify code snippet is captured."""
        source = """import requests
requests.get("http://example.com")
"""
        tree = ast.parse(source)
        lines = source.splitlines()
        alias_map = {"requests": "requests"}
        patterns = {"requests": REQUESTS_PATTERN}

        visitor = DependencyCallVisitor(
            alias_map=alias_map,
            patterns=patterns,
            rel_path="test.py",
            lines=lines,
        )
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        # Snippet should contain the call


class TestGroupCallsByLibrary:
    """Tests for group_calls_by_library function."""

    @staticmethod
    def test_groups_by_library() -> None:
        """Verify calls are grouped by library."""
        calls = [
            DependencyCall(
                library="requests",
                target="get",
                modes=["read"],
                severity=None,
                criticality=None,
            ),
            DependencyCall(
                library="requests",
                target="post",
                modes=["write"],
                severity=None,
                criticality=None,
            ),
            DependencyCall(
                library="pandas",
                target="read_csv",
                modes=["read"],
                severity=None,
                criticality=None,
            ),
        ]
        grouped = group_calls_by_library(calls)

        assert len(grouped) == EXPECTED_GROUPED_LIBRARIES
        assert len(grouped["requests"]) == EXPECTED_REQUESTS_CALLS
        assert len(grouped["pandas"]) == 1

    @staticmethod
    def test_empty_list() -> None:
        """Verify empty list returns empty dict."""
        grouped = group_calls_by_library([])
        assert grouped == {}

    @staticmethod
    def test_single_library() -> None:
        """Verify single library grouping."""
        calls = [
            DependencyCall(
                library="requests",
                target="get",
                modes=["read"],
                severity=None,
                criticality=None,
            ),
        ]
        grouped = group_calls_by_library(calls)
        assert len(grouped) == 1
        assert "requests" in grouped


class TestBuildAliasMaps:
    """Tests for build_alias_maps function."""

    @staticmethod
    def test_builds_maps_for_multiple_files(tmp_path: Path) -> None:
        """Verify alias maps are built for multiple files."""
        # Create test files
        file1 = tmp_path / "module1.py"
        file1.write_text("import requests\nimport pandas as pd\n")

        file2 = tmp_path / "module2.py"
        file2.write_text("from sqlalchemy import create_engine\n")

        module_map = {
            "module1.py": "module1",
            "module2.py": "module2",
        }

        alias_maps = build_alias_maps(tmp_path, module_map)

        assert "module1.py" in alias_maps
        assert "module2.py" in alias_maps
        assert alias_maps["module1.py"]["requests"] == "requests"
        assert alias_maps["module1.py"]["pd"] == "pandas"
        assert alias_maps["module2.py"]["create_engine"] == "sqlalchemy"

    @staticmethod
    def test_handles_missing_files(tmp_path: Path) -> None:
        """Verify missing files are skipped."""
        module_map = {
            "nonexistent.py": "nonexistent",
        }
        alias_maps = build_alias_maps(tmp_path, module_map)
        assert "nonexistent.py" not in alias_maps

    @staticmethod
    def test_handles_syntax_errors(tmp_path: Path) -> None:
        """Verify files with syntax errors are skipped."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n")  # Syntax error

        module_map = {"bad.py": "bad"}
        alias_maps = build_alias_maps(tmp_path, module_map)
        assert "bad.py" not in alias_maps

    @staticmethod
    def test_empty_module_map(tmp_path: Path) -> None:
        """Verify empty module map returns empty result."""
        alias_maps = build_alias_maps(tmp_path, {})
        assert alias_maps == {}


class TestDependencyCallDataclass:
    """Tests for DependencyCall dataclass."""

    @staticmethod
    def test_required_fields() -> None:
        """Verify required fields create valid instance."""
        call = DependencyCall(
            library="requests",
            target="get",
            modes=["read"],
            severity="medium",
            criticality=2.0,
        )
        assert call.library == "requests"
        assert call.target == "get"
        assert call.modes == ["read"]
        assert call.severity == "medium"
        assert call.criticality == DEFAULT_CRITICALITY

    @staticmethod
    def test_optional_fields_defaults() -> None:
        """Verify optional fields have correct defaults."""
        call = DependencyCall(
            library="test",
            target="func",
            modes=[],
            severity=None,
            criticality=None,
        )
        assert call.matched_pattern is None
        assert call.risk_score is None
        assert call.lineno is None
        assert call.end_lineno is None
        assert not call.snippet

    @staticmethod
    def test_all_fields() -> None:
        """Verify all fields can be set."""
        call = DependencyCall(
            library="sqlalchemy",
            target="execute",
            modes=["query"],
            severity="high",
            criticality=HIGH_CRITICALITY,
            matched_pattern="execute",
            risk_score=HIGH_RISK_SCORE,
            lineno=LINE_NUMBER_START,
            end_lineno=LINE_NUMBER_END,
            snippet="conn.execute('SELECT * FROM users')",
        )
        assert call.matched_pattern == "execute"
        assert call.risk_score == HIGH_RISK_SCORE
        assert call.lineno == LINE_NUMBER_START
        assert call.end_lineno == LINE_NUMBER_END
        assert "SELECT" in call.snippet

    @staticmethod
    def test_frozen_dataclass() -> None:
        """Verify DependencyCall is frozen/immutable."""
        call = DependencyCall(
            library="test",
            target="func",
            modes=[],
            severity=None,
            criticality=None,
        )
        # Should raise AttributeError (FrozenInstanceError is a subclass)
        assert_frozen(call, "library", "changed")
