"""Typed row factory helpers for analytics/storage contract tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.config.datasets.rows.profiles import (
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
)
from codeintel.config.datasets.rows.test import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
)

if TYPE_CHECKING:
    from codeintel.config.datasets.rows.profiles import (
        FileProfileRowModel,
        FunctionProfileRowModel,
        ModuleProfileRowModel,
    )
    from codeintel.config.datasets.rows.test import (
        BehavioralCoverageRowModel,
        ProfileRowModel,
    )


def blank_file_profile_row() -> FileProfileRowModel:
    """
    Return an empty FileProfileRowModel with all keys initialized.

    Returns
    -------
    FileProfileRowModel
        Empty file profile row with every column set to None.
    """
    return cast("FileProfileRowModel", dict.fromkeys(FILE_PROFILE_COLUMNS))


def blank_module_profile_row() -> ModuleProfileRowModel:
    """
    Return an empty ModuleProfileRowModel with all keys initialized.

    Returns
    -------
    ModuleProfileRowModel
        Empty module profile row with every column set to None.
    """
    return cast("ModuleProfileRowModel", dict.fromkeys(MODULE_PROFILE_COLUMNS))


def blank_test_profile_row() -> ProfileRowModel:
    """
    Return an empty ProfileRowModel with all keys initialized.

    Returns
    -------
    ProfileRowModel
        Empty test profile row with every column set to None.
    """
    return cast("ProfileRowModel", dict.fromkeys(TEST_PROFILE_COLUMNS))


def blank_behavioral_coverage_row() -> BehavioralCoverageRowModel:
    """
    Return an empty BehavioralCoverageRowModel with all keys initialized.

    Returns
    -------
    BehavioralCoverageRowModel
        Empty behavioral coverage row with every column set to None.
    """
    return cast("BehavioralCoverageRowModel", dict.fromkeys(BEHAVIORAL_COVERAGE_COLUMNS))


def blank_function_profile_row() -> FunctionProfileRowModel:
    """
    Return an empty FunctionProfileRowModel with all keys initialized.

    Returns
    -------
    FunctionProfileRowModel
        Empty function profile row with every column set to None.
    """
    return cast("FunctionProfileRowModel", dict.fromkeys(FUNCTION_PROFILE_COLUMNS))


def sample_function_profile_rows(repo: str, commit: str) -> list[FunctionProfileRowModel]:
    """Build realistic function profile rows with varied content.

    Returns
    -------
    list[FunctionProfileRowModel]
        Rows containing unicode identifiers, optional fields, and multiple entries.
    """
    return [
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 101,
                "urn": "urn:fn:alpha::helper",
                "rel_path": "pkg/alpha.py",
                "language": "python",
                "kind": "function",
                "qualname": "pkg.alpha.helper",
                "tags": '["io","auth"]',
                "owners": '["team-data"]',
                "created_at": None,
            },
        ),
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 202,
                "urn": "urn:fn:beta::process",
                "rel_path": "pkg/beta.py",
                "language": "python",
                "kind": "method",
                "qualname": "pkg.beta.B.process",
                "tags": "[]",
                "owners": None,
                "created_at": None,
            },
        ),
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 303,
                "urn": "urn:fn:unicode::δelta",
                "rel_path": "pkg/unicode/δ.py",
                "language": "python",
                "kind": "function",
                "qualname": "pkg.unicode.δ.fn",
                "tags": '["unicode","core"]',
                "owners": '["team-δ"]',
                "created_at": None,
            },
        ),
    ]


def sample_file_profile_rows(repo: str, commit: str) -> list[FileProfileRowModel]:
    """Build realistic file profile rows with unicode and nulls.

    Returns
    -------
    list[FileProfileRowModel]
        File profile rows with varied tags, owners, and optional fields.
    """
    return [
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/alpha.py",
                "module": "pkg.alpha_mod",
                "tags": '["core","io"]',
                "owners": '["team-analytics"]',
                "created_at": None,
            },
        ),
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/beta.py",
                "module": "pkg.beta",
                "tags": "[]",
                "owners": None,
                "created_at": None,
            },
        ),
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/unicode/δ.py",
                "module": "pkg.unicode.delta",
                "tags": '["unicode"]',
                "owners": None,
                "created_at": None,
            },
        ),
    ]


def sample_module_profile_rows(repo: str, commit: str) -> list[ModuleProfileRowModel]:
    """Build realistic module profile rows with edge-case paths.

    Returns
    -------
    list[ModuleProfileRowModel]
        Module rows covering tagged and untagged modules.
    """
    return [
        cast(
            "ModuleProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "module": "pkg.alpha_mod",
                "path": "pkg/alpha.py",
                "tags": '["core"]',
                "owners": '["team-alpha"]',
                "created_at": None,
            },
        ),
        cast(
            "ModuleProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "module": "pkg.beta",
                "path": "pkg/beta.py",
                "tags": "[]",
                "owners": None,
                "created_at": None,
            },
        ),
        cast(
            "ModuleProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "module": "pkg.unicode.delta",
                "path": "pkg/unicode/δ.py",
                "tags": '["unicode"]',
                "owners": '["team-unicode"]',
                "created_at": None,
            },
        ),
    ]


def sample_test_profile_rows(repo: str, commit: str) -> list[ProfileRowModel]:
    """Build realistic test profile rows including unicode markers.

    Returns
    -------
    list[ProfileRowModel]
        Test profile rows with markers, coverage, and subsystem details.
    """
    return [
        cast(
            "ProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_alpha.py::TestAlpha::test_io",
                "rel_path": "tests/test_alpha.py",
                "markers": ["slow", "unicode"],
                "functions_covered": [101, 202],
                "primary_function_goids": [101],
                "subsystems_covered": ["analytics", "io"],
                "created_at": None,
            },
        ),
        cast(
            "ProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_beta.py::test_beta_fast",
                "rel_path": "tests/test_beta.py",
                "markers": [],
                "functions_covered": [],
                "primary_function_goids": [],
                "subsystems_covered": [],
                "created_at": None,
            },
        ),
        cast(
            "ProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/unicode/test_delta.py::test_δelta_flow",
                "rel_path": "tests/unicode/test_delta.py",
                "markers": ["unicode"],
                "functions_covered": [303],
                "primary_function_goids": [303],
                "subsystems_covered": ["intl"],
                "created_at": None,
            },
        ),
    ]


def sample_behavioral_coverage_rows(repo: str, commit: str) -> list[BehavioralCoverageRowModel]:
    """Build realistic behavioral coverage rows with multiple tags.

    Returns
    -------
    list[BehavioralCoverageRowModel]
        Behavioral coverage rows capturing tagged and untagged scenarios.
    """
    return [
        cast(
            "BehavioralCoverageRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_alpha.py::TestAlpha::test_io",
                "rel_path": "tests/test_alpha.py",
                "qualname": "TestAlpha::test_io",
                "behavior_tags": ["io", "auth"],
                "tag_source": "heuristic",
                "created_at": None,
            },
        ),
        cast(
            "BehavioralCoverageRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_beta.py::test_beta_fast",
                "rel_path": "tests/test_beta.py",
                "qualname": "test_beta_fast",
                "behavior_tags": [],
                "tag_source": "manual",
                "created_at": None,
            },
        ),
        cast(
            "BehavioralCoverageRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/unicode/test_delta.py::test_δelta_flow",
                "rel_path": "tests/unicode/test_delta.py",
                "qualname": "test_δelta_flow",
                "behavior_tags": ["unicode", "edge"],
                "tag_source": "heuristic",
                "created_at": None,
            },
        ),
    ]


def sample_pytest_tests() -> list[dict[str, object]]:
    """
    Return multi-row pytest tests payload with unicode and nullable longrepr.

    Returns
    -------
    list[dict[str, object]]
        Test rows containing varied outcomes and longrepr coverage.
    """
    return [
        cast(
            "dict[str, object]",
            {"nodeid": "tests/pkg/mod.py::test_ok", "outcome": "passed", "duration": 0.1},
        ),
        cast(
            "dict[str, object]",
            {
                "nodeid": "tests/pkg/test_unicode.py::test_fail_umbrella_umbrella",
                "outcome": "failed",
                "duration": 0.25,
                "longrepr": "x" * 1500,
            },
        ),
        cast(
            "dict[str, object]",
            {
                "nodeid": "tests/pkg/test_gamma.py::test_skipped",
                "outcome": "skipped",
                "duration": 0.05,
                "longrepr": None,
            },
        ),
        cast(
            "dict[str, object]",
            {
                "nodeid": "tests/pkg/test_error.py::test_raises",
                "outcome": "error",
                "duration": 0.02,
                "longrepr": None,
            },
        ),
    ]


def sample_pytest_summary() -> dict[str, object]:
    """
    Return a pytest summary payload with counts and duration.

    Returns
    -------
    dict[str, object]
        Summary including counts and total duration.
    """
    return {"passed": 1, "failed": 1, "skipped": 1, "error": 1, "duration": 0.42}


def sample_coverage_payload() -> dict[str, object]:
    """
    Return coverage-like JSON payload keyed by relative path.

    Returns
    -------
    dict[str, object]
        Mapping of file path to covered/missing line lists.
    """
    return {
        "pkg/mod.py": {"covered_lines": [1, 2, 4], "missing_lines": [3]},
        "pkg/naive.py": {"covered_lines": [1], "missing_lines": [2, 3, 5]},
        "pkg/unicode/δ.py": {"covered_lines": [1, 3], "missing_lines": [2]},
    }


def sample_scip_documents() -> list[dict[str, object]]:
    """
    Build SCIP documents with symbols and occurrences for multiple files.

    Returns
    -------
    list[dict[str, object]]
        SCIP document payloads with symbols and occurrences.
    """
    return [
        {
            "relativePath": "pkg/a.py",
            "symbols": [
                {"symbol": "pkg/a.py:func", "documentation": ["doc"]},
                {"symbol": "pkg/a.py:helper", "documentation": ["helper doc"]},
            ],
            "occurrences": [
                {"symbol": "pkg/a.py:func", "range": [1, 0, 1, 4], "symbolRoles": 1},
                {"symbol": "pkg/a.py:helper", "range": [2, 0, 2, 6], "symbolRoles": 1},
            ],
        },
        {
            "relativePath": "pkg/naive.py",
            "symbols": [{"symbol": "pkg/naive.py:helper", "documentation": ["naive helper"]}],
            "occurrences": [
                {"symbol": "pkg/naive.py:helper", "range": [2, 0, 2, 6], "symbolRoles": 1}
            ],
        },
        {
            "relativePath": "pkg/unicode/δ.py",
            "symbols": [{"symbol": "pkg/unicode/δ.py:δelta", "documentation": ["Δ doc"]}],
            "occurrences": [
                {"symbol": "pkg/unicode/δ.py:δelta", "range": [1, 0, 1, 6], "symbolRoles": 1},
                {"symbol": "pkg/unicode/δ.py:δelta", "range": [2, 0, 2, 6], "symbolRoles": 0},
            ],
        },
    ]
