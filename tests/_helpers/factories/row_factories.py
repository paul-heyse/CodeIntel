"""Typed row factory helpers for analytics/storage contract tests."""

from __future__ import annotations

from typing import cast

from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    FileProfileRowModel,
    FunctionProfileRowModel,
    ModuleProfileRowModel,
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
                "urn": "urn:fn:β::process",
                "rel_path": "pkg/beta.py",
                "language": "python",
                "kind": "method",
                "qualname": "pkg.beta.B.process",
                "tags": "[]",
                "owners": None,
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
    ]
