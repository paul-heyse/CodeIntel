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

# Alias for backward compatibility
TestProfileRowModel = ProfileRowModel


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


def blank_test_profile_row() -> TestProfileRowModel:
    """
    Return an empty TestProfileRowModel with all keys initialized.

    Returns
    -------
    TestProfileRowModel
        Empty test profile row with every column set to None.
    """
    return cast("TestProfileRowModel", dict.fromkeys(TEST_PROFILE_COLUMNS))


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
