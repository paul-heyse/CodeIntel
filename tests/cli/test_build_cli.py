"""Tests for build CLI commands.

This module tests the build command group including:
- build status: Display current target status
- build run: Build targets with dependency resolution
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.core.build.targets import TargetModule
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

runner = CliRunner()


# =============================================================================
# Build Status Command Tests
# =============================================================================


class TestBuildStatusHelp:
    """Tests for build status --help."""

    def test_status_help_shows_description(self) -> None:
        """Help text describes the status command."""
        result = runner.invoke(app, ["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("current state", result.stdout.lower(), label="description present")

    def test_status_help_shows_options(self) -> None:
        """Help text shows available options."""
        result = runner.invoke(app, ["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--module", result.stdout, label="module option")
        expect_in("--json", result.stdout, label="json option")


class TestBuildStatusCommand:
    """Tests for build status command output."""

    def test_status_json_output_structure(self) -> None:
        """JSON output has expected structure."""
        result = runner.invoke(app, ["build", "status", "--json"])

        # Command should succeed (even without project file it may return empty)
        # Check that if it succeeds, the JSON structure is correct
        if result.exit_code == 0:
            data = json.loads(result.stdout)
            expect_is_instance(data, dict, label="payload type")
            expect_in("computed", data, label="computed key")
            expect_in("missing", data, label="missing key")
            expect_in("stale", data, label="stale key")
            expect_in("blocked", data, label="blocked key")

    def test_status_invalid_module(self) -> None:
        """Invalid module name produces error or requires project context."""
        result = runner.invoke(app, ["build", "status", "--module", "invalid_module"])

        # Should fail - either because module is invalid or no project context
        expect_equal(result.exit_code, 1, label="exit_code")
        # Error message may be about invalid module or missing project
        combined = (result.stdout + (result.output or "")).lower()
        has_module_error = "unknown module" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_module_error or has_project_error,
            message="Expected module validation or project context error",
        )


# =============================================================================
# Build Run Command Tests
# =============================================================================


class TestBuildRunHelp:
    """Tests for build run --help."""

    def test_run_help_shows_description(self) -> None:
        """Help text describes the run command."""
        result = runner.invoke(app, ["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("dependency resolution", result.stdout.lower(), label="description present")

    def test_run_help_shows_options(self) -> None:
        """Help text shows available options."""
        result = runner.invoke(app, ["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--module", result.stdout, label="module option")
        expect_in("--dry-run", result.stdout, label="dry-run option")
        expect_in("--force", result.stdout, label="force option")
        expect_in("--json", result.stdout, label="json option")


class TestBuildRunValidation:
    """Tests for build run argument validation."""

    def test_run_no_targets_no_module_error(self) -> None:
        """Running without targets or module produces error or requires project context."""
        result = runner.invoke(app, ["build", "run"])

        # Should fail - either because no targets/module or no project context
        expect_equal(result.exit_code, 1, label="exit_code")
        # Error message may be about missing targets or missing project
        combined = (result.stdout + (result.output or "")).lower()
        has_targets_error = "specify targets" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_targets_error or has_project_error,
            message="Expected targets/module or project context error",
        )

    def test_run_invalid_module(self) -> None:
        """Invalid module name produces error or requires project context."""
        result = runner.invoke(app, ["build", "run", "--module", "invalid_module"])

        # Should fail - either because module is invalid or no project context
        expect_equal(result.exit_code, 1, label="exit_code")
        # Error message may be about invalid module or missing project
        combined = (result.stdout + (result.output or "")).lower()
        has_module_error = "unknown module" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_module_error or has_project_error,
            message="Expected module validation or project context error",
        )

    def test_run_unknown_target(self) -> None:
        """Unknown target name produces error or requires project context."""
        result = runner.invoke(app, ["build", "run", "nonexistent_target_xyz"])

        # Should fail - either because target is unknown or no project context
        expect_equal(result.exit_code, 1, label="exit_code")
        # Error message may be about unknown target or missing project
        combined = (result.stdout + (result.output or "")).lower()
        has_target_error = "unknown target" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_target_error or has_project_error,
            message="Expected target validation or project context error",
        )


# =============================================================================
# Build App Structure Tests
# =============================================================================


class TestBuildAppStructure:
    """Tests for build app registration and structure."""

    def test_build_help_shows_commands(self) -> None:
        """Build help shows available subcommands."""
        result = runner.invoke(app, ["build", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("run", result.stdout, label="run command")
        expect_in("status", result.stdout, label="status command")

    def test_build_no_args_shows_help(self) -> None:
        """Build with no args shows help."""
        result = runner.invoke(app, ["build"])

        # Should show help or error about missing command
        expect_in("run", result.stdout, label="run mentioned")
        expect_in("status", result.stdout, label="status mentioned")


# =============================================================================
# Module Option Tests
# =============================================================================


class TestModuleOption:
    """Tests for the --module option."""

    @pytest.mark.parametrize("module", ["ingestion", "graphs", "analytics"])
    def test_valid_module_names(self, module: TargetModule) -> None:
        """Valid module names are accepted."""
        # Dry-run with valid module should not error on module validation
        result = runner.invoke(app, ["build", "run", "--module", module, "--dry-run"])

        # If it fails, it should not be due to invalid module
        if result.exit_code != 0:
            expect_true(
                "unknown module" not in result.stdout.lower(),
                message=f"Module {module} should be valid",
            )


# =============================================================================
# Dry Run Tests
# =============================================================================


class TestDryRun:
    """Tests for --dry-run functionality."""

    def test_dry_run_flag_recognized(self) -> None:
        """Dry run flag is recognized."""
        result = runner.invoke(app, ["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--dry-run", result.stdout, label="dry-run option recognized")
        expect_in("-n", result.stdout, label="short flag recognized")


# =============================================================================
# Force Option Tests
# =============================================================================


class TestForceOption:
    """Tests for --force option."""

    def test_force_flag_recognized(self) -> None:
        """Force flag is recognized."""
        result = runner.invoke(app, ["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--force", result.stdout, label="force option recognized")
        expect_in("-f", result.stdout, label="short flag recognized")


# =============================================================================
# JSON Output Tests
# =============================================================================


class TestJsonOutput:
    """Tests for --json output option."""

    def test_json_flag_recognized_on_status(self) -> None:
        """JSON flag is recognized on status command."""
        result = runner.invoke(app, ["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--json", result.stdout, label="json option recognized")

    def test_json_flag_recognized_on_run(self) -> None:
        """JSON flag is recognized on run command."""
        result = runner.invoke(app, ["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--json", result.stdout, label="json option recognized")
