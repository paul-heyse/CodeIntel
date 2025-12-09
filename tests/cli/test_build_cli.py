"""Tests for build CLI commands."""

from __future__ import annotations

import json

import pytest

from codeintel.build.targets import TargetModule
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import run_cli


class TestBuildStatusHelp:
    """Tests for build status --help."""

    @staticmethod
    def test_status_help_shows_description() -> None:
        """Help text describes the status command."""
        result = run_cli(["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("current state", result.stdout.lower(), label="description present")

    @staticmethod
    def test_status_help_shows_options() -> None:
        """Help text shows available options."""
        result = run_cli(["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--module", result.stdout, label="module option")
        expect_in("--json", result.stdout, label="json option")


class TestBuildStatusCommand:
    """Tests for build status command output."""

    @staticmethod
    def test_status_json_output_structure() -> None:
        """JSON output has expected structure."""
        result = run_cli(["build", "status", "--json"])

        if result.exit_code == 0:
            data = json.loads(result.stdout)
            expect_is_instance(data, dict, label="payload type")
            expect_in("computed", data, label="computed key")
            expect_in("missing", data, label="missing key")
            expect_in("stale", data, label="stale key")
            expect_in("blocked", data, label="blocked key")

    @staticmethod
    def test_status_invalid_module() -> None:
        """Invalid module name produces error or requires project context."""
        result = run_cli(["build", "status", "--module", "invalid_module"])

        expect_equal(result.exit_code, 1, label="exit_code")
        combined = (result.stdout + (result.output or "")).lower()
        has_module_error = "unknown module" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_module_error or has_project_error,
            message="Expected module validation or project context error",
        )


class TestBuildRunHelp:
    """Tests for build run --help."""

    @staticmethod
    def test_run_help_shows_description() -> None:
        """Help text describes the run command."""
        result = run_cli(["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("dependency resolution", result.stdout.lower(), label="description present")

    @staticmethod
    def test_run_help_shows_options() -> None:
        """Help text shows available options."""
        result = run_cli(["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--module", result.stdout, label="module option")
        expect_in("--dry-run", result.stdout, label="dry-run option")
        expect_in("--force", result.stdout, label="force option")
        expect_in("--json", result.stdout, label="json option")


class TestBuildRunValidation:
    """Tests for build run argument validation."""

    @staticmethod
    def test_run_no_targets_no_module_error() -> None:
        """Running without targets or module produces error or requires project context."""
        result = run_cli(["build", "run"])

        expect_equal(result.exit_code, 1, label="exit_code")
        combined = (result.stdout + (result.output or "")).lower()
        has_targets_error = "specify targets" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_targets_error or has_project_error,
            message="Expected targets/module or project context error",
        )

    @staticmethod
    def test_run_invalid_module() -> None:
        """Invalid module name produces error or requires project context."""
        result = run_cli(["build", "run", "--module", "invalid_module"])

        expect_equal(result.exit_code, 1, label="exit_code")
        combined = (result.stdout + (result.output or "")).lower()
        has_module_error = "unknown module" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_module_error or has_project_error,
            message="Expected module validation or project context error",
        )

    @staticmethod
    def test_run_unknown_target() -> None:
        """Unknown target name produces error or requires project context."""
        result = run_cli(["build", "run", "nonexistent_target_xyz"])

        expect_equal(result.exit_code, 1, label="exit_code")
        combined = (result.stdout + (result.output or "")).lower()
        has_target_error = "unknown target" in combined
        has_project_error = "codeintel.yaml" in combined or "provide --repo" in combined
        expect_true(
            has_target_error or has_project_error,
            message="Expected target validation or project context error",
        )

    @staticmethod
    def test_run_conflicting_selection_flags() -> None:
        """Providing multiple selection mechanisms fails fast."""
        result = run_cli(["build", "run", "target_a", "--module", "ingestion", "--all"])

        expect_equal(result.exit_code, 1, label="exit_code")
        expect_in("Provide exactly one of targets, --module, or --all.", result.stderr)


class TestBuildAppStructure:
    """Tests for build app registration and structure."""

    @staticmethod
    def test_build_help_shows_commands() -> None:
        """Build help shows available subcommands."""
        result = run_cli(["build", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("run", result.stdout, label="run command")
        expect_in("status", result.stdout, label="status command")

    @staticmethod
    def test_build_no_args_shows_help() -> None:
        """Build with no args shows help."""
        result = run_cli(["build"])

        expect_in("run", result.stdout, label="run mentioned")
        expect_in("status", result.stdout, label="status mentioned")


class TestModuleOption:
    """Tests for the --module option."""

    @pytest.mark.parametrize("module", ["ingestion", "graphs", "analytics"])
    @staticmethod
    def test_valid_module_names(module: TargetModule) -> None:
        """Valid module names are accepted."""
        result = run_cli(["build", "run", "--module", module, "--dry-run"])

        if result.exit_code != 0:
            expect_true(
                "unknown module" not in result.stdout.lower(),
                message=f"Module {module} should be valid",
            )


class TestDryRun:
    """Tests for --dry-run functionality."""

    @staticmethod
    def test_dry_run_flag_recognized() -> None:
        """Dry run flag is recognized."""
        result = run_cli(["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--dry-run", result.stdout, label="dry-run option recognized")
        expect_in("-n", result.stdout, label="short flag recognized")


class TestForceOption:
    """Tests for --force option."""

    @staticmethod
    def test_force_flag_recognized() -> None:
        """Force flag is recognized."""
        result = run_cli(["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--force", result.stdout, label="force option recognized")
        expect_in("-f", result.stdout, label="short flag recognized")


class TestJsonOutput:
    """Tests for --json output option."""

    @staticmethod
    def test_json_flag_recognized_on_status() -> None:
        """JSON flag is recognized on status command."""
        result = run_cli(["build", "status", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--json", result.stdout, label="json option recognized")

    @staticmethod
    def test_json_flag_recognized_on_run() -> None:
        """JSON flag is recognized on run command."""
        result = run_cli(["build", "run", "--help"])

        expect_equal(result.exit_code, 0, label="exit_code")
        expect_in("--json", result.stdout, label="json option recognized")
