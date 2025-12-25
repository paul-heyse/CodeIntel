"""Tests for scip target hashing inputs."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.ingestion.ingest_targets import ModuleScanResult
from codeintel.build.hamilton.native.ingestion.scip import scip__hash_options
from codeintel.build.hamilton.native.options.ingestion import ScipIngestOptions
from codeintel.build.providers import Providers
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.service import ToolService
from tests._helpers.assertions import expect_true
from tests._helpers.context import create_test_context
from tests._helpers.fakes.tools import FakeToolRunner, FakeToolRunnerConfig
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env


def _build_env(tmp_path: Path, *, scip_stdout: str) -> tuple[ModuleScanResult, BuildEnv]:
    ctx = create_test_context(tmp_path)
    runner = FakeToolRunner(
        cache_dir=tmp_path,
        config=FakeToolRunnerConfig(payloads={"scip-python": scip_stdout}),
    )
    providers = Providers(
        tool_runner=runner,
        tool_service=ToolService(runner, tools_config=ToolsConfig.default()),
    )
    env = build_test_env(
        BuildEnvSpec(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
            providers=providers,
        )
    )
    scan = ModuleScanResult(success=True, file_state_hash="state-hash")
    return scan, env


def test_scip_hash_options_changes_with_tool_version(tmp_path: Path) -> None:
    """Tool version changes should alter the scip target options hash."""
    scan_a, env_a = _build_env(tmp_path / "a", scip_stdout="scip-python 1.0.0")
    scan_b, env_b = _build_env(tmp_path / "b", scip_stdout="scip-python 2.0.0")

    options = ScipIngestOptions()
    hash_a = scip__hash_options(env_a, scan_a, options)
    hash_b = scip__hash_options(env_b, scan_b, options)

    expect_true(hash_a.options_hash is not None)
    expect_true(hash_b.options_hash is not None)
    expect_true(hash_a.options_hash != hash_b.options_hash)
