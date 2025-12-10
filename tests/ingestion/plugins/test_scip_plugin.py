"""Tests for ScipIngestPlugin behavior and fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.protocols import ScipOccurrence, ScipSymbol
from codeintel.build.providers import Providers
from codeintel.ingestion.plugins.scip_plugin import (
    ScipIngestPlugin,
    get_module_paths,
    paths_to_modules,
)
from tests._helpers import build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.factories.row_factories import sample_scip_documents
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.fake_providers import FakeProviders
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    make_resource_case_params,
    run_ingestion_scenario,
    write_scip_index,
)
from tests.ingestion.plugins._wiring import run_module_path_resolution_scenarios


def test_paths_to_modules_creates_records(tmp_path: Path) -> None:
    """Path conversion should map to module names with file paths."""
    repo_root = tmp_path / "repo"
    paths = ["pkg/a.py", "pkg/util/b.py"]
    modules = paths_to_modules(paths, repo_root)

    expect_equal(modules[0].module_name, "pkg.a")
    expect_equal(modules[0].file_path, repo_root / "pkg/a.py")
    expect_equal(modules[1].module_name, "pkg.util.b")


RESOURCE_CASES = make_resource_case_params()


@pytest.mark.parametrize(
    "options",
    [params for _, params in RESOURCE_CASES],
    ids=[name for name, _ in RESOURCE_CASES],
)
def test_module_path_resolution_scenarios(
    tmp_path: Path, options: dict[str, bool], ingestion_gateway
) -> None:
    """Shared module path resolution coverage for ScipIngestPlugin."""
    run_module_path_resolution_scenarios(
        lambda _capture: ScipIngestPlugin(),
        get_module_paths,
        tmp_path,
        resources_path="pkg/a.py",
        options=options,
        gateway=ingestion_gateway,
    )


@pytest.mark.anyio
async def test_execute_raises_when_indexer_missing(tmp_path: Path) -> None:
    """Missing scip_indexer should raise ToolNotAvailableError."""
    plugin = ScipIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)

    with pytest.raises(ToolNotAvailableError):
        await plugin.execute(ctx)


def _write_scip_json(target_dir: Path) -> Path:
    docs = sample_scip_documents()
    return write_scip_index(target_dir, docs)


@pytest.mark.anyio
async def test_execute_ingests_symbols_and_occurrences(tmp_path: Path) -> None:
    """SCIP ingestion should write symbols/occurrences and artifacts."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/a.py": "def a():\n    return 1\n"})
    fake_providers = FakeProviders()
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/a.py",),
    )
    fake_providers.scip_indexer.symbols = (
        ScipSymbol(symbol="pkg/a.py:func", name="func", kind="function"),
    )
    fake_providers.scip_indexer.occurrences = (
        ScipOccurrence(
            symbol="pkg/a.py:func",
            path="pkg/a.py",
            line=1,
            character=0,
            end_line=1,
            end_character=4,
            role="definition",
        ),
    )
    ctx, result = await run_ingestion_scenario(
        ScipIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
        seed_fn=lambda context: _write_scip_json(context.scip_dir),
    )

    expect_true(result.success is True)
    expect_true("index.scip" in result.artifacts_written)
    expect_true("index.json" in result.artifacts_written)
    expect_true(len(fake_providers.scip_indexer.index_calls.calls) >= 1)


@pytest.mark.anyio
async def test_execute_fails_when_indexer_returns_error(tmp_path: Path) -> None:
    """Failed index run should propagate as failed TargetResult."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/a.py": "def a():\n    return 1\n"})
    fake_providers = FakeProviders()
    fake_providers.scip_indexer.index_success = False
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/a.py",),
    )
    ctx, result = await run_ingestion_scenario(
        ScipIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
        seed_fn=lambda context: _write_scip_json(context.scip_dir),
    )

    expect_true(result.success is False)
    expect_true("SCIP ingest failed" in (result.error_message or ""))
