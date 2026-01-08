"""Tests for incremental SCIP indexing behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolRunOptions,
    ToolRunResult,
)
from codeintel.ingestion.engine.infrastructure.runner import ToolRunner
from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.scip.incremental import ScipIncrementalConfig, update_index_incremental
from codeintel.ingestion.scip.index_store import load_index_proto
from codeintel.ingestion.scip.manifest import load_manifest, manifest_path
from codeintel.ingestion.scip.telemetry import ScipRunIdentity, ScipRunTelemetry
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest


@dataclass(frozen=True)
class _ProtoIndexRunnerConfig:
    proto_module_path: Path
    doc_map: Mapping[str, Mapping[str, object]]
    version_stdout: str = "scip-python 0.1.0"
    fail_target_only: bool = False


class ProtoIndexToolRunner(ToolRunner):
    """ToolRunner that emits real SCIP protobufs for tests."""

    def __init__(self, cache_dir: Path, *, config: _ProtoIndexRunnerConfig) -> None:
        super().__init__(cache_dir=cache_dir, tools_config=ToolsConfig.default())
        self.config = config
        self.calls: list[tuple[ToolName, tuple[str, ...], ToolRunOptions]] = []

    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        options: ToolRunOptions | None = None,
    ) -> ToolRunResult:
        """Run scip-python for tests and write a protobuf index.

        Returns
        -------
        ToolRunResult
            Tool run metadata for the invocation.

        Raises
        ------
        ValueError
            If an unsupported tool is requested or the output path is missing.
        ToolExecutionError
            If the runner is configured to fail target-only calls.
        """
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        args_list = list(args)
        run_options = options or ToolRunOptions()
        self.calls.append((tool_enum, tuple(args_list), run_options))
        if tool_enum is not ToolName.SCIP_PYTHON:
            msg = f"Unsupported tool: {tool_enum.value}"
            raise ValueError(msg)
        if "--version" in args_list:
            return ToolRunResult(
                tool=tool_enum,
                args=tuple(args_list),
                returncode=0,
                stdout=self.config.version_stdout,
                stderr="",
                output_path=run_options.output_path,
                duration_s=0.0,
            )

        output_path = run_options.output_path
        if output_path is None:
            message = "Output path required for scip-python test runner"
            raise ValueError(message)

        target_only = _extract_target_only(args_list)
        if target_only and self.config.fail_target_only:
            result = ToolRunResult(
                tool=tool_enum,
                args=tuple(args_list),
                returncode=1,
                stdout="",
                stderr="failed",
                output_path=output_path,
                duration_s=0.0,
            )
            raise ToolExecutionError(result)

        if target_only:
            docs = [_doc_payload(self.config.doc_map, rel_path) for rel_path in target_only]
        else:
            docs = list(self.config.doc_map.values())

        write_scip_index(
            output_path,
            proto_module_path=self.config.proto_module_path,
            documents=docs,
        )
        return ToolRunResult(
            tool=tool_enum,
            args=tuple(args_list),
            returncode=0,
            stdout="",
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )


def _extract_target_only(args: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for idx, arg in enumerate(args):
        if arg != "--target-only":
            continue
        if idx + 1 < len(args):
            targets.append(args[idx + 1])
    return targets


def _count_target_calls(runner: ProtoIndexToolRunner) -> int:
    return sum(1 for _, args, _ in runner.calls if "--target-only" in args)


def _doc_payload(
    doc_map: Mapping[str, Mapping[str, object]],
    rel_path: str,
) -> Mapping[str, object]:
    payload = doc_map.get(rel_path)
    if payload is None:
        msg = f"Missing document payload for {rel_path}"
        raise ValueError(msg)
    return payload


def _module_record(repo_root: Path, rel_path: str, index: int, total: int) -> ModuleRecord:
    return ModuleRecord(
        rel_path=rel_path,
        module_name=rel_path.replace("/", ".").removesuffix(".py"),
        file_path=repo_root / rel_path,
        index=index,
        total=total,
    )


def _scip_identity(*, run_id: str, options_hash: str | None = "options-hash") -> ScipRunIdentity:
    return ScipRunIdentity(
        repo="test/repo",
        commit="abc123",
        run_id=run_id,
        options_hash=options_hash,
        project_version=None,
        project_namespace=None,
    )


def _write_module(repo_root: Path, rel_path: str, body: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_incremental_updates_and_deletes_documents(tmp_path: Path) -> None:
    """Incremental updates should replace changed docs and drop deleted ones."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    change_set = ChangeSet(
        modified=[_module_record(repo_root, "a.py", 1, 2)],
        deleted=[_module_record(repo_root, "b.py", 2, 2)],
    )
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(
            proto_module_path=proto_module_path,
            doc_map={"a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA_new"}]}},
        ),
    )
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=(
            _module_record(repo_root, "a.py", 1, 2),
            _module_record(repo_root, "b.py", 2, 2),
        ),
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_true(result.full_rebuild is False)

    merged = load_index_proto(output_scip, proto_module_path=proto_module_path)
    rel_paths = sorted(doc.relative_path for doc in merged.documents)
    expect_equal(rel_paths, ["a.py"])
    symbols = [sym.symbol for sym in merged.documents[0].symbols]
    expect_equal(symbols, ["symA_new"])

    manifest = load_manifest(manifest_path(output_scip.parent))
    expect_true("a.py" in manifest.records)
    expect_true("b.py" not in manifest.records)


def test_incremental_batches_changed_modules(tmp_path: Path) -> None:
    """Incremental indexing batches target-only calls."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")
    _write_module(repo_root, "c.py", "def c():\n    return 3\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
        "c.py": {"relative_path": "c.py", "symbols": [{"symbol": "symC"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (
        _module_record(repo_root, "a.py", 1, 3),
        _module_record(repo_root, "b.py", 2, 3),
        _module_record(repo_root, "c.py", 3, 3),
    )
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(
            proto_module_path=proto_module_path,
            doc_map={
                "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA_new"}]},
                "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB_new"}]},
                "c.py": {"relative_path": "c.py", "symbols": [{"symbol": "symC_new"}]},
            },
        ),
    )
    telemetry = ScipRunTelemetry.create(identity=_scip_identity(run_id="run-1"))
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        batch_size=2,
        batch_max_bytes=0,
        full_rebuild_threshold_count=10,
        full_rebuild_threshold_ratio=1.0,
        telemetry=telemetry,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_equal(telemetry.batch_count, 2)
    expect_equal(_count_target_calls(runner), 2)


def test_batch_shard_paths_stable_across_runs(tmp_path: Path) -> None:
    """Batch shard paths should be deterministic for the same inputs."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")
    _write_module(repo_root, "c.py", "def c():\n    return 3\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
        "c.py": {"relative_path": "c.py", "symbols": [{"symbol": "symC"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (
        _module_record(repo_root, "a.py", 1, 3),
        _module_record(repo_root, "b.py", 2, 3),
        _module_record(repo_root, "c.py", 3, 3),
    )
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(proto_module_path=proto_module_path, doc_map=base_docs),
    )
    file_state_by_path = {
        "a.py": FileDigest(size_bytes=10, mtime_ns=1, content_hash="hash-a"),
        "b.py": FileDigest(size_bytes=10, mtime_ns=1, content_hash="hash-b"),
        "c.py": FileDigest(size_bytes=10, mtime_ns=1, content_hash="hash-c"),
    }
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        batch_size=2,
        batch_max_bytes=0,
        full_rebuild_threshold_count=10,
        full_rebuild_threshold_ratio=1.0,
        file_state_by_path=file_state_by_path,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    manifest_first = load_manifest(manifest_path(output_scip.parent))

    result = update_index_incremental(config=config)
    expect_true(result.success)
    manifest_second = load_manifest(manifest_path(output_scip.parent))

    shard_paths_first = {path: record.shard_path for path, record in manifest_first.records.items()}
    shard_paths_second = {
        path: record.shard_path for path, record in manifest_second.records.items()
    }
    expect_equal(shard_paths_first, shard_paths_second)


def test_full_rebuild_thresholds_trigger_full_rebuild(tmp_path: Path) -> None:
    """Thresholds force a single full rebuild invocation."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (
        _module_record(repo_root, "a.py", 1, 2),
        _module_record(repo_root, "b.py", 2, 2),
    )
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(proto_module_path=proto_module_path, doc_map=base_docs),
    )
    telemetry = ScipRunTelemetry.create(identity=_scip_identity(run_id="run-2"))
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        full_rebuild_threshold_count=1,
        full_rebuild_threshold_ratio=1.0,
        telemetry=telemetry,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_true(result.full_rebuild)
    expect_equal(_count_target_calls(runner), 0)


def test_hash_reuse_from_file_state(tmp_path: Path) -> None:
    """File state rows should provide content hashes for planning."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {"a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]}}
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (_module_record(repo_root, "a.py", 1, 1),)
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(proto_module_path=proto_module_path, doc_map=base_docs),
    )
    telemetry = ScipRunTelemetry.create(identity=_scip_identity(run_id="run-3"))
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        file_state_by_path={
            "a.py": FileDigest(size_bytes=10, mtime_ns=1, content_hash="hash-a"),
        },
        telemetry=telemetry,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_equal(telemetry.hash_source, "file_state")
    expect_equal(telemetry.hash_reused, 1)
    expect_equal(telemetry.hash_computed, 0)


def test_hash_resolver_fallback_to_disk(tmp_path: Path) -> None:
    """Digest resolution should fall back to on-disk hashing when state is missing."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {"a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]}}
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (_module_record(repo_root, "a.py", 1, 1),)
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(proto_module_path=proto_module_path, doc_map=base_docs),
    )
    telemetry = ScipRunTelemetry.create(identity=_scip_identity(run_id="run-3b"))
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        telemetry=telemetry,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_equal(telemetry.hash_source, "computed")
    expect_equal(telemetry.hash_reused, 0)
    expect_equal(telemetry.hash_computed, 1)
    expect_equal(telemetry.hash_source_breakdown, "computed=1")


def test_ratio_gate_prevents_small_repo_full_rebuild(tmp_path: Path) -> None:
    """Ratio thresholds should be gated for small repos."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    modules = (
        _module_record(repo_root, "a.py", 1, 2),
        _module_record(repo_root, "b.py", 2, 2),
    )
    change_set = ChangeSet(modified=list(modules))
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(proto_module_path=proto_module_path, doc_map=base_docs),
    )
    telemetry = ScipRunTelemetry.create(identity=_scip_identity(run_id="run-4"))
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=modules,
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        full_rebuild_threshold_ratio=0.1,
        full_rebuild_ratio_min_modules=10,
        full_rebuild_ratio_min_changed=5,
        telemetry=telemetry,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_true(result.full_rebuild is False)
    expect_equal(telemetry.decision, "incremental")
    expect_true(telemetry.ratio_gate_applied is False)


def test_incremental_plan_summary_logs_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Plan summary logs counts and decision."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    change_set = ChangeSet(modified=[_module_record(repo_root, "a.py", 1, 2)])
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(
            proto_module_path=proto_module_path,
            doc_map={"a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA_new"}]}},
        ),
    )
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=(
            _module_record(repo_root, "a.py", 1, 2),
            _module_record(repo_root, "b.py", 2, 2),
        ),
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
        full_rebuild_threshold_count=10,
        full_rebuild_threshold_ratio=1.0,
    )

    with caplog.at_level(logging.INFO, logger="codeintel.ingestion.scip.incremental"):
        result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_in("SCIP incremental plan", caplog.text)
    expect_in("changed=1", caplog.text)
    expect_in("decision=incremental", caplog.text)


def test_incremental_falls_back_to_full_rebuild(tmp_path: Path) -> None:
    """Failures during per-module indexing should trigger a full rebuild."""
    repo_root = tmp_path / "repo"
    _write_module(repo_root, "a.py", "def a():\n    return 1\n")
    _write_module(repo_root, "b.py", "def b():\n    return 2\n")

    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "build" / "scip" / "index.scip"
    base_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB"}]},
    }
    write_scip_index(output_scip, proto_module_path=proto_module_path, documents=base_docs.values())

    change_set = ChangeSet(
        modified=[_module_record(repo_root, "a.py", 1, 2)],
    )
    full_docs = {
        "a.py": {"relative_path": "a.py", "symbols": [{"symbol": "symA_full"}]},
        "b.py": {"relative_path": "b.py", "symbols": [{"symbol": "symB_full"}]},
    }
    runner = ProtoIndexToolRunner(
        cache_dir=tmp_path,
        config=_ProtoIndexRunnerConfig(
            proto_module_path=proto_module_path,
            doc_map=full_docs,
            fail_target_only=True,
        ),
    )
    config = ScipIncrementalConfig(
        repo_root=repo_root,
        output_scip=output_scip,
        proto_module_path=proto_module_path,
        change_set=change_set,
        modules=(
            _module_record(repo_root, "a.py", 1, 2),
            _module_record(repo_root, "b.py", 2, 2),
        ),
        options_hash="options-hash",
        tools_config=runner.tools_config,
        tool_runner=runner,
        scope_paths=None,
        max_file_size_kb=1024,
        timeout_seconds=30,
        target_dir=None,
    )

    result = update_index_incremental(config=config)
    expect_true(result.success)
    expect_true(result.full_rebuild)

    merged = load_index_proto(output_scip, proto_module_path=proto_module_path)
    rel_paths = sorted(doc.relative_path for doc in merged.documents)
    expect_equal(rel_paths, ["a.py", "b.py"])
    symbols = {sym.symbol for doc in merged.documents for sym in doc.symbols}
    expect_true("symA_full" in symbols)
    expect_true("symB_full" in symbols)
