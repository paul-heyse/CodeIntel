"""Unit tests for SCIP ingest result building."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.ingestion.scip import (
    ScipIngestInputs,
    ScipRunResult,
    t__scip__ingest,
)
from codeintel.build.hamilton.native.options.ingestion import ScipIngestOptions
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.core.hamilton.records import TargetRunRecord
from tests._helpers.assertions import expect_true
from tests._helpers.context import create_test_context
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index


def _build_env(tmp_path: Path) -> BuildEnv:
    ctx = create_test_context(tmp_path)
    return build_test_env(
        BuildEnvSpec(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
        )
    )


def _modules_record() -> TargetRunRecord:
    return TargetRunRecord(
        target="modules",
        impl_kind="native",
        status="succeeded",
        input_hash="hash",
    )


def test_scip_ingest_skips_when_run_skipped(tmp_path: Path) -> None:
    """Skipped SCIP runs should return a skipped ingest result."""
    env = _build_env(tmp_path)
    inputs = ScipIngestInputs(
        modules=_modules_record(),
        run=ScipRunResult(
            result=ExecutionResult.skip("SCIP target skipped"),
            outputs={"scip_index": tmp_path / "index.scip"},
        ),
        proto_module_path=None,
        options=ScipIngestOptions(),
    )

    result = t__scip__ingest(env, inputs)

    expect_true(result.result.skipped)
    expect_true(result.payload is None)


def test_scip_ingest_parses_explicit_index_path(tmp_path: Path) -> None:
    """SCIP ingest should parse the explicit index.scip path from run results."""
    env = _build_env(tmp_path / "env")
    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "external" / "index.scip"
    write_scip_index(output_scip, proto_module_path=proto_module_path)

    inputs = ScipIngestInputs(
        modules=_modules_record(),
        run=ScipRunResult(
            result=ExecutionResult.ok(),
            outputs={"scip_index": output_scip},
        ),
        proto_module_path=proto_module_path,
        options=ScipIngestOptions(),
    )

    result = t__scip__ingest(env, inputs)

    expect_true(result.result.success)
    expect_true(not result.result.skipped)
    payload = result.payload
    expect_true(payload is not None)
    if payload is None:
        return
    symbols = tabular_to_lazyframe(payload["core.scip_symbols"]).collect()
    occurrences = tabular_to_lazyframe(payload["core.scip_occurrences"]).collect()
    expect_true(symbols.height > 0)
    expect_true(occurrences.height > 0)


def test_scip_ingest_payload_is_columnar(tmp_path: Path) -> None:
    """SCIP ingest should emit LazyFrame payloads for all tables."""
    env = _build_env(tmp_path / "env")
    proto_module_path = ensure_proto_module(tmp_path)
    output_scip = tmp_path / "columnar" / "index.scip"
    write_scip_index(output_scip, proto_module_path=proto_module_path)

    inputs = ScipIngestInputs(
        modules=_modules_record(),
        run=ScipRunResult(
            result=ExecutionResult.ok(),
            outputs={"scip_index": output_scip},
        ),
        proto_module_path=proto_module_path,
        options=ScipIngestOptions(),
    )

    result = t__scip__ingest(env, inputs)

    payload = result.payload
    expect_true(payload is not None)
    if payload is None:
        return
    for table_key, frame in payload.items():
        lazy = tabular_to_lazyframe(frame)
        expect_true(
            isinstance(lazy, pl.LazyFrame),
            message=f"Expected LazyFrame for {table_key}",
        )
