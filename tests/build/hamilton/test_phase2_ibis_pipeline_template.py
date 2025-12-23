"""Phase 2: Ibis subDAG template smoke test.

This test validates that the reusable subDAG in
``codeintel.build.hamilton.templates.materialize_template`` can be instantiated via
Hamilton's ``@subdag`` decorator and executed end-to-end.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import cast

import hamilton.driver as h_driver
import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import source, subdag, tag, value

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.templates import materialize_template
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import assert_record_row_counts, assert_target_ok
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


class _EphemeralModule(ModuleType):
    ir: ModuleType
    BuildEnv: type[BuildEnv]
    TargetRunRecord: type[TargetRunRecord]
    t__modules__compute: object
    t__modules: object


def _make_env(harness: HamiltonBuildHarness) -> BuildEnv:
    return replace(harness.build_env(), force_targets=frozenset({"modules"}))


def _make_graph() -> TargetGraph:
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="modules",
            module="ingestion",
            contract=OutputContract.simple(table_keys=("core.modules",)),
        )
    )
    return graph


def _build_module() -> ModuleType:
    mod = _EphemeralModule("tests.build.hamilton._phase2_ibis_pipeline_case")
    mod.__doc__ = "Ephemeral module for testing materialize_template via @subdag."
    sys.modules[mod.__name__] = mod

    # Inject types into the module namespace for Hamilton type resolution
    mod.ir = ir
    mod.BuildEnv = BuildEnv
    mod.TargetRunRecord = TargetRunRecord

    @tag(domain="ingestion", target="modules", node_type="compute")
    def t__modules__compute(env: BuildEnv) -> ir.Table:
        """Return a temporary Ibis table expression for materialization.

        Returns
        -------
        ir.Table
            Ibis expression backed by the registered ``tmp_modules`` relation.
        """
        return env.gateway.ibis.con.table("tmp_modules")

    @tag(domain="ingestion", target="modules", node_type="materialize")
    @subdag(
        materialize_template,
        inputs={
            "env": source("env"),
            "graph": source("graph"),
            "target_name": value("modules"),
            "table_key": value("core.modules"),
            "expr": source("t__modules__compute"),
        },
    )
    def t__modules(duckdb_record: TargetRunRecord) -> TargetRunRecord:
        """Return the subDAG-produced record.

        Returns
        -------
        TargetRunRecord
            Target execution record produced by the subDAG pipeline.
        """
        return duckdb_record

    # Hamilton discovers callables from the module; ensure the functions appear to
    # originate from this ephemeral module (not the test module).
    t__modules__compute.__module__ = mod.__name__
    t__modules.__module__ = mod.__name__

    mod.t__modules__compute = t__modules__compute
    mod.t__modules = t__modules
    return mod


def test_phase2_ibis_pipeline_template_executes(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Instantiate materialize_template via @subdag and execute a simple materialization."""
    env = _make_env(build_harness)
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    graph = _make_graph()
    module = _build_module()

    df = pd.DataFrame(
        [
            {
                "module": "m0",
                "path": "pkg/mod_0.py",
                "repo": repo,
                "commit": commit,
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
            {
                "module": "m1",
                "path": "pkg/mod_1.py",
                "repo": repo,
                "commit": commit,
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
        ]
    )
    env.gateway.con.register("tmp_modules", df)
    expected_row_count = len(df)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__modules"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__modules"])

    assert_target_ok(record)
    assert_record_row_counts(record, {"core.modules": expected_row_count})

    row = env.gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row is not None, message="Expected COUNT(*) query to return a row")
    row_tuple = cast("tuple[int, ...]", row)
    expect_equal(row_tuple[0], expected_row_count)
