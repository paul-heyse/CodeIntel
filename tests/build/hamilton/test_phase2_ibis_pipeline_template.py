"""Phase 2: Ibis subDAG template smoke test.

This test validates that the reusable subDAG in
``codeintel.build.hamilton.templates.ibis_pipeline`` can be instantiated via
Hamilton's ``@subdag`` decorator and executed end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import hamilton.driver as h_driver
import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import source, subdag, tag, value

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.templates import ibis_pipeline
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from codeintel.hamilton.records import TargetRunRecord
from tests._helpers.build import make_build_config, make_build_paths
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway


def _make_env(*, gateway: StorageGateway, snapshot: SnapshotRef) -> BuildEnv:
    paths = make_build_paths(snapshot.repo_root)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=frozenset({"modules"}),
    )


def _make_graph() -> TargetGraph:
    graph = TargetGraph()
    graph.register(
        OutputTarget.from_tables(
            name="modules",
            module="ingestion",
            tables=("core.modules",),
        )
    )
    return graph


def _build_module() -> ModuleType:
    mod = ModuleType("tests.build.hamilton._phase2_ibis_pipeline_case")
    mod.__doc__ = "Ephemeral module for testing ibis_pipeline via @subdag."
    sys.modules[mod.__name__] = mod

    @tag(domain="ingestion", target="modules", node_type="compute")
    def t__modules__compute(env: BuildEnv) -> ir.Table:
        """Return a temporary Ibis table expression for materialization."""
        return env.gateway.ibis.con.table("tmp_modules")

    @tag(domain="ingestion", target="modules", node_type="materialize")
    @subdag(
        ibis_pipeline,
        inputs={
            "env": source("env"),
            "graph": source("graph"),
            "target_name": value("modules"),
            "table_key": value("core.modules"),
            "expr": source("t__modules__compute"),
        },
    )
    def t__modules(record: TargetRunRecord) -> TargetRunRecord:
        """Return the subDAG-produced record."""
        return record

    # Hamilton discovers callables from the module; ensure the functions appear to
    # originate from this ephemeral module (not the test module).
    t__modules__compute.__module__ = mod.__name__
    t__modules.__module__ = mod.__name__

    mod.t__modules__compute = t__modules__compute
    mod.t__modules = t__modules
    return mod


def test_phase2_ibis_pipeline_template_executes(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Instantiate ibis_pipeline via @subdag and execute a simple materialization."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
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
    fresh_gateway.con.register("tmp_modules", df)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__modules"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__modules"])

    if record.status != "succeeded":
        raise AssertionError(f"Expected succeeded record, got {record.status}: {record.error}")
    if record.row_counts.get("core.modules") != 2:
        raise AssertionError(f"Expected row count 2, got {record.row_counts}")

    row = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    if row is None:
        raise AssertionError("Expected COUNT(*) query to return a row")
    if int(row[0]) != 2:
        raise AssertionError(f"Expected 2 rows in core.modules, got {row[0]}")
