"""Tests for provenance reporting on duplicate outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.validate import validate_nodes
from codeintel.core.hamilton import tags as ht
from codeintel.runtime.module_resolver import ModuleProvenance

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class DummyNode:
    """Minimal node stub for validation tests."""

    name: str
    tags: dict[str, object]
    dependencies: Sequence[DummyNode] = ()
    originating_module: str | None = None


def test_duplicate_output_reports_origin() -> None:
    """Report module provenance for duplicate output issues."""
    materialize_alpha = DummyNode(
        name="m__alpha",
        tags={
            ht.TAG_NODE_TYPE: ht.NODE_TYPE_MATERIALIZE,
            ht.TAG_DOMAIN: "analytics",
            ht.TAG_TARGET: "alpha",
            ht.TAG_TARGET_SPEC_VERSION: "1",
        },
        originating_module="alpha_pack.targets",
    )
    materialize_beta = DummyNode(
        name="m__beta",
        tags={
            ht.TAG_NODE_TYPE: ht.NODE_TYPE_MATERIALIZE,
            ht.TAG_DOMAIN: "analytics",
            ht.TAG_TARGET: "beta",
            ht.TAG_TARGET_SPEC_VERSION: "1",
        },
        originating_module="beta_pack.targets",
    )
    dataset_alpha = DummyNode(
        name="d__alpha",
        tags={
            ht.TAG_NODE_TYPE: ht.NODE_TYPE_DATASET,
            ht.TAG_DOMAIN: "analytics",
            ht.TAG_TABLE_KEY: "analytics.dupe_table",
        },
        dependencies=(materialize_alpha,),
        originating_module="alpha_pack.targets",
    )
    dataset_beta = DummyNode(
        name="d__beta",
        tags={
            ht.TAG_NODE_TYPE: ht.NODE_TYPE_DATASET,
            ht.TAG_DOMAIN: "analytics",
            ht.TAG_TABLE_KEY: "analytics.dupe_table",
        },
        dependencies=(materialize_beta,),
        originating_module="beta_pack.targets",
    )

    provenance = {
        "alpha_pack.targets": ModuleProvenance(
            origin="plugin",
            module_import="alpha_pack.targets",
            file_path=None,
            plugin_name="alpha_pack",
            dist_name=None,
            dist_version=None,
        ),
        "beta_pack.targets": ModuleProvenance(
            origin="plugin",
            module_import="beta_pack.targets",
            file_path=None,
            plugin_name="beta_pack",
            dist_name=None,
            dist_version=None,
        ),
    }

    nodes = {
        materialize_alpha.name: materialize_alpha,
        materialize_beta.name: materialize_beta,
        dataset_alpha.name: dataset_alpha,
        dataset_beta.name: dataset_beta,
    }

    result = validate_nodes(
        nodes,
        validate_schema=False,
        module_provenance=provenance,
    )

    dupe_issue = next(issue for issue in result.errors if issue.code == "duplicate_table_key")
    assert dupe_issue.plugin_name == "beta_pack"
    assert dupe_issue.module_import == "beta_pack.targets"
