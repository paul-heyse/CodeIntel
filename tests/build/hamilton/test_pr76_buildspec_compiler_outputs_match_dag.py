"""Tests for PR-76: BuildSpec compiler compiles from Hamilton DAG."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import get_schema_provider
from codeintel.build.spec import compile_buildspec
from codeintel.runtime.runtime_bundle import RuntimeBundle


def test_buildspec_compiler_outputs_match_dag(hamilton_runtime: RuntimeBundle) -> None:
    """Compile BuildSpec and verify key targets/outputs are present."""
    spec = compile_buildspec(
        catalog=hamilton_runtime.catalog,
        provider=get_schema_provider(),
    )
    by_name = {t.name: t for t in spec.targets}

    risk = by_name.get("risk_factors")
    if risk is None:
        pytest.fail("Expected BuildSpec to include risk_factors target")
    if "analytics.goid_risk_factors" not in set(risk.outputs):
        pytest.fail("Expected risk_factors outputs to include analytics.goid_risk_factors")

    export = by_name.get("export_jsonl")
    if export is None:
        pytest.fail("Expected BuildSpec to include export_jsonl target")
    artifact_names = {a.name for a in export.artifacts}
    if "datasets_manifest_jsonl" not in artifact_names:
        pytest.fail("Expected export_jsonl artifacts to include datasets_manifest_jsonl")
