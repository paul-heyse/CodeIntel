"""Tests for inference observation guardrail helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Protocol

    class GuardrailsApi(Protocol):
        """Protocol for guardrails helpers used by tests."""

        def schema_observations_available(self, schemas: object) -> bool:
            """Return whether observation summaries are queryable."""
            ...

        def missing_schema_observations(
            self,
            table_targets: Mapping[str, str],
            *,
            schemas: object,
        ) -> list[str]:
            """Return missing observation keys for the given targets."""
            ...


guardrails = cast("GuardrailsApi", importlib.import_module("tools.guardrails"))


class _StubSchemaCatalog:
    def __init__(
        self,
        *,
        observations: Mapping[str, SchemaObservationRecord],
        contracts: set[str],
        total_tables: int,
    ) -> None:
        self._observations = observations
        self._contracts = contracts
        self._total_tables = total_tables

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        return self._observations.get(table_key)

    def drift_summary_report(self, *, limit: int = 50) -> dict[str, object]:
        _ = limit
        return {"total_tables": self._total_tables}

    def has_contract_arrow_schema(self, *, table_key: str) -> bool:
        return table_key in self._contracts


def _observation_for(table_key: str) -> SchemaObservationRecord:
    return SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64="payload",
    )


def test_schema_observations_available_when_empty() -> None:
    """Return True when observation summaries are available."""
    schemas = _StubSchemaCatalog(observations={}, contracts=set(), total_tables=0)
    assert guardrails.schema_observations_available(schemas)


def test_missing_schema_observations_reports_missing_keys() -> None:
    """Return missing table keys when observations are incomplete."""
    schemas = _StubSchemaCatalog(
        observations={"core.modules": _observation_for("core.modules")},
        contracts=set(),
        total_tables=1,
    )
    missing = guardrails.missing_schema_observations(
        {
            "core.modules": "ingestion",
            "core.repo_map": "ingestion",
        },
        schemas=schemas,
    )
    assert missing == ["core.repo_map (target=ingestion)"]


def test_missing_schema_observations_accepts_contract_renderer_cache() -> None:
    """Treat renderer cache Arrow schemas as satisfying the guardrail."""
    schemas = _StubSchemaCatalog(
        observations={"core.modules": _observation_for("core.modules")},
        contracts={"core.repo_map"},
        total_tables=1,
    )
    missing = guardrails.missing_schema_observations(
        {
            "core.modules": "ingestion",
            "core.repo_map": "ingestion",
        },
        schemas=schemas,
    )
    assert missing == []
