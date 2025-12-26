"""Phase 5 replication tests for analytics metadata targets."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.native.analytics.metadata_targets import (
    DATA_MODELS_TABLE_KEYS,
    FUNCTION_AST_FEATURES_TABLE_KEY,
)
from codeintel.storage.repositories import RepositoryFactory
from tests._helpers.assertions import (
    assert_record_has_datasets,
    assert_table_schema_valid,
    assert_target_ok,
    expect_true,
)
from tests._helpers.harnesses.analytics_harness import AnalyticsTargetHarness
from tests._helpers.harnesses.hamilton_build import HarnessOpenOptions
from tests._helpers.seeds import CORE_PACK, DATA_MODELS_PACK


def test_function_ast_features_target_materializes_schema(tmp_path: Path) -> None:
    """Validate function_ast_features writes schema-aligned rows."""
    options = HarnessOpenOptions(
        repo_strategy="canonical",
        seed_packs=(CORE_PACK,),
    )
    with AnalyticsTargetHarness.open(tmp_path, options=options) as harness:
        records = harness.run_targets(("function_ast_features",))
        record = records["function_ast_features"]

        assert_target_ok(record)
        assert_record_has_datasets(record, [FUNCTION_AST_FEATURES_TABLE_KEY])

        gateway = harness.harness.ctx.gateway
        assert_table_schema_valid(gateway, FUNCTION_AST_FEATURES_TABLE_KEY)

        row_count = record.row_counts.get(FUNCTION_AST_FEATURES_TABLE_KEY)
        expect_true(
            row_count is not None and row_count >= 0,
            message="Expected row count for function_ast_features",
        )


def test_data_models_target_materializes_tables(tmp_path: Path) -> None:
    """Validate data_models writes all tables and serves repository reads."""
    options = HarnessOpenOptions(
        repo_strategy="canonical",
        seed_packs=(DATA_MODELS_PACK,),
    )
    with AnalyticsTargetHarness.open(tmp_path, options=options) as harness:
        records = harness.run_targets(("data_models",))
        record = records["data_models"]

        assert_target_ok(record)
        assert_record_has_datasets(record, DATA_MODELS_TABLE_KEYS)

        gateway = harness.harness.ctx.gateway
        for table_key in DATA_MODELS_TABLE_KEYS:
            assert_table_schema_valid(gateway, table_key)
            row_count = record.row_counts.get(table_key)
            expect_true(
                row_count is not None and row_count >= 0,
                message=f"Expected row count for {table_key}",
            )

        repo = RepositoryFactory(
            gateway,
            repo=harness.harness.ctx.snapshot.repo,
            commit=harness.harness.ctx.snapshot.commit,
        ).data_models
        _ = repo.list_models()
