"""Persistence helpers for function AST features."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.analytics.ast_features.model import FunctionAstFeatures
from codeintel.config.dataset_contract import FunctionAstFeaturesRow


def features_to_row(
    *,
    repo: str,
    commit: str,
    features: FunctionAstFeatures,
    created_at: datetime | None = None,
) -> FunctionAstFeaturesRow:
    """
    Convert FunctionAstFeatures into a FunctionAstFeaturesRow.

    Returns
    -------
    FunctionAstFeaturesRow
        Serialized row ready for storage.
    """
    timestamp = created_at or datetime.now(tz=UTC)
    return FunctionAstFeaturesRow(
        repo=repo,
        commit=commit,
        function_goid_h128=int(features.goid),
        rel_path=features.rel_path,
        qualname=features.qualname,
        is_async=features.is_async,
        uses_network=features.io_flags.uses_network,
        uses_db=features.io_flags.uses_db,
        uses_filesystem=features.io_flags.uses_filesystem,
        uses_subprocess=features.io_flags.uses_subprocess,
        uses_concurrency_lib=features.uses_concurrency_lib,
        uses_threading=features.uses_threading,
        uses_asyncio_lib=features.uses_asyncio_lib,
        http_client_libs=sorted(features.http_client_libs),
        http_server_libs=sorted(features.http_server_libs),
        db_libs=sorted(features.db_libs),
        message_libs=sorted(features.message_libs),
        config_read_count=features.config_read_count,
        feature_flag_count=features.feature_flag_count,
        decorators=list(features.decorators),
        libraries_used=sorted(features.libraries_used),
        created_at=timestamp,
    )


__all__ = ["features_to_row"]
