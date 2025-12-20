"""Feature gating helpers for serving surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class ServingFeatureSet:
    """Derived feature toggles for serving adapters."""

    enable_http_export: bool
    enable_mcp_export: bool
    enable_mcp_export_tasks: bool
    enable_mcp_search: bool
    enable_mcp_explain: bool
    enable_mcp_meta: bool
    enable_mcp_sampling: bool
    enable_mcp_progress: bool
    enable_mcp_event_store: bool

    @classmethod
    def from_settings(cls, settings: ServingSettings) -> ServingFeatureSet:
        """Compute feature flags from serving settings.

        Returns
        -------
        ServingFeatureSet
            Derived feature flags based on settings.
        """
        return cls(
            enable_http_export=settings.enable_export_endpoints,
            enable_mcp_export=settings.mcp_enable_export,
            enable_mcp_export_tasks=settings.mcp_export_enable_tasks,
            enable_mcp_search=settings.mcp_enable_search,
            enable_mcp_explain=settings.mcp_enable_explain,
            enable_mcp_meta=settings.mcp_enable_meta,
            enable_mcp_sampling=settings.mcp_enable_sampling,
            enable_mcp_progress=settings.mcp_progress_reporting,
            enable_mcp_event_store=settings.mcp_enable_event_store,
        )

    @classmethod
    def all_enabled(cls) -> ServingFeatureSet:
        """Return a feature set with all toggles enabled.

        Returns
        -------
        ServingFeatureSet
            Feature set with all flags enabled.
        """
        return cls(
            enable_http_export=True,
            enable_mcp_export=True,
            enable_mcp_export_tasks=True,
            enable_mcp_search=True,
            enable_mcp_explain=True,
            enable_mcp_meta=True,
            enable_mcp_sampling=True,
            enable_mcp_progress=True,
            enable_mcp_event_store=True,
        )


__all__ = ["ServingFeatureSet"]
