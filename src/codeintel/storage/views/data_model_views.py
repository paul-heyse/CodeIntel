"""Docs views for data models and configuration data flow.

View definitions are now managed via Ibis in ibis_views.py.
This module only exports the view name constants for compatibility.
"""

from __future__ import annotations

DATA_MODEL_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_data_models",
    "docs.v_data_model_fields",
    "docs.v_data_model_relationships",
    "docs.v_data_models_normalized",
    "docs.v_data_model_usage",
    "docs.v_config_data_flow",
)

__all__ = ["DATA_MODEL_VIEW_NAMES"]
