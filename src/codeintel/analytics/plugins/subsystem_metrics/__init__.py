"""Subsystem metrics plugins.

Compute graph metrics and agreement for subsystems.
"""

from __future__ import annotations

from codeintel.analytics.plugins.subsystem_metrics.agreement import (
    SubsystemAgreementPlugin,
)
from codeintel.analytics.plugins.subsystem_metrics.graph_metrics import (
    SubsystemGraphMetricsPlugin,
)

__all__ = ["SubsystemAgreementPlugin", "SubsystemGraphMetricsPlugin"]
