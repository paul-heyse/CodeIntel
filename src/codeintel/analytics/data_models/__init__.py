"""Data model analytics plugins package.

For Hamilton native execution, use the pure compute functions:
- ``compute_data_models_pure`` returns ``DataModelsResult`` without writing

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.data_models``
"""

from codeintel.analytics.data_models.compute import (
    DataModelsResult,
    compute_data_models_pure,
)
from codeintel.analytics.data_models.core import ClassMeta, ModelRecord

__all__ = [
    "ClassMeta",
    "DataModelsResult",
    "ModelRecord",
    "compute_data_models_pure",
]
