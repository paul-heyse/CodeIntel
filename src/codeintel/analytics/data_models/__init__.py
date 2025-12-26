"""Data model analytics plugins package.

For Hamilton native execution, use the pure compute functions:
- ``load_data_models_inputs`` loads storage-backed inputs
- ``compute_data_models_from_inputs`` returns ``DataModelsResult`` without writing
- ``compute_data_models_pure`` combines both steps for convenience

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.data_models``
"""

from codeintel.analytics.data_models.compute import (
    DataModelsInputs,
    DataModelsResult,
    compute_data_models_from_inputs,
    compute_data_models_pure,
    load_data_models_inputs,
)
from codeintel.analytics.data_models.core import ClassMeta, ModelRecord

__all__ = [
    "ClassMeta",
    "DataModelsInputs",
    "DataModelsResult",
    "ModelRecord",
    "compute_data_models_from_inputs",
    "compute_data_models_pure",
    "load_data_models_inputs",
]
