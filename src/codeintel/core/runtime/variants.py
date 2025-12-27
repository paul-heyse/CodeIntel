"""Variant configuration for Hamilton DAG composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from codeintel.core.hashing.fingerprint import fingerprint

DataFrameBackend = Literal["pandas", "polars", "polars_lazy"]
CleanMode = Literal["off", "lenient", "strict"]
NullPolicy = Literal["preserve", "drop_bad_rows"]
FeatureSetName = str

_ALLOWED_BACKENDS: set[str] = {"pandas", "polars", "polars_lazy"}
_ALLOWED_CLEAN_MODES: set[str] = {"off", "lenient", "strict"}
_ALLOWED_NULL_POLICIES: set[str] = {"preserve", "drop_bad_rows"}


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """Typed configuration surface for DAG variants.

    Attributes
    ----------
    df_backend
        DataFrame backend for column-subDAG execution.
    clean_mode
        Cleaning strictness for table input pipelines.
    feature_sets
        Mapping from table key to selected column-op node names.
    enable_common_columns
        Whether to append common columns in table decorators.
    enable_schema_enforcement
        Whether to enforce schema casts post-compute.
    enable_canonicalization
        Whether to apply canonical ordering/post-processing.
    enable_value_clipping
        Whether to apply numeric clipping transforms.
    max_loc_clip
        Upper bound for loc clipping.
    null_policy
        Null handling policy for cleaning steps.
    variant_fingerprint
        Deterministic hash of the normalized variant configuration.
    """

    df_backend: DataFrameBackend = "polars"
    clean_mode: CleanMode = "lenient"
    feature_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    enable_common_columns: bool = True
    enable_schema_enforcement: bool = True
    enable_canonicalization: bool = True
    enable_value_clipping: bool = True
    max_loc_clip: int = 10_000
    null_policy: NullPolicy = "preserve"
    variant_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        for table_key, ops in self.feature_sets.items():
            normalized[table_key] = tuple(sorted(str(op) for op in ops))
        object.__setattr__(self, "feature_sets", normalized)
        object.__setattr__(self, "variant_fingerprint", self._compute_fingerprint())

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> VariantConfig:
        """Build a VariantConfig from a raw mapping.

        Parameters
        ----------
        data
            Mapping containing variant configuration keys.

        Returns
        -------
        VariantConfig
            Parsed variant configuration.
        """
        df_backend = str(data.get("df_backend", "polars"))
        clean_mode = str(data.get("clean_mode", "lenient"))
        raw_feature_sets = data.get("feature_sets", {})
        feature_sets: dict[str, tuple[str, ...]] = {}
        if isinstance(raw_feature_sets, Mapping):
            for table_key, ops in raw_feature_sets.items():
                if ops is None:
                    feature_sets[str(table_key)] = tuple()
                elif isinstance(ops, (list, tuple)):
                    feature_sets[str(table_key)] = tuple(str(op) for op in ops)
                else:
                    msg = f"feature_sets[{table_key!s}] must be a sequence"
                    raise TypeError(msg)
        else:
            msg = "feature_sets must be a mapping"
            raise TypeError(msg)
        return cls(
            df_backend=df_backend,
            clean_mode=clean_mode,
            feature_sets=feature_sets,
            enable_common_columns=bool(data.get("enable_common_columns", True)),
            enable_schema_enforcement=bool(data.get("enable_schema_enforcement", True)),
            enable_canonicalization=bool(data.get("enable_canonicalization", True)),
            enable_value_clipping=bool(data.get("enable_value_clipping", True)),
            max_loc_clip=int(data.get("max_loc_clip", 10_000)),
            null_policy=str(data.get("null_policy", "preserve")),
        )

    def validate(
        self,
        *,
        allowed_ops: Mapping[str, set[str]] | None = None,
    ) -> VariantConfig:
        """Validate variant configuration invariants.

        Parameters
        ----------
        allowed_ops
            Optional mapping of table keys to allowed column-op names.

        Returns
        -------
        VariantConfig
            Validated configuration (self).

        Raises
        ------
        ValueError
            If any configuration invariant is violated.
        """
        if self.df_backend not in _ALLOWED_BACKENDS:
            msg = f"Unsupported df_backend={self.df_backend!r}"
            raise ValueError(msg)
        if self.clean_mode not in _ALLOWED_CLEAN_MODES:
            msg = f"Unsupported clean_mode={self.clean_mode!r}"
            raise ValueError(msg)
        if self.null_policy not in _ALLOWED_NULL_POLICIES:
            msg = f"Unsupported null_policy={self.null_policy!r}"
            raise ValueError(msg)
        if self.max_loc_clip <= 0:
            msg = "max_loc_clip must be positive"
            raise ValueError(msg)

        for table_key, ops in self.feature_sets.items():
            if not ops:
                continue
            if len(set(ops)) != len(ops):
                msg = f"Duplicate feature ops for table {table_key}"
                raise ValueError(msg)
            if allowed_ops is None:
                continue
            allowed = allowed_ops.get(table_key)
            if allowed is None:
                msg = f"Unknown feature set table key {table_key}"
                raise ValueError(msg)
            invalid = sorted(op for op in ops if op not in allowed)
            if invalid:
                msg = f"Invalid feature ops for {table_key}: {invalid}"
                raise ValueError(msg)
        return self

    def as_hamilton_config(self) -> dict[str, object]:
        """Return configuration keys used by resolve_from_config.

        Returns
        -------
        dict[str, object]
            Mapping of configuration keys to values.
        """
        return {
            "df_backend": self.df_backend,
            "clean_mode": self.clean_mode,
            "feature_sets": self.feature_sets,
            "enable_common_columns": self.enable_common_columns,
            "enable_schema_enforcement": self.enable_schema_enforcement,
            "enable_canonicalization": self.enable_canonicalization,
            "enable_value_clipping": self.enable_value_clipping,
            "max_loc_clip": self.max_loc_clip,
            "null_policy": self.null_policy,
        }

    def _compute_fingerprint(self) -> str:
        payload = {
            "df_backend": self.df_backend,
            "clean_mode": self.clean_mode,
            "feature_sets": {
                key: list(self.feature_sets[key]) for key in sorted(self.feature_sets)
            },
            "enable_common_columns": self.enable_common_columns,
            "enable_schema_enforcement": self.enable_schema_enforcement,
            "enable_canonicalization": self.enable_canonicalization,
            "enable_value_clipping": self.enable_value_clipping,
            "max_loc_clip": self.max_loc_clip,
            "null_policy": self.null_policy,
        }
        return fingerprint(payload)


__all__ = [
    "CleanMode",
    "DataFrameBackend",
    "FeatureSetName",
    "NullPolicy",
    "VariantConfig",
]
