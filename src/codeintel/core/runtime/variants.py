"""Variant configuration for Hamilton DAG composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, cast

from codeintel.core.hashing.fingerprint import fingerprint

DataFrameBackend = Literal["polars_lazy"]
CleanMode = Literal["off", "lenient", "strict"]
NullPolicy = Literal["preserve", "drop_bad_rows"]
FeatureSetName = str

_ALLOWED_BACKENDS: set[str] = {"polars_lazy"}
_ALLOWED_CLEAN_MODES: set[str] = {"off", "lenient", "strict"}
_ALLOWED_NULL_POLICIES: set[str] = {"preserve", "drop_bad_rows"}


def _validate_choice(value: str, allowed: set[str], label: str) -> None:
    if value not in allowed:
        msg = f"Unsupported {label}={value!r}"
        raise ValueError(msg)


def _validate_max_loc_clip(max_loc_clip: int) -> None:
    if max_loc_clip <= 0:
        msg = "max_loc_clip must be positive"
        raise ValueError(msg)


def _coerce_choice(
    value: object | None,
    *,
    default: str,
    allowed: set[str],
    label: str,
) -> str:
    if value is None:
        resolved = default
    elif isinstance(value, str):
        resolved = value
    else:
        msg = f"{label} must be a string"
        raise TypeError(msg)
    _validate_choice(resolved, allowed, label)
    return resolved


def _coerce_int(value: object | None, *, default: int, label: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        msg = f"{label} must be an int"
        raise TypeError(msg)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            msg = f"{label} must be an int"
            raise ValueError(msg) from exc
    msg = f"{label} must be an int"
    raise TypeError(msg)


def _validate_feature_sets(
    feature_sets: Mapping[str, tuple[str, ...]],
    allowed_ops: Mapping[str, set[str]] | None,
) -> None:
    for table_key, ops in feature_sets.items():
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

    df_backend: DataFrameBackend = "polars_lazy"
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
        """Normalize feature sets and compute the variant fingerprint."""
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

        Raises
        ------
        TypeError
            If feature_sets or its entries are not valid sequences/mappings.
        """
        df_backend_raw = _coerce_choice(
            data.get("df_backend"),
            default="polars_lazy",
            allowed=_ALLOWED_BACKENDS,
            label="df_backend",
        )
        clean_mode_raw = _coerce_choice(
            data.get("clean_mode"),
            default="lenient",
            allowed=_ALLOWED_CLEAN_MODES,
            label="clean_mode",
        )
        raw_feature_sets = data.get("feature_sets", {})
        feature_sets: dict[str, tuple[str, ...]] = {}
        if isinstance(raw_feature_sets, Mapping):
            for table_key, ops in raw_feature_sets.items():
                if ops is None:
                    feature_sets[str(table_key)] = ()
                elif isinstance(ops, (list, tuple)):
                    feature_sets[str(table_key)] = tuple(str(op) for op in ops)
                else:
                    msg = f"feature_sets[{table_key!s}] must be a sequence"
                    raise TypeError(msg)
        else:
            msg = "feature_sets must be a mapping"
            raise TypeError(msg)
        max_loc_clip = _coerce_int(
            data.get("max_loc_clip"),
            default=10_000,
            label="max_loc_clip",
        )
        null_policy_raw = _coerce_choice(
            data.get("null_policy"),
            default="preserve",
            allowed=_ALLOWED_NULL_POLICIES,
            label="null_policy",
        )

        return cls(
            df_backend=cast("DataFrameBackend", df_backend_raw),
            clean_mode=cast("CleanMode", clean_mode_raw),
            feature_sets=feature_sets,
            enable_common_columns=bool(data.get("enable_common_columns", True)),
            enable_schema_enforcement=bool(data.get("enable_schema_enforcement", True)),
            enable_canonicalization=bool(data.get("enable_canonicalization", True)),
            enable_value_clipping=bool(data.get("enable_value_clipping", True)),
            max_loc_clip=max_loc_clip,
            null_policy=cast("NullPolicy", null_policy_raw),
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

        """
        _validate_choice(self.df_backend, _ALLOWED_BACKENDS, "df_backend")
        _validate_choice(self.clean_mode, _ALLOWED_CLEAN_MODES, "clean_mode")
        _validate_choice(self.null_policy, _ALLOWED_NULL_POLICIES, "null_policy")
        _validate_max_loc_clip(self.max_loc_clip)
        _validate_feature_sets(self.feature_sets, allowed_ops)
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
