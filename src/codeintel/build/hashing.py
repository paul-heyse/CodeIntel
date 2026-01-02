"""Hash helpers for build configuration and options."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import orjson

from codeintel.build.parameters import TargetParameters


def compute_options_hash(options: object | None) -> str | None:
    """Compute hash of implementation configuration options.

    Serializes the options to JSON and hashes the result. This allows
    detecting when implementation configuration has changed.

    Parameters
    ----------
    options
        Options object (must be JSON-serializable).
        Returns None if options is None.

    Returns
    -------
    str | None
        16-character hex hash string, or None if no options.

    Examples
    --------
    >>> compute_options_hash({"threshold": 0.5})
    '7b226e616d65223a...'
    >>> compute_options_hash(None) is None
    True
    """
    if options is None:
        return None

    try:
        serialized = orjson.dumps(options, option=orjson.OPT_SORT_KEYS, default=str)
    except TypeError:
        serialized = str(options).encode("utf-8")

    hasher = hashlib.sha256()
    hasher.update(serialized)
    return hasher.hexdigest()[:16]


def compute_options_hash_for_parameters(params: TargetParameters) -> str | None:
    """Compute options hash for a TargetParameters instance.

    Parameters
    ----------
    params
        TargetParameters to hash.

    Returns
    -------
    str | None
        16-character hex hash string, or None when parameters are empty.
    """
    if len(params) == 0:
        return None
    return compute_options_hash(params.as_dict())


def compute_target_options_hash(
    options: TargetParameters | Mapping[str, object] | None,
) -> str | None:
    """Compute options hash for target configuration parameters.

    Parameters
    ----------
    options
        TargetParameters or mapping of configuration values.

    Returns
    -------
    str | None
        16-character hex hash string, or None when options are empty.
    """
    if options is None:
        return None
    if isinstance(options, TargetParameters):
        return compute_options_hash_for_parameters(options)
    if len(options) == 0:
        return None
    return compute_options_hash(options)


__all__ = [
    "compute_options_hash",
    "compute_options_hash_for_parameters",
    "compute_target_options_hash",
]
