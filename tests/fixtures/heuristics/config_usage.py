"""Fixture module that simulates config access patterns for heuristic tests."""

import os
from collections.abc import MutableMapping
from typing import Any


def config_checks(settings: MutableMapping[str, Any]) -> bool:
    """
    Apply simple feature flag checks to mimic configuration usage.

    Parameters
    ----------
    settings : MutableMapping[str, Any]
        Mutable settings mapping that may hold feature toggles.

    Returns
    -------
    bool
        True when an explicit flag is found; False after defaulting the flag.
    """
    feature = settings.get("feature")
    flag = feature.get("flag") if isinstance(feature, MutableMapping) else None
    if isinstance(flag, bool) and flag:
        return True
    if bool(settings.get("feature.flag")):
        return True
    if os.getenv("FEATURE_FLAG"):
        return True
    settings.setdefault("feature.flag", False)
    settings.update({"feature.flag": True})
    return False
