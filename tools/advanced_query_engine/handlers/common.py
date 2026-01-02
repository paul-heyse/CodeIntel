"""Shared handler helpers."""

from __future__ import annotations

from tools.advanced_query_engine.packs.catalog import PackCatalog


def default_rpygrep_preset() -> dict[str, object]:
    """Return a safe default rpygrep preset when none is configured.

    Returns
    -------
    dict[str, object]
        Default rpygrep preset payload.
    """
    return {
        "preset_id": "rg.default_interactive",
        "description": "Fallback preset for interactive searches.",
        "options": {
            "case_sensitive": False,
            "patterns_are_not_regex": False,
            "auto_hybrid_regex": True,
            "before_context": 1,
            "after_context": 1,
            "max_count": 200,
            "max_file_size_bytes": 2_097_152,
            "exclude_globs": [
                "**/.git/**",
                "**/node_modules/**",
                "**/.venv/**",
                "**/dist/**",
                "**/build/**",
            ],
            "extra_args": ["--no-config"],
            "as_json": True,
        },
    }


def load_rpygrep_preset(catalog: PackCatalog, preset_id: str) -> dict[str, object]:
    """Load a rpygrep preset by id, falling back to defaults.

    Parameters
    ----------
    catalog:
        Pack catalog to query.
    preset_id:
        Preset identifier to load.

    Returns
    -------
    dict[str, object]
        Preset payload, or the default preset if not found.
    """
    try:
        return catalog.preset(preset_id)
    except ValueError:
        return default_rpygrep_preset()


__all__ = ["default_rpygrep_preset", "load_rpygrep_preset"]
