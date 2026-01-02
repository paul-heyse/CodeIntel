from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class PackPaths:
    pack_root: Path

    def resolve(self, rel: str) -> Path:
        p = (self.pack_root / rel).resolve()
        if not str(p).startswith(str(self.pack_root.resolve())):
            raise ValueError(f"Refuses to resolve outside pack_root: {rel}")
        return p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load .yaml rule packs; install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_preset(pack_paths: PackPaths, preset_id: str) -> Dict[str, Any]:
    # preset_id is a logical id; we store presets under rpygrep/presets/*.json
    preset_map = {
        "rg.default_interactive": "rpygrep/presets/default_interactive.json",
        "rg.audit_deterministic": "rpygrep/presets/audit_deterministic.json",
    }
    if preset_id not in preset_map:
        raise ValueError(f"Unknown preset_id: {preset_id}")
    return load_json(pack_paths.resolve(preset_map[preset_id]))


def load_pattern_group(pack_paths: PackPaths, rel: str) -> Dict[str, Any]:
    return load_json(pack_paths.resolve(rel))


def load_ast_grep_rules(pack_paths: PackPaths, rel: str) -> Dict[str, Any]:
    p = pack_paths.resolve(rel)
    if p.suffix.lower() in (".yaml", ".yml"):
        return load_yaml(p)
    return load_json(p)
