"""Pack catalog loading and resolution utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tools.advanced_query_engine.contracts import JSONValue

try:
    import yaml  # type: ignore[import-not-found]
except ModuleNotFoundError:
    yaml = None


@dataclass(frozen=True)
class PackCatalog:
    """Catalog of pack files resolved from a pack root."""

    root: Path
    presets: dict[str, Path]
    pattern_groups: dict[str, Path]
    ast_grep_rules: dict[str, Path]
    tree_sitter_packs: dict[str, Path]
    wiring_packs: dict[str, Path]

    @staticmethod
    def load_json(path: Path) -> dict[str, JSONValue]:
        """Load a JSON file into a JSONValue dict.

        Returns
        -------
        dict[str, JSONValue]
            Parsed JSON payload.
        """
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_yaml(path: Path) -> dict[str, JSONValue]:
        """Load a YAML file into a JSONValue dict.

        Returns
        -------
        dict[str, JSONValue]
            Parsed YAML payload.

        Raises
        ------
        RuntimeError
            If PyYAML is not installed.
        """
        if yaml is None:
            msg = "PyYAML is required to load YAML pack files."
            raise RuntimeError(msg)
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def preset(self, preset_id: str) -> dict[str, JSONValue]:
        """Return the preset payload for a preset id.

        Returns
        -------
        dict[str, JSONValue]
            Preset payload.

        Raises
        ------
        ValueError
            If the preset id is unknown.
        """
        path = self.presets.get(preset_id)
        if path is None:
            msg = f"Unknown preset_id: {preset_id}"
            raise ValueError(msg)
        return self.load_json(path)

    def pattern_group(self, group_id: str) -> dict[str, JSONValue]:
        """Return the pattern group payload for a group id.

        Returns
        -------
        dict[str, JSONValue]
            Pattern group payload.

        Raises
        ------
        ValueError
            If the pattern group id is unknown.
        """
        path = self.pattern_groups.get(group_id)
        if path is None:
            msg = f"Unknown pattern_group_id: {group_id}"
            raise ValueError(msg)
        return self.load_json(path)

    def ast_grep_rule_pack(self, pack_id: str) -> dict[str, JSONValue]:
        """Return the ast-grep rule pack payload for a pack id.

        Returns
        -------
        dict[str, JSONValue]
            Rule pack payload.

        Raises
        ------
        ValueError
            If the pack id is unknown.
        """
        path = self.ast_grep_rules.get(pack_id)
        if path is None:
            msg = f"Unknown ast_grep pack id: {pack_id}"
            raise ValueError(msg)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return self.load_yaml(path)
        return self.load_json(path)

    def tree_sitter_pack(self, pack_id: str) -> str:
        """Return the query text for a tree-sitter pack id.

        Returns
        -------
        str
            Tree-sitter query text.

        Raises
        ------
        ValueError
            If the pack id is unknown.
        """
        path = self.tree_sitter_packs.get(pack_id)
        if path is None:
            msg = f"Unknown tree_sitter pack id: {pack_id}"
            raise ValueError(msg)
        return path.read_text(encoding="utf-8")

    def wiring_pack(self, pack_id: str) -> dict[str, JSONValue]:
        """Return the wiring pack payload for a pack id.

        Returns
        -------
        dict[str, JSONValue]
            Wiring pack payload.

        Raises
        ------
        ValueError
            If the pack id is unknown.
        """
        path = self.wiring_packs.get(pack_id)
        if path is None:
            msg = f"Unknown wiring pack id: {pack_id}"
            raise ValueError(msg)
        return self.load_json(path)


@dataclass
class _PackMaps:
    presets: dict[str, Path]
    pattern_groups: dict[str, Path]
    ast_grep_rules: dict[str, Path]
    tree_sitter_packs: dict[str, Path]
    wiring_packs: dict[str, Path]


def _resolve_under_root(root: Path, rel: str) -> Path:
    resolved = (root / rel).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        msg = f"Path escapes pack root: {rel}"
        raise ValueError(msg)
    return resolved


def _load_json(path: Path) -> dict[str, JSONValue]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(root: Path) -> dict[str, JSONValue] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    return _load_json(manifest_path)


def build_pack_catalog(root: Path) -> PackCatalog:
    """Build a pack catalog from a root directory.

    Returns
    -------
    PackCatalog
        Pack catalog resolved from the root directory.
    """
    manifest = _load_manifest(root)
    maps = _PackMaps(
        presets={},
        pattern_groups={},
        ast_grep_rules={},
        tree_sitter_packs={},
        wiring_packs={},
    )

    if manifest:
        _populate_from_manifest(root, manifest, maps)
    else:
        _scan_default_dirs(root, maps)

    return PackCatalog(
        root=root,
        presets=maps.presets,
        pattern_groups=maps.pattern_groups,
        ast_grep_rules=maps.ast_grep_rules,
        tree_sitter_packs=maps.tree_sitter_packs,
        wiring_packs=maps.wiring_packs,
    )


def _populate_from_manifest(root: Path, manifest: dict[str, JSONValue], maps: _PackMaps) -> None:
    _register_presets(root, manifest.get("rpygrep_presets") or [], maps)
    _register_ast_grep(root, manifest, maps)
    _register_tree_sitter(root, manifest, maps)
    _register_wiring(root, manifest.get("wiring_packs") or [], maps)
    _register_pattern_groups(root, maps)


def _register_presets(root: Path, entries: object, maps: _PackMaps) -> None:
    for entry in _coerce_list(entries):
        if isinstance(entry, dict):
            preset_id = str(entry.get("preset_id"))
            rel_path = str(entry.get("file"))
        else:
            rel_path = str(entry)
            preset_id = _preset_id_from_file(_resolve_under_root(root, rel_path))
        maps.presets[preset_id] = _resolve_under_root(root, rel_path)


def _register_ast_grep(root: Path, manifest: dict[str, JSONValue], maps: _PackMaps) -> None:
    entries = manifest.get("ast_grep_packs") or manifest.get("ast_grep_rule_files") or []
    for entry in _coerce_list(entries):
        if isinstance(entry, dict):
            pack_id = str(entry.get("pack_id"))
            rel_path = str(entry.get("file"))
        else:
            rel_path = str(entry)
            pack_id = _ast_grep_pack_id_from_file(_resolve_under_root(root, rel_path))
        maps.ast_grep_rules[pack_id] = _resolve_under_root(root, rel_path)


def _register_tree_sitter(root: Path, manifest: dict[str, JSONValue], maps: _PackMaps) -> None:
    entries = manifest.get("tree_sitter_packs") or []
    for entry in _coerce_list(entries):
        if isinstance(entry, dict):
            pack_id = str(entry.get("pack_id"))
            rel_path = str(entry.get("file"))
        else:
            rel_path = str(entry)
            pack_id = Path(rel_path).stem
        maps.tree_sitter_packs[pack_id] = _resolve_under_root(root, rel_path)


def _register_wiring(root: Path, entries: object, maps: _PackMaps) -> None:
    for entry in _coerce_list(entries):
        if isinstance(entry, dict):
            pack_id = str(entry.get("pack_id"))
            rel_path = str(entry.get("file"))
        else:
            rel_path = str(entry)
            pack_id = _wiring_pack_id_from_file(_resolve_under_root(root, rel_path))
        maps.wiring_packs[pack_id] = _resolve_under_root(root, rel_path)


def _register_pattern_groups(root: Path, maps: _PackMaps) -> None:
    pattern_root = root / "rpygrep" / "patterns"
    if not pattern_root.exists():
        return
    for path in sorted(pattern_root.glob("*.json")):
        group_id = _pattern_group_id_from_file(path)
        maps.pattern_groups[group_id] = path


def _scan_default_dirs(root: Path, maps: _PackMaps) -> None:
    preset_root = root / "rpygrep" / "presets"
    if preset_root.exists():
        for path in sorted(preset_root.glob("*.json")):
            preset_id = _preset_id_from_file(path)
            maps.presets[preset_id] = path

    _register_pattern_groups(root, maps)

    ast_root = root / "ast_grep"
    if ast_root.exists():
        for path in sorted(ast_root.rglob("*.y*ml")):
            pack_id = _ast_grep_pack_id_from_file(path)
            maps.ast_grep_rules[pack_id] = path

    ts_root = root / "tree_sitter"
    if ts_root.exists():
        for path in sorted(ts_root.rglob("*.scm")):
            pack_id = f"{path.parent.name}.{path.stem}"
            maps.tree_sitter_packs[pack_id] = path

    wiring_root = root / "wiring_packs"
    if wiring_root.exists():
        for path in sorted(wiring_root.rglob("*.json")):
            pack_id = _wiring_pack_id_from_file(path)
            maps.wiring_packs[pack_id] = path


def _coerce_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _preset_id_from_file(path: Path) -> str:
    payload = _load_json(path)
    preset_id = payload.get("preset_id")
    if isinstance(preset_id, str) and preset_id:
        return preset_id
    return path.stem


def _pattern_group_id_from_file(path: Path) -> str:
    payload = _load_json(path)
    group_id = payload.get("pattern_group_id")
    if isinstance(group_id, str) and group_id:
        return group_id
    return path.stem


def _ast_grep_pack_id_from_file(path: Path) -> str:
    payload = _load_json(path) if yaml is None else yaml.safe_load(path.read_text(encoding="utf-8"))
    pack_id = payload.get("pack_id") if isinstance(payload, dict) else None
    if isinstance(pack_id, str) and pack_id:
        return pack_id
    return path.stem


def _wiring_pack_id_from_file(path: Path) -> str:
    payload = _load_json(path)
    pack_id = payload.get("pack_id") if isinstance(payload, dict) else None
    if isinstance(pack_id, str) and pack_id:
        return pack_id
    return path.stem


__all__ = ["PackCatalog", "build_pack_catalog"]
