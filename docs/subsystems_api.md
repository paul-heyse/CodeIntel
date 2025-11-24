Subsystem data access cheatsheet
===============================

What changed
------------
- New architecture/subsystem views in DuckDB: `docs.v_subsystem_summary`, `docs.v_module_with_subsystem`, `docs.v_ide_hints`.
- FastAPI endpoints:
  - `GET /architecture/subsystems?limit=&role=&q=` – list subsystems with name/description/role filters.
  - `GET /architecture/subsystem?subsystem_id=` – detail + modules for a subsystem.
  - `GET /architecture/module-subsystems?module=` – memberships for a module.
  - `GET /ide/hints?rel_path=` – IDE-friendly hints for a file (module + subsystem context).
- MCP tools:
  - `list_subsystems(limit=50, role=None, q=None)`
  - `get_subsystem_modules(subsystem_id)`
  - `get_module_subsystems(module)`
  - `get_file_hints(rel_path)`
- CLI helpers:
  - `codeintel subsystem list --repo <slug> --commit <sha> --db-path <db> [--role ROLE] [--q TEXT] [--limit N]`
  - `codeintel subsystem show --repo ... --commit ... --db-path ... --subsystem-id <id>`
  - `codeintel subsystem module-memberships --repo ... --commit ... --db-path ... --module <pkg.mod>`
  - `codeintel ide hints --repo ... --commit ... --db-path ... --rel-path <path>`

Field expectations
------------------
- Subsystem rows surface: `subsystem_id`, `name`, `description`, `module_count`, `risk_level`, `entrypoints_json`.
- Module-subsystem rows surface: `module`, `role`, `import_fan_in`, `import_fan_out`, `symbol_fan_in`, `symbol_fan_out`, `subsystem_name`, `subsystem_risk_level`.
- IDE hints rows include: `rel_path`, `module`, subsystem name/role/description, import fan-in/out, symbol fan-in/out, coverage/risk summary if available.

Heuristics (naming/roles)
-------------------------
- Names are derived from common module prefixes; if none, we prefix with the dominant role (from tags or module segments).
- Roles are inferred from tags and module segments using the ROLE_TAGS map (`api`, `core`, `infra`, `platform`, `data`, `ml`, `cli`, `tests`, `service`, `endpoint`, etc.).

UI wiring tips
--------------
- Subsystem list view: show `name` with role badge, `module_count`, `risk_level`, and `description`; sort by `module_count` then `risk_level`.
- Subsystem detail view: show modules with `role` (highlight when different from dominant), import fan-in/out, and risk; surface entrypoints from `entrypoints_json`.
- Module detail: render memberships from `/architecture/module-subsystems` with subsystem name/role/risk and link to subsystem detail.
- Search/filter: pass `role` and `q` to `/architecture/subsystems` (or MCP `list_subsystems`) to filter by role and name/description substring.
- Graph overlays: color or group modules by subsystem name/role using `v_module_with_subsystem`; use `subsystem_description` for tooltips.
