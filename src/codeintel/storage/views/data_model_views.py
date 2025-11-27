"""Docs views for data models and configuration data flow."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

DATA_MODEL_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_data_models",
    "docs.v_data_model_fields",
    "docs.v_data_model_relationships",
    "docs.v_data_models_normalized",
    "docs.v_data_model_usage",
    "docs.v_config_data_flow",
)


def create_data_model_views(con: DuckDBPyConnection) -> None:
    """Create or replace data model docs views."""
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_models AS
        SELECT
            dm.repo,
            dm.commit,
            dm.model_id,
            dm.goid_h128,
            dm.model_name,
            dm.module,
            dm.rel_path,
            dm.model_kind,
            coalesce(dm.base_classes_json, '[]') AS base_classes_json,
            (
                SELECT to_json(
                           coalesce(
                               list(
                                   struct_pack(
                                       name := f.field_name,
                                       type := f.field_type,
                                       required := f.required,
                                       has_default := f.has_default,
                                       default_expr := f.default_expr,
                                       constraints := f.constraints_json,
                                       source := f.source,
                                       lineno := f.lineno
                                   )
                                   ORDER BY f.field_name
                               ),
                               []
                           )
                       )
                FROM analytics.data_model_fields f
                WHERE f.repo = dm.repo AND f.commit = dm.commit AND f.model_id = dm.model_id
            ) AS fields,
            (
                SELECT to_json(
                           coalesce(
                               list(
                                   struct_pack(
                                       field := r.field_name,
                                       target_model_id := r.target_model_id,
                                       target_model_name := r.target_model_name,
                                       target_module := r.target_module,
                                       multiplicity := r.multiplicity,
                                       kind := r.relationship_kind,
                                       via := r.via,
                                       rel_path := r.rel_path,
                                       lineno := r.lineno,
                                       evidence := r.evidence_json
                                   )
                                   ORDER BY r.field_name
                               ),
                               []
                           )
                       )
                FROM analytics.data_model_relationships r
                WHERE r.repo = dm.repo
                  AND r.commit = dm.commit
                  AND r.source_model_id = dm.model_id
            ) AS relationships,
            dm.doc_short,
            dm.doc_long,
            dm.created_at
        FROM analytics.data_models dm;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_fields AS
        SELECT
            repo,
            commit,
            model_id,
            field_name,
            field_type,
            required,
            has_default,
            default_expr,
            constraints_json,
            source,
            rel_path,
            lineno,
            created_at
        FROM analytics.data_model_fields;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_relationships AS
        SELECT
            repo,
            commit,
            source_model_id,
            target_model_id,
            target_module,
            target_model_name,
            field_name,
            relationship_kind,
            multiplicity,
            via,
            evidence_json,
            rel_path,
            lineno,
            created_at
        FROM analytics.data_model_relationships;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_models_normalized AS
        SELECT
            dm.repo,
            dm.commit,
            dm.model_id,
            dm.goid_h128,
            dm.model_name,
            dm.module,
            dm.rel_path,
            dm.model_kind,
            coalesce(dm.base_classes_json, '[]') AS base_classes_json,
            (SELECT coalesce(list(
                        struct_pack(
                            field_name := f.field_name,
                            field_type := f.field_type,
                            required := f.required,
                            has_default := f.has_default,
                            default_expr := f.default_expr,
                            constraints := f.constraints_json,
                            source := f.source,
                            rel_path := f.rel_path,
                            lineno := f.lineno,
                            created_at := f.created_at
                        )
                        ORDER BY f.field_name
                    )
                    , []
                    )
             FROM analytics.data_model_fields f
             WHERE f.repo = dm.repo AND f.commit = dm.commit AND f.model_id = dm.model_id
            ) AS fields,
            (SELECT coalesce(list(
                        struct_pack(
                            field_name := r.field_name,
                            target_model_id := r.target_model_id,
                            target_module := r.target_module,
                            target_model_name := r.target_model_name,
                            relationship_kind := r.relationship_kind,
                            multiplicity := r.multiplicity,
                            via := r.via,
                            evidence := r.evidence_json,
                            rel_path := r.rel_path,
                            lineno := r.lineno,
                            created_at := r.created_at
                        )
                        ORDER BY r.field_name
                    )
                    , []
                    )
             FROM analytics.data_model_relationships r
             WHERE r.repo = dm.repo AND r.commit = dm.commit AND r.source_model_id = dm.model_id
            ) AS relationships,
            dm.doc_short,
            dm.doc_long,
            dm.created_at
        FROM analytics.data_models dm;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_usage AS
        SELECT
            u.repo,
            u.commit,
            u.model_id,
            dm.model_name,
            dm.model_kind,
            u.function_goid_h128,
            fp.qualname        AS function_qualname,
            fp.rel_path        AS function_rel_path,
            fp.risk_score,
            fp.coverage_ratio,
            u.usage_kinds_json,
            u.context_json,
            u.evidence_json,
            u.created_at
        FROM analytics.data_model_usage u
        LEFT JOIN analytics.data_models dm
          ON dm.repo = u.repo
         AND dm.commit = u.commit
         AND dm.model_id = u.model_id
        LEFT JOIN analytics.function_profile fp
          ON fp.repo = u.repo
         AND fp.commit = u.commit
         AND fp.function_goid_h128 = u.function_goid_h128;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_data_flow AS
        SELECT
            c.repo,
            c.commit,
            c.config_key,
            c.config_path,
            c.function_goid_h128,
            fp.qualname        AS function_qualname,
            fp.rel_path        AS function_rel_path,
            fp.risk_score,
            fp.coverage_ratio,
            c.usage_kind,
            c.evidence_json,
            c.call_chain_id,
            c.call_chain_json,
            c.created_at
        FROM analytics.config_data_flow c
        LEFT JOIN analytics.function_profile fp
          ON fp.repo = c.repo
         AND fp.commit = c.commit
         AND fp.function_goid_h128 = c.function_goid_h128;
        """
    )
