"""
Parameterized SQL templates for the CodeIntel server.

The mappings here drive MPC/CLI queries for summaries, coverage gaps, and
graph neighborhoods against the DuckDB views under `docs.*`.
"""

# src/codeintel_core/ai/query_templates.py

QUERY_TEMPLATES = {
    # 1) Basic: look up a function summary by its GOID URN.
    "function_summary_by_urn": """
        SELECT *
        FROM docs.v_function_summary
        WHERE urn = :urn;
    """,
    # 2) Look up a function by file + qualified name (e.g., for a code lens).
    "function_summary_by_path_and_qualname": """
        SELECT *
        FROM docs.v_function_summary
        WHERE rel_path = :rel_path
          AND qualname = :qualname;
    """,
    # 3) Global top-N high-risk functions.
    "top_high_risk_functions": """
        SELECT *
        FROM docs.v_function_summary
        WHERE risk_level IN ('high', 'medium')
        ORDER BY risk_score DESC
        LIMIT :limit;
    """,
    # 4) High-risk functions within a specific file.
    "high_risk_functions_in_file": """
        SELECT *
        FROM docs.v_function_summary
        WHERE rel_path = :rel_path
          AND risk_level IN ('high', 'medium')
        ORDER BY risk_score DESC;
    """,
    # 5) Untested or very low coverage functions in a file.
    "untested_functions_in_file": """
        SELECT *
        FROM docs.v_function_summary
        WHERE rel_path = :rel_path
          AND (
                tested = FALSE
                OR coverage_ratio IS NULL
                OR coverage_ratio < :min_coverage
          )
        ORDER BY coverage_ratio ASC NULLS FIRST;
    """,
    # 6) Untested/low coverage functions for a module (via modules path).
    "untested_functions_in_module": """
        SELECT f.*
        FROM docs.v_function_summary f
        JOIN core.modules m
          ON f.rel_path = m.path
        WHERE m.module = :module
          AND (
                f.tested = FALSE
                OR f.coverage_ratio IS NULL
                OR f.coverage_ratio < :min_coverage
          )
        ORDER BY f.coverage_ratio ASC NULLS FIRST;
    """,
    # 7) Call graph neighborhood around a function (one hop).
    # direction: 'incoming', 'outgoing', or 'both'
    "callgraph_neighbors": """
        SELECT *
        FROM docs.v_call_graph_enriched
        WHERE
          (:direction = 'outgoing' AND caller_goid_h128 = :goid_h128)
          OR (:direction = 'incoming' AND callee_goid_h128 = :goid_h128)
          OR (:direction = 'both' AND (caller_goid_h128 = :goid_h128
                                       OR callee_goid_h128 = :goid_h128));
    """,
    # 8) Tests that exercise a function (by GOID).
    "tests_for_function": """
        SELECT *
        FROM docs.v_test_to_function
        WHERE function_goid_h128 = :goid_h128
        ORDER BY test_status DESC, coverage_ratio DESC;
    """,
    # 9) Functions impacted by changes to a given file:
    #   - Functions defined in this file
    #   - Plus their direct callers and callees.
    "impact_analysis_one_hop": """
        WITH changed_functions AS (
            SELECT function_goid_h128
            FROM docs.v_function_summary
            WHERE rel_path = :rel_path
        ),
        neighbors AS (
            SELECT
                e.caller_goid_h128 AS src_goid_h128,
                e.callee_goid_h128 AS dst_goid_h128
            FROM docs.v_call_graph_enriched e
            JOIN changed_functions c
              ON e.caller_goid_h128 = c.function_goid_h128
              OR e.callee_goid_h128 = c.function_goid_h128
        ),
        impacted AS (
            SELECT src_goid_h128 AS goid_h128 FROM neighbors
            UNION
            SELECT dst_goid_h128 AS goid_h128 FROM neighbors
        )
        SELECT f.*
        FROM docs.v_function_summary f
        JOIN impacted i
          ON f.function_goid_h128 = i.goid_h128
        ORDER BY f.risk_score DESC;
    """,
    # 10) Files with the most high-risk functions.
    "files_with_most_high_risk_functions": """
        SELECT
            rel_path,
            function_count,
            high_risk_functions,
            medium_risk_functions,
            low_risk_functions,
            max_risk_score
        FROM docs.v_file_summary
        WHERE high_risk_functions > 0
        ORDER BY high_risk_functions DESC, max_risk_score DESC
        LIMIT :limit;
    """,
    # 11) Fetch a full function profile by GOID.
    "function_profile_by_goid": """
        SELECT *
        FROM docs.v_function_profile
        WHERE repo = :repo
          AND commit = :commit
          AND function_goid_h128 = :function_goid_h128;
    """,
    # 12) Fetch a file profile by relative path.
    "file_profile_by_path": """
        SELECT *
        FROM docs.v_file_profile
        WHERE repo = :repo
          AND commit = :commit
          AND rel_path = :rel_path;
    """,
    # 13) Fetch a module profile by module name.
    "module_profile_by_name": """
        SELECT *
        FROM docs.v_module_profile
        WHERE repo = :repo
          AND commit = :commit
          AND module = :module;
    """,
}
