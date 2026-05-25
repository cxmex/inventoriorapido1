-- ============================================================
-- Natural Language Query — logging + feedback for RL
-- Run in Supabase SQL Editor
-- ============================================================

CREATE TABLE IF NOT EXISTS nl_query_logs (
    id          BIGSERIAL PRIMARY KEY,
    query       TEXT NOT NULL,                          -- raw user input
    query_norm  TEXT,                                   -- lowercased / cleaned
    intent      TEXT,                                   -- resolved intent key
    params      JSONB DEFAULT '{}'::jsonb,              -- extracted params
    result_type TEXT,                                   -- text / table / chart / combo / unknown
    result_preview TEXT,                                -- first 500 chars of response text
    feedback    SMALLINT DEFAULT 0,                     -- -1 = bad, 0 = none, 1 = good
    feedback_comment TEXT,                              -- optional user note on why it was bad
    response_ms INTEGER,                                -- how long the query took
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_nl_logs_created   ON nl_query_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nl_logs_intent    ON nl_query_logs (intent);
CREATE INDEX IF NOT EXISTS idx_nl_logs_feedback  ON nl_query_logs (feedback);

-- ── Insert a log row and return its id ──────────────────────
CREATE OR REPLACE FUNCTION nl_log_query(
    p_query       TEXT,
    p_query_norm  TEXT,
    p_intent      TEXT,
    p_params      JSONB,
    p_result_type TEXT,
    p_result_preview TEXT,
    p_response_ms INTEGER
) RETURNS BIGINT
LANGUAGE sql AS $$
    INSERT INTO nl_query_logs
        (query, query_norm, intent, params, result_type, result_preview, response_ms)
    VALUES
        (p_query, p_query_norm, p_intent, p_params, p_result_type, p_result_preview, p_response_ms)
    RETURNING id;
$$;

-- ── Record feedback on a query ──────────────────────────────
CREATE OR REPLACE FUNCTION nl_log_feedback(
    p_id       BIGINT,
    p_feedback SMALLINT,
    p_comment  TEXT DEFAULT NULL
) RETURNS VOID
LANGUAGE sql AS $$
    UPDATE nl_query_logs
    SET feedback = p_feedback,
        feedback_comment = COALESCE(p_comment, feedback_comment)
    WHERE id = p_id;
$$;

-- ── Get recent logs for admin review ────────────────────────
CREATE OR REPLACE FUNCTION nl_get_logs(
    p_limit  INTEGER DEFAULT 100,
    p_offset INTEGER DEFAULT 0
) RETURNS SETOF nl_query_logs
LANGUAGE sql STABLE AS $$
    SELECT * FROM nl_query_logs
    ORDER BY created_at DESC
    LIMIT p_limit OFFSET p_offset;
$$;

-- ── Aggregated stats: intent accuracy ───────────────────────
CREATE OR REPLACE FUNCTION nl_get_stats()
RETURNS TABLE(
    intent        TEXT,
    total         BIGINT,
    thumbs_up     BIGINT,
    thumbs_down   BIGINT,
    no_feedback   BIGINT,
    accuracy_pct  NUMERIC
)
LANGUAGE sql STABLE AS $$
    SELECT
        intent,
        COUNT(*)                                       AS total,
        COUNT(*) FILTER (WHERE feedback = 1)           AS thumbs_up,
        COUNT(*) FILTER (WHERE feedback = -1)          AS thumbs_down,
        COUNT(*) FILTER (WHERE feedback = 0)           AS no_feedback,
        ROUND(
            COUNT(*) FILTER (WHERE feedback = 1) * 100.0
            / NULLIF(COUNT(*) FILTER (WHERE feedback != 0), 0),
            1
        )                                              AS accuracy_pct
    FROM nl_query_logs
    GROUP BY intent
    ORDER BY total DESC;
$$;

-- ── Missed queries: unknown intent or thumbs down ───────────
CREATE OR REPLACE FUNCTION nl_get_missed_queries(
    p_limit INTEGER DEFAULT 50
) RETURNS SETOF nl_query_logs
LANGUAGE sql STABLE AS $$
    SELECT * FROM nl_query_logs
    WHERE intent = 'unknown' OR feedback = -1
    ORDER BY created_at DESC
    LIMIT p_limit;
$$;
