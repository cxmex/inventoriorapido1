-- ============================================================
-- YEARLY FORECAST — Run this in Supabase SQL Editor
-- ============================================================

DROP FUNCTION IF EXISTS get_monthly_sales_by_branch();

CREATE OR REPLACE FUNCTION get_monthly_sales_by_branch()
RETURNS TABLE(
    month_date TEXT,
    year_num INTEGER,
    month_num INTEGER,
    t1_revenue DOUBLE PRECISION,
    t1_qty BIGINT,
    t2_revenue DOUBLE PRECISION,
    t2_qty BIGINT,
    total_revenue DOUBLE PRECISION,
    total_qty BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH t1_clean AS (
        SELECT fecha::text AS fecha_txt, price, qty
        FROM ventas_terex1
        WHERE fecha IS NOT NULL
          AND fecha::text ~ '^\d{4}-\d{2}-\d{2}'
    ),
    t2_clean AS (
        SELECT fecha::text AS fecha_txt, price, qty
        FROM ventas_terex2
        WHERE fecha IS NOT NULL
          AND fecha::text ~ '^\d{4}-\d{2}-\d{2}'
    ),
    t1 AS (
        SELECT
            TO_CHAR(fecha_txt::date, 'YYYY-MM') AS ym,
            EXTRACT(YEAR FROM fecha_txt::date)::INTEGER AS yr,
            EXTRACT(MONTH FROM fecha_txt::date)::INTEGER AS mn,
            COALESCE(SUM(price * qty), 0)::DOUBLE PRECISION AS revenue,
            COALESCE(SUM(qty), 0)::BIGINT AS units
        FROM t1_clean
        GROUP BY ym, yr, mn
    ),
    t2 AS (
        SELECT
            TO_CHAR(fecha_txt::date, 'YYYY-MM') AS ym,
            EXTRACT(YEAR FROM fecha_txt::date)::INTEGER AS yr,
            EXTRACT(MONTH FROM fecha_txt::date)::INTEGER AS mn,
            COALESCE(SUM(price * qty), 0)::DOUBLE PRECISION AS revenue,
            COALESCE(SUM(qty), 0)::BIGINT AS units
        FROM t2_clean
        GROUP BY ym, yr, mn
    ),
    all_months AS (
        SELECT DISTINCT ym, yr, mn FROM t1
        UNION
        SELECT DISTINCT ym, yr, mn FROM t2
    )
    SELECT
        am.ym AS month_date,
        am.yr AS year_num,
        am.mn AS month_num,
        COALESCE(t1.revenue, 0)::DOUBLE PRECISION AS t1_revenue,
        COALESCE(t1.units, 0)::BIGINT AS t1_qty,
        COALESCE(t2.revenue, 0)::DOUBLE PRECISION AS t2_revenue,
        COALESCE(t2.units, 0)::BIGINT AS t2_qty,
        (COALESCE(t1.revenue, 0) + COALESCE(t2.revenue, 0))::DOUBLE PRECISION AS total_revenue,
        (COALESCE(t1.units, 0) + COALESCE(t2.units, 0))::BIGINT AS total_qty
    FROM all_months am
    LEFT JOIN t1 ON am.ym = t1.ym
    LEFT JOIN t2 ON am.ym = t2.ym
    ORDER BY am.ym;
END;
$$ LANGUAGE plpgsql;
