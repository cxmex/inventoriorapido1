-- ============================================================
-- ML PREDICTION TABLES — Run this in Supabase SQL Editor
-- ============================================================

-- 1. Daily records: single source of truth for all ML models
CREATE TABLE IF NOT EXISTS daily_records (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    modelo TEXT NOT NULL,
    estilo TEXT NOT NULL,
    tienda TEXT NOT NULL DEFAULT 'all',  -- 'Terex 1', 'Terex 2', 'all'

    units_sold_today NUMERIC DEFAULT 0,
    revenue_today NUMERIC DEFAULT 0,
    units_sold_7d NUMERIC DEFAULT 0,
    units_sold_14d NUMERIC DEFAULT 0,
    units_sold_30d NUMERIC DEFAULT 0,
    revenue_7d NUMERIC DEFAULT 0,
    revenue_30d NUMERIC DEFAULT 0,

    days_in_stock NUMERIC DEFAULT 0,
    current_stock_level INTEGER DEFAULT 0,

    lost_sales_today NUMERIC DEFAULT 0,
    was_out_of_stock BOOLEAN DEFAULT FALSE,
    restock_recommended BOOLEAN DEFAULT FALSE,

    price_avg NUMERIC DEFAULT 0,
    price_min NUMERIC DEFAULT 0,
    price_max NUMERIC DEFAULT 0,

    avg_daily_sales_7d NUMERIC DEFAULT 0,
    avg_daily_sales_30d NUMERIC DEFAULT 0,
    turnover_rate NUMERIC DEFAULT 0,
    sell_through_pct NUMERIC DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, modelo, estilo, tienda)
);

CREATE INDEX IF NOT EXISTS idx_daily_records_date ON daily_records(date);
CREATE INDEX IF NOT EXISTS idx_daily_records_modelo ON daily_records(modelo);
CREATE INDEX IF NOT EXISTS idx_daily_records_estilo ON daily_records(estilo);

-- 2. Model scores: track ML model performance over time
CREATE TABLE IF NOT EXISTS model_scores (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name TEXT NOT NULL,
    target TEXT NOT NULL,           -- 'sales' or 'stockout'
    mae NUMERIC,
    rmse NUMERIC,
    accuracy NUMERIC,              -- for classification (stockout)
    r2_score NUMERIC,              -- for regression (sales)
    rank INTEGER,
    num_predictions INTEGER DEFAULT 0,
    training_rows INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, model_name, target)
);

CREATE INDEX IF NOT EXISTS idx_model_scores_date ON model_scores(date);

-- 3. Stockout alerts: auto-generated warnings
CREATE TABLE IF NOT EXISTS stockout_alerts (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    modelo TEXT NOT NULL,
    estilo TEXT NOT NULL,
    tienda TEXT DEFAULT 'all',
    probability NUMERIC NOT NULL,
    days_until_stockout NUMERIC,
    recommended_order_qty INTEGER DEFAULT 0,
    current_stock INTEGER DEFAULT 0,
    avg_daily_sales NUMERIC DEFAULT 0,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stockout_alerts_date ON stockout_alerts(date);

-- 4. ML predictions log: store each day's predictions for comparison
CREATE TABLE IF NOT EXISTS ml_predictions (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,                -- date the prediction was made
    target_date DATE NOT NULL,         -- date being predicted
    modelo TEXT NOT NULL,
    estilo TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_sales NUMERIC DEFAULT 0,
    actual_sales NUMERIC,              -- filled in next day
    predicted_stockout_prob NUMERIC DEFAULT 0,
    actual_was_stockout BOOLEAN,       -- filled in next day
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, target_date, modelo, estilo, model_name)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_date ON ml_predictions(date);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_name);

-- 5. Helper: upsert function for daily_records
CREATE OR REPLACE FUNCTION upsert_daily_record(p_data JSONB)
RETURNS VOID AS $$
BEGIN
    INSERT INTO daily_records (
        date, modelo, estilo, tienda,
        units_sold_today, revenue_today,
        units_sold_7d, units_sold_14d, units_sold_30d,
        revenue_7d, revenue_30d,
        days_in_stock, current_stock_level,
        lost_sales_today, was_out_of_stock, restock_recommended,
        price_avg, price_min, price_max,
        avg_daily_sales_7d, avg_daily_sales_30d,
        turnover_rate, sell_through_pct
    ) VALUES (
        (p_data->>'date')::DATE,
        p_data->>'modelo',
        p_data->>'estilo',
        COALESCE(p_data->>'tienda', 'all'),
        COALESCE((p_data->>'units_sold_today')::NUMERIC, 0),
        COALESCE((p_data->>'revenue_today')::NUMERIC, 0),
        COALESCE((p_data->>'units_sold_7d')::NUMERIC, 0),
        COALESCE((p_data->>'units_sold_14d')::NUMERIC, 0),
        COALESCE((p_data->>'units_sold_30d')::NUMERIC, 0),
        COALESCE((p_data->>'revenue_7d')::NUMERIC, 0),
        COALESCE((p_data->>'revenue_30d')::NUMERIC, 0),
        COALESCE((p_data->>'days_in_stock')::NUMERIC, 0),
        COALESCE((p_data->>'current_stock_level')::INTEGER, 0),
        COALESCE((p_data->>'lost_sales_today')::NUMERIC, 0),
        COALESCE((p_data->>'was_out_of_stock')::BOOLEAN, FALSE),
        COALESCE((p_data->>'restock_recommended')::BOOLEAN, FALSE),
        COALESCE((p_data->>'price_avg')::NUMERIC, 0),
        COALESCE((p_data->>'price_min')::NUMERIC, 0),
        COALESCE((p_data->>'price_max')::NUMERIC, 0),
        COALESCE((p_data->>'avg_daily_sales_7d')::NUMERIC, 0),
        COALESCE((p_data->>'avg_daily_sales_30d')::NUMERIC, 0),
        COALESCE((p_data->>'turnover_rate')::NUMERIC, 0),
        COALESCE((p_data->>'sell_through_pct')::NUMERIC, 0)
    )
    ON CONFLICT (date, modelo, estilo, tienda)
    DO UPDATE SET
        units_sold_today = EXCLUDED.units_sold_today,
        revenue_today = EXCLUDED.revenue_today,
        units_sold_7d = EXCLUDED.units_sold_7d,
        units_sold_14d = EXCLUDED.units_sold_14d,
        units_sold_30d = EXCLUDED.units_sold_30d,
        revenue_7d = EXCLUDED.revenue_7d,
        revenue_30d = EXCLUDED.revenue_30d,
        days_in_stock = EXCLUDED.days_in_stock,
        current_stock_level = EXCLUDED.current_stock_level,
        lost_sales_today = EXCLUDED.lost_sales_today,
        was_out_of_stock = EXCLUDED.was_out_of_stock,
        restock_recommended = EXCLUDED.restock_recommended,
        price_avg = EXCLUDED.price_avg,
        price_min = EXCLUDED.price_min,
        price_max = EXCLUDED.price_max,
        avg_daily_sales_7d = EXCLUDED.avg_daily_sales_7d,
        avg_daily_sales_30d = EXCLUDED.avg_daily_sales_30d,
        turnover_rate = EXCLUDED.turnover_rate,
        sell_through_pct = EXCLUDED.sell_through_pct;
END;
$$ LANGUAGE plpgsql;

-- 6. Batch upsert for daily records (accepts array)
CREATE OR REPLACE FUNCTION upsert_daily_records_batch(p_records JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_records)
    LOOP
        PERFORM upsert_daily_record(rec);
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 7. Upsert model scores
CREATE OR REPLACE FUNCTION upsert_model_scores_batch(p_scores JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_scores)
    LOOP
        INSERT INTO model_scores (date, model_name, target, mae, rmse, accuracy, r2_score, rank, num_predictions, training_rows)
        VALUES (
            (rec->>'date')::DATE,
            rec->>'model_name',
            rec->>'target',
            (rec->>'mae')::NUMERIC,
            (rec->>'rmse')::NUMERIC,
            (rec->>'accuracy')::NUMERIC,
            (rec->>'r2_score')::NUMERIC,
            (rec->>'rank')::INTEGER,
            COALESCE((rec->>'num_predictions')::INTEGER, 0),
            COALESCE((rec->>'training_rows')::INTEGER, 0)
        )
        ON CONFLICT (date, model_name, target)
        DO UPDATE SET
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            accuracy = EXCLUDED.accuracy,
            r2_score = EXCLUDED.r2_score,
            rank = EXCLUDED.rank,
            num_predictions = EXCLUDED.num_predictions,
            training_rows = EXCLUDED.training_rows;
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 8. Upsert ML predictions
CREATE OR REPLACE FUNCTION upsert_ml_predictions_batch(p_preds JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_preds)
    LOOP
        INSERT INTO ml_predictions (date, target_date, modelo, estilo, model_name, predicted_sales, predicted_stockout_prob)
        VALUES (
            (rec->>'date')::DATE,
            (rec->>'target_date')::DATE,
            rec->>'modelo',
            rec->>'estilo',
            rec->>'model_name',
            COALESCE((rec->>'predicted_sales')::NUMERIC, 0),
            COALESCE((rec->>'predicted_stockout_prob')::NUMERIC, 0)
        )
        ON CONFLICT (date, target_date, modelo, estilo, model_name)
        DO UPDATE SET
            predicted_sales = EXCLUDED.predicted_sales,
            predicted_stockout_prob = EXCLUDED.predicted_stockout_prob;
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 9. Insert stockout alerts
CREATE OR REPLACE FUNCTION insert_stockout_alerts_batch(p_alerts JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_alerts)
    LOOP
        INSERT INTO stockout_alerts (date, modelo, estilo, tienda, probability, days_until_stockout, recommended_order_qty, current_stock, avg_daily_sales)
        VALUES (
            (rec->>'date')::DATE,
            rec->>'modelo',
            rec->>'estilo',
            COALESCE(rec->>'tienda', 'all'),
            (rec->>'probability')::NUMERIC,
            (rec->>'days_until_stockout')::NUMERIC,
            COALESCE((rec->>'recommended_order_qty')::INTEGER, 0),
            COALESCE((rec->>'current_stock')::INTEGER, 0),
            COALESCE((rec->>'avg_daily_sales')::NUMERIC, 0)
        );
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 10. Fetch daily records for ML training
CREATE OR REPLACE FUNCTION get_daily_records_for_training(p_days_back INTEGER DEFAULT 90)
RETURNS SETOF daily_records AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM daily_records
    WHERE date >= CURRENT_DATE - p_days_back
    ORDER BY date, modelo, estilo;
END;
$$ LANGUAGE plpgsql;

-- 11. Fetch model scores for leaderboard
CREATE OR REPLACE FUNCTION get_model_leaderboard(p_days_back INTEGER DEFAULT 30)
RETURNS TABLE (
    model_name TEXT,
    target TEXT,
    avg_mae NUMERIC,
    avg_rmse NUMERIC,
    avg_accuracy NUMERIC,
    avg_r2 NUMERIC,
    days_ranked_first INTEGER,
    total_days INTEGER,
    latest_rank INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ms.model_name,
        ms.target,
        ROUND(AVG(ms.mae), 2) AS avg_mae,
        ROUND(AVG(ms.rmse), 2) AS avg_rmse,
        ROUND(AVG(ms.accuracy), 2) AS avg_accuracy,
        ROUND(AVG(ms.r2_score), 4) AS avg_r2,
        COUNT(*) FILTER (WHERE ms.rank = 1)::INTEGER AS days_ranked_first,
        COUNT(*)::INTEGER AS total_days,
        (SELECT ms2.rank FROM model_scores ms2
         WHERE ms2.model_name = ms.model_name AND ms2.target = ms.target
         ORDER BY ms2.date DESC LIMIT 1) AS latest_rank
    FROM model_scores ms
    WHERE ms.date >= CURRENT_DATE - p_days_back
    GROUP BY ms.model_name, ms.target
    ORDER BY ms.target, avg_mae ASC NULLS LAST;
END;
$$ LANGUAGE plpgsql;

-- 12. Fetch model scores history for chart
CREATE OR REPLACE FUNCTION get_model_scores_history(p_days_back INTEGER DEFAULT 30)
RETURNS SETOF model_scores AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM model_scores
    WHERE date >= CURRENT_DATE - p_days_back
    ORDER BY date, target, rank;
END;
$$ LANGUAGE plpgsql;

-- 13. Fetch latest predictions
CREATE OR REPLACE FUNCTION get_latest_predictions()
RETURNS SETOF ml_predictions AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM ml_predictions
    WHERE date = (SELECT MAX(date) FROM ml_predictions)
    ORDER BY predicted_sales DESC;
END;
$$ LANGUAGE plpgsql;

-- 14. Fetch active stockout alerts
CREATE OR REPLACE FUNCTION get_active_stockout_alerts()
RETURNS SETOF stockout_alerts AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM stockout_alerts
    WHERE resolved = FALSE AND date >= CURRENT_DATE - 7
    ORDER BY probability DESC;
END;
$$ LANGUAGE plpgsql;

-- 15. Get daily_records count (to check data sufficiency)
CREATE OR REPLACE FUNCTION get_daily_records_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(DISTINCT date) FROM daily_records);
END;
$$ LANGUAGE plpgsql;
