-- ============================================================
-- MARKET INTELLIGENCE TABLES — Run this in Supabase SQL Editor
-- ============================================================

-- 1. Market phones: phones available in Mexico from retailers/brands
CREATE TABLE IF NOT EXISTS market_phones (
    id BIGSERIAL PRIMARY KEY,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    price_mxn NUMERIC,
    launch_date DATE,
    category TEXT DEFAULT 'smartphone',  -- smartphone, tablet
    segment TEXT,                          -- gama_baja, gama_media, gama_alta, flagship
    source TEXT,                           -- samsung.com, walmart, coppel, etc.
    source_url TEXT,
    in_stock BOOLEAN DEFAULT TRUE,
    is_bestseller BOOLEAN DEFAULT FALSE,
    specs_summary TEXT,
    last_seen DATE,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(brand, model, source)
);

CREATE INDEX IF NOT EXISTS idx_market_phones_brand ON market_phones(brand);
CREATE INDEX IF NOT EXISTS idx_market_phones_model ON market_phones(model);

-- 2. Market snapshots: periodic price/availability tracking
CREATE TABLE IF NOT EXISTS market_snapshots (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    source TEXT NOT NULL,
    price_mxn NUMERIC,
    in_stock BOOLEAN DEFAULT TRUE,
    is_bestseller BOOLEAN DEFAULT FALSE,
    bestseller_rank INTEGER,
    price_change NUMERIC DEFAULT 0,       -- vs previous snapshot
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, brand, model, source)
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_date ON market_snapshots(date);

-- 3. Market gaps: phones in market that we don't have cases for
CREATE TABLE IF NOT EXISTS market_gaps (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    phone_price NUMERIC,
    segment TEXT,
    num_retailers INTEGER DEFAULT 0,        -- how many retailers carry it
    is_bestseller BOOLEAN DEFAULT FALSE,
    opportunity_score NUMERIC DEFAULT 0,    -- higher = should order sooner
    reason TEXT,                             -- 'new_launch', 'bestseller_no_cases', 'trending'
    estimated_monthly_demand INTEGER,
    we_carry_cases BOOLEAN DEFAULT FALSE,   -- cross-ref with inventario_modelos
    our_modelo_match TEXT,                   -- matching modelo name if found
    our_stock_level INTEGER,
    our_sales_30d INTEGER,
    our_was_stockout BOOLEAN DEFAULT FALSE,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, brand, model)
);

CREATE INDEX IF NOT EXISTS idx_market_gaps_date ON market_gaps(date);
CREATE INDEX IF NOT EXISTS idx_market_gaps_score ON market_gaps(opportunity_score DESC);

-- 4. Market alerts: significant events (launches, price drops, trending)
CREATE TABLE IF NOT EXISTS market_alerts (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    alert_type TEXT NOT NULL,  -- 'new_launch', 'price_drop', 'bestseller', 'stockout_spike', 'demand_correlation'
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    impact_score NUMERIC DEFAULT 0,
    our_modelo_match TEXT,
    our_action TEXT,            -- 'order_cases', 'restock', 'monitor', 'investigate'
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_market_alerts_date ON market_alerts(date);

-- 5. Demand correlations: link market events to our sales changes
CREATE TABLE IF NOT EXISTS demand_correlations (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    market_event TEXT NOT NULL,        -- 'phone_launch', 'price_drop', 'promo_campaign', 'new_carrier'
    event_date DATE,
    our_modelo TEXT,
    sales_before_7d NUMERIC DEFAULT 0,
    sales_after_7d NUMERIC DEFAULT 0,
    sales_change_pct NUMERIC DEFAULT 0,
    stock_before INTEGER DEFAULT 0,
    stock_after INTEGER DEFAULT 0,
    was_stockout BOOLEAN DEFAULT FALSE,
    correlation_strength TEXT,         -- 'strong', 'moderate', 'weak', 'none'
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(date, brand, model, market_event)
);

-- 6. Upsert market phones batch
CREATE OR REPLACE FUNCTION upsert_market_phones_batch(p_phones JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_phones)
    LOOP
        INSERT INTO market_phones (brand, model, price_mxn, launch_date, category, segment, source, source_url, in_stock, is_bestseller, specs_summary, last_seen)
        VALUES (
            rec->>'brand', rec->>'model',
            (rec->>'price_mxn')::NUMERIC,
            (rec->>'launch_date')::DATE,
            COALESCE(rec->>'category', 'smartphone'),
            rec->>'segment',
            rec->>'source', rec->>'source_url',
            COALESCE((rec->>'in_stock')::BOOLEAN, TRUE),
            COALESCE((rec->>'is_bestseller')::BOOLEAN, FALSE),
            rec->>'specs_summary',
            COALESCE((rec->>'last_seen')::DATE, CURRENT_DATE)
        )
        ON CONFLICT (brand, model, source) DO UPDATE SET
            price_mxn = EXCLUDED.price_mxn,
            in_stock = EXCLUDED.in_stock,
            is_bestseller = EXCLUDED.is_bestseller,
            specs_summary = EXCLUDED.specs_summary,
            last_seen = EXCLUDED.last_seen;
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 7. Upsert market gaps batch
CREATE OR REPLACE FUNCTION upsert_market_gaps_batch(p_gaps JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_gaps)
    LOOP
        INSERT INTO market_gaps (date, brand, model, phone_price, segment, num_retailers, is_bestseller, opportunity_score, reason, estimated_monthly_demand, we_carry_cases, our_modelo_match, our_stock_level, our_sales_30d, our_was_stockout)
        VALUES (
            (rec->>'date')::DATE, rec->>'brand', rec->>'model',
            (rec->>'phone_price')::NUMERIC,
            rec->>'segment',
            COALESCE((rec->>'num_retailers')::INTEGER, 0),
            COALESCE((rec->>'is_bestseller')::BOOLEAN, FALSE),
            COALESCE((rec->>'opportunity_score')::NUMERIC, 0),
            rec->>'reason',
            (rec->>'estimated_monthly_demand')::INTEGER,
            COALESCE((rec->>'we_carry_cases')::BOOLEAN, FALSE),
            rec->>'our_modelo_match',
            (rec->>'our_stock_level')::INTEGER,
            (rec->>'our_sales_30d')::INTEGER,
            COALESCE((rec->>'our_was_stockout')::BOOLEAN, FALSE)
        )
        ON CONFLICT (date, brand, model) DO UPDATE SET
            phone_price = EXCLUDED.phone_price,
            opportunity_score = EXCLUDED.opportunity_score,
            we_carry_cases = EXCLUDED.we_carry_cases,
            our_modelo_match = EXCLUDED.our_modelo_match,
            our_stock_level = EXCLUDED.our_stock_level,
            our_sales_30d = EXCLUDED.our_sales_30d,
            our_was_stockout = EXCLUDED.our_was_stockout;
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 8. Insert market alerts batch
CREATE OR REPLACE FUNCTION insert_market_alerts_batch(p_alerts JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_alerts)
    LOOP
        INSERT INTO market_alerts (date, alert_type, brand, model, title, description, impact_score, our_modelo_match, our_action)
        VALUES (
            (rec->>'date')::DATE,
            rec->>'alert_type', rec->>'brand', rec->>'model',
            rec->>'title', rec->>'description',
            COALESCE((rec->>'impact_score')::NUMERIC, 0),
            rec->>'our_modelo_match',
            rec->>'our_action'
        );
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 9. Upsert demand correlations
CREATE OR REPLACE FUNCTION upsert_demand_correlations_batch(p_corrs JSONB)
RETURNS INTEGER AS $$
DECLARE
    rec JSONB;
    cnt INTEGER := 0;
BEGIN
    FOR rec IN SELECT * FROM jsonb_array_elements(p_corrs)
    LOOP
        INSERT INTO demand_correlations (date, brand, model, market_event, event_date, our_modelo, sales_before_7d, sales_after_7d, sales_change_pct, stock_before, stock_after, was_stockout, correlation_strength)
        VALUES (
            (rec->>'date')::DATE, rec->>'brand', rec->>'model',
            rec->>'market_event', (rec->>'event_date')::DATE,
            rec->>'our_modelo',
            COALESCE((rec->>'sales_before_7d')::NUMERIC, 0),
            COALESCE((rec->>'sales_after_7d')::NUMERIC, 0),
            COALESCE((rec->>'sales_change_pct')::NUMERIC, 0),
            COALESCE((rec->>'stock_before')::INTEGER, 0),
            COALESCE((rec->>'stock_after')::INTEGER, 0),
            COALESCE((rec->>'was_stockout')::BOOLEAN, FALSE),
            rec->>'correlation_strength'
        )
        ON CONFLICT (date, brand, model, market_event) DO UPDATE SET
            sales_after_7d = EXCLUDED.sales_after_7d,
            sales_change_pct = EXCLUDED.sales_change_pct,
            stock_after = EXCLUDED.stock_after,
            was_stockout = EXCLUDED.was_stockout,
            correlation_strength = EXCLUDED.correlation_strength;
        cnt := cnt + 1;
    END LOOP;
    RETURN cnt;
END;
$$ LANGUAGE plpgsql;

-- 10. Get active market gaps (not resolved, ordered by opportunity)
CREATE OR REPLACE FUNCTION get_market_gaps_active(p_days_back INTEGER DEFAULT 7)
RETURNS SETOF market_gaps AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM market_gaps
    WHERE date >= CURRENT_DATE - p_days_back AND resolved = FALSE
    ORDER BY opportunity_score DESC;
END;
$$ LANGUAGE plpgsql;

-- 11. Get recent market alerts
CREATE OR REPLACE FUNCTION get_market_alerts_recent(p_days_back INTEGER DEFAULT 30)
RETURNS SETOF market_alerts AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM market_alerts
    WHERE date >= CURRENT_DATE - p_days_back
    ORDER BY date DESC, impact_score DESC;
END;
$$ LANGUAGE plpgsql;

-- 12. Get demand correlations
CREATE OR REPLACE FUNCTION get_demand_correlations_recent(p_days_back INTEGER DEFAULT 30)
RETURNS SETOF demand_correlations AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM demand_correlations
    WHERE date >= CURRENT_DATE - p_days_back
    ORDER BY date DESC, sales_change_pct DESC;
END;
$$ LANGUAGE plpgsql;
