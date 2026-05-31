-- ─────────────────────────────────────────────────────────────────────────────
-- Daily Stock Snapshots — Run in Supabase SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- One row per day: tracks total inventory, asset value, SKU count, and sales
CREATE TABLE IF NOT EXISTS stock_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    fecha           DATE NOT NULL,
    terex1_pzas     INT DEFAULT 0,
    terex2_pzas     INT DEFAULT 0,
    total_pzas      INT DEFAULT 0,
    terex1_skus     INT DEFAULT 0,
    terex2_skus     INT DEFAULT 0,
    total_skus      INT DEFAULT 0,
    terex1_asset    NUMERIC(12,2) DEFAULT 0,    -- stock × price
    terex2_asset    NUMERIC(12,2) DEFAULT 0,
    total_asset     NUMERIC(12,2) DEFAULT 0,
    day_sales_pzas  INT DEFAULT 0,              -- units sold that day
    day_sales_rev   NUMERIC(12,2) DEFAULT 0,    -- revenue that day
    day_entradas    INT DEFAULT 0,              -- units entered that day
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(fecha)
);

CREATE INDEX IF NOT EXISTS stock_snapshots_fecha_idx ON stock_snapshots (fecha DESC);
