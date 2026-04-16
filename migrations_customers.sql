-- Customer master table: one row per WhatsApp customer with a scannable barcode.
-- The customer_barcode is 13 digits EAN-13 format: '9000' + 8-digit id + 1 check digit.
-- (Products use 7xxx, loyalty uses 8xxx, customers use 9xxx.)

CREATE TABLE IF NOT EXISTS customers (
    id BIGSERIAL PRIMARY KEY,
    phone_number    TEXT UNIQUE NOT NULL,
    customer_barcode TEXT UNIQUE,
    name            TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customers_barcode ON customers(customer_barcode);
CREATE INDEX IF NOT EXISTS idx_customers_phone   ON customers(phone_number);

-- Backfill customer rows for existing linked phone numbers in qr_rewards.
-- customer_barcode is filled in by the app when the customer is first queried.
INSERT INTO customers (phone_number)
SELECT DISTINCT phone_number
FROM qr_rewards
WHERE phone_number IS NOT NULL
ON CONFLICT (phone_number) DO NOTHING;
