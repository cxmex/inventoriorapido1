-- QR rewards table: each ticket gets a unique QR token.
-- Customer scans QR -> WhatsApp -> bot captures phone -> row gets phone_number
-- On next purchase, reward_amount can be redeemed.

CREATE TABLE IF NOT EXISTS qr_rewards (
    id BIGSERIAL PRIMARY KEY,
    qr_token        TEXT UNIQUE NOT NULL,
    order_id        INTEGER NOT NULL,
    purchase_amount NUMERIC(10,2) NOT NULL,
    reward_amount   NUMERIC(10,2) NOT NULL,
    phone_number    TEXT,
    status          TEXT DEFAULT 'pending',   -- 'pending' | 'claimed' | 'redeemed'
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    claimed_at      TIMESTAMPTZ,
    redeemed_at     TIMESTAMPTZ,
    redeemed_order_id INTEGER   -- the next order the credit was used on
);

CREATE INDEX IF NOT EXISTS idx_qr_rewards_token ON qr_rewards(qr_token);
CREATE INDEX IF NOT EXISTS idx_qr_rewards_phone ON qr_rewards(phone_number);
CREATE INDEX IF NOT EXISTS idx_qr_rewards_status ON qr_rewards(status);

-- Status values:
--   'pending'  = ticket was generated, nobody scanned the QR yet
--   'linked'   = customer scanned QR, phone is associated, credit is available to use
--   'redeemed' = credit was applied to a new purchase (see redeemed_order_id)

-- One-time rename for existing rows created before the linked/redeemed split:
UPDATE qr_rewards SET status = 'linked' WHERE status = 'claimed';
