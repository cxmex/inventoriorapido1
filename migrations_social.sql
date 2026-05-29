-- ─────────────────────────────────────────────────────────────────────────────
-- Social / Customer Messaging — Run in Supabase SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- Tracks every message sent to customers to avoid spam
CREATE TABLE IF NOT EXISTS customer_messages (
    id            BIGSERIAL PRIMARY KEY,
    sent_at       TIMESTAMPTZ DEFAULT now(),
    channel       TEXT NOT NULL,           -- 'whatsapp', 'email', 'facebook', 'instagram'
    customer_name TEXT,
    customer_phone TEXT,
    customer_email TEXT,
    message_text  TEXT,
    image_url     TEXT,
    estilo        TEXT,
    modelo        TEXT,
    campaign_type TEXT,                    -- 'top_sellers', 'new_arrivals', 'restock', 'promo'
    delivered     BOOLEAN DEFAULT false,
    opted_out     BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS customer_messages_phone_idx
    ON customer_messages (customer_phone, sent_at DESC);

CREATE INDEX IF NOT EXISTS customer_messages_channel_idx
    ON customer_messages (channel, sent_at DESC);

-- Tracks customer opt-outs (reply STOP)
CREATE TABLE IF NOT EXISTS customer_optouts (
    id         BIGSERIAL PRIMARY KEY,
    phone      TEXT UNIQUE,
    email      TEXT,
    opted_out_at TIMESTAMPTZ DEFAULT now(),
    channel    TEXT                        -- which channel they opted out from
);

-- Tracks social media posts for analytics
CREATE TABLE IF NOT EXISTS social_posts (
    id          BIGSERIAL PRIMARY KEY,
    posted_at   TIMESTAMPTZ DEFAULT now(),
    platform    TEXT NOT NULL,             -- 'facebook', 'instagram'
    post_type   TEXT,                      -- 'top_sellers', 'new_arrivals', 'promo'
    caption     TEXT,
    image_url   TEXT,
    estilos     JSONB,                     -- which estilos were featured
    success     BOOLEAN DEFAULT false
);
