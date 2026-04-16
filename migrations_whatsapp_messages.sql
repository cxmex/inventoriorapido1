-- Log table for all WhatsApp messages (incoming + outgoing)
CREATE TABLE IF NOT EXISTS whatsapp_messages (
    id BIGSERIAL PRIMARY KEY,
    message_id      TEXT,                       -- Meta's wamid (NULL for bot-origin)
    direction       TEXT NOT NULL,              -- 'in' | 'out'
    phone_number    TEXT NOT NULL,              -- customer phone (from or to)
    message_type    TEXT,                       -- 'text' | 'image' | 'document' | 'interactive' | 'button_reply'
    text_body       TEXT,
    command_matched TEXT,                       -- 'CANJEAR' | 'CLIENTE' | 'COMPRAS' | 'QUERY' | 'BUTTON_REPLY' | NULL
    extra           JSONB,                      -- raw payload or response info
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wm_phone_time ON whatsapp_messages(phone_number, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wm_command   ON whatsapp_messages(command_matched);
CREATE INDEX IF NOT EXISTS idx_wm_direction ON whatsapp_messages(direction);
