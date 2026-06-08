-- ─────────────────────────────────────────────────────────────────────────────
-- Team Chat — Internal messaging for Fundastock team
-- Run in Supabase SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- Team members
CREATE TABLE IF NOT EXISTS chat_users (
    id          SERIAL PRIMARY KEY,
    username    TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    display_name TEXT,
    avatar_color TEXT DEFAULT '#1565C0',
    role        TEXT DEFAULT 'staff',         -- 'admin', 'staff'
    last_seen   TIMESTAMPTZ DEFAULT now(),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Insert the 6 team members
INSERT INTO chat_users (username, password, display_name, avatar_color, role) VALUES
    ('horacio', 'horacio123', 'Horacio', '#1565C0', 'admin'),
    ('rocio', 'rocio123', 'Rocio', '#E91E63', 'staff'),
    ('carlos', 'carlos123', 'Carlos', '#2E7D32', 'staff'),
    ('luis', 'luis123', 'Luis', '#E65100', 'staff'),
    ('marcos', 'marcos123', 'Marcos', '#7B1FA2', 'staff'),
    ('nayeli', 'nayeli123', 'Nayeli', '#00897B', 'staff')
ON CONFLICT (username) DO NOTHING;

-- Messages (private + group)
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    sender_id   INT REFERENCES chat_users(id),
    sender_name TEXT,
    recipient_id INT REFERENCES chat_users(id),  -- NULL = group message to all
    channel     TEXT DEFAULT 'general',           -- 'general', 'ventas', 'inventario', 'private'
    content     TEXT,
    file_url    TEXT,                              -- attachment URL in Supabase storage
    file_type   TEXT,                              -- 'image', 'pdf', 'doc', etc.
    file_name   TEXT,
    is_ai       BOOLEAN DEFAULT false,            -- true if sent by ARGOS/AI
    read_by     JSONB DEFAULT '[]',               -- array of user_ids who read it
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_channel_idx ON chat_messages (channel, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_messages_private_idx ON chat_messages (sender_id, recipient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_messages_created_idx ON chat_messages (created_at DESC);

-- Channels
CREATE TABLE IF NOT EXISTS chat_channels (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    icon        TEXT DEFAULT '💬',
    created_at  TIMESTAMPTZ DEFAULT now()
);

INSERT INTO chat_channels (name, description, icon) VALUES
    ('general', 'Chat general del equipo', '💬'),
    ('ventas', 'Discusion de ventas y clientes', '🛒'),
    ('inventario', 'Stock, entradas, conteo', '📦'),
    ('fotos', 'Fotos de productos para Emanuel', '📷'),
    ('argos', 'Reportes automaticos de ARGOS', '🧠')
ON CONFLICT (name) DO NOTHING;
