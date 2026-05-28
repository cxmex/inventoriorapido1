-- Agent Insights table — stores hourly intelligence and daily visual report records
CREATE TABLE IF NOT EXISTS agent_insights (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    type TEXT NOT NULL,
    branch TEXT,
    summary TEXT,
    data JSONB,
    sent_telegram BOOLEAN DEFAULT false
);

-- Index for fast lookups by type and time
CREATE INDEX IF NOT EXISTS agent_insights_type_idx ON agent_insights (type, created_at DESC);
