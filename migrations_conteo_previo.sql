-- Conteo Previo de Mercancía — pre-arrival box counts
CREATE TABLE IF NOT EXISTS conteo_previo (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    caja_numero INT  NOT NULL,
    fecha       DATE NOT NULL DEFAULT CURRENT_DATE,
    modelo      TEXT NOT NULL,
    color       TEXT NOT NULL,
    qty         INT  NOT NULL,
    notas       TEXT,
    reconciled  BOOLEAN DEFAULT false,
    reconciled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_conteo_previo_caja      ON conteo_previo(caja_numero);
CREATE INDEX IF NOT EXISTS idx_conteo_previo_fecha     ON conteo_previo(fecha);
CREATE INDEX IF NOT EXISTS idx_conteo_previo_reconciled ON conteo_previo(reconciled);
