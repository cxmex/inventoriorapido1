-- Link entrada_mercancia rows to a conteo_previo caja
ALTER TABLE entrada_mercancia   ADD COLUMN IF NOT EXISTS conteo_previo_caja INT;
ALTER TABLE entrada_mercancia_2 ADD COLUMN IF NOT EXISTS conteo_previo_caja INT;
