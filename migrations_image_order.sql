-- Add display_order column to image_uploads so the user can pick which image
-- shows first per (estilo, color). Lower number = earlier. Default 100.
-- Setting an image as "primary" = display_order = 0 (and we reset all other
-- images for that estilo+color back to 100 so there's only one primary).

ALTER TABLE image_uploads
    ADD COLUMN IF NOT EXISTS display_order INTEGER NOT NULL DEFAULT 100;

CREATE INDEX IF NOT EXISTS idx_image_uploads_order
    ON image_uploads (estilo_id, color_id, display_order);

-- Optional helper RPC (called from the new "set primary" endpoint).
-- Atomically sets the chosen image to 0 and resets siblings to 100.
CREATE OR REPLACE FUNCTION set_primary_image(p_image_id BIGINT)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    v_estilo_id BIGINT;
    v_color_id  BIGINT;
BEGIN
    SELECT estilo_id, color_id INTO v_estilo_id, v_color_id
    FROM image_uploads WHERE id = p_image_id;

    IF v_estilo_id IS NULL THEN
        RAISE EXCEPTION 'image_uploads % not found', p_image_id;
    END IF;

    UPDATE image_uploads
    SET display_order = 100
    WHERE estilo_id = v_estilo_id AND color_id = v_color_id AND id <> p_image_id;

    UPDATE image_uploads
    SET display_order = 0
    WHERE id = p_image_id;
END;
$$;
