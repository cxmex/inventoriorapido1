-- Create barcode_photos table for ML training data
CREATE TABLE IF NOT EXISTS barcode_photos (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    barcode TEXT NOT NULL,
    product_name TEXT,
    estilo TEXT,
    estilo_id INT,
    color TEXT,
    file_path TEXT NOT NULL,
    public_url TEXT NOT NULL
);

-- Index for lookups by barcode and estilo
CREATE INDEX IF NOT EXISTS idx_barcode_photos_barcode ON barcode_photos(barcode);
CREATE INDEX IF NOT EXISTS idx_barcode_photos_estilo_id ON barcode_photos(estilo_id);

-- Create storage bucket (run in Supabase dashboard > Storage if not via SQL)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('barcode-photos', 'barcode-photos', true);
