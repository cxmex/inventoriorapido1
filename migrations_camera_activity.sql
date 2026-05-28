-- ─────────────────────────────────────────────────────────────────────────────
-- Camera Activity Monitoring — Run this in the Supabase SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- Stores one row per captured frame with face-detection results and ML features
CREATE TABLE IF NOT EXISTS shop_activity_frames (
    id            BIGSERIAL PRIMARY KEY,
    branch        TEXT        NOT NULL,           -- 'terex1' or 'terex2'
    camera_name   TEXT,                           -- e.g. 'Camera_1'
    captured_at   TIMESTAMPTZ DEFAULT now(),
    face_count    INT         DEFAULT 0,
    known_faces   INT         DEFAULT 0,          -- staff faces detected
    unknown_faces INT         DEFAULT 0,          -- customer/visitor faces
    activity_type TEXT,                           -- 'selling','counting','idle','busy'
    confidence    FLOAT,                          -- classifier confidence 0-1
    face_features JSONB,                          -- face bboxes + encodings for ML
    image_url     TEXT,                           -- path in Supabase 'camera-activity' bucket
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS shop_activity_frames_branch_time_idx
    ON shop_activity_frames (branch, captured_at DESC);

CREATE INDEX IF NOT EXISTS shop_activity_frames_type_idx
    ON shop_activity_frames (activity_type, captured_at DESC);


-- Stores face encodings for known staff — used by local camera_service.py
-- to distinguish staff (familiar faces) from customers (new faces)
CREATE TABLE IF NOT EXISTS staff_faces (
    id             BIGSERIAL PRIMARY KEY,
    name           TEXT        NOT NULL,
    branch         TEXT,                          -- 'terex1', 'terex2', or NULL for both
    face_encoding  JSONB       NOT NULL,          -- 128-dim face_recognition encoding
    added_at       TIMESTAMPTZ DEFAULT now(),
    active         BOOLEAN     DEFAULT true
);

CREATE INDEX IF NOT EXISTS staff_faces_branch_idx ON staff_faces (branch, active);
