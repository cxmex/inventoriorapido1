"""
camera_service.py — Local Camera Service
=========================================
Run this on the machine where the IP cameras are accessible.
It exposes a simple HTTP API that the Railway app calls via ngrok.

Usage:
    pip install flask opencv-python face-recognition numpy requests pillow
    python camera_service.py

Then expose with ngrok:
    ngrok http 5050

Paste the ngrok URL into LOCAL_CAMERA_SERVICE in app.py.

Endpoints:
    POST /activity-snapshot          → capture + analyze one frame from all cameras
    POST /capture_telegram           → capture frame and send to Telegram directly
    POST /trigger                    → start 1-minute capture loop for a barcode
    POST /enroll-staff               → enroll a staff member's face from current camera view
    GET  /health                     → health check
"""

import os, io, json, time, base64, logging
from datetime import datetime
import threading
import requests
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("Install opencv-python:  pip install opencv-python")

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except ImportError:
    FACE_RECOG_AVAILABLE = False
    logging.warning("face_recognition not installed — face ID disabled. "
                    "Install: pip install face-recognition")

from flask import Flask, request, jsonify
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "admin123!"
CAMERAS = [
    {"ip": "192.168.1.103", "name": "Camera_1",  "branch": "terex1"},
    {"ip": "192.168.1.106", "name": "Camera_2",  "branch": "terex1"},
    {"ip": "192.168.1.107", "name": "Camera_3",  "branch": "terex2"},
]

SUPABASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdia2hrYmZiYXJzbnBiZGt4emlpIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3MzQzODAzNzMsImV4cCI6MjA0OTk1NjM3M30."
    "mcOcC2GVEu_wD3xNBzSCC3MwDck3CIdmz4D8adU-bpI"
)
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

TELEGRAM_TOKEN    = "8487551934:AAGOw4FLIgXKolbeiFmAsRuyBS8mJ-3kSQk"
TELEGRAM_CHAT_IDS = ["7204722077", "7145539843", "8133878707"]

ACTIVITY_BUCKET   = "camera-activity"
CAPTURE_BUCKET    = "camera-captures"

PORT = 5050

# ── Face recognition cache (loaded at startup) ────────────────────────────────
_staff_encodings: list = []  # [{"name": str, "branch": str, "encoding": np.array}]
_staff_cache_loaded = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera_service")

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rtsp_url(cam: dict) -> str:
    ip = cam["ip"]
    return (
        f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{ip}:554"
        "/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
    )


def _grab_frame(cam: dict):
    """Capture one frame from an IP camera. Returns (frame_bgr, camera_name) or (None, name)."""
    cap = cv2.VideoCapture(_rtsp_url(cam))
    frame = None
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame = None
    cap.release()
    return frame, cam["name"]


def _detect_faces_opencv(frame_bgr):
    """Fast face detection with Haar cascade (no external lib needed)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    bboxes = []
    for (x, y, w, h) in faces if len(faces) > 0 else []:
        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return bboxes


def _load_staff_encodings():
    """Load staff face encodings from Supabase staff_faces table."""
    global _staff_encodings, _staff_cache_loaded
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/staff_faces?active=eq.true&select=name,branch,face_encoding",
            headers=SUPABASE_HEADERS,
            timeout=10,
        )
        if resp.status_code < 400:
            rows = resp.json()
            _staff_encodings = []
            for row in rows:
                enc = row.get("face_encoding")
                if enc:
                    arr = np.array(enc if isinstance(enc, list) else json.loads(enc))
                    _staff_encodings.append({
                        "name": row["name"],
                        "branch": row.get("branch"),
                        "encoding": arr,
                    })
            log.info("Loaded %d staff face encodings", len(_staff_encodings))
        _staff_cache_loaded = True
    except Exception as e:
        log.warning("Could not load staff encodings: %s", e)
        _staff_cache_loaded = True


def _classify_faces(frame_bgr, bboxes: list):
    """
    If face_recognition is available, match detected faces against known staff.
    Returns (known_count, unknown_count, face_features).
    """
    if not FACE_RECOG_AVAILABLE or not _staff_cache_loaded:
        return 0, len(bboxes), [{"bbox": b} for b in bboxes]

    if not bboxes:
        return 0, 0, []

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locations = [(b["y"], b["x"] + b["w"], b["y"] + b["h"], b["x"]) for b in bboxes]

    try:
        encodings = face_recognition.face_encodings(rgb, locations)
    except Exception:
        return 0, len(bboxes), [{"bbox": b} for b in bboxes]

    known = 0
    unknown = 0
    features = []

    staff_encs = [s["encoding"] for s in _staff_encodings]

    for enc, bbox in zip(encodings, bboxes):
        matched_name = "unknown"
        if staff_encs:
            matches = face_recognition.compare_faces(staff_encs, enc, tolerance=0.5)
            if any(matches):
                idx = matches.index(True)
                matched_name = _staff_encodings[idx]["name"]
                known += 1
            else:
                unknown += 1
        else:
            unknown += 1

        features.append({
            "bbox": bbox,
            "identity": matched_name,
            "encoding": enc.tolist()[:16],  # store first 16 dims to save space
        })

    return known, unknown, features


def _classify_activity(face_count: int, known: int, unknown: int) -> tuple[str, float]:
    """
    Classify shop activity based on face detection results.
    Returns (activity_type, confidence).
    """
    if face_count == 0:
        return "idle", 0.9
    if unknown > 0 and known > 0:
        return "selling", 0.85   # staff + customers = selling in progress
    if unknown > 0 and known == 0:
        return "busy", 0.7       # only unrecognised people (maybe customers browsing)
    if known > 0 and unknown == 0:
        return "counting", 0.8   # only staff, no customers
    return "idle", 0.5


def _upload_to_supabase(image_bytes: bytes, path: str, bucket: str) -> str | None:
    """Upload JPEG bytes to a Supabase Storage bucket. Returns public URL or None."""
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "image/jpeg",
                "x-upsert": "true",
            },
            data=image_bytes,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
    except Exception as e:
        log.warning("Upload failed (%s/%s): %s", bucket, path, e)
    return None


def _compress_frame(frame_bgr, quality: int = 60, max_width: int = 640) -> bytes:
    """Resize and JPEG-compress a frame. Returns bytes."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    success, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if success else b""


def _send_telegram_photo(image_bytes: bytes, caption: str):
    """Send a JPEG to all Telegram chat IDs."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            requests.post(
                url,
                files={"photo": ("frame.jpg", image_bytes, "image/jpeg")},
                data={"chat_id": chat_id, "caption": caption},
                timeout=15,
            )
        except Exception as e:
            log.warning("Telegram photo error (chat %s): %s", chat_id, e)


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cameras": len(CAMERAS),
                    "face_recog": FACE_RECOG_AVAILABLE,
                    "staff_loaded": len(_staff_encodings)})


@app.route("/activity-snapshot", methods=["POST"])
def activity_snapshot():
    """
    Called by Railway app every minute.
    Captures one frame per camera, runs face detection, classifies activity.
    Uploads compressed frame to Supabase.
    Returns analysis results as JSON so Railway app can store them.
    """
    results = []
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for cam in CAMERAS:
        frame, cam_name = _grab_frame(cam)
        if frame is None:
            log.warning("No frame from %s", cam_name)
            results.append({
                "camera": cam_name,
                "branch": cam["branch"],
                "error": "no_frame",
            })
            continue

        bboxes = _detect_faces_opencv(frame)
        face_count = len(bboxes)
        known, unknown, features = _classify_faces(frame, bboxes)
        activity, confidence = _classify_activity(face_count, known, unknown)

        # Upload compressed image (only when people detected, to save storage)
        image_url = None
        if face_count > 0:
            img_bytes = _compress_frame(frame, quality=55, max_width=480)
            path = f"{cam['branch']}/{ts}-{cam_name.lower()}.jpg"
            image_url = _upload_to_supabase(img_bytes, path, ACTIVITY_BUCKET)

        results.append({
            "camera": cam_name,
            "branch": cam["branch"],
            "face_count": face_count,
            "known_faces": known,
            "unknown_faces": unknown,
            "activity_type": activity,
            "confidence": confidence,
            "face_features": features,
            "image_url": image_url,
        })
        log.info("%s → %s faces (%d known, %d unknown) → %s",
                 cam_name, face_count, known, unknown, activity)

    return jsonify({"ok": True, "timestamp": ts, "results": results})


@app.route("/capture_telegram", methods=["POST"])
def capture_telegram():
    """
    Capture frames from all cameras and send to Telegram.
    Used by barcode-scan triggered captures.
    """
    data = request.get_json(silent=True) or {}
    barcode = data.get("barcode", "unknown")
    sent = []

    for cam in CAMERAS:
        frame, cam_name = _grab_frame(cam)
        if frame is None:
            continue
        img_bytes = _compress_frame(frame, quality=70, max_width=800)
        caption = f"📷 {cam_name} | Código: {barcode} | {datetime.now().strftime('%H:%M:%S')}"
        _send_telegram_photo(img_bytes, caption)
        sent.append(cam_name)

    return jsonify({"ok": True, "telegram_sent": sent})


@app.route("/trigger", methods=["POST"])
def trigger_capture():
    """Start a 1-minute background capture loop for a barcode."""
    data = request.get_json(silent=True) or {}
    barcode = data.get("barcode", "unknown")

    def _loop():
        end = time.time() + 60
        while time.time() < end:
            for cam in CAMERAS:
                frame, cam_name = _grab_frame(cam)
                if frame is None:
                    continue
                ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                path = f"captures/{ts}-{cam_name.lower()}-{barcode}.jpg"
                img_bytes = _compress_frame(frame, quality=65)
                _upload_to_supabase(img_bytes, path, CAPTURE_BUCKET)
            time.sleep(5)

    threading.Thread(target=_loop, daemon=True).start()
    return jsonify({"ok": True, "barcode": barcode, "duration_sec": 60})


@app.route("/enroll-staff", methods=["POST"])
def enroll_staff():
    """
    Enroll a staff member's face by capturing from the camera right now.
    POST body: {"name": "María", "branch": "terex1", "camera_ip": "192.168.1.103"}

    This requires face_recognition to be installed.
    """
    if not FACE_RECOG_AVAILABLE:
        return jsonify({"error": "face_recognition not installed"}), 400

    data = request.get_json(silent=True) or {}
    name = data.get("name")
    branch = data.get("branch")
    camera_ip = data.get("camera_ip")

    if not name:
        return jsonify({"error": "name is required"}), 400

    cam_cfg = next((c for c in CAMERAS if c["ip"] == camera_ip), CAMERAS[0])
    frame, cam_name = _grab_frame(cam_cfg)
    if frame is None:
        return jsonify({"error": f"Could not capture from {cam_name}"}), 500

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        return jsonify({"error": "No face detected in frame"}), 400
    if len(encodings) > 1:
        return jsonify({"error": f"{len(encodings)} faces detected — ensure only the staff member is in frame"}), 400

    enc = encodings[0].tolist()

    # Save to Supabase
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/staff_faces",
        headers={**SUPABASE_HEADERS, "Prefer": "return=representation"},
        json={"name": name, "branch": branch, "face_encoding": enc},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        return jsonify({"error": "Supabase insert failed", "detail": resp.text}), 500

    # Reload cache
    _load_staff_encodings()

    return jsonify({"ok": True, "name": name, "branch": branch, "camera": cam_name})


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Loading staff face encodings…")
    _load_staff_encodings()
    log.info("Starting camera service on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)
