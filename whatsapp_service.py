"""
whatsapp_service.py — Local WhatsApp Web Automation via Playwright
==================================================================
Run this on your local machine. It opens a persistent browser session
with WhatsApp Web — scan the QR code ONCE, then it stays logged in.

Each WhatsApp number uses its own browser profile folder, so you can
run multiple instances for different numbers.

Usage:
    pip install playwright flask requests
    playwright install chromium

    # Default (first WhatsApp number):
    python whatsapp_service.py

    # Second WhatsApp number on a different port:
    python whatsapp_service.py --profile wa_profile_2 --port 5051

    # Use an existing browser profile (e.g. Brave, Chrome):
    python whatsapp_service.py --profile "/Users/you/Library/Application Support/BraveSoftware/Brave-Browser/Default"

Then expose with ngrok:
    ngrok http 5050

Endpoints:
    POST /whatsapp/send         → send text message
    POST /whatsapp/send-image   → send image from URL with caption
    GET  /whatsapp/status       → check if logged in
    GET  /health                → health check
"""

import argparse
import os
import sys
import time
import logging
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

import requests
from flask import Flask, request, jsonify

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
except ImportError:
    print("Install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whatsapp_service")

app = Flask(__name__)

# ── Globals (set at startup) ──────────────────────────────────────────────────
_browser = None
_context = None
_page = None
_playwright = None
_ready = False
_profile_dir = None


def _ensure_whatsapp_ready(timeout: int = 15):
    """Make sure WhatsApp Web is loaded and logged in."""
    global _page, _ready

    if _page is None:
        return False

    try:
        # Check if we're on WhatsApp and logged in
        url = _page.url
        if "web.whatsapp.com" not in url:
            _page.goto("https://web.whatsapp.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)

        # Wait for the search/chat bar to appear (means logged in)
        try:
            _page.wait_for_selector('[data-tab="3"]', timeout=timeout * 1000)
            _ready = True
            return True
        except PwTimeout:
            # Check if QR code is showing
            qr = _page.query_selector('canvas[aria-label="Scan this QR code to link a device!"]')
            if qr:
                log.warning("QR code visible — please scan it in the browser window!")
                _ready = False
                return False
            # Might still be loading
            _ready = False
            return False
    except Exception as e:
        log.error("WhatsApp check failed: %s", e)
        return False


def _open_chat(phone: str) -> bool:
    """Navigate to a specific chat using the WhatsApp URL scheme."""
    global _page
    # Remove any non-numeric characters
    phone_clean = "".join(c for c in phone if c.isdigit())
    try:
        _page.goto(
            f"https://web.whatsapp.com/send?phone={phone_clean}",
            wait_until="domcontentloaded",
            timeout=20000,
        )
        # Wait for the message input box
        _page.wait_for_selector(
            'div[contenteditable="true"][data-tab="10"]',
            timeout=15000,
        )
        time.sleep(1)
        return True
    except Exception as e:
        log.error("Could not open chat with %s: %s", phone_clean, e)
        return False


def _send_text_message(phone: str, text: str) -> bool:
    """Type and send a text message to the given phone number."""
    global _page
    if not _open_chat(phone):
        return False

    try:
        # Find the message input
        msg_box = _page.query_selector('div[contenteditable="true"][data-tab="10"]')
        if not msg_box:
            log.error("Message box not found")
            return False

        # Type message line by line (Shift+Enter for newlines)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            msg_box.type(line, delay=5)
            if i < len(lines) - 1:
                _page.keyboard.down("Shift")
                _page.keyboard.press("Enter")
                _page.keyboard.up("Shift")

        # Press Enter to send
        _page.keyboard.press("Enter")
        time.sleep(1)
        log.info("Text message sent to %s", phone)
        return True
    except Exception as e:
        log.error("Send text failed: %s", e)
        return False


def _send_image_message(phone: str, image_url: str, caption: str = "") -> bool:
    """
    Download an image from URL, then send it via WhatsApp Web.
    Uses the attachment button → image upload flow.
    """
    global _page
    if not _open_chat(phone):
        return False

    # Download image to temp file
    tmp_path = None
    try:
        resp = requests.get(image_url, timeout=15)
        if resp.status_code >= 400:
            log.error("Image download failed: %s", resp.status_code)
            return False

        suffix = ".jpg" if "jpeg" in (resp.headers.get("content-type", "")) else ".png"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(resp.content)
        tmp.close()
        tmp_path = tmp.name
    except Exception as e:
        log.error("Image download failed: %s", e)
        return False

    try:
        # Click the attachment (paperclip) button
        attach_btn = _page.query_selector('div[title="Attach"]') or _page.query_selector('button[aria-label="Attach"]')
        if not attach_btn:
            # Try Spanish locale
            attach_btn = _page.query_selector('div[title="Adjuntar"]') or _page.query_selector('button[aria-label="Adjuntar"]')
        if not attach_btn:
            # Fallback: look for the + button
            attach_btn = _page.query_selector('[data-icon="plus"]') or _page.query_selector('[data-icon="clip"]')

        if not attach_btn:
            log.error("Attach button not found")
            return False

        attach_btn.click()
        time.sleep(1)

        # Find the file input for images/photos
        file_input = _page.query_selector('input[accept="image/*,video/mp4,video/3gpp,video/quicktime"]')
        if not file_input:
            # Try broader selector
            file_input = _page.query_selector('input[type="file"][accept*="image"]')
        if not file_input:
            # Last resort: any file input
            inputs = _page.query_selector_all('input[type="file"]')
            file_input = inputs[0] if inputs else None

        if not file_input:
            log.error("File input not found")
            _page.keyboard.press("Escape")
            return False

        file_input.set_input_files(tmp_path)
        time.sleep(2)

        # Add caption if provided
        if caption:
            caption_box = _page.query_selector(
                'div[contenteditable="true"][data-tab="10"]'
            ) or _page.query_selector(
                'div.copyable-text[contenteditable="true"]'
            )
            if caption_box:
                caption_box.type(caption, delay=5)
                time.sleep(0.5)

        # Click send button
        send_btn = _page.query_selector('[data-icon="send"]') or _page.query_selector(
            'span[data-icon="send"]'
        )
        if send_btn:
            send_btn.click()
        else:
            _page.keyboard.press("Enter")

        time.sleep(2)
        log.info("Image sent to %s", phone)
        return True

    except Exception as e:
        log.error("Send image failed: %s", e)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Flask Endpoints ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "whatsapp_ready": _ready,
        "profile": _profile_dir,
    })


@app.route("/whatsapp/status", methods=["GET"])
def whatsapp_status():
    ready = _ensure_whatsapp_ready(timeout=5)
    return jsonify({
        "logged_in": ready,
        "profile": _profile_dir,
        "hint": "Scan the QR code in the browser window" if not ready else "Ready to send",
    })


@app.route("/whatsapp/send", methods=["POST"])
def whatsapp_send():
    data = request.get_json(silent=True) or {}
    to = data.get("to", "")
    text = data.get("text", "")

    if not to or not text:
        return jsonify({"error": "Both 'to' and 'text' are required"}), 400

    if not _ready and not _ensure_whatsapp_ready():
        return jsonify({"error": "WhatsApp not logged in — scan QR in browser"}), 503

    ok = _send_text_message(to, text)
    return jsonify({"ok": ok})


@app.route("/whatsapp/send-image", methods=["POST"])
def whatsapp_send_image():
    data = request.get_json(silent=True) or {}
    to = data.get("to", "")
    image_url = data.get("image_url", "")
    caption = data.get("caption", "")

    if not to or not image_url:
        return jsonify({"error": "'to' and 'image_url' are required"}), 400

    if not _ready and not _ensure_whatsapp_ready():
        return jsonify({"error": "WhatsApp not logged in — scan QR in browser"}), 503

    ok = _send_image_message(to, image_url, caption)
    return jsonify({"ok": ok})


# ── Startup ───────────────────────────────────────────────────────────────────

def _start_browser(profile_dir: str, headless: bool = False):
    """Launch Playwright with a persistent context (keeps WhatsApp session)."""
    global _playwright, _browser, _context, _page, _profile_dir

    _profile_dir = profile_dir
    _playwright = sync_playwright().start()

    # Use a persistent context so WhatsApp stays logged in across restarts
    _context = _playwright.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
        locale="es-MX",
        viewport={"width": 1280, "height": 800},
    )

    _page = _context.pages[0] if _context.pages else _context.new_page()
    _page.goto("https://web.whatsapp.com", wait_until="domcontentloaded", timeout=30000)
    log.info("WhatsApp Web opened — profile: %s", profile_dir)

    # Wait for login (up to 60s for QR scan)
    _ensure_whatsapp_ready(timeout=60)
    if _ready:
        log.info("WhatsApp is logged in and ready!")
    else:
        log.warning("WhatsApp waiting for QR scan — open the browser window and scan the code")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatsApp Web Automation Service")
    parser.add_argument(
        "--profile",
        default="wa_profile_1",
        help="Browser profile directory (relative or absolute path). "
             "Use different profiles for different WhatsApp numbers.",
    )
    parser.add_argument("--port", type=int, default=5050, help="HTTP port (default: 5050)")
    parser.add_argument("--headless", action="store_true", help="Run browser headless (no window)")
    args = parser.parse_args()

    # Resolve profile path
    if os.path.isabs(args.profile):
        profile_path = args.profile
    else:
        profile_path = os.path.join(os.path.dirname(__file__), args.profile)
    os.makedirs(profile_path, exist_ok=True)

    # Start browser in background thread
    threading.Thread(
        target=_start_browser,
        args=(profile_path, args.headless),
        daemon=True,
    ).start()

    log.info("Starting WhatsApp service on port %d", args.port)
    app.run(host="0.0.0.0", port=args.port, debug=False)
