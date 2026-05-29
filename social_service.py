"""
social_service.py — Automated Social Media & Email Service
==========================================================
Run this locally. It opens persistent browser sessions for Facebook,
Instagram, and email. Log in ONCE, then everything is automated.

Usage:
    pip install playwright flask requests pillow
    playwright install chromium

    python social_service.py

    # First run: browser windows open — log into Facebook, Instagram, Gmail
    # After that: sessions persist, no more manual login needed

Then expose with ngrok:
    ngrok http 5052

Endpoints:
    POST /facebook/post          → post text + image to Facebook Page
    POST /instagram/post         → post image + caption to Instagram
    POST /email/send             → send email via Gmail/Outlook
    GET  /status                 → check which platforms are logged in
"""

import argparse
import os
import sys
import time
import json
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime

import requests as req
from flask import Flask, request, jsonify

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
except ImportError:
    print("Install: pip install playwright && playwright install chromium")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("social_service")

app = Flask(__name__)

# ── Browser globals ───────────────────────────────────────────────────────────
_playwright = None
_browser = None
_fb_page = None
_ig_page = None
_email_page = None
_fb_ready = False
_ig_ready = False
_email_ready = False
_profile_dir = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_image(url: str) -> str | None:
    """Download image to temp file, return path."""
    try:
        resp = req.get(url, timeout=15)
        if resp.status_code >= 400:
            return None
        suffix = ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        log.error("Image download failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FACEBOOK
# ═══════════════════════════════════════════════════════════════════════════════

def _check_fb_logged_in() -> bool:
    global _fb_ready, _fb_page
    if not _fb_page:
        return False
    try:
        url = _fb_page.url
        if "facebook.com" not in url:
            _fb_page.goto("https://www.facebook.com", wait_until="domcontentloaded", timeout=15000)
            time.sleep(2)
        # Check for the compose box or profile icon (means logged in)
        logged_in = _fb_page.query_selector('[aria-label="Create a post"]') or \
                    _fb_page.query_selector('[aria-label="Crear una publicación"]') or \
                    _fb_page.query_selector('[aria-label="Your profile"]') or \
                    _fb_page.query_selector('[aria-label="Tu perfil"]') or \
                    _fb_page.query_selector('div[role="banner"]')
        _fb_ready = logged_in is not None
        return _fb_ready
    except Exception as e:
        log.warning("FB login check failed: %s", e)
        return False


def _fb_post_to_page(page_name: str, text: str, image_path: str = None) -> bool:
    """Post to a Facebook Page. page_name is the page URL slug or ID."""
    global _fb_page
    try:
        # Navigate to the page
        _fb_page.goto(
            f"https://www.facebook.com/{page_name}",
            wait_until="domcontentloaded",
            timeout=15000,
        )
        time.sleep(3)

        # Click "Create post" or the post composer
        create_btn = (
            _fb_page.query_selector('[aria-label="Create a post"]') or
            _fb_page.query_selector('[aria-label="Crear una publicación"]') or
            _fb_page.query_selector('div[class*="sjgh65i0"]') or  # fallback
            _fb_page.query_selector('[role="button"]:has-text("Create post")') or
            _fb_page.query_selector('[role="button"]:has-text("Crear publicación")')
        )

        if not create_btn:
            # Try clicking the text area that says "What's on your mind" / "¿Qué estás pensando?"
            post_area = _fb_page.query_selector(
                'span:has-text("What\'s on your mind"),'
                'span:has-text("¿Qué estás pensando")'
            )
            if post_area:
                post_area.click()
            else:
                log.error("Could not find Facebook post composer")
                return False
        else:
            create_btn.click()

        time.sleep(2)

        # Wait for the post dialog
        post_box = _fb_page.wait_for_selector(
            'div[contenteditable="true"][role="textbox"]',
            timeout=8000,
        )
        if not post_box:
            log.error("Post dialog did not open")
            return False

        # Type the message
        post_box.click()
        post_box.type(text, delay=15)
        time.sleep(1)

        # Add image if provided
        if image_path and os.path.exists(image_path):
            # Click "Photo/Video" button
            photo_btn = (
                _fb_page.query_selector('[aria-label="Photo/video"]') or
                _fb_page.query_selector('[aria-label="Foto/video"]') or
                _fb_page.query_selector('[aria-label="Photo/Video"]')
            )
            if photo_btn:
                photo_btn.click()
                time.sleep(1)

            # Find file input
            file_input = _fb_page.query_selector('input[type="file"][accept*="image"]')
            if not file_input:
                inputs = _fb_page.query_selector_all('input[type="file"]')
                file_input = inputs[0] if inputs else None

            if file_input:
                file_input.set_input_files(image_path)
                time.sleep(3)  # wait for upload
            else:
                log.warning("Could not find file input for image")

        # Click "Post" / "Publicar"
        time.sleep(1)
        post_btn = (
            _fb_page.query_selector('[aria-label="Post"]') or
            _fb_page.query_selector('[aria-label="Publicar"]') or
            _fb_page.query_selector('div[aria-label="Post"][role="button"]') or
            _fb_page.query_selector('div[aria-label="Publicar"][role="button"]')
        )
        if post_btn:
            post_btn.click()
            time.sleep(3)
            log.info("Facebook post published to %s", page_name)
            return True
        else:
            # Try pressing Ctrl+Enter as fallback
            _fb_page.keyboard.press("Control+Enter")
            time.sleep(3)
            log.info("Facebook post published (via Ctrl+Enter)")
            return True

    except Exception as e:
        log.error("Facebook post failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# INSTAGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def _check_ig_logged_in() -> bool:
    global _ig_ready, _ig_page
    if not _ig_page:
        return False
    try:
        url = _ig_page.url
        if "instagram.com" not in url:
            _ig_page.goto("https://www.instagram.com", wait_until="domcontentloaded", timeout=15000)
            time.sleep(3)
        # Check for the new post button or profile link
        logged_in = (
            _ig_page.query_selector('svg[aria-label="New post"]') or
            _ig_page.query_selector('svg[aria-label="Nueva publicación"]') or
            _ig_page.query_selector('[aria-label="Home"]') or
            _ig_page.query_selector('[aria-label="Inicio"]')
        )
        _ig_ready = logged_in is not None
        return _ig_ready
    except Exception:
        return False


def _ig_post(image_path: str, caption: str) -> bool:
    """Post an image to Instagram."""
    global _ig_page
    try:
        _ig_page.goto("https://www.instagram.com", wait_until="domcontentloaded", timeout=15000)
        time.sleep(2)

        # Click "New post" / "Nueva publicación"
        new_post = (
            _ig_page.query_selector('svg[aria-label="New post"]') or
            _ig_page.query_selector('svg[aria-label="Nueva publicación"]')
        )
        if new_post:
            new_post.click()
        else:
            # Try the "Create" link in sidebar
            create_link = _ig_page.query_selector('a[href="/create/style/"]') or \
                          _ig_page.query_selector('span:has-text("Create")') or \
                          _ig_page.query_selector('span:has-text("Crear")')
            if create_link:
                create_link.click()
            else:
                log.error("Could not find Instagram new post button")
                return False

        time.sleep(2)

        # Upload image via file input
        file_input = _ig_page.wait_for_selector(
            'input[type="file"][accept*="image"]',
            timeout=5000,
        )
        if file_input:
            file_input.set_input_files(image_path)
        else:
            log.error("Instagram file input not found")
            return False

        time.sleep(3)

        # Click "Next" / "Siguiente" (crop screen)
        next_btn = (
            _ig_page.query_selector('button:has-text("Next")') or
            _ig_page.query_selector('button:has-text("Siguiente")') or
            _ig_page.query_selector('div[role="button"]:has-text("Next")') or
            _ig_page.query_selector('div[role="button"]:has-text("Siguiente")')
        )
        if next_btn:
            next_btn.click()
            time.sleep(2)

        # Click "Next" again (filters screen)
        next_btn2 = (
            _ig_page.query_selector('button:has-text("Next")') or
            _ig_page.query_selector('button:has-text("Siguiente")') or
            _ig_page.query_selector('div[role="button"]:has-text("Next")') or
            _ig_page.query_selector('div[role="button"]:has-text("Siguiente")')
        )
        if next_btn2:
            next_btn2.click()
            time.sleep(2)

        # Type caption
        caption_box = _ig_page.query_selector('textarea[aria-label*="caption"]') or \
                      _ig_page.query_selector('textarea[aria-label*="pie de foto"]') or \
                      _ig_page.query_selector('div[contenteditable="true"][role="textbox"]')
        if caption_box:
            caption_box.click()
            caption_box.type(caption, delay=10)
            time.sleep(1)

        # Click "Share" / "Compartir"
        share_btn = (
            _ig_page.query_selector('button:has-text("Share")') or
            _ig_page.query_selector('button:has-text("Compartir")') or
            _ig_page.query_selector('div[role="button"]:has-text("Share")') or
            _ig_page.query_selector('div[role="button"]:has-text("Compartir")')
        )
        if share_btn:
            share_btn.click()
            time.sleep(5)
            log.info("Instagram post published")
            return True

        log.error("Share button not found")
        return False

    except Exception as e:
        log.error("Instagram post failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# EMAIL (Gmail)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_email_logged_in() -> bool:
    global _email_ready, _email_page
    if not _email_page:
        return False
    try:
        url = _email_page.url
        if "mail.google" not in url:
            _email_page.goto("https://mail.google.com", wait_until="domcontentloaded", timeout=15000)
            time.sleep(3)
        logged_in = (
            _email_page.query_selector('div[gh="cm"]') or  # compose button
            _email_page.query_selector('[aria-label="Compose"]') or
            _email_page.query_selector('[aria-label="Redactar"]')
        )
        _email_ready = logged_in is not None
        return _email_ready
    except Exception:
        return False


def _send_email(to: str, subject: str, body: str, image_path: str = None) -> bool:
    """Send an email via Gmail web interface."""
    global _email_page
    try:
        _email_page.goto("https://mail.google.com", wait_until="domcontentloaded", timeout=15000)
        time.sleep(2)

        # Click Compose
        compose_btn = (
            _email_page.query_selector('[aria-label="Compose"]') or
            _email_page.query_selector('[aria-label="Redactar"]') or
            _email_page.query_selector('div[gh="cm"]')
        )
        if compose_btn:
            compose_btn.click()
        else:
            log.error("Compose button not found")
            return False

        time.sleep(2)

        # Fill To field
        to_field = _email_page.query_selector('input[aria-label="To recipients"]') or \
                   _email_page.query_selector('input[aria-label="Para"]') or \
                   _email_page.query_selector('textarea[name="to"]')
        if to_field:
            to_field.fill(to)
            _email_page.keyboard.press("Tab")
            time.sleep(0.5)

        # Fill Subject
        subj_field = _email_page.query_selector('input[name="subjectbox"]') or \
                     _email_page.query_selector('input[aria-label="Subject"]') or \
                     _email_page.query_selector('input[aria-label="Asunto"]')
        if subj_field:
            subj_field.fill(subject)
            time.sleep(0.5)

        # Fill Body
        body_field = _email_page.query_selector('div[aria-label="Message Body"]') or \
                     _email_page.query_selector('div[aria-label="Cuerpo del mensaje"]') or \
                     _email_page.query_selector('div[contenteditable="true"][role="textbox"]')
        if body_field:
            body_field.click()
            body_field.type(body, delay=5)

        # Attach image if provided
        if image_path and os.path.exists(image_path):
            attach_btn = _email_page.query_selector('[aria-label="Attach files"]') or \
                         _email_page.query_selector('[aria-label="Adjuntar archivos"]')
            if attach_btn:
                attach_btn.click()
                time.sleep(1)
                file_input = _email_page.query_selector('input[type="file"]')
                if file_input:
                    file_input.set_input_files(image_path)
                    time.sleep(3)

        # Click Send
        time.sleep(1)
        send_btn = _email_page.query_selector('[aria-label="Send"]') or \
                   _email_page.query_selector('[aria-label="Enviar"]') or \
                   _email_page.query_selector('div[data-tooltip="Send"]')
        if send_btn:
            send_btn.click()
            time.sleep(2)
            log.info("Email sent to %s", to)
            return True

        # Fallback: Ctrl+Enter
        _email_page.keyboard.press("Control+Enter")
        time.sleep(2)
        return True

    except Exception as e:
        log.error("Email send failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "facebook": _fb_ready,
        "instagram": _ig_ready,
        "email": _email_ready,
        "profile": _profile_dir,
    })


@app.route("/facebook/post", methods=["POST"])
def fb_post():
    data = request.get_json(silent=True) or {}
    page_name = data.get("page_name", "")
    text = data.get("text", "")
    image_url = data.get("image_url", "")

    if not page_name or not text:
        return jsonify({"error": "'page_name' and 'text' required"}), 400
    if not _fb_ready and not _check_fb_logged_in():
        return jsonify({"error": "Not logged into Facebook — open browser and log in"}), 503

    img_path = _download_image(image_url) if image_url else None
    ok = _fb_post_to_page(page_name, text, img_path)
    if img_path:
        os.unlink(img_path)
    return jsonify({"ok": ok})


@app.route("/instagram/post", methods=["POST"])
def ig_post():
    data = request.get_json(silent=True) or {}
    caption = data.get("caption", "")
    image_url = data.get("image_url", "")

    if not image_url:
        return jsonify({"error": "'image_url' required"}), 400
    if not _ig_ready and not _check_ig_logged_in():
        return jsonify({"error": "Not logged into Instagram — open browser and log in"}), 503

    img_path = _download_image(image_url)
    if not img_path:
        return jsonify({"error": "Could not download image"}), 400

    ok = _ig_post(img_path, caption)
    os.unlink(img_path)
    return jsonify({"ok": ok})


@app.route("/email/send", methods=["POST"])
def email_send():
    data = request.get_json(silent=True) or {}
    to = data.get("to", "")
    subject = data.get("subject", "")
    body = data.get("body", "")
    image_url = data.get("image_url", "")

    if not to or not subject:
        return jsonify({"error": "'to' and 'subject' required"}), 400
    if not _email_ready and not _check_email_logged_in():
        return jsonify({"error": "Not logged into Gmail — open browser and log in"}), 503

    img_path = _download_image(image_url) if image_url else None
    ok = _send_email(to, subject, body, img_path)
    if img_path:
        os.unlink(img_path)
    return jsonify({"ok": ok})


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def _start_browsers(profile_dir: str):
    """Launch one browser with 3 tabs: Facebook, Instagram, Gmail."""
    global _playwright, _browser, _fb_page, _ig_page, _email_page, _profile_dir

    _profile_dir = profile_dir
    _playwright = sync_playwright().start()

    context = _playwright.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,  # must be visible for first-time login
        args=["--disable-blink-features=AutomationControlled"],
        locale="es-MX",
        viewport={"width": 1280, "height": 800},
    )

    # Tab 1: Facebook
    _fb_page = context.pages[0] if context.pages else context.new_page()
    _fb_page.goto("https://www.facebook.com", wait_until="domcontentloaded", timeout=20000)
    log.info("Facebook tab opened")

    # Tab 2: Instagram
    _ig_page = context.new_page()
    _ig_page.goto("https://www.instagram.com", wait_until="domcontentloaded", timeout=20000)
    log.info("Instagram tab opened")

    # Tab 3: Gmail
    _email_page = context.new_page()
    _email_page.goto("https://mail.google.com", wait_until="domcontentloaded", timeout=20000)
    log.info("Gmail tab opened")

    time.sleep(5)

    # Check login status
    _check_fb_logged_in()
    _check_ig_logged_in()
    _check_email_logged_in()

    platforms = []
    if _fb_ready:
        platforms.append("Facebook ✓")
    else:
        platforms.append("Facebook ✗ (log in manually)")
    if _ig_ready:
        platforms.append("Instagram ✓")
    else:
        platforms.append("Instagram ✗ (log in manually)")
    if _email_ready:
        platforms.append("Gmail ✓")
    else:
        platforms.append("Gmail ✗ (log in manually)")

    log.info("Platform status: %s", " | ".join(platforms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Media Automation Service")
    parser.add_argument("--profile", default="social_profile", help="Browser profile directory")
    parser.add_argument("--port", type=int, default=5052, help="HTTP port (default: 5052)")
    args = parser.parse_args()

    if os.path.isabs(args.profile):
        profile_path = args.profile
    else:
        profile_path = os.path.join(os.path.dirname(__file__), args.profile)
    os.makedirs(profile_path, exist_ok=True)

    threading.Thread(
        target=_start_browsers,
        args=(profile_path,),
        daemon=True,
    ).start()

    log.info("Starting social service on port %d", args.port)
    app.run(host="0.0.0.0", port=args.port, debug=False)
