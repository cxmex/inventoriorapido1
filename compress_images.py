"""
Compress all images in Supabase storage bucket 'images_estilos'.

Downloads each image, compresses it (max 400px wide, JPEG quality 50),
and re-uploads it, replacing the original.

Usage:
    python compress_images.py              # compress all
    python compress_images.py --dry-run    # preview without uploading
    python compress_images.py --estilo 42  # compress only estilo ID 42
"""

import httpx
import io
import os
import sys
import json
import asyncio
from PIL import Image

SUPABASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co"
# Anon key for reading; service role key required for writes (bypasses RLS)
ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdia2hrYmZiYXJzbnBiZGt4emlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQzODAzNzMsImV4cCI6MjA0OTk1NjM3M30.mcOcC2GVEu_wD3xNBzSCC3MwDck3CIdmz4D8adU-bpI"
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

BUCKET = "images_estilos"
MAX_WIDTH = 400
JPEG_QUALITY = 50

def get_headers(for_write=False):
    key = SERVICE_KEY if for_write else ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def list_folders(client):
    resp = await client.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}",
        headers=get_headers(),
        json={"prefix": "", "limit": 10000},
    )
    resp.raise_for_status()
    return [f["name"] for f in resp.json() if f.get("name") and not f.get("id")]


async def list_files(client, folder):
    resp = await client.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}",
        headers=get_headers(),
        json={"prefix": f"{folder}/", "limit": 100},
    )
    resp.raise_for_status()
    return [f["name"] for f in resp.json() if f.get("name") and f.get("id")]


def compress_image(data):
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    original_size = len(data)

    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_height = int(img.height * ratio)
        img = img.resize((MAX_WIDTH, new_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    compressed = buf.getvalue()
    return compressed, original_size, len(compressed)


async def process_image(client, folder, filename, dry_run):
    path = f"{folder}/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.content

        if len(data) < 5000:
            return None  # skip tiny

        compressed, orig_size, comp_size = compress_image(data)
        savings = ((orig_size - comp_size) / orig_size * 100) if orig_size > 0 else 0

        if comp_size >= orig_size:
            print(f"  SKIP {path} ({orig_size/1024:.0f}KB, no savings)")
            return None

        print(f"  {path}: {orig_size/1024:.0f}KB -> {comp_size/1024:.0f}KB ({savings:.0f}% saved)")

        if not dry_run:
            write_key = SERVICE_KEY
            # Step 1: Delete old file
            del_resp = await client.request(
                "DELETE",
                f"{SUPABASE_URL}/storage/v1/object/{BUCKET}",
                headers={
                    "apikey": write_key,
                    "Authorization": f"Bearer {write_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps({"prefixes": [path]}).encode(),
            )
            if del_resp.status_code >= 400:
                print(f"    Delete error {del_resp.status_code}: {del_resp.text[:200]}")
                return None

            # Step 2: Upload compressed version
            resp = await client.post(
                f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}",
                headers={
                    "apikey": write_key,
                    "Authorization": f"Bearer {write_key}",
                    "Content-Type": "image/jpeg",
                },
                content=compressed,
            )
            if resp.status_code >= 400:
                print(f"    Upload error {resp.status_code}: {resp.text[:200]}")
                return None

        return (orig_size, comp_size)

    except Exception as e:
        print(f"  ERROR {path}: {e}")
        return None


async def main():
    dry_run = "--dry-run" in sys.argv
    filter_estilo = None
    if "--estilo" in sys.argv:
        idx = sys.argv.index("--estilo")
        filter_estilo = sys.argv[idx + 1]

    if not dry_run and not SERVICE_KEY:
        print("ERROR: Set SUPABASE_SERVICE_KEY env var to compress images.")
        print("  Find it in Supabase Dashboard > Settings > API > service_role key")
        print("  Usage: SUPABASE_SERVICE_KEY=eyJ... python compress_images.py")
        print("  Or use --dry-run to preview without uploading.")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Compressing images in {BUCKET}...")
    print(f"  Max width: {MAX_WIDTH}px, JPEG quality: {JPEG_QUALITY}")
    print()

    async with httpx.AsyncClient(timeout=30) as client:
        folders = await list_folders(client)
        if filter_estilo:
            folders = [f for f in folders if f == filter_estilo]

        print(f"Found {len(folders)} estilo folders")

        total_original = 0
        total_compressed = 0
        count = 0

        for folder in folders:
            files = await list_files(client, folder)
            if not files:
                continue

            # Process files in this folder concurrently
            tasks = [process_image(client, folder, f, dry_run) for f in files]
            results = await asyncio.gather(*tasks)

            for r in results:
                if r:
                    total_original += r[0]
                    total_compressed += r[1]
                    count += 1

    print()
    if total_original > 0:
        overall = ((total_original - total_compressed) / total_original * 100)
        print(f"{'[DRY RUN] ' if dry_run else ''}Done! {count} images compressed")
        print(f"  Total: {total_original/1024/1024:.1f}MB -> {total_compressed/1024/1024:.1f}MB ({overall:.0f}% saved)")
    else:
        print("No images needed compression.")


if __name__ == "__main__":
    asyncio.run(main())
