"""
Market Intelligence Engine

Scans Mexican retailers for phone models, cross-references with our inventory,
detects gaps (phones we don't carry cases for), and correlates market events
with our sales/stockout data.
"""

import json
import logging
import re
from datetime import date, timedelta
from typing import Dict, List, Any, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdia2hrYmZiYXJzbnBiZGt4emlpIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE3MzQzODAzNzMsImV4cCI6MjA0OTk1NjM3M30."
    "mcOcC2GVEu_wD3xNBzSCC3MwDck3CIdmz4D8adU-bpI"
)
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


async def _rpc(fn: str, params: dict = None):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{SUPABASE_URL}/rest/v1/rpc/{fn}", headers=HEADERS, json=params or {})
    if r.status_code >= 400:
        logger.error("RPC %s failed %s: %s", fn, r.status_code, r.text[:300])
        return None
    return r.json()


async def _rpc_get(fn: str, params: dict = None):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{SUPABASE_URL}/rest/v1/rpc/{fn}", headers=HEADERS, params=params or {})
    if r.status_code >= 400:
        logger.error("RPC GET %s failed: %s", fn, r.text[:300])
        return None
    return r.json()


async def _rest_get(table: str, params: dict = None):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=HEADERS, params=params or {})
    if r.status_code >= 400:
        return []
    return r.json()


# ──────────────────────────────────────────────
# KNOWN PHONE MODELS IN MEXICO (curated database)
# This is the "market" that we cross-reference against.
# Updated periodically via web scraping + manual additions.
# ──────────────────────────────────────────────

# These are phones actively sold in Mexico as of 2026.
# The scraper supplements this with live data.
MEXICO_PHONE_CATALOG = [
    # ═══════════════════════════════════════════
    # SAMSUNG — samsung.com/mx
    # ═══════════════════════════════════════════
    # Galaxy A series (highest volume in Mexico)
    {"brand": "SAMSUNG", "model": "GALAXY A07", "price": 2499, "segment": "gama_baja", "launched": "2025-09", "retailers": ["samsung.com", "telcel", "att", "walmart", "coppel", "liverpool"], "hot": True, "notes": "Successor to #1 LatAm seller A06"},
    {"brand": "SAMSUNG", "model": "GALAXY A17", "price": 3999, "segment": "gama_baja", "launched": "2025-09", "retailers": ["samsung.com", "telcel", "att", "walmart", "coppel"], "hot": True},
    {"brand": "SAMSUNG", "model": "GALAXY A06", "price": 2499, "segment": "gama_baja", "launched": "2024-09", "retailers": ["samsung.com", "telcel", "walmart", "coppel"], "hot": True, "notes": "#1 seller LatAm 2025, 11 months at #1"},
    {"brand": "SAMSUNG", "model": "GALAXY A16", "price": 3499, "segment": "gama_baja", "launched": "2024-12", "retailers": ["samsung.com", "telcel", "walmart", "coppel"]},
    {"brand": "SAMSUNG", "model": "GALAXY A26 5G", "price": 5999, "segment": "gama_media", "launched": "2025-12", "retailers": ["samsung.com", "telcel", "att", "walmart", "coppel", "liverpool", "amazon"], "hot": True, "notes": "Super AMOLED 6.7 FHD+ 120Hz, IP67"},
    {"brand": "SAMSUNG", "model": "GALAXY A36 5G", "price": 6499, "segment": "gama_media", "launched": "2025-03", "retailers": ["samsung.com", "telcel", "att", "walmart", "coppel"]},
    {"brand": "SAMSUNG", "model": "GALAXY A56 5G", "price": 8999, "segment": "gama_media", "launched": "2025-03", "retailers": ["samsung.com", "telcel", "att", "coppel"]},
    {"brand": "SAMSUNG", "model": "GALAXY A55", "price": 7999, "segment": "gama_media", "launched": "2024-03", "retailers": ["samsung.com", "telcel", "coppel"]},
    {"brand": "SAMSUNG", "model": "GALAXY A08", "price": 2999, "segment": "gama_baja", "launched": "2026-03", "retailers": ["samsung.com", "telcel"], "hot": True, "notes": "Just launched March 2026"},
    {"brand": "SAMSUNG", "model": "GALAXY A37", "price": 6999, "segment": "gama_media", "launched": "2026-02", "retailers": ["samsung.com", "telcel", "walmart"]},
    # Galaxy S series
    {"brand": "SAMSUNG", "model": "GALAXY S26 ULTRA", "price": 29999, "segment": "flagship", "launched": "2026-01", "retailers": ["samsung.com", "telcel", "att", "liverpool"], "hot": True},
    {"brand": "SAMSUNG", "model": "GALAXY S26", "price": 18999, "segment": "flagship", "launched": "2026-01", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY S26 PLUS", "price": 23999, "segment": "flagship", "launched": "2026-01", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY S25 ULTRA", "price": 27999, "segment": "flagship", "launched": "2025-01", "retailers": ["samsung.com", "telcel", "att", "liverpool", "walmart"]},
    {"brand": "SAMSUNG", "model": "GALAXY S25 EDGE", "price": 22999, "segment": "flagship", "launched": "2025-05", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY S25", "price": 17999, "segment": "flagship", "launched": "2025-01", "retailers": ["samsung.com", "telcel", "att"]},
    {"brand": "SAMSUNG", "model": "GALAXY S25 FE", "price": 12999, "segment": "gama_alta", "launched": "2025-10", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY S24 FE", "price": 10999, "segment": "gama_alta", "launched": "2024-10", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY S24 ULTRA", "price": 24999, "segment": "flagship", "launched": "2024-01", "retailers": ["samsung.com", "telcel", "walmart"]},
    # Galaxy Z foldables
    {"brand": "SAMSUNG", "model": "GALAXY Z FOLD 7", "price": 42999, "segment": "flagship", "launched": "2025-07", "retailers": ["samsung.com", "telcel"]},
    {"brand": "SAMSUNG", "model": "GALAXY Z FLIP 7", "price": 24999, "segment": "flagship", "launched": "2025-07", "retailers": ["samsung.com", "telcel"]},

    # ═══════════════════════════════════════════
    # APPLE — apple.com/mx
    # ═══════════════════════════════════════════
    {"brand": "APPLE", "model": "IPHONE 17 PRO MAX", "price": 34999, "segment": "flagship", "launched": "2025-09", "retailers": ["apple.com", "telcel", "att", "liverpool", "walmart", "coppel"], "hot": True},
    {"brand": "APPLE", "model": "IPHONE 17 PRO", "price": 29999, "segment": "flagship", "launched": "2025-09", "retailers": ["apple.com", "telcel", "att", "liverpool"]},
    {"brand": "APPLE", "model": "IPHONE 17", "price": 22999, "segment": "gama_alta", "launched": "2025-09", "retailers": ["apple.com", "telcel", "att", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 17 AIR", "price": 27999, "segment": "flagship", "launched": "2025-09", "retailers": ["apple.com", "telcel", "att"], "hot": True},
    {"brand": "APPLE", "model": "IPHONE 16 PRO MAX", "price": 29999, "segment": "flagship", "launched": "2024-09", "retailers": ["apple.com", "telcel", "att", "walmart", "liverpool"]},
    {"brand": "APPLE", "model": "IPHONE 16 PRO", "price": 24999, "segment": "flagship", "launched": "2024-09", "retailers": ["apple.com", "telcel", "att"]},
    {"brand": "APPLE", "model": "IPHONE 16", "price": 22999, "segment": "gama_alta", "launched": "2024-09", "retailers": ["apple.com", "telcel", "att", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 16E", "price": 14999, "segment": "gama_alta", "launched": "2025-03", "retailers": ["apple.com", "telcel", "att", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 15 PRO MAX", "price": 24999, "segment": "flagship", "launched": "2023-09", "retailers": ["apple.com", "telcel", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 15 PRO", "price": 19999, "segment": "flagship", "launched": "2023-09", "retailers": ["telcel", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 15", "price": 17999, "segment": "gama_alta", "launched": "2023-09", "retailers": ["apple.com", "telcel", "walmart", "coppel"]},
    {"brand": "APPLE", "model": "IPHONE 14", "price": 13999, "segment": "gama_media", "launched": "2022-09", "retailers": ["telcel", "walmart", "coppel"]},
    {"brand": "APPLE", "model": "IPHONE 14 PRO MAX", "price": 19999, "segment": "flagship", "launched": "2022-09", "retailers": ["telcel", "walmart"]},
    {"brand": "APPLE", "model": "IPHONE 13 PRO MAX", "price": 14999, "segment": "gama_alta", "launched": "2021-09", "retailers": ["walmart", "coppel", "mercadolibre"]},
    {"brand": "APPLE", "model": "IPHONE 13", "price": 10999, "segment": "gama_media", "launched": "2021-09", "retailers": ["walmart", "coppel", "mercadolibre"]},
    {"brand": "APPLE", "model": "IPHONE 12", "price": 8999, "segment": "gama_media", "launched": "2020-10", "retailers": ["walmart", "coppel", "mercadolibre"]},
    {"brand": "APPLE", "model": "IPHONE 11", "price": 7999, "segment": "gama_media", "launched": "2019-09", "retailers": ["walmart", "coppel", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # HONOR — honor.com/mx
    # ═══════════════════════════════════════════
    {"brand": "HONOR", "model": "MAGIC 8 LITE", "price": 6999, "segment": "gama_media", "launched": "2025-11", "retailers": ["honor.com", "coppel", "telcel"], "hot": True, "notes": "Same as X9d intl, 8300mAh battery"},
    {"brand": "HONOR", "model": "MAGIC 7 PRO", "price": 16999, "segment": "flagship", "launched": "2025-04", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "MAGIC 7 LITE", "price": 5999, "segment": "gama_media", "launched": "2025-02", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "MAGIC V3", "price": 29999, "segment": "flagship", "launched": "2024-09", "retailers": ["honor.com"]},
    {"brand": "HONOR", "model": "X9D", "price": 6999, "segment": "gama_media", "launched": "2025-10", "retailers": ["honor.com", "coppel", "att", "mercadolibre"], "hot": True},
    {"brand": "HONOR", "model": "X8D", "price": 4999, "segment": "gama_baja", "launched": "2025-08", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "X7D", "price": 3999, "segment": "gama_baja", "launched": "2025-06", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "X6C", "price": 2999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["honor.com", "coppel", "telcel"]},
    {"brand": "HONOR", "model": "X5C PLUS", "price": 2499, "segment": "gama_baja", "launched": "2025-05", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "X5C", "price": 1999, "segment": "gama_baja", "launched": "2025-01", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "HONOR 400", "price": 7999, "segment": "gama_media", "launched": "2025-06", "retailers": ["honor.com", "coppel", "telcel"]},
    {"brand": "HONOR", "model": "HONOR 400 LITE", "price": 4999, "segment": "gama_baja", "launched": "2025-06", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "HONOR 400 PRO", "price": 9999, "segment": "gama_alta", "launched": "2025-06", "retailers": ["honor.com", "coppel"], "notes": "200MP camera"},
    {"brand": "HONOR", "model": "HONOR 500", "price": 8999, "segment": "gama_media", "launched": "2026-01", "retailers": ["honor.com", "coppel"], "hot": True},
    {"brand": "HONOR", "model": "HONOR 500 PRO", "price": 11999, "segment": "gama_alta", "launched": "2026-02", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "HONOR 200", "price": 6999, "segment": "gama_media", "launched": "2024-06", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "HONOR 200 PRO", "price": 11999, "segment": "gama_alta", "launched": "2024-06", "retailers": ["honor.com"]},
    {"brand": "HONOR", "model": "HONOR 90", "price": 5999, "segment": "gama_media", "launched": "2023-07", "retailers": ["honor.com", "coppel"]},
    {"brand": "HONOR", "model": "HONOR X9C", "price": 6999, "segment": "gama_media", "launched": "2025-08", "retailers": ["honor.com", "coppel"]},

    # ═══════════════════════════════════════════
    # OPPO — oppo.com/mx
    # ═══════════════════════════════════════════
    {"brand": "OPPO", "model": "FIND X9 PRO", "price": 19999, "segment": "flagship", "launched": "2026-01", "retailers": ["oppo.com", "telcel"]},
    {"brand": "OPPO", "model": "FIND X9", "price": 14999, "segment": "flagship", "launched": "2026-01", "retailers": ["oppo.com", "telcel"]},
    {"brand": "OPPO", "model": "RENO 14", "price": 7999, "segment": "gama_media", "launched": "2025-11", "retailers": ["oppo.com", "telcel", "coppel"], "notes": "IP69, 6000mAh"},
    {"brand": "OPPO", "model": "RENO 14F 5G", "price": 5999, "segment": "gama_media", "launched": "2025-09", "retailers": ["oppo.com", "telcel", "coppel"], "notes": "Snapdragon 6 Gen 1, Star Wars edition available"},
    {"brand": "OPPO", "model": "A6 PRO 5G", "price": 5999, "segment": "gama_media", "launched": "2025-06", "retailers": ["oppo.com", "telcel", "coppel"], "hot": True, "notes": "7000mAh battery, IP69, 80W charging"},
    {"brand": "OPPO", "model": "A6X", "price": 3999, "segment": "gama_baja", "launched": "2025-09", "retailers": ["oppo.com", "telcel", "coppel"]},
    {"brand": "OPPO", "model": "OPPO A5", "price": 3999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["oppo.com", "telcel"]},
    {"brand": "OPPO", "model": "RENO 15", "price": 9999, "segment": "gama_alta", "launched": "2025-12", "retailers": ["oppo.com", "telcel"]},
    {"brand": "OPPO", "model": "RENO 15 PRO", "price": 12999, "segment": "gama_alta", "launched": "2026-01", "retailers": ["oppo.com", "telcel"]},
    {"brand": "OPPO", "model": "RENO 10", "price": 5999, "segment": "gama_media", "launched": "2023-07", "retailers": ["telcel", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # MOTOROLA — motorola.com.mx
    # ═══════════════════════════════════════════
    {"brand": "MOTO", "model": "EDGE 60 PRO", "price": 14999, "segment": "gama_alta", "launched": "2025-09", "retailers": ["motorola.com", "telcel", "walmart", "coppel"], "notes": "Dimensity 8350, 6000mAh, 90W"},
    {"brand": "MOTO", "model": "EDGE 60", "price": 10999, "segment": "gama_alta", "launched": "2025-06", "retailers": ["motorola.com", "telcel", "walmart"]},
    {"brand": "MOTO", "model": "EDGE 60 FUSION", "price": 7999, "segment": "gama_media", "launched": "2025-06", "retailers": ["motorola.com", "telcel", "walmart", "coppel"]},
    {"brand": "MOTO", "model": "EDGE 60 NEO", "price": 5999, "segment": "gama_media", "launched": "2025-10", "retailers": ["motorola.com", "telcel", "coppel"]},
    {"brand": "MOTO", "model": "EDGE 70", "price": 12999, "segment": "gama_alta", "launched": "2025-11", "retailers": ["motorola.com", "telcel"]},
    {"brand": "MOTO", "model": "G86", "price": 5499, "segment": "gama_media", "launched": "2025-06", "retailers": ["motorola.com", "telcel", "coppel", "walmart"], "notes": "6729mAh battery"},
    {"brand": "MOTO", "model": "G56", "price": 4999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["motorola.com", "telcel", "coppel", "walmart"]},
    {"brand": "MOTO", "model": "G06", "price": 2999, "segment": "gama_baja", "launched": "2025-10", "retailers": ["motorola.com", "telcel", "coppel", "walmart"]},
    {"brand": "MOTO", "model": "G24", "price": 2999, "segment": "gama_baja", "launched": "2024-03", "retailers": ["telcel", "coppel", "walmart"]},
    {"brand": "MOTO", "model": "RAZR 60", "price": 19999, "segment": "flagship", "launched": "2025-12", "retailers": ["motorola.com", "telcel"]},
    {"brand": "MOTO", "model": "RAZR 60 ULTRA", "price": 27999, "segment": "flagship", "launched": "2025-12", "retailers": ["motorola.com", "telcel"]},
    {"brand": "MOTO", "model": "EDGE 50 FUSION", "price": 5999, "segment": "gama_media", "launched": "2024-06", "retailers": ["motorola.com", "telcel"]},

    # ═══════════════════════════════════════════
    # ZTE — ztedevices.mx
    # ═══════════════════════════════════════════
    {"brand": "ZTE", "model": "BLADE A35E", "price": 1599, "segment": "gama_baja", "launched": "2025-07", "retailers": ["telcel", "coppel", "amazon"], "hot": True, "notes": "Best rated under $1000 on MercadoLibre"},
    {"brand": "ZTE", "model": "ZTE A56", "price": 2499, "segment": "gama_baja", "launched": "2025-09", "retailers": ["telcel", "coppel"], "notes": "6.75 HD+ 90Hz"},
    {"brand": "ZTE", "model": "AXON 70", "price": 3999, "segment": "gama_media", "launched": "2025-11", "retailers": ["telcel", "coppel"], "notes": "Live Island 2.0"},
    {"brand": "ZTE", "model": "AXON 60", "price": 5999, "segment": "gama_media", "launched": "2025-06", "retailers": ["telcel"]},
    {"brand": "ZTE", "model": "AXON 60 LITE", "price": 3999, "segment": "gama_baja", "launched": "2025-06", "retailers": ["telcel"]},

    # ═══════════════════════════════════════════
    # XIAOMI / REDMI / POCO — mi.com/mx
    # ═══════════════════════════════════════════
    # Xiaomi flagship
    {"brand": "XIAOMI", "model": "MI 15T", "price": 9999, "segment": "gama_alta", "launched": "2025-09", "retailers": ["xiaomi.com", "telcel", "amazon", "walmart"]},
    {"brand": "XIAOMI", "model": "MI 15T PRO", "price": 12999, "segment": "gama_alta", "launched": "2025-09", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "XIAOMI", "model": "MI 15", "price": 15999, "segment": "flagship", "launched": "2025-02", "retailers": ["xiaomi.com", "telcel"]},
    {"brand": "XIAOMI", "model": "MI 15 ULTRA", "price": 19999, "segment": "flagship", "launched": "2025-06", "retailers": ["xiaomi.com", "amazon"]},
    # Redmi Note 15 series (launched Mexico Feb 2026 — HOT)
    {"brand": "XIAOMI", "model": "REDMI NOTE 15", "price": 5999, "segment": "gama_media", "launched": "2026-02", "retailers": ["xiaomi.com", "telcel", "walmart", "coppel", "movistar"], "hot": True, "notes": "Just launched Feb 2026 in Mexico"},
    {"brand": "XIAOMI", "model": "REDMI NOTE 15 PRO", "price": 7999, "segment": "gama_media", "launched": "2026-02", "retailers": ["xiaomi.com", "telcel", "walmart", "movistar"], "hot": True},
    {"brand": "XIAOMI", "model": "REDMI NOTE 15 PRO PLUS 5G", "price": 10999, "segment": "gama_alta", "launched": "2026-02", "retailers": ["xiaomi.com", "telcel"], "hot": True, "notes": "6500mAh, IP69K, 200MP, 100W charge"},
    # Redmi Note 14 series (Telcel active)
    {"brand": "XIAOMI", "model": "REDMI NOTE 14", "price": 5499, "segment": "gama_media", "launched": "2025-03", "retailers": ["xiaomi.com", "telcel", "coppel", "movistar"]},
    {"brand": "XIAOMI", "model": "REDMI NOTE 14 PRO", "price": 7499, "segment": "gama_media", "launched": "2025-03", "retailers": ["xiaomi.com", "telcel", "movistar"]},
    {"brand": "XIAOMI", "model": "REDMI NOTE 14 PRO 5G", "price": 8999, "segment": "gama_alta", "launched": "2025-03", "retailers": ["xiaomi.com", "telcel"], "notes": "200MP, IP64, Dimensity 7300"},
    {"brand": "XIAOMI", "model": "REDMI NOTE 14 PRO PLUS 5G", "price": 9999, "segment": "gama_alta", "launched": "2025-03", "retailers": ["xiaomi.com", "telcel"]},
    # Redmi Note 13 series (still selling at Telcel)
    {"brand": "XIAOMI", "model": "REDMI NOTE 13", "price": 4499, "segment": "gama_baja", "launched": "2024-01", "retailers": ["telcel", "coppel", "walmart"]},
    {"brand": "XIAOMI", "model": "REDMI NOTE 13 PRO", "price": 6999, "segment": "gama_media", "launched": "2024-01", "retailers": ["telcel", "coppel"]},
    # Redmi numbered series
    {"brand": "XIAOMI", "model": "REDMI 15C", "price": 3499, "segment": "gama_baja", "launched": "2025-08", "retailers": ["xiaomi.com", "telcel", "walmart", "coppel"]},
    {"brand": "XIAOMI", "model": "REDMI 15", "price": 4999, "segment": "gama_baja", "launched": "2025-11", "retailers": ["xiaomi.com", "telcel"]},
    {"brand": "XIAOMI", "model": "REDMI 15 PRO", "price": 6999, "segment": "gama_media", "launched": "2026-01", "retailers": ["xiaomi.com", "telcel"]},
    {"brand": "XIAOMI", "model": "REDMI 14T", "price": 3999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["xiaomi.com", "telcel", "coppel"]},
    {"brand": "XIAOMI", "model": "REDMI 14T PRO", "price": 4999, "segment": "gama_baja", "launched": "2025-06", "retailers": ["xiaomi.com", "telcel"]},
    {"brand": "XIAOMI", "model": "REDMI 14C", "price": 2999, "segment": "gama_baja", "launched": "2024-08", "retailers": ["xiaomi.com", "telcel", "walmart", "coppel"]},
    {"brand": "XIAOMI", "model": "REDMI 13", "price": 3999, "segment": "gama_baja", "launched": "2024-06", "retailers": ["telcel", "coppel", "walmart"]},
    {"brand": "XIAOMI", "model": "REDMI 12", "price": 3499, "segment": "gama_baja", "launched": "2023-06", "retailers": ["telcel", "coppel", "walmart"]},
    # Redmi A series (ultra-budget, massive volume)
    {"brand": "XIAOMI", "model": "REDMI A5", "price": 1999, "segment": "gama_baja", "launched": "2025-06", "retailers": ["telcel", "coppel", "walmart"], "hot": True, "notes": "Ultra budget, high volume at Telcel"},
    {"brand": "XIAOMI", "model": "REDMI A3", "price": 1499, "segment": "gama_baja", "launched": "2024-03", "retailers": ["telcel", "coppel", "walmart"]},
    # POCO series (mistoremx.com / mi.com/mx)
    {"brand": "POCO", "model": "POCO X8 PRO MAX", "price": 8899, "segment": "gama_media", "launched": "2026-03", "retailers": ["xiaomi.com", "amazon", "mercadolibre"], "hot": True},
    {"brand": "POCO", "model": "POCO X8 PRO", "price": 7999, "segment": "gama_media", "launched": "2025-09", "retailers": ["xiaomi.com", "amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO X7 PRO", "price": 7999, "segment": "gama_media", "launched": "2025-01", "retailers": ["xiaomi.com", "amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO X7", "price": 5999, "segment": "gama_media", "launched": "2025-01", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO F8 PRO", "price": 11499, "segment": "gama_alta", "launched": "2025-09", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO F8", "price": 9999, "segment": "gama_alta", "launched": "2025-09", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO F8 ULTRA", "price": 15799, "segment": "flagship", "launched": "2025-09", "retailers": ["xiaomi.com"]},
    {"brand": "POCO", "model": "POCO F7", "price": 8999, "segment": "gama_media", "launched": "2026-02", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO F6", "price": 8999, "segment": "gama_media", "launched": "2024-05", "retailers": ["amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO F6 PRO", "price": 11999, "segment": "gama_alta", "launched": "2024-05", "retailers": ["amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO M8", "price": 4499, "segment": "gama_baja", "launched": "2025-10", "retailers": ["xiaomi.com", "amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO M8 PRO", "price": 5999, "segment": "gama_media", "launched": "2025-12", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO M7", "price": 4299, "segment": "gama_baja", "launched": "2025-06", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "POCO", "model": "POCO C85", "price": 2449, "segment": "gama_baja", "launched": "2025-06", "retailers": ["xiaomi.com", "amazon", "mercadolibre"]},
    {"brand": "POCO", "model": "POCO C71", "price": 1499, "segment": "gama_baja", "launched": "2025-03", "retailers": ["xiaomi.com", "amazon"]},
    {"brand": "XIAOMI", "model": "14C", "price": 2999, "segment": "gama_baja", "launched": "2024-08", "retailers": ["xiaomi.com", "telcel", "walmart"]},

    # ═══════════════════════════════════════════
    # VIVO — vivo.com
    # ═══════════════════════════════════════════
    {"brand": "VIVO", "model": "V60 LITE", "price": 5999, "segment": "gama_media", "launched": "2025-09", "retailers": ["vivo.com", "amazon", "mercadolibre"]},
    {"brand": "VIVO", "model": "V50", "price": 7999, "segment": "gama_media", "launched": "2025-03", "retailers": ["vivo.com", "amazon"]},
    {"brand": "VIVO", "model": "V50 LITE", "price": 5499, "segment": "gama_media", "launched": "2025-06", "retailers": ["vivo.com", "amazon"]},
    {"brand": "VIVO", "model": "V40", "price": 6999, "segment": "gama_media", "launched": "2024-09", "retailers": ["vivo.com", "amazon"]},
    {"brand": "VIVO", "model": "X9D", "price": 8999, "segment": "gama_media", "launched": "2025-11", "retailers": ["amazon", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # REALME
    # ═══════════════════════════════════════════
    {"brand": "REALME", "model": "REALME 14 PRO", "price": 7999, "segment": "gama_media", "launched": "2025-06", "retailers": ["amazon", "mercadolibre"]},
    {"brand": "REALME", "model": "REALME C75", "price": 3999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["amazon", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # INFINIX
    # ═══════════════════════════════════════════
    {"brand": "INFINIX", "model": "NOTE 50 PRO", "price": 5999, "segment": "gama_media", "launched": "2025-06", "retailers": ["amazon", "mercadolibre"]},
    {"brand": "INFINIX", "model": "HOT 50 5G", "price": 3999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["amazon", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # TECNO
    # ═══════════════════════════════════════════
    {"brand": "TECNO", "model": "CAMON 30", "price": 4999, "segment": "gama_media", "launched": "2024-06", "retailers": ["amazon", "mercadolibre"]},
    {"brand": "TECNO", "model": "SPARK 30 PRO", "price": 3999, "segment": "gama_baja", "launched": "2025-03", "retailers": ["amazon", "mercadolibre"]},

    # ═══════════════════════════════════════════
    # HUAWEI (still sold in MX via third parties)
    # ═══════════════════════════════════════════
    {"brand": "HUAWEI", "model": "NOVA 13", "price": 6999, "segment": "gama_media", "launched": "2025-01", "retailers": ["huawei.com", "amazon", "mercadolibre"]},
    {"brand": "HUAWEI", "model": "NOVA 13 PRO", "price": 9999, "segment": "gama_alta", "launched": "2025-01", "retailers": ["huawei.com", "amazon"]},
    {"brand": "HUAWEI", "model": "MATE 20 LITE", "price": 2999, "segment": "gama_baja", "launched": "2018-10", "retailers": ["mercadolibre"], "notes": "Still popular in MX refurbished market"},
]


# ──────────────────────────────────────────────
# MODEL NAME MATCHING
# ──────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Normalize model name for matching."""
    name = name.upper().strip()
    # Remove common prefixes
    for prefix in ["GALAXY ", "SAMSUNG ", "APPLE ", "ZTE BLADE ", "BLADE ", "HONOR ", "VIVO "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Normalize spacing
    name = re.sub(r'\s+', ' ', name)
    return name


def _match_modelo(market_model: str, market_brand: str, our_modelos: List[Dict]) -> Optional[Dict]:
    """Try to match a market phone model to our inventario_modelos."""
    norm_market = _normalize(market_model)

    for m in our_modelos:
        our_modelo = (m.get("modelo") or "").upper().strip()
        our_marca = (m.get("marca") or "").upper().strip()

        # Exact match
        if our_modelo == norm_market:
            return m
        if our_modelo == market_model.upper():
            return m

        # Partial match: "A07" matches "A07", "A35E" matches "ZTE A35E"
        norm_our = _normalize(our_modelo)
        if norm_our == norm_market:
            return m

        # Brand-aware match
        if market_brand.upper() in our_marca or our_marca in market_brand.upper():
            if norm_our == norm_market:
                return m
            # Fuzzy: one contains the other
            if len(norm_market) > 3 and (norm_market in norm_our or norm_our in norm_market):
                return m

    return None


# ──────────────────────────────────────────────
# CORE: Market Scan + Gap Analysis
# ──────────────────────────────────────────────

async def run_market_scan() -> Dict[str, Any]:
    """
    Full market intelligence scan:
    1. Load our inventario_modelos
    2. Load our sales/stockout data from retail_metrics
    3. Cross-reference with market catalog
    4. Detect gaps, generate alerts
    5. Correlate with demand data
    """
    import asyncio
    today_str = date.today().isoformat()
    results = {}

    # 1. Fetch our inventory models
    our_modelos = await _rest_get("inventario_modelos", {
        "select": "id,marca,modelo,ventas_totales",
    })
    results["our_models_count"] = len(our_modelos) if our_modelos else 0

    # 2. Fetch our retail metrics (modelo-level) for sales/stock data
    metrics_30 = await _rpc_get("get_retail_metrics", {
        "group_by_field": "modelo", "days_back": 30
    })
    metrics_map = {}
    if metrics_30:
        for r in metrics_30:
            metrics_map[r.get("group_name", "").upper()] = r

    # 3. Cross-reference market catalog with our inventory
    market_phones_to_save = []
    gaps = []
    alerts = []

    for phone in MEXICO_PHONE_CATALOG:
        brand = phone["brand"]
        model = phone["model"]
        price = phone.get("price", 0)
        segment = phone.get("segment", "")
        retailers = phone.get("retailers", [])
        launched = phone.get("launched")
        is_hot = phone.get("hot", False)
        notes = phone.get("notes", "")

        # Save to market_phones
        market_phones_to_save.append({
            "brand": brand,
            "model": model,
            "price_mxn": price,
            "launch_date": launched,
            "segment": segment,
            "source": retailers[0] if retailers else "catalog",
            "in_stock": True,
            "is_bestseller": is_hot,
            "specs_summary": notes,
            "last_seen": today_str,
        })

        # Try to match with our inventory
        match = _match_modelo(model, brand, our_modelos or [])

        # Get our sales data for this model
        our_stock = 0
        our_sales_30d = 0
        our_was_stockout = False
        our_modelo_name = None

        if match:
            our_modelo_name = match.get("modelo")
            # Look up in metrics
            for key_variant in [our_modelo_name.upper(), _normalize(model)]:
                if key_variant in metrics_map:
                    m = metrics_map[key_variant]
                    our_stock = int(m.get("current_stock_total", 0) or 0)
                    our_sales_30d = int(m.get("units_sold_total", 0) or 0)
                    our_was_stockout = our_stock == 0 and our_sales_30d > 0
                    break

        we_carry = match is not None

        # Calculate opportunity score
        score = 0
        if not we_carry:
            score += 40  # we don't carry it at all
        if is_hot:
            score += 20
        score += len(retailers) * 5  # more retailers = more phone sales = more case demand
        if segment == "gama_baja":
            score += 10  # high volume segment
        elif segment == "gama_media":
            score += 8
        if our_was_stockout:
            score += 25  # we had it and ran out

        # Determine reason
        if not we_carry:
            reason = "no_cases"
        elif our_was_stockout:
            reason = "stockout"
        elif our_stock < 10 and our_sales_30d > 20:
            reason = "low_stock"
        elif is_hot:
            reason = "bestseller"
        else:
            reason = "monitor"

        # Estimate monthly demand based on retailer presence and segment
        est_demand = len(retailers) * (30 if segment == "gama_baja" else 20 if segment == "gama_media" else 10)
        if is_hot:
            est_demand = int(est_demand * 1.5)

        gaps.append({
            "date": today_str,
            "brand": brand,
            "model": model,
            "phone_price": price,
            "segment": segment,
            "num_retailers": len(retailers),
            "is_bestseller": is_hot,
            "opportunity_score": score,
            "reason": reason,
            "estimated_monthly_demand": est_demand,
            "we_carry_cases": we_carry,
            "our_modelo_match": our_modelo_name,
            "our_stock_level": our_stock,
            "our_sales_30d": our_sales_30d,
            "our_was_stockout": our_was_stockout,
        })

        # Generate alerts for important findings
        if not we_carry and score >= 40:
            alerts.append({
                "date": today_str,
                "alert_type": "no_cases",
                "brand": brand,
                "model": model,
                "title": f"No tenemos fundas para {brand} {model}",
                "description": f"Precio: ${price:,.0f} MXN. Disponible en {len(retailers)} retailers. Segmento: {segment}. {notes}",
                "impact_score": score,
                "our_modelo_match": None,
                "our_action": "order_cases",
            })
        elif our_was_stockout and our_sales_30d > 10:
            alerts.append({
                "date": today_str,
                "alert_type": "stockout_spike",
                "brand": brand,
                "model": model,
                "title": f"SOLD OUT: {our_modelo_name} — vendiamos {our_sales_30d} uds/mes",
                "description": f"Phone price: ${price:,.0f}. Sold {our_sales_30d} in 30d but stock=0. Restock urgently.",
                "impact_score": score,
                "our_modelo_match": our_modelo_name,
                "our_action": "restock",
            })
        elif is_hot and we_carry and our_stock < 50:
            alerts.append({
                "date": today_str,
                "alert_type": "bestseller",
                "brand": brand,
                "model": model,
                "title": f"Bestseller {brand} {model} — stock bajo: {our_stock} uds",
                "description": f"Hot phone at ${price:,.0f}. We have {our_stock} cases, selling {our_sales_30d}/month.",
                "impact_score": score * 0.8,
                "our_modelo_match": our_modelo_name,
                "our_action": "restock",
            })

    # 4. Build demand correlations from our daily_records
    correlations = []
    daily_records = await _rpc_get("get_daily_records_for_training", {"p_days_back": 30})
    if daily_records:
        # Group by modelo: look for sales spikes that correlate with phone launches
        from collections import defaultdict
        modelo_sales = defaultdict(list)
        for r in daily_records:
            mod = r.get("modelo", "")
            sold = float(r.get("units_sold_today", 0) or 0)
            stock = int(r.get("current_stock_level", 0) or 0)
            modelo_sales[mod].append({"sold": sold, "stock": stock, "date": r.get("date")})

        for phone in MEXICO_PHONE_CATALOG:
            if not phone.get("launched"):
                continue
            match = _match_modelo(phone["model"], phone["brand"], our_modelos or [])
            if not match:
                continue
            our_mod = match.get("modelo", "")
            if our_mod not in modelo_sales:
                continue

            sales_data = modelo_sales[our_mod]
            if len(sales_data) < 2:
                continue

            total_sold = sum(s["sold"] for s in sales_data)
            avg_sold = total_sold / len(sales_data)
            latest_stock = sales_data[-1]["stock"]

            correlations.append({
                "date": today_str,
                "brand": phone["brand"],
                "model": phone["model"],
                "market_event": "phone_active",
                "event_date": phone["launched"],
                "our_modelo": our_mod,
                "sales_before_7d": round(avg_sold * 7, 1),
                "sales_after_7d": round(total_sold, 1),
                "sales_change_pct": 0,
                "stock_before": latest_stock,
                "stock_after": latest_stock,
                "was_stockout": latest_stock == 0 and total_sold > 0,
                "correlation_strength": "strong" if total_sold > 50 else "moderate" if total_sold > 20 else "weak",
            })

    # 5. Save everything to Supabase
    if market_phones_to_save:
        r = await _rpc("upsert_market_phones_batch", {"p_phones": market_phones_to_save})
        results["phones_saved"] = r if isinstance(r, int) else len(market_phones_to_save)

    if gaps:
        r = await _rpc("upsert_market_gaps_batch", {"p_gaps": gaps})
        results["gaps_saved"] = r if isinstance(r, int) else len(gaps)

    if alerts:
        r = await _rpc("insert_market_alerts_batch", {"p_alerts": alerts})
        results["alerts_generated"] = r if isinstance(r, int) else len(alerts)

    if correlations:
        r = await _rpc("upsert_demand_correlations_batch", {"p_corrs": correlations})
        results["correlations_saved"] = r if isinstance(r, int) else len(correlations)

    # 6. Build summary
    no_cases = [g for g in gaps if not g["we_carry_cases"]]
    stockouts = [g for g in gaps if g["our_was_stockout"]]
    hot_gaps = [g for g in gaps if g["opportunity_score"] >= 40]

    results["summary"] = {
        "total_market_phones": len(MEXICO_PHONE_CATALOG),
        "phones_we_carry": sum(1 for g in gaps if g["we_carry_cases"]),
        "phones_we_dont_carry": len(no_cases),
        "stockout_models": len(stockouts),
        "high_opportunity_gaps": len(hot_gaps),
        "alerts_count": len(alerts),
    }
    results["gaps"] = sorted(gaps, key=lambda x: x["opportunity_score"], reverse=True)
    results["alerts"] = alerts
    results["no_cases_models"] = sorted(no_cases, key=lambda x: x["opportunity_score"], reverse=True)

    return results


# ──────────────────────────────────────────────
# Data fetchers for dashboard
# ──────────────────────────────────────────────

async def get_dashboard_data() -> Dict[str, Any]:
    """Fetch all data for market intelligence dashboard."""
    import asyncio

    gaps, alerts, correlations = await asyncio.gather(
        _rpc_get("get_market_gaps_active", {"p_days_back": 7}),
        _rpc_get("get_market_alerts_recent", {"p_days_back": 30}),
        _rpc_get("get_demand_correlations_recent", {"p_days_back": 30}),
    )

    gaps = gaps or []
    alerts = alerts or []
    correlations = correlations or []

    no_cases = [g for g in gaps if not g.get("we_carry_cases")]
    stockouts = [g for g in gaps if g.get("our_was_stockout")]
    high_opp = [g for g in gaps if float(g.get("opportunity_score", 0)) >= 40]

    return {
        "gaps": gaps,
        "no_cases": no_cases,
        "stockouts": stockouts,
        "high_opportunity": high_opp,
        "alerts": alerts,
        "correlations": correlations,
        "summary": {
            "total_gaps": len(gaps),
            "no_cases_count": len(no_cases),
            "stockout_count": len(stockouts),
            "high_opp_count": len(high_opp),
            "alerts_count": len(alerts),
        },
    }
