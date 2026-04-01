# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Run locally
uvicorn app:app --reload --port 8000

# Production (Railway)
uvicorn app:app --host 0.0.0.0 --port ${PORT}
```

No test suite or linter is configured. The app is deployed on Railway.

## Architecture

This is a **single-file FastAPI application** (`app.py`, ~8200 lines) serving a retail inventory management system for a two-branch fast-fashion business (Terex1, Terex2). It uses **server-side rendered Jinja2 templates** with a Supabase PostgreSQL backend accessed via REST API and RPC calls.

### Backend Pattern

All data access goes through two async helper functions:
- `supabase_request(method, endpoint, params, json_data)` — direct REST API calls to Supabase tables
- `supabase_rpc(function_name, params)` — calls Supabase PostgreSQL RPC functions

**Critical**: Always prefer `supabase_rpc()` over direct REST queries. REST queries hit Supabase pagination limits and silently truncate results. All analytics and aggregation queries should use server-side SQL functions.

### Supabase Configuration

The Supabase URL and anon key are hardcoded at the top of `app.py` (lines 67-68). Headers are set in the `HEADERS` dict (lines 76-81).

### Key Database Tables

- `inventario_estilos` — master style catalog
- `inventario` — current stock per barcode
- `ventas_terex1` / `ventas_terex2` — separate sales tables per branch
- `inventario_daily` — daily stock snapshots
- `conteo_efectivo` — cash count tracking
- `entradas` — merchandise intake records

### Route Organization

Routes are defined directly on the `app` FastAPI instance (no blueprint/router separation). Major feature areas:

- `/` — main inventory dashboard with style prioritization
- `/verventasxdia`, `/verventasxsemana` — daily/weekly sales views
- `/secretmenu/*` — admin panel (style management, image uploads, daily sales)
- `/entradamercancia` — merchandise intake with image upload
- `/ventasviaje` — travel sales (mobile POS)
- `/conteorapido`, `/conteoefectivo` — quick count and cash count
- `/analytics`, `/flores3-analytics`, `/analyticstravel` — analytics dashboards
- `/dashboard-power` — advanced analytics with turnover/demand metrics
- `/api/save` — core sales transaction endpoint (decrements inventory, records sale)
- `/nota`, `/nota1` — receipt/ticket generation (PDF via ReportLab)

### Frontend

Templates extend `base.html` and use inline JavaScript. No build step or bundler. Static files served from `/static/`.

### Timezone

All date/time operations use `America/Mexico_City` timezone via the `_now_strs()` helper.
