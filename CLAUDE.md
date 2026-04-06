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

This is a **FastAPI application** serving a retail inventory management system for a two-branch fast-fashion phone case business (Terex1, Terex2) in Mexico. It uses **server-side rendered Jinja2 templates** with a Supabase PostgreSQL backend, plus ML prediction and market intelligence modules.

### Core Files

- `app.py` (~9000+ lines) — main FastAPI app with all routes
- `ml_engine.py` — ML prediction pipeline (5 competing models, daily snapshots, stockout alerts)
- `market_intel.py` — market intelligence engine (phone catalog for Mexico, gap analysis vs `inventario_modelos`)

### Backend Pattern

All data access goes through two async helper functions:
- `supabase_request(method, endpoint, params, json_data)` — direct REST API calls to Supabase tables
- `supabase_rpc(function_name, params)` — calls Supabase PostgreSQL RPC functions

**Critical**: Always prefer `supabase_rpc()` over direct REST queries. REST queries hit Supabase pagination limits and silently truncate results. All analytics and aggregation queries should use server-side SQL functions.

**Critical**: When passing data to Supabase RPC batch functions (e.g., `upsert_daily_records_batch`), pass Python lists/dicts directly — do NOT wrap in `json.dumps()`. Supabase PostgREST handles JSON serialization.

### Supabase Configuration

The Supabase URL and anon key are hardcoded at the top of `app.py`. Headers are set in the `HEADERS` dict. The same credentials are duplicated in `ml_engine.py` and `market_intel.py`.

### Key Database Tables

- `inventario_estilos` — master style catalog (estilo = case design)
- `inventario_modelos` — master phone model catalog with `marca` (brand) and `modelo` columns
- `inventario` — current stock per barcode
- `ventas_terex1` / `ventas_terex2` — separate sales tables per branch
- `inventario_daily` — daily stock snapshots
- `daily_records` — ML training data (daily snapshots of sales/stock per estilo+modelo)
- `model_scores` — ML model competition scores (5 models ranked daily)
- `ml_predictions` — stored predictions for comparison
- `stockout_alerts` — auto-generated stockout warnings
- `market_phones`, `market_gaps`, `market_alerts` — market intelligence data

### SQL Migrations

Two migration files must be run in Supabase SQL Editor:
- `migrations.sql` — ML tables + RPC functions
- `migrations_market.sql` — market intelligence tables + RPC functions

### Route Organization

Routes are defined directly on the `app` FastAPI instance. Major feature areas:

- `/` — main inventory dashboard with style prioritization
- `/verventasxdia`, `/verventasxsemana` — daily/weekly sales views
- `/verinventariostock` — daily stock levels (estilo > modelo > color AND modelo > estilo > color tables)
- `/retail_metrics` — retail metrics with estilo table + modelo-led table (DOI, turnover, sell-through)
- `/should_order` — purchase recommendations by supplier with urgency levels
- `/secretmenu/*` — admin panel:
  - `/secretmenu/estilos` — style management
  - `/secretmenu/dailysales` — daily sales by estilo/modelo
  - `/secretmenu/ml` — ML predictions (demand forecast, dead stock, branch transfers, sold-out detection)
  - `/secretmenu/market-intel` — market intelligence (phone catalog vs inventory gap analysis)
  - `/secretmenu/order-planner` — interactive order planning with brand filters and editable quantities
- `/ml/dashboard` — ML model competition dashboard (5 models, MAE charts, champion badge)
- `/ml/daily-snapshot`, `/ml/run-pipeline` — ML pipeline triggers
- `/ml/leaderboard`, `/ml/predictions/tomorrow` — ML API endpoints
- `/api/save` — core sales transaction endpoint (decrements inventory, records sale)

### ML Pipeline (`ml_engine.py`)

5 models compete daily: Linear Regression, Random Forest, XGBoost, Exponential Smoothing, MLP Neural Net. APScheduler runs the full pipeline at 11pm Mexico City time. Models predict `units_sold_tomorrow` and stockout probability.

When returning data from `run_full_pipeline()`, internal prediction dicts use tuple keys `(modelo, estilo)` which are not JSON-serializable. These are stripped before returning to endpoints.

### Market Intelligence (`market_intel.py`)

Contains a curated `MEXICO_PHONE_CATALOG` of ~137 phone models actively sold in Mexico with brand, price, segment, launch date, and retailers. The `_match_modelo()` function cross-references market phones against `inventario_modelos` to find gaps (phones we don't carry cases for).

### Frontend

Templates extend `base.html` and use inline JavaScript with Chart.js and Bootstrap 5. No build step or bundler. Static files served from `/static/`.

### Timezone

All date/time operations use `America/Mexico_City` timezone via the `_now_strs()` helper.
