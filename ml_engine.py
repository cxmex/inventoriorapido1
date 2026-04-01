"""
ML Engine for inventory demand forecasting and stockout prediction.

Models compete daily:
  1. Linear Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. Exponential Smoothing (Prophet alternative — lightweight)
  5. MLP Neural Network (LSTM alternative — lightweight)

Each model predicts:
  (a) units_sold_tomorrow per modelo/estilo
  (b) probability of stockout within 7 days
"""

import math
import logging
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import httpx

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Supabase helpers (mirrors app.py config)
# ──────────────────────────────────────────────

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


async def _rpc(fn: str, params: dict = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{SUPABASE_URL}/rest/v1/rpc/{fn}", headers=HEADERS, json=params or {})
    if r.status_code >= 400:
        logger.error("RPC %s failed %s: %s", fn, r.status_code, r.text[:300])
        return None
    return r.json()


async def _rpc_get(fn: str, params: dict = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{SUPABASE_URL}/rest/v1/rpc/{fn}", headers=HEADERS, params=params or {})
    if r.status_code >= 400:
        logger.error("RPC GET %s failed %s: %s", fn, r.status_code, r.text[:300])
        return None
    return r.json()


# ──────────────────────────────────────────────
# STEP 1 — Daily Snapshot
# ──────────────────────────────────────────────

async def take_daily_snapshot() -> Dict[str, Any]:
    """
    Capture today's state from existing RPCs into daily_records table.
    Returns summary of what was saved.
    """
    import asyncio

    today_str = date.today().isoformat()

    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch all data sources in parallel
        resp_em_30, resp_em_14, resp_em_7, resp_em_1, resp_order = await asyncio.gather(
            client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                       params={"group_by_field": "estilo_modelo", "days_back": 30}),
            client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                       params={"group_by_field": "estilo_modelo", "days_back": 14}),
            client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                       params={"group_by_field": "estilo_modelo", "days_back": 7}),
            client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                       params={"group_by_field": "estilo_modelo", "days_back": 1}),
            client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_order_analysis", headers=HEADERS,
                       params={"days_back": 30}),
        )

    def safe(r):
        return r.json() if r.status_code < 400 else []

    em30 = {r["group_name"]: r for r in safe(resp_em_30)}
    em14 = {r["group_name"]: r for r in safe(resp_em_14)}
    em7 = {r["group_name"]: r for r in safe(resp_em_7)}
    em1 = {r["group_name"]: r for r in safe(resp_em_1)}

    # Build order analysis lookup: estilo|modelo -> row
    order_lookup = {}
    for r in safe(resp_order):
        key = f"{r.get('estilo', '')} > {r.get('modelo', '')}"
        if key not in order_lookup:
            order_lookup[key] = r

    # Build daily_records batch
    records = []
    for gn, m30 in em30.items():
        if " > " not in gn:
            continue
        estilo, modelo = gn.split(" > ", 1)

        m14_r = em14.get(gn, {})
        m7_r = em7.get(gn, {})
        m1_r = em1.get(gn, {})
        order_r = order_lookup.get(gn, {})

        stock = int(m30.get("current_stock_total", 0) or 0)
        sold_30 = int(m30.get("units_sold_total", 0) or 0)
        sold_14 = int(m14_r.get("units_sold_total", 0) or 0)
        sold_7 = int(m7_r.get("units_sold_total", 0) or 0)
        sold_1 = int(m1_r.get("units_sold_total", 0) or 0)
        rev_30 = float(m30.get("revenue_total", 0) or 0)
        rev_7 = float(m7_r.get("revenue_total", 0) or 0)
        rev_1 = float(m1_r.get("revenue_today", 0) or m1_r.get("revenue_total", 0) or 0)
        doi = m30.get("days_of_inventory")
        avg_daily_30 = float(m30.get("avg_daily_sales", 0) or 0)
        avg_daily_7 = float(m7_r.get("avg_daily_sales", 0) or 0)
        turnover = float(m30.get("turnover_rate", 0) or 0)
        sell_through = float(m30.get("sell_through_pct", 0) or 0)
        rev_per_unit = float(m30.get("revenue_per_unit", 0) or 0)

        # Lost sales from order analysis
        order_doi = order_r.get("days_of_inventory")
        avg_daily_order = float(order_r.get("avg_daily_sales", 0) or 0)
        order_rev = float(order_r.get("revenue_total", 0) or 0)
        order_sold = float(order_r.get("sold_total", 0) or 0)
        avg_price = order_rev / order_sold if order_sold > 0 else rev_per_unit
        lost_sales = 0
        if stock == 0 and sold_30 > 0:
            lost_sales = round(avg_daily_30 * avg_price, 0)
        elif order_doi is not None and float(order_doi) < 30 and avg_daily_order > 0:
            stockout_days = 30 - float(order_doi)
            lost_sales = round(avg_daily_order * stockout_days * avg_price / 30, 0)

        records.append({
            "date": today_str,
            "modelo": modelo,
            "estilo": estilo,
            "tienda": "all",
            "units_sold_today": sold_1,
            "revenue_today": rev_1,
            "units_sold_7d": sold_7,
            "units_sold_14d": sold_14,
            "units_sold_30d": sold_30,
            "revenue_7d": rev_7,
            "revenue_30d": rev_30,
            "days_in_stock": float(doi) if doi is not None else 0,
            "current_stock_level": stock,
            "lost_sales_today": lost_sales,
            "was_out_of_stock": stock == 0 and sold_30 > 0,
            "restock_recommended": (doi is not None and float(doi) < 30 and avg_daily_30 > 0) or (stock == 0 and sold_30 > 0),
            "price_avg": round(avg_price, 2),
            "price_min": round(avg_price * 0.8, 2),  # estimated
            "price_max": round(avg_price * 1.2, 2),  # estimated
            "avg_daily_sales_7d": avg_daily_7,
            "avg_daily_sales_30d": avg_daily_30,
            "turnover_rate": turnover,
            "sell_through_pct": sell_through,
        })

    # Save to Supabase in batches of 200
    total_saved = 0
    for i in range(0, len(records), 200):
        batch = records[i:i + 200]
        result = await _rpc("upsert_daily_records_batch", {"p_records": batch})
        if result is not None:
            total_saved += int(result) if isinstance(result, (int, float)) else len(batch)

    return {
        "date": today_str,
        "records_saved": total_saved,
        "total_combos": len(records),
    }


# ──────────────────────────────────────────────
# STEP 2 — ML Models
# ──────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Build feature matrix from daily_records DataFrame."""
    feature_cols = [
        "units_sold_7d", "units_sold_14d", "units_sold_30d",
        "revenue_7d", "revenue_30d",
        "current_stock_level", "days_in_stock",
        "avg_daily_sales_7d", "avg_daily_sales_30d",
        "turnover_rate", "sell_through_pct",
        "price_avg", "was_out_of_stock_num",
        "day_of_week", "day_of_month",
    ]

    df = df.copy()
    df["was_out_of_stock_num"] = df["was_out_of_stock"].astype(int)
    df["date_parsed"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date_parsed"].dt.dayofweek
    df["day_of_month"] = df["date_parsed"].dt.day

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, feature_cols


class MLModelRunner:
    """Trains and scores all 5 models."""

    def __init__(self):
        self.models = {}
        self.scores = {}

    def train_and_predict(
        self, df: pd.DataFrame, today_str: str
    ) -> Dict[str, Any]:
        """
        Train on all data up to yesterday, predict today, score against actuals.
        Returns predictions and scores.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        try:
            import xgboost as xgb
            has_xgb = True
        except ImportError:
            has_xgb = False
            logger.warning("XGBoost not available, skipping")

        df, feature_cols = _build_features(df)

        today = pd.to_datetime(today_str).date()
        yesterday = today - timedelta(days=1)

        # Split: train on everything before today, test on today
        train_mask = df["date_parsed"].dt.date < today
        test_mask = df["date_parsed"].dt.date == today

        df_train = df[train_mask]
        df_test = df[test_mask]

        # If only 1 day of data, train on all data (no test split yet)
        if len(df_train) < 10 and len(df) >= 10:
            logger.info("Only 1 day — training on all data, no test evaluation")
            df_train = df
            df_test = df  # self-evaluation for initial predictions
        elif len(df_train) < 10 and len(df) < 10:
            logger.warning("Not enough training data: %d rows", len(df))
            return {"error": "insufficient_data", "rows": len(df)}

        X_train = df_train[feature_cols].values
        y_train_sales = df_train["units_sold_today"].values
        y_train_stockout = df_train["was_out_of_stock_num"].values

        X_test = df_test[feature_cols].values if len(df_test) > 0 else None
        y_test_sales = df_test["units_sold_today"].values if len(df_test) > 0 else None
        y_test_stockout = df_test["was_out_of_stock_num"].values if len(df_test) > 0 else None

        # Define models
        model_defs = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "MLP_Neural_Net": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
        }
        if has_xgb:
            model_defs["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            )

        # Exponential smoothing per-series (aggregate approach)
        # We use a simple weighted average as proxy
        model_defs["ExpSmoothing"] = None  # handled separately

        all_predictions = {}  # model_name -> {(modelo, estilo): predicted_sales}
        all_scores = []

        for model_name, model_obj in model_defs.items():
            try:
                if model_name == "ExpSmoothing":
                    # Simple exponential smoothing: predict using weighted recent history
                    preds_sales = self._exp_smoothing_predict(df, today, feature_cols)
                    preds_stockout = np.zeros(len(preds_sales)) if preds_sales is not None else None

                    if X_test is not None and len(X_test) > 0 and preds_sales is not None:
                        # Match predictions to test set
                        test_keys = list(zip(df_test["modelo"].values, df_test["estilo"].values))
                        pred_vals = [preds_sales.get(k, 0) for k in test_keys]
                        pred_vals = np.array(pred_vals)
                    else:
                        pred_vals = None
                else:
                    # Standard sklearn models
                    model_obj.fit(X_train, y_train_sales)
                    self.models[model_name] = model_obj

                    if X_test is not None and len(X_test) > 0:
                        pred_vals = model_obj.predict(X_test)
                        pred_vals = np.maximum(pred_vals, 0)  # no negative sales
                    else:
                        pred_vals = None

                # Score if we have test data
                if pred_vals is not None and y_test_sales is not None and len(y_test_sales) > 0:
                    mae = mean_absolute_error(y_test_sales, pred_vals)
                    rmse = math.sqrt(mean_squared_error(y_test_sales, pred_vals))
                    r2 = r2_score(y_test_sales, pred_vals) if len(y_test_sales) > 1 else 0

                    # Stockout prediction: threshold at stock < 7 days of predicted sales
                    if model_name != "ExpSmoothing":
                        stockout_preds = (pred_vals * 7 > df_test["current_stock_level"].values).astype(int)
                    else:
                        stockout_preds = np.zeros(len(pred_vals))

                    stockout_acc = np.mean(stockout_preds == y_test_stockout) if y_test_stockout is not None else 0

                    all_scores.append({
                        "model_name": model_name,
                        "target": "sales",
                        "mae": round(float(mae), 4),
                        "rmse": round(float(rmse), 4),
                        "r2_score": round(float(r2), 4),
                        "accuracy": round(float(stockout_acc), 4),
                        "num_predictions": len(pred_vals),
                        "training_rows": len(X_train),
                    })

                # Store predictions for tomorrow
                if model_name != "ExpSmoothing" and model_obj is not None:
                    # Use today's features to predict tomorrow
                    if X_test is not None and len(X_test) > 0:
                        tomorrow_preds = model_obj.predict(X_test)
                        tomorrow_preds = np.maximum(tomorrow_preds, 0)
                        keys = list(zip(df_test["modelo"].values, df_test["estilo"].values))
                        all_predictions[model_name] = dict(zip(keys, tomorrow_preds))
                elif model_name == "ExpSmoothing" and preds_sales is not None:
                    all_predictions[model_name] = preds_sales

            except Exception as e:
                logger.error("Model %s failed: %s", model_name, str(e))
                continue

        # Rank models by MAE
        sales_scores = [s for s in all_scores if s["target"] == "sales"]
        sales_scores.sort(key=lambda x: x["mae"])
        for i, s in enumerate(sales_scores):
            s["rank"] = i + 1
            s["date"] = today_str

        return {
            "scores": all_scores,
            "_predictions_internal": all_predictions,  # tuple keys — not JSON-safe, used internally only
            "training_rows": len(df_train),
            "test_rows": len(df_test) if df_test is not None else 0,
            "models_trained": list(all_predictions.keys()),
        }

    def _exp_smoothing_predict(
        self, df: pd.DataFrame, today: date, feature_cols: list
    ) -> Optional[Dict[Tuple[str, str], float]]:
        """Simple exponential smoothing per modelo/estilo combo."""
        predictions = {}
        grouped = df.groupby(["modelo", "estilo"])

        for (modelo, estilo), group in grouped:
            group = group.sort_values("date_parsed")
            sales = group["units_sold_today"].values

            if len(sales) < 3:
                predictions[(modelo, estilo)] = float(sales[-1]) if len(sales) > 0 else 0
                continue

            # Simple exponential smoothing with alpha=0.3
            alpha = 0.3
            smoothed = sales[0]
            for val in sales[1:]:
                smoothed = alpha * val + (1 - alpha) * smoothed
            predictions[(modelo, estilo)] = max(0, float(smoothed))

        return predictions


# ──────────────────────────────────────────────
# STEP 3 — Full Pipeline
# ──────────────────────────────────────────────

async def run_full_pipeline() -> Dict[str, Any]:
    """
    Complete daily pipeline:
    1. Take snapshot
    2. Fetch training data
    3. Train models & score
    4. Save scores & predictions
    5. Generate stockout alerts
    """
    results = {}

    # 1. Snapshot
    try:
        snap = await take_daily_snapshot()
        results["snapshot"] = snap
        logger.info("Snapshot done: %s", snap)
    except Exception as e:
        logger.error("Snapshot failed: %s", e)
        results["snapshot"] = {"error": str(e)}

    # 2. Fetch training data
    today_str = date.today().isoformat()
    raw = await _rpc_get("get_daily_records_for_training", {"p_days_back": 90})
    if not raw or len(raw) < 10:
        results["ml"] = {"error": "insufficient_data", "rows": len(raw) if raw else 0}
        return results

    df = pd.DataFrame(raw)
    results["training_data_rows"] = len(df)
    results["unique_dates"] = df["date"].nunique()

    # 3. Train and predict
    runner = MLModelRunner()
    ml_results = runner.train_and_predict(df, today_str)
    # Store ml_results but strip internal tuple-keyed dict for serialization
    results["ml"] = {k: v for k, v in ml_results.items() if not k.startswith("_")}
    ml_results_full = ml_results  # keep full version for internal use

    if "error" in ml_results:
        return results

    # 4. Save scores to Supabase
    if ml_results.get("scores"):
        score_result = await _rpc("upsert_model_scores_batch", {
            "p_scores": ml_results["scores"]
        })
        results["scores_saved"] = score_result

    # 5. Save predictions
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    pred_records = []
    best_model = None
    if ml_results.get("scores"):
        sales_scores = [s for s in ml_results["scores"] if s["target"] == "sales"]
        if sales_scores:
            best_model = min(sales_scores, key=lambda x: x["mae"])["model_name"]

    for model_name, preds in ml_results_full.get("_predictions_internal", {}).items():
        for (modelo, estilo), pred_sales in preds.items():
            # Stockout probability: predicted_sales * 7 > current_stock
            stock_row = df[(df["modelo"] == modelo) & (df["estilo"] == estilo)]
            current_stock = int(stock_row["current_stock_level"].iloc[-1]) if len(stock_row) > 0 else 0
            stockout_prob = min(1.0, max(0, (float(pred_sales) * 7 - current_stock) / max(float(pred_sales) * 7, 1)))

            pred_records.append({
                "date": today_str,
                "target_date": tomorrow_str,
                "modelo": modelo,
                "estilo": estilo,
                "model_name": model_name,
                "predicted_sales": round(float(pred_sales), 2),
                "predicted_stockout_prob": round(float(stockout_prob), 4),
            })

    if pred_records:
        for i in range(0, len(pred_records), 200):
            batch = pred_records[i:i + 200]
            await _rpc("upsert_ml_predictions_batch", {"p_preds": batch})
        results["predictions_saved"] = len(pred_records)

    # 6. Generate stockout alerts (>70% probability from best model)
    if best_model:
        alerts = []
        for rec in pred_records:
            if rec["model_name"] == best_model and rec["predicted_stockout_prob"] > 0.70:
                stock_row = df[(df["modelo"] == rec["modelo"]) & (df["estilo"] == rec["estilo"])]
                avg_daily = float(stock_row["avg_daily_sales_30d"].iloc[-1]) if len(stock_row) > 0 else 0
                current_stock = int(stock_row["current_stock_level"].iloc[-1]) if len(stock_row) > 0 else 0
                days_until = current_stock / avg_daily if avg_daily > 0 else 0

                alerts.append({
                    "date": today_str,
                    "modelo": rec["modelo"],
                    "estilo": rec["estilo"],
                    "tienda": "all",
                    "probability": rec["predicted_stockout_prob"],
                    "days_until_stockout": round(days_until, 1),
                    "recommended_order_qty": max(0, int(avg_daily * 30 - current_stock)),
                    "current_stock": current_stock,
                    "avg_daily_sales": round(avg_daily, 2),
                })

        if alerts:
            await _rpc("insert_stockout_alerts_batch", {"p_alerts": alerts})
            results["alerts_generated"] = len(alerts)

    results["best_model"] = best_model
    results.pop("ml", None)  # remove raw ml results with tuple keys
    if "ml" in results and "_predictions_internal" in results.get("ml", {}):
        del results["ml"]["_predictions_internal"]

    # Sanitize: convert numpy/pandas types to native Python for JSON serialization
    clean = {k: v for k, v in results.items() if not k.startswith("_")}
    return json.loads(json.dumps(clean, default=str))


# ──────────────────────────────────────────────
# Data fetchers for endpoints
# ──────────────────────────────────────────────

async def get_leaderboard(days_back: int = 30) -> Dict[str, Any]:
    """Fetch model leaderboard + score history for charts."""
    import asyncio
    leaderboard, history, record_count = await asyncio.gather(
        _rpc_get("get_model_leaderboard", {"p_days_back": days_back}),
        _rpc_get("get_model_scores_history", {"p_days_back": days_back}),
        _rpc("get_daily_records_count"),
    )
    return {
        "leaderboard": leaderboard or [],
        "history": history or [],
        "total_days_data": record_count if isinstance(record_count, int) else 0,
    }


async def get_predictions_tomorrow() -> Dict[str, Any]:
    """Fetch latest predictions from best model."""
    preds = await _rpc_get("get_latest_predictions")
    if not preds:
        return {"predictions": [], "best_model": None}

    # Group by model, find best (lowest MAE from latest scores)
    leaderboard = await _rpc_get("get_model_leaderboard", {"p_days_back": 7})
    best_model = None
    if leaderboard:
        sales_lb = [r for r in leaderboard if r.get("target") == "sales"]
        if sales_lb:
            best_model = sales_lb[0].get("model_name")

    # Filter predictions from best model
    if best_model:
        best_preds = [p for p in preds if p.get("model_name") == best_model]
    else:
        best_preds = preds

    # Sort by predicted_sales desc
    best_preds.sort(key=lambda x: float(x.get("predicted_sales", 0) or 0), reverse=True)

    return {
        "predictions": best_preds,
        "best_model": best_model,
        "all_predictions": preds,
    }


async def get_lost_sales_data() -> Dict[str, Any]:
    """Fetch lost sales from daily_records history."""
    raw = await _rpc_get("get_daily_records_for_training", {"p_days_back": 60})
    if not raw:
        return {"daily": [], "by_modelo": [], "total_lost": 0}

    df = pd.DataFrame(raw)
    df["lost_sales_today"] = pd.to_numeric(df["lost_sales_today"], errors="coerce").fillna(0)
    df["revenue_today"] = pd.to_numeric(df["revenue_today"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"])

    # Daily totals
    daily = df.groupby("date").agg(
        lost=("lost_sales_today", "sum"),
        revenue=("revenue_today", "sum")
    ).reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily_list = daily.to_dict("records")

    # By modelo (last 30 days)
    last_30 = df[df["date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
    by_modelo = last_30.groupby("modelo").agg(
        total_lost=("lost_sales_today", "sum"),
        total_rev=("revenue_today", "sum"),
        days_oos=("was_out_of_stock", "sum"),
    ).reset_index()
    by_modelo = by_modelo.sort_values("total_lost", ascending=False)
    by_modelo_list = by_modelo.head(30).to_dict("records")

    return {
        "daily": daily_list,
        "by_modelo": by_modelo_list,
        "total_lost_30d": float(last_30["lost_sales_today"].sum()),
    }


async def get_stockout_alerts() -> List[Dict]:
    """Fetch active stockout alerts."""
    alerts = await _rpc_get("get_active_stockout_alerts")
    return alerts or []
