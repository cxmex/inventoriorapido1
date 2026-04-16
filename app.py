# app.py - FastAPI with templates for inventory management
import ml_engine
import market_intel
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import jwt
import requests
from fastapi import HTTPException,APIRouter
from datetime import datetime
import time
import hashlib
from fastapi.responses import FileResponse
import uuid


import urllib.parse


import qrcode
import asyncio

from fastapi import FastAPI, HTTPException, Request, Form, Depends,UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse,StreamingResponse  # Optional for debugging
from collections import defaultdict
import httpx
from datetime import datetime, timedelta
import pytz
import os
import shutil
import traceback
import io
import logging
from typing import List, Dict, Optional

from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.graphics.barcode import code128
from reportlab.lib import colors

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.barcode import getCodes
import secrets
import re
import json

print(f"Templates directory exists: {os.path.exists('templates')}", flush=True)
print(f"Templates directory contents: {os.listdir('templates')}", flush=True)

app = FastAPI(
    title="Inventory Management System",
    description="API for inventory management with style prioritization",
    version="1.0.0"
)

# ── APScheduler: ML pipeline at 11pm daily ──
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

ml_scheduler = AsyncIOScheduler()

async def _scheduled_ml_pipeline():
    """Run full ML pipeline — called by scheduler at 11pm."""
    logger.info("Running scheduled ML pipeline...")
    try:
        result = await ml_engine.run_full_pipeline()
        logger.info("ML pipeline complete: %s", {k: v for k, v in result.items() if k != "ml"})
    except Exception as e:
        logger.error("ML pipeline error: %s", e)

ml_scheduler.add_job(
    _scheduled_ml_pipeline,
    CronTrigger(hour=23, minute=0, timezone="America/Mexico_City"),
    id="ml_daily_pipeline",
    replace_existing=True,
)

@app.on_event("startup")
async def start_scheduler():
    ml_scheduler.start()
    logger.info("ML scheduler started — runs daily at 11pm Mexico City time")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Supabase configuration
SUPABASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdia2hrYmZiYXJzbnBiZGt4emlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQzODAzNzMsImV4cCI6MjA0OTk1NjM3M30.mcOcC2GVEu_wD3xNBzSCC3MwDck3CIdmz4D8adU-bpI"

LOCAL_CAMERA_SERVICE = "https://fred-nonchalky-fatally.ngrok-free.dev"

logger = logging.getLogger(__name__)


# Supabase client headershello world
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

router = APIRouter()


# Data models
class ConteoEfectivoCreate(BaseModel):
    tipo: str  # 'credito', 'debito', or 'conteo'
    nombre: str
    amount: float

class EntradaPayload(BaseModel):
    qty:         int
    barcode:     str
    numero_caja: Optional[int]  = None
    notas:       Optional[str]  = None
    imagen_url:  Optional[str]  = None

class ConteoEfectivoResponse(BaseModel):
    id: int
    nombre: str
    tipo: str
    amount: float
    balance: float
    created_at: str
    order_id: Optional[int] = None
    descripcion: Optional[str] = None
    diferencia: Optional[float] = None  # NEW: for conteo tipo


class GoogleAuthRequest(BaseModel):
    google_token: str

class AuthenticatedRedeemRequest(BaseModel):
    google_token: str
    redemption_token: str


class ProductIn(BaseModel):
    qty: int = Field(..., ge=1)
    name: str
    codigo: str
    price: float = Field(..., ge=0)

class SavePayload(BaseModel):
    products: list[dict]
    payment_method: Optional[str] = "efectivo"


class InventarioEstilo(BaseModel):
    id: int
    nombre: str


class Product(BaseModel):
    qty: int
    name: str
    codigo: str
    price: float
    customer_email: Optional[str] = None  # Add this field

class FCMTokenRegistration(BaseModel):
    fcm_token: str
    device_name: Optional[str] = None


# Helper function for Supabase requests
async def supabase_request(
    method: str, 
    endpoint: str, 
    params: Dict[str, Any] = None, 
    json_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}{endpoint}"
    
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url, headers=HEADERS, params=params)
        elif method == "POST":
            response = await client.post(url, headers=HEADERS, json=json_data)
        elif method == "PUT":
            response = await client.put(url, headers=HEADERS, json=json_data)
        elif method == "PATCH":
            response = await client.patch(url, headers=HEADERS, json=json_data)
        elif method == "DELETE":
            response = await client.delete(url, headers=HEADERS, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Supabase API error: {response.text}"
        )
    
    return response.json()

# Helper function for RPC calls
async def supabase_rpc(function_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/rpc/{function_name}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=HEADERS, json=params or {})
    
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Supabase RPC error: {response.text}"
        )
    
    return response.json()

def _now_strs():
    mexico_tz = pytz.timezone("America/Mexico_City")
    now = dt.datetime.now(mexico_tz)  # ADD TIMEZONE HERE
    fecha = f"{now.year}-{now.month:02d}-{now.day:02d}"
    hora = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    return now, fecha, hora

def _build_receipt_pdf(items: List[dict], total: float, order_id: int) -> io.BytesIO:
    # 58mm receipt width
    width = 58 * mm
    height = 200 * mm  # generous page; we add more pages if needed
    margin = 2 * mm

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    def new_page_header():
        y = height - margin
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(width/2, y, "TICKET DE VENTA")
        y -= 12
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(width/2, y, f"Orden #{order_id}")
        y -= 10
        _, fecha, hora = _now_strs()
        c.setFont("Helvetica", 8)
        c.drawString(margin, y, f"Fecha: {fecha}")
        y -= 10
        c.drawString(margin, y, f"Hora: {hora}")
        y -= 6
        c.setStrokeColor(colors.black)
        c.line(margin, y, width - margin, y)
        y -= 10
        # headers
        c.setFont("Helvetica-Bold", 8)
        c.drawString(margin, y, "Cant")
        c.drawString(margin + 20, y, "Descripción")
        c.drawRightString(width - margin, y, "Precio")
        y -= 10
        c.setFont("Helvetica", 8)
        return y

    y = new_page_header()

    for it in items:
        # line
        if y < 25 * mm:
            c.showPage()
            y = new_page_header()

        qty = str(it["qty"])
        name = str(it["name"])
        price = it["price"]
        subtotal = it["subtotal"]

        c.drawString(margin, y, qty)
        # wrap/clip description
        max_desc_chars = 28
        c.drawString(margin + 20, y, (name[:max_desc_chars] + ("…" if len(name) > max_desc_chars else "")))
        c.drawRightString(width - margin, y, f"${price:0.2f}")
        y -= 10
        c.drawRightString(width - margin, y, f"Subtotal: ${subtotal:0.2f}")
        y -= 12

    y -= 4
    c.line(margin, y, width - margin, y)
    y -= 12
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "TOTAL:")
    c.drawRightString(width - margin, y, f"${total:0.2f}")
    y -= 14
    c.setFont("Helvetica", 8)
    c.drawCentredString(width/2, y, "¡Gracias por su compra!")
    y -= 10

    # Barcode
    try:
        bc = code128.Code128(f"ORDER-{order_id}", barHeight=12 * mm, barWidth=0.35)
        x = (width - bc.width) / 2
        bc.drawOn(c, x, max(margin, y - 16 * mm))
    except Exception:
        # If barcode fails for any reason, just ignore and finish
        pass

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

user_sessions = {}


TELEGRAM_TOKEN = "8487551934:AAGOw4FLIgXKolbeiFmAsRuyBS8mJ-3kSQk"
TELEGRAM_CHAT_IDS = ["7204722077", "7145539843","8133878707"]

STORAGE_BUCKET   = "entrada-mercancia"          # bucket name (create in Supabase Dashboard)
STORAGE_BASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co/storage/v1"

async def send_telegram_picture(barcode: str = None, order_id: int = None):
    """Trigger camera capture and send to Telegram"""
    try:
        # Determine which URL to use (local or ngrok)
        # Use ngrok if running on server, local if testing locally
        camera_url = "https://fred-nonchalky-fatally.ngrok-free.dev"  # Your ngrok URL
        # camera_url = LOCAL_CAMERA_SERVICE  # Use this if running locally
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{camera_url}/capture_telegram",
                json={
                    "barcode": barcode,
                    "order_id": order_id
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"📸 Camera capture: {len(data.get('telegram_sent', []))} photos sent to Telegram")
                return data
            else:
                print(f"Camera service error: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"Camera/Telegram error: {str(e)}")
        return None

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        except Exception as e:
            print(f"Telegram error: {e}")

# Global variables for camera capture
current_capture_task = None
current_barcode = None

# Camera configuration
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "admin123!"
CAMERAS = [
    {"ip": "192.168.1.103", "name": "Camera_1"},
    {"ip": "192.168.1.106", "name": "Camera_2"},
    {"ip": "192.168.1.107", "name": "Camera_3"},
]

def initialize_cameras():
    """Initialize all camera connections"""
    import cv2
    caps = []
    for cam in CAMERAS:
        ip = cam["ip"]
        url = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{ip}:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            caps.append({"cap": cap, "name": cam['name'], "ip": ip})
    return caps

async def capture_images_for_one_minute(barcode: str, task_id: int):
    """Capture images for 1 minute with the given barcode"""
    import cv2
    import time
    from datetime import datetime
    
    global current_barcode
    
    print(f"Starting capture for barcode: {barcode}")
    caps = initialize_cameras()
    
    if len(caps) == 0:
        print("No cameras connected!")
        return
    
    start_time = time.time()
    capture_interval = 5
    last_capture_time = time.time()
    duration = 60
    
    try:
        while time.time() - start_time < duration:
            if current_barcode != barcode or task_id != id(current_capture_task):
                print(f"Capture cancelled for {barcode}")
                break
            
            current_time = time.time()
            
            if current_time - last_capture_time >= capture_interval:
                for cam_data in caps:
                    if current_barcode != barcode:
                        break
                    
                    cap = cam_data["cap"]
                    name = cam_data["name"]
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    timestamp = datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
                    camera_name = name.lower().replace("camera_", "cam")
                    filename = f"{timestamp}-{camera_name}-{barcode}.jpg"
                    
                    success, buffer = cv2.imencode('.jpg', frame)
                    
                    if success:
                        image_bytes = buffer.tobytes()
                        try:
                            supabase.storage.from_("camera-captures").upload(
                                path=filename,
                                file=image_bytes,
                                file_options={"content-type": "image/jpeg"}
                            )
                            print(f"✓ Uploaded: {filename}")
                        except Exception as e:
                            print(f"✗ Upload failed: {str(e)}")
                
                last_capture_time = current_time
            
            await asyncio.sleep(0.1)
    
    finally:
        for cam_data in caps:
            cam_data["cap"].release()

@app.post("/api/start_camera_capture")
async def start_camera_capture(barcode: str):
    """
    Trigger the local camera service to start capturing images.
    This endpoint can be called from your server, and it will communicate
    with your local machine where the cameras are connected.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{LOCAL_CAMERA_SERVICE}/trigger",
                json={"barcode": barcode}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "message": f"Camera capture started for barcode: {barcode}",
                    "details": data
                }
            else:
                return {
                    "status": "error",
                    "message": f"Camera service returned status {response.status_code}"
                }
                
    except httpx.ConnectError:
        # Local camera service is not reachable
        print(f"Cannot connect to camera service at {LOCAL_CAMERA_SERVICE}")
        return {
            "status": "error",
            "message": "Camera service unavailable"
        }
    except httpx.TimeoutException:
        print(f"Timeout connecting to camera service")
        return {
            "status": "error",
            "message": "Camera service timeout"
        }
    except Exception as e:
        print(f"Camera service error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/camera_status")
async def camera_status():
    """Check if the local camera service is online"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{LOCAL_CAMERA_SERVICE}/")
            return {
                "status": "online",
                "camera_service": LOCAL_CAMERA_SERVICE
            }
    except Exception as e:
        return {
            "status": "offline",
            "camera_service": LOCAL_CAMERA_SERVICE,
            "error": str(e)
        }



# Home page / Menu
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        # Fetch inventory styles with prioridad=1
        inventario_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario_estilos",
            params={
                "select": "id,nombre",
                "prioridad": "eq.1"
            }
        )
        
        # Fetch total terex1 sums
        totals_map = {}
        try:
            total_terex1_response = await supabase_rpc("sum_terex1_by_estilo")
            
            for item in total_terex1_response:
                estilo_name = item.get('estilo', '')
                sum_value = 0
                
                if item.get('sum') is not None:
                    if isinstance(item['sum'], int):
                        sum_value = item['sum']
                    elif isinstance(item['sum'], float):
                        sum_value = round(item['sum'])
                    else:
                        try:
                            sum_value = int(item['sum'])
                        except (ValueError, TypeError):
                            sum_value = 0
                
                if estilo_name:
                    totals_map[estilo_name] = sum_value
        except Exception as e:
            print(f"Error fetching total terex1: {str(e)}", flush=True)
        
        # Fetch negative terex1 counts
        negatives_map = {}
        try:
            negative_counts_response = await supabase_rpc("count_negative_terex1_by_estilo")
            
            for item in negative_counts_response:
                estilo_name = item.get('estilo', '')
                count = 0
                
                if item.get('count') is not None:
                    if isinstance(item['count'], int):
                        count = item['count']
                    else:
                        try:
                            count = int(item['count'])
                        except (ValueError, TypeError):
                            count = 0
                
                if estilo_name:
                    negatives_map[estilo_name] = count
        except Exception as e:
            print(f"Error fetching negative counts: {str(e)}", flush=True)
        
        # Fetch sales data for the last 7 days
        ventas_por_estilo = {}
        try:
            seven_days_ago = datetime.now() - timedelta(days=7)
            formatted_date = seven_days_ago.strftime("%Y-%m-%d")
            
            ventas_response = await supabase_request(
                method="GET",
                endpoint="/rest/v1/ventas_terex1",
                params={
                    "select": "estilo,qty",
                    "fecha": f"gte.{formatted_date}"
                }
            )
            
            for venta in ventas_response:
                estilo = venta.get('estilo', '')
                qty = 0
                
                if venta.get('qty') is not None:
                    if isinstance(venta['qty'], int):
                        qty = venta['qty']
                    else:
                        try:
                            qty = int(venta['qty'])
                        except (ValueError, TypeError):
                            qty = 0
                
                if estilo:
                    ventas_por_estilo[estilo] = ventas_por_estilo.get(estilo, 0) + qty
        except Exception as e:
            print(f"Error fetching sales data: {str(e)}", flush=True)
        
        # Calculate turnover rates and enhance the inventory items
        enhanced_items = []
        for item in inventario_response:
            nombre_estilo = item.get('nombre', '')
            ventas_count = ventas_por_estilo.get(nombre_estilo, 0)
            negatives_count = negatives_map.get(nombre_estilo, 0)
            total_terex1 = totals_map.get(nombre_estilo, 0)
            
            # Calculate turnover rate
            turnover_rate = 0
            if total_terex1 > 0:
                turnover_rate = (ventas_count / total_terex1) * 100
            
            # Add to enhanced items
            enhanced_items.append({
                'id': item.get('id'),
                'nombre': nombre_estilo,
                'ventas_count': ventas_count,
                'negatives_count': negatives_count,
                'total_terex1': total_terex1,
                'turnover_rate': turnover_rate
            })
        
        # Sort by turnover rate (high to low)
        enhanced_items.sort(key=lambda x: x['turnover_rate'], reverse=True)
        
        return templates.TemplateResponse(
            request=request,
            name="menu.html",
            context={
                "inventory_styles": enhanced_items
            }
        )
    except Exception as e:
        print(f"Error loading menu: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading menu: {str(e)}"
            }
        )


# Inventory detail page
# Inventory detail page - FIXED VERSION
# Fixed Inventory detail page - Version 2
@app.get("/inventory/{estilo_id}", response_class=HTMLResponse)
async def inventory_detail(request: Request, estilo_id: int):
    try:
        # Get the style name
        style_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario_estilos",
            params={
                "select": "nombre",
                "id": f"eq.{estilo_id}"
            }
        )
        
        style_name = "Unknown Style"
        if style_response and len(style_response) > 0:
            style_name = style_response[0].get('nombre', "Unknown Style")
        
        # CRITICAL FIX: Use the SAME method that works in your debug
        # Get ALL inventory items WITHOUT any conversion/processing
        inventory_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "barcode,name,terex1,marca,estilo_id",
                "estilo_id": f"eq.{estilo_id}"
            }
        )
        
        print(f"Retrieved {len(inventory_response)} total items for estilo_id {estilo_id}", flush=True)
        
        # Process inventory data with MINIMAL conversion
        brands = set()
        processed_items = []
        
        negative_count = 0
        zero_count = 0
        positive_count = 0
        
        for item in inventory_response:
            # CRITICAL: Don't modify the terex1 value unless absolutely necessary
            terex1_value = item.get('terex1')
            
            # Only process if terex1 is not None
            if terex1_value is not None:
                # Keep the original value but ensure it's the right type
                if isinstance(terex1_value, (int, float)):
                    # It's already a number, keep it as is
                    final_terex1 = int(terex1_value)
                elif isinstance(terex1_value, str):
                    # Convert string to int
                    try:
                        final_terex1 = int(terex1_value.strip()) if terex1_value.strip() else 0
                    except ValueError:
                        try:
                            final_terex1 = int(float(terex1_value.strip()))
                        except (ValueError, AttributeError):
                            final_terex1 = 0
                else:
                    final_terex1 = 0
            else:
                final_terex1 = 0
            
            # Update the item with the processed value
            item['terex1'] = final_terex1
            
            # Count items by type
            if final_terex1 < 0:
                negative_count += 1
                print(f"NEGATIVE ITEM: {item.get('name', 'No name')} - Barcode: {item.get('barcode')} - Value: {final_terex1}", flush=True)
            elif final_terex1 == 0:
                zero_count += 1
            else:
                positive_count += 1
            
            # Add brand to set
            if item.get('marca'):
                brands.add(item.get('marca').upper())
            
            processed_items.append(item)
        
        # Sort brands alphabetically
        sorted_brands = sorted(list(brands))
        
        print(f"FINAL COUNTS for estilo_id {estilo_id}: Negative: {negative_count}, Zero: {zero_count}, Positive: {positive_count}", flush=True)
        
        # Double-check by filtering the processed items
        actual_negative_items = [item for item in processed_items if item.get('terex1', 0) < 0]
        print(f"Double-check: Found {len(actual_negative_items)} negative items in processed list", flush=True)
        
        if len(actual_negative_items) != negative_count:
            print(f"WARNING: Count mismatch! negative_count={negative_count}, actual_negative_items={len(actual_negative_items)}", flush=True)
        
        return templates.TemplateResponse(
            request=request,
            name="inventory_detail.html",
            context={
                "inventory_items": processed_items,
                "estilo_id": estilo_id,
                "estilo_nombre": style_name,
                "brands": sorted_brands,
                # Add debug info to template
                "debug_counts": {
                    "negative": negative_count,
                    "zero": zero_count,
                    "positive": positive_count,
                    "total": len(processed_items)
                }
            }
        )
    except Exception as e:
        print(f"Error loading inventory: {str(e)}", flush=True)
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading inventory: {str(e)}"
            }
        )


# Alternative approach: Get negative items separately and merge
@app.get("/inventory/{estilo_id}/alternative", response_class=HTMLResponse)
async def inventory_detail_alternative(request: Request, estilo_id: int):
    """Alternative approach: Get all items and negative items separately, then merge"""
    try:
        # Get the style name
        style_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario_estilos",
            params={
                "select": "nombre",
                "id": f"eq.{estilo_id}"
            }
        )
        
        style_name = "Unknown Style"
        if style_response and len(style_response) > 0:
            style_name = style_response[0].get('nombre', "Unknown Style")
        
        # Method 1: Get ALL items
        all_items = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "barcode,name,terex1,marca,estilo_id",
                "estilo_id": f"eq.{estilo_id}"
            }
        )
        
        # Method 2: Get ONLY negative items (we know this works from your debug)
        negative_items = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "barcode,name,terex1,marca,estilo_id",
                "estilo_id": f"eq.{estilo_id}",
                "terex1": "lt.0"
            }
        )
        
        print(f"Alternative method: All items: {len(all_items)}, Negative items: {len(negative_items)}", flush=True)
        
        # Create a set of negative barcodes for quick lookup
        negative_barcodes = {item.get('barcode') for item in negative_items}
        
        # Process all items, but use the negative items data where applicable
        processed_items = []
        brands = set()
        
        for item in all_items:
            barcode = item.get('barcode')
            
            # If this item is in our negative items list, use that data
            if barcode in negative_barcodes:
                # Find the matching negative item
                negative_item = next((neg for neg in negative_items if neg.get('barcode') == barcode), None)
                if negative_item:
                    # Use the negative item data (which we know has the correct terex1 value)
                    item = negative_item.copy()
                    print(f"Using negative item data for {item.get('name', 'No name')}: {item.get('terex1')}", flush=True)
            
            # Ensure terex1 is an integer
            terex1_val = item.get('terex1')
            if terex1_val is not None:
                try:
                    item['terex1'] = int(terex1_val)
                except (ValueError, TypeError):
                    item['terex1'] = 0
            else:
                item['terex1'] = 0
            
            # Add brand
            if item.get('marca'):
                brands.add(item.get('marca').upper())
            
            processed_items.append(item)
        
        # Count final items
        final_negative = sum(1 for item in processed_items if item.get('terex1', 0) < 0)
        final_zero = sum(1 for item in processed_items if item.get('terex1', 0) == 0)
        final_positive = sum(1 for item in processed_items if item.get('terex1', 0) > 0)
        
        print(f"Alternative method final counts: Negative: {final_negative}, Zero: {final_zero}, Positive: {final_positive}", flush=True)
        
        return templates.TemplateResponse(
            request=request,
            name="inventory_detail.html",
            context={
                "inventory_items": processed_items,
                "estilo_id": estilo_id,
                "estilo_nombre": style_name,
                "brands": sorted(list(brands)),
                "debug_counts": {
                    "negative": final_negative,
                    "zero": final_zero,
                    "positive": final_positive,
                    "total": len(processed_items)
                }
            }
        )
        
    except Exception as e:
        print(f"Error in alternative method: {str(e)}", flush=True)
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Alternative method error: {str(e)}"
            }
        )
    
    

# Update terex1 endpoint - IMPROVED VERSION WITH DIRECT REQUESTS
@app.post("/update-terex1", response_class=HTMLResponse)
async def update_terex1(
    request: Request,
    barcode: str = Form(...),
    new_value: int = Form(...),
    name: str = Form(...),
    estilo_id: int = Form(...)
):

    try:
        mexico_tz = pytz.timezone("America/Mexico_City")
        print(f"Processing update for barcode: {barcode}, new value: {new_value}", flush=True)
        
        # Get current value before update
        current_item = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "terex1,name",
                "barcode": f"eq.{barcode}"
            }
        )
        
        old_value = 0
        if current_item and len(current_item) > 0:
            old_value = current_item[0].get('terex1', 0)
            if old_value is None:
                old_value = 0
            else:
                try:
                    old_value = int(old_value)
                except (ValueError, TypeError):
                    old_value = 0
        
        print(f"Current value for {barcode}: {old_value}, updating to: {new_value}", flush=True)
        
        # Update the inventory1 table
        try:
            # Create the complete URL for debugging
            update_url = f"{SUPABASE_URL}/rest/v1/inventario1?barcode=eq.{barcode}"
            print(f"Trying to update at URL: {update_url}", flush=True)
            print(f"With data: {{'terex1': {new_value}}}", flush=True)
            
            # Make direct httpx request to see the full response
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    update_url,
                    headers=HEADERS,
                    json={"terex1": new_value}
                )
                
                print(f"Response status: {response.status_code}", flush=True)
                print(f"Response body: {response.text}", flush=True)
                
                if response.status_code >= 400:
                    error_detail = f"API error: Status {response.status_code}, {response.text}"
                    print(f"Error detail: {error_detail}", flush=True)
                    return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                            "error_message": f"Failed to update inventory: {error_detail}"
                        }
        )
                    
                update_response = response.json()
            
            print(f"Inventory update successful", flush=True)
        except Exception as update_error:
            error_msg = str(update_error)
            print(f"Error updating inventory: {error_msg}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                    "error_message": f"Failed to update inventory: {error_msg}"
                }
        )
        
        # Make sure we have the name if it wasn't provided or was empty
        if not name or name.strip() == "":
            if current_item and len(current_item) > 0 and current_item[0].get('name'):
                name = current_item[0].get('name')
            else:
                name = f"Item {barcode}"  # Fallback name
        
        # Record the update in ventas_terex1_update table with error handling
        try:
            # Create the tracking data
            tracking_data = {
                "name_id": barcode,
                "name": name,
                "new_qty": new_value,
                "old_qty": old_value,
                "fecha": datetime.now(mexico_tz).isoformat()  # Mexico Add timestamp
            }
            
            print(f"Sending tracking data: {tracking_data}", flush=True)
            
            # Make direct httpx request for tracking
            tracking_url = f"{SUPABASE_URL}/rest/v1/ventas_terex1_update"
            print(f"Sending to URL: {tracking_url}", flush=True)
            
            async with httpx.AsyncClient() as client:
                tracking_response = await client.post(
                    tracking_url,
                    headers=HEADERS,
                    json=tracking_data
                )
                
                print(f"Tracking response status: {tracking_response.status_code}", flush=True)
                print(f"Tracking response body: {tracking_response.text}", flush=True)
                
                if tracking_response.status_code >= 400:
                    print(f"Warning: Failed to record tracking data: Status {tracking_response.status_code}, {tracking_response.text}", flush=True)
                    # Continue execution even if tracking fails
                else:
                    print(f"Tracking recorded successfully", flush=True)
                    
        except Exception as tracking_error:
            error_msg = str(tracking_error)
            print(f"Error recording tracking data: {error_msg}", flush=True)
            traceback.print_exc()
            # Continue execution even if tracking fails - don't return an error here
            # since the inventory was already updated
        
        # Redirect back to the inventory detail page
        return RedirectResponse(url=f"/inventory/{estilo_id}?success=true", status_code=303)
        
    except Exception as e:
        print(f"Update error: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error updating inventory: {str(e)}"
            }
        )






    # Directory Structure Checker

@app.get("/check-static")
async def check_static():
    """Endpoint to check static files directory structure"""
    static_dir = "static"
    result = {"exists": os.path.exists(static_dir), "files": []}
    
    if result["exists"]:
        # List main directory
        result["files"] = os.listdir(static_dir)
        
        # Check for subdirectories
        if "css" in result["files"] and os.path.isdir(os.path.join(static_dir, "css")):
            result["css_files"] = os.listdir(os.path.join(static_dir, "css"))
        
        if "js" in result["files"] and os.path.isdir(os.path.join(static_dir, "js")):
            result["js_files"] = os.listdir(os.path.join(static_dir, "js"))
    
    # Check for specific files
    css_files = ["bootstrap.min.css", "styles.css"]
    js_files = ["jquery-3.6.0.min.js", "bootstrap.bundle.min.js"]
    
    result["missing_files"] = []
    
    # Check in root static directory
    for file in css_files + js_files:
        if not os.path.exists(os.path.join(static_dir, file)):
            result["missing_files"].append(file)
    
    # Check in css subdirectory if it exists
    if "css_files" in result:
        for file in css_files:
            if file in result["css_files"]:
                result["css_subdirectory_has"] = result.get("css_subdirectory_has", []) + [file]
    
    # Check in js subdirectory if it exists
    if "js_files" in result:
        for file in js_files:
            if file in result["js_files"]:
                result["js_subdirectory_has"] = result.get("js_subdirectory_has", []) + [file]
    
    return result

# Static Files Copy Helper
@app.get("/fix-static-files")
async def fix_static_files():
    """Copy files from css/js subdirectories to the main static directory"""
    static_dir = "static"
    result = {"success": True, "copied_files": []}
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        result["created_static_dir"] = True
    
    # Check and copy from CSS subdirectory
    css_dir = os.path.join(static_dir, "css")
    if os.path.exists(css_dir) and os.path.isdir(css_dir):
        for file in os.listdir(css_dir):
            source = os.path.join(css_dir, file)
            destination = os.path.join(static_dir, file)
            
            if os.path.isfile(source) and not os.path.exists(destination):
                try:
                    shutil.copy2(source, destination)
                    result["copied_files"].append(f"css/{file}")
                except Exception as e:
                    result["errors"] = result.get("errors", []) + [f"Error copying {file}: {str(e)}"]
    
    # Check and copy from JS subdirectory
    js_dir = os.path.join(static_dir, "js")
    if os.path.exists(js_dir) and os.path.isdir(js_dir):
        for file in os.listdir(js_dir):
            source = os.path.join(js_dir, file)
            destination = os.path.join(static_dir, file)
            
            if os.path.isfile(source) and not os.path.exists(destination):
                try:
                    shutil.copy2(source, destination)
                    result["copied_files"].append(f"js/{file}")
                except Exception as e:
                    result["errors"] = result.get("errors", []) + [f"Error copying {file}: {str(e)}"]
    
    return result

# Test Tracking Endpoint
@app.get("/test-tracking")
async def test_tracking():
    """Test endpoint to try inserting into the tracking table directly"""
    try:
        # Create test tracking data
        tracking_data = {
            "name_id": "test-barcode",
            "name": "Test Item",
            "new_qty": 10,
            "old_qty": 5,
            "fecha": datetime.now().isoformat()
        }
        
        print(f"Testing with tracking data: {tracking_data}", flush=True)
        
        # Make direct httpx request for tracking
        tracking_url = f"{SUPABASE_URL}/rest/v1/ventas_terex1_update"
        
        async with httpx.AsyncClient() as client:
            tracking_response = await client.post(
                tracking_url,
                headers=HEADERS,
                json=tracking_data
            )
            
            response_detail = {
                "status_code": tracking_response.status_code,
                "headers": dict(tracking_response.headers),
                "body": tracking_response.text
            }
            
            if tracking_response.status_code >= 400:
                return {
                    "success": False, 
                    "message": "Failed to insert test tracking data",
                    "response": response_detail
                }
            else:
                return {
                    "success": True,
                    "message": "Successfully inserted test tracking data",
                    "response": response_detail
                }
                
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        }

# Table Description Endpoint
@app.get("/describe-table/{table_name}")
async def describe_table(table_name: str):
    """Endpoint to get table information to help diagnose schema issues"""
    try:
        # Query for information about the table
        definitions_url = f"{SUPABASE_URL}/rest/v1/{table_name}"
        
        # Use multiple requests to get different info
        async with httpx.AsyncClient() as client:
            # Get a single row to understand the structure
            sample_response = await client.get(
                definitions_url,
                headers=HEADERS,
                params={"limit": 1}
            )
            
            # Get the columns definition using OPTIONS request
            options_response = await client.options(
                definitions_url,
                headers=HEADERS
            )
            
            result = {
                "table_name": table_name,
                "sample_data": None,
                "columns_definition": None,
                "errors": []
            }
            
            # Process sample data
            if sample_response.status_code < 400:
                try:
                    result["sample_data"] = sample_response.json()
                except Exception as e:
                    result["errors"].append(f"Error parsing sample data: {str(e)}")
            else:
                result["errors"].append(f"Error fetching sample data: {sample_response.status_code}, {sample_response.text}")
            
            # Process OPTIONS response
            if options_response.status_code < 400:
                try:
                    # Extract definition from headers
                    if "Content-Profile" in options_response.headers:
                        result["content_profile"] = options_response.headers["Content-Profile"]
                        
                    result["options_headers"] = dict(options_response.headers)
                        
                except Exception as e:
                    result["errors"].append(f"Error processing OPTIONS response: {str(e)}")
            else:
                result["errors"].append(f"Error in OPTIONS request: {options_response.status_code}, {options_response.text}")
                
            return result
            
    except Exception as e:
        return {
            "table_name": table_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Endpoint to test a direct SQL query via RPC
@app.get("/test-sql-update/{barcode}/{new_value}")
async def test_sql_update(barcode: str, new_value: int):
    """Test endpoint to update inventory using direct SQL via RPC"""
    try:
        # First, get current value
        current_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "terex1,name",
                "barcode": f"eq.{barcode}"
            }
        )
        
        current_value = 0
        item_name = f"Item {barcode}"
        
        if current_response and len(current_response) > 0:
            current_value = current_response[0].get('terex1', 0)
            if current_value is None:
                current_value = 0
                
            if current_response[0].get('name'):
                item_name = current_response[0].get('name')
        
        # Try updating via direct SQL (if an RPC function exists)
        try:
            sql_update_result = await supabase_rpc(
                "update_inventory_with_tracking",
                {
                    "p_barcode": barcode,
                    "p_new_value": new_value,
                    "p_name": item_name,
                    "p_old_value": current_value
                }
            )
            
            return {
                "success": True,
                "message": "SQL update succeeded",
                "result": sql_update_result,
                "barcode": barcode,
                "old_value": current_value,
                "new_value": new_value,
                "item_name": item_name
            }
            
        except Exception as sql_error:
            return {
                "success": False,
                "message": f"SQL update failed: {str(sql_error)}",
                "error_details": str(sql_error),
                "traceback": traceback.format_exc()
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/verventasxdia", response_class=HTMLResponse)
async def ver_ventas_por_dia(request: Request):
    try:
        print("Fetching daily sales by branch + estilo + modelo (RPC)", flush=True)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp_branch, resp_estilo, resp_modelo = await asyncio.gather(
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_branch",
                        headers=HEADERS,
                        params={"days_back": 14}
                    ),
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_estilo_branch",
                        headers=HEADERS,
                        params={"days_back": 14}
                    ),
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_modelo_branch",
                        headers=HEADERS,
                        params={"days_back": 14}
                    ),
                )

            if resp_branch.status_code >= 400:
                raise Exception(f"RPC branch error {resp_branch.status_code}: {resp_branch.text}")

            rows = resp_branch.json()

            # --- Branch totals chart (existing) ---
            day_totals_t1 = {}
            day_totals_t2 = {}
            day_totals_total = {}
            labels = []
            data_t1 = []
            data_t2 = []
            data_total = []

            for row in rows:
                full_display = row.get("day_date", "")
                try:
                    display = datetime.strptime(full_display, "%d/%m/%Y").strftime("%d/%m")
                except ValueError:
                    display = full_display

                t1_val = float(row.get("t1_revenue", 0) or 0)
                t2_val = float(row.get("t2_revenue", 0) or 0)
                tot_val = float(row.get("total_revenue", 0) or 0)

                labels.append(display)
                data_t1.append(round(t1_val, 2))
                data_t2.append(round(t2_val, 2))
                data_total.append(round(tot_val, 2))

                day_totals_t1[full_display] = round(t1_val, 2)
                day_totals_t2[full_display] = round(t2_val, 2)
                day_totals_total[full_display] = round(tot_val, 2)

            chart_data = {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Terex 1',
                        'data': data_t1,
                        'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': 'Terex 2',
                        'data': data_t2,
                        'backgroundColor': 'rgba(255, 159, 64, 0.7)',
                        'borderColor': 'rgba(255, 159, 64, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': 'Total',
                        'data': data_total,
                        'backgroundColor': 'rgba(75, 192, 192, 0.7)',
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'borderWidth': 2
                    }
                ]
            }

            # --- Process estilo/modelo breakdown data ---
            # Structure: { "Terex 1": { "16/03": { "EstiloA": 500, "EstiloB": 300 }, ... }, "Terex 2": {...} }
            def process_breakdown(resp, key_field):
                if resp.status_code >= 400:
                    return {}
                raw = resp.json()
                # Build: { branch: { day_display: { name: revenue } } }
                result = {}
                for r in raw:
                    branch = r.get("branch", "")
                    day_full = r.get("day_date", "")
                    name = r.get(key_field, "") or "Sin dato"
                    rev = float(r.get("revenue", 0) or 0)
                    try:
                        day_display = datetime.strptime(day_full, "%d/%m/%Y").strftime("%d/%m")
                    except ValueError:
                        day_display = day_full
                    if branch not in result:
                        result[branch] = {}
                    if day_display not in result[branch]:
                        result[branch][day_display] = {}
                    result[branch][day_display][name] = result[branch][day_display].get(name, 0) + round(rev, 2)
                return result

            estilo_data = process_breakdown(resp_estilo, "estilo")
            modelo_data = process_breakdown(resp_modelo, "modelo")

            return templates.TemplateResponse(
                request=request,
                name="ventas_por_dia.html",
                context={
                    "day_totals_t1": day_totals_t1,
                    "day_totals_t2": day_totals_t2,
                    "day_totals_total": day_totals_total,
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True,
                    "estilo_data": estilo_data,
                    "modelo_data": modelo_data,
                    "chart_labels": labels,
                }
            )

        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            import traceback
            traceback.print_exc()
            return templates.TemplateResponse(
                request=request,
                name="ventas_por_dia.html",
                context={
                    "day_totals_t1": {},
                    "day_totals_t2": {},
                    "day_totals_total": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "chart_data_by_estilo": False,
                    "estilo_data": {},
                    "modelo_data": {},
                    "chart_labels": [],
                }
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading daily sales: {str(e)}"
            }
        )

@app.get("/verventasxsemana", response_class=HTMLResponse)
async def ver_ventas_por_semana(request: Request):
    try:
        print("Fetching weekly sales by branch + estilo + modelo (RPC)", flush=True)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp_branch, resp_estilo, resp_modelo = await asyncio.gather(
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_by_branch",
                        headers=HEADERS,
                        params={"weeks_back": 15}
                    ),
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_by_estilo_branch",
                        headers=HEADERS,
                        params={"weeks_back": 15}
                    ),
                    client.get(
                        f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_by_modelo_branch",
                        headers=HEADERS,
                        params={"weeks_back": 15}
                    ),
                )

            if resp_branch.status_code >= 400:
                raise Exception(f"RPC branch error {resp_branch.status_code}: {resp_branch.text}")

            rows = resp_branch.json()

            week_totals_t1 = {}
            week_totals_t2 = {}
            week_totals_total = {}
            labels = []
            data_t1 = []
            data_t2 = []
            data_total = []

            for row in rows:
                full_display = row.get("week_start", "")
                try:
                    display = datetime.strptime(full_display, "%d/%m/%Y").strftime("%d/%m")
                except ValueError:
                    display = full_display

                t1_val = float(row.get("t1_revenue", 0) or 0)
                t2_val = float(row.get("t2_revenue", 0) or 0)
                tot_val = float(row.get("total_revenue", 0) or 0)

                labels.append(display)
                data_t1.append(round(t1_val, 2))
                data_t2.append(round(t2_val, 2))
                data_total.append(round(tot_val, 2))

                week_totals_t1[full_display] = round(t1_val, 2)
                week_totals_t2[full_display] = round(t2_val, 2)
                week_totals_total[full_display] = round(tot_val, 2)

            chart_data = {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Terex 1',
                        'data': data_t1,
                        'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': 'Terex 2',
                        'data': data_t2,
                        'backgroundColor': 'rgba(255, 159, 64, 0.7)',
                        'borderColor': 'rgba(255, 159, 64, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': 'Total',
                        'data': data_total,
                        'backgroundColor': 'rgba(75, 192, 192, 0.7)',
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'borderWidth': 2
                    }
                ]
            }

            # Process estilo/modelo breakdown data
            def process_breakdown(resp, key_field):
                if resp.status_code >= 400:
                    return {}
                raw = resp.json()
                result = {}
                for r in raw:
                    branch = r.get("branch", "")
                    week_full = r.get("week_start", "")
                    name = r.get(key_field, "") or "Sin dato"
                    rev = float(r.get("revenue", 0) or 0)
                    try:
                        week_display = datetime.strptime(week_full, "%d/%m/%Y").strftime("%d/%m")
                    except ValueError:
                        week_display = week_full
                    if branch not in result:
                        result[branch] = {}
                    if week_display not in result[branch]:
                        result[branch][week_display] = {}
                    result[branch][week_display][name] = result[branch][week_display].get(name, 0) + round(rev, 2)
                return result

            estilo_data = process_breakdown(resp_estilo, "estilo")
            modelo_data = process_breakdown(resp_modelo, "modelo")

            return templates.TemplateResponse(
                request=request,
                name="ventas_por_semana.html",
                context={
                    "week_totals_t1": week_totals_t1,
                    "week_totals_t2": week_totals_t2,
                    "week_totals_total": week_totals_total,
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True,
                    "estilo_data": estilo_data,
                    "modelo_data": modelo_data,
                    "chart_labels": labels,
                }
            )

        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            import traceback
            traceback.print_exc()
            return templates.TemplateResponse(
                request=request,
                name="ventas_por_semana.html",
                context={
                    "week_totals_t1": {},
                    "week_totals_t2": {},
                    "week_totals_total": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "chart_data_by_estilo": False,
                    "estilo_data": {},
                    "modelo_data": {},
                    "chart_labels": [],
                }
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading weekly sales: {str(e)}"
            }
        )



@app.get("/should_order", response_class=HTMLResponse)
async def should_order(request: Request):
    try:
        days = int(request.query_params.get("days", 30))
        lead_time = 30  # days

        async with httpx.AsyncClient(timeout=30) as client:
            resp_analysis, resp_suppliers = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_order_analysis",
                    headers=HEADERS,
                    params={"days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                    headers={**HEADERS, "Range": "0-9999"},
                    params={"select": "nombre,supplier", "order": "nombre.asc"}
                ),
            )

        if resp_analysis.status_code >= 400:
            raise Exception(f"RPC error {resp_analysis.status_code}: {resp_analysis.text}")

        rows = resp_analysis.json()

        # Build supplier map: estilo -> supplier
        supplier_map = {}
        if resp_suppliers.status_code < 400:
            for s in resp_suppliers.json():
                if s.get("nombre"):
                    supplier_map[s["nombre"]] = s.get("supplier") or "Sin proveedor"

        # Compute modelo-level total stock across ALL estilos (cross-coverage)
        modelo_total_stock = {}
        modelo_total_sold = {}
        for r in rows:
            mod = r.get("modelo", "")
            stock = float(r.get("stock_total", 0) or 0)
            sold = float(r.get("sold_total", 0) or 0)
            modelo_total_stock[mod] = modelo_total_stock.get(mod, 0) + stock
            # Only count sold once per modelo (sales are at modelo level, duplicated per color)
            if mod not in modelo_total_sold:
                modelo_total_sold[mod] = sold

        # Build structured data: supplier -> estilo -> modelo -> [colors]
        structured = {}
        for r in rows:
            est = r.get("estilo", "") or "Sin estilo"
            mod = r.get("modelo", "") or "Sin modelo"
            color = r.get("color", "") or "Sin color"
            supplier = supplier_map.get(est, "Sin proveedor")

            stock = float(r.get("stock_total", 0) or 0)
            sold = float(r.get("sold_total", 0) or 0)
            rev = float(r.get("revenue_total", 0) or 0)
            avg_daily = float(r.get("avg_daily_sales", 0) or 0)
            doi = r.get("days_of_inventory")
            doi = float(doi) if doi is not None else None

            # Lost sales estimate: if DOI < lead_time, we'll run out before restock
            # Lost sales = avg_daily_sales * (lead_time - DOI) * avg_price_per_unit
            avg_price = rev / sold if sold > 0 else 0
            lost_sales = 0
            if doi is not None and doi < lead_time and avg_daily > 0:
                stockout_days = lead_time - doi
                lost_sales = round(avg_daily * stockout_days * avg_price, 0)
            elif stock == 0 and sold > 0:
                # Already sold out: losing avg_daily * lead_time * avg_price
                lost_sales = round(avg_daily * lead_time * avg_price, 0)

            # Cross-coverage: is this modelo available in other estilos?
            modelo_cross_stock = modelo_total_stock.get(mod, 0) - stock
            modelo_has_coverage = modelo_cross_stock > 0

            # Urgency score (lower = more urgent)
            # Factors: DOI (lower=urgent), cross-coverage (none=urgent), sold volume
            if stock == 0 and sold > 0:
                urgency = 0  # SOLD OUT
            elif doi is not None and doi < 7:
                urgency = 1  # CRITICAL
            elif doi is not None and doi < lead_time:
                urgency = 2  # ORDER NOW
            elif doi is not None and doi < lead_time * 2:
                urgency = 3  # PLAN ORDER
            else:
                urgency = 4  # OK

            # Boost urgency if no cross-coverage for this modelo
            if not modelo_has_coverage and urgency <= 2:
                urgency = max(0, urgency - 1)

            if supplier not in structured:
                structured[supplier] = {}
            if est not in structured[supplier]:
                structured[supplier][est] = {}
            if mod not in structured[supplier][est]:
                structured[supplier][est][mod] = {
                    "colors": [],
                    "total_stock": 0,
                    "total_sold": sold,
                    "total_rev": rev,
                    "avg_daily": avg_daily,
                    "doi": doi,
                    "lost_sales": 0,
                    "urgency": 4,
                    "modelo_cross_stock": modelo_cross_stock,
                }
            structured[supplier][est][mod]["colors"].append({
                "color": color,
                "stock": int(stock),
                "stock_t1": int(float(r.get("stock_t1", 0) or 0)),
                "stock_t2": int(float(r.get("stock_t2", 0) or 0)),
                "sold_pct": round(sold / max(sold, 1) * 100, 0) if color != "SOLD OUT" else 0,
            })
            structured[supplier][est][mod]["total_stock"] += int(stock)
            structured[supplier][est][mod]["lost_sales"] += lost_sales
            structured[supplier][est][mod]["urgency"] = min(structured[supplier][est][mod]["urgency"], urgency)

        # Sort suppliers, estilos, modelos by urgency then revenue
        sorted_suppliers = sorted(structured.items(), key=lambda x: x[0])

        # Compute totals
        total_lost_sales = 0
        total_items_to_order = 0
        for sup, estilos in structured.items():
            for est, modelos in estilos.items():
                for mod, data in modelos.items():
                    total_lost_sales += data["lost_sales"]
                    if data["urgency"] <= 2:
                        total_items_to_order += 1

        return templates.TemplateResponse(
            request=request,
            name="should_order.html",
            context={
                "structured": sorted_suppliers,
                "days": days,
                "lead_time": lead_time,
                "total_lost_sales": total_lost_sales,
                "total_items_to_order": total_items_to_order,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading order analysis: {str(e)}"}
        )


@app.get("/retail_metrics", response_class=HTMLResponse)
async def retail_metrics(request: Request):
    try:
        group_by = request.query_params.get("group", "estilo")
        days = int(request.query_params.get("days", 30))

        async with httpx.AsyncClient(timeout=30) as client:
            resp, resp_modelo, resp_em = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics",
                    headers=HEADERS,
                    params={"group_by_field": group_by, "days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics",
                    headers=HEADERS,
                    params={"group_by_field": "modelo", "days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics",
                    headers=HEADERS,
                    params={"group_by_field": "estilo_modelo", "days_back": days}
                ),
            )

        if resp.status_code >= 400:
            raise Exception(f"RPC error {resp.status_code}: {resp.text}")

        rows = resp.json()

        # Build modelo-led table: modelo rows + drill-down by estilo
        modelo_rows = resp_modelo.json() if resp_modelo.status_code < 400 else []
        modelo_rows.sort(key=lambda r: float(r.get("revenue_total", 0) or 0), reverse=True)
        em_rows = resp_em.json() if resp_em.status_code < 400 else []
        # Group estilo_modelo rows by modelo
        modelo_estilo_detail = {}
        for r in em_rows:
            gn = r.get("group_name", "")
            if " > " in gn:
                estilo, modelo = gn.split(" > ", 1)
            else:
                continue
            if modelo not in modelo_estilo_detail:
                modelo_estilo_detail[modelo] = []
            modelo_estilo_detail[modelo].append({**r, "estilo_name": estilo})
        # Sort each modelo's estilos by revenue desc
        for mod in modelo_estilo_detail:
            modelo_estilo_detail[mod].sort(key=lambda r: float(r.get("revenue_total", 0) or 0), reverse=True)

        # Compute summary KPIs
        total_stock = sum(int(r.get("current_stock_total", 0) or 0) for r in rows)
        total_sold = sum(int(r.get("units_sold_total", 0) or 0) for r in rows)
        total_rev = sum(float(r.get("revenue_total", 0) or 0) for r in rows)
        dead_count = sum(1 for r in rows if r.get("is_dead_stock"))
        soldout_count = sum(1 for r in rows if int(r.get("current_stock_total", 0) or 0) == 0 and int(r.get("units_sold_total", 0) or 0) > 0)
        low_count = sum(1 for r in rows if r.get("days_of_inventory") is not None and float(r["days_of_inventory"]) < 7 and int(r.get("current_stock_total", 0) or 0) > 0)
        avg_doi = None
        doi_values = [float(r["days_of_inventory"]) for r in rows if r.get("days_of_inventory") is not None]
        if doi_values:
            avg_doi = round(sum(doi_values) / len(doi_values), 1)
        avg_turnover = None
        turn_values = [float(r["turnover_rate"]) for r in rows if r.get("turnover_rate") is not None]
        if turn_values:
            avg_turnover = round(sum(turn_values) / len(turn_values), 2)

        return templates.TemplateResponse(
            request=request,
            name="retail_metrics.html",
            context={
                "rows": rows,
                "group_by": group_by,
                "days": days,
                "total_stock": total_stock,
                "total_sold": total_sold,
                "total_rev": total_rev,
                "dead_count": dead_count,
                "soldout_count": soldout_count,
                "low_count": low_count,
                "avg_doi": avg_doi,
                "avg_turnover": avg_turnover,
                "modelo_rows": modelo_rows,
                "modelo_estilo_detail": modelo_estilo_detail,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading retail metrics: {str(e)}"}
        )


@app.get("/verinventariostock", response_class=HTMLResponse)
async def ver_inventario_stock(request: Request):
    try:
        print("Inventory stock snapshot page — triggering daily snapshot + loading data", flush=True)

        async with httpx.AsyncClient(timeout=30) as client:
            # 1. Trigger today's snapshot (idempotent — skips if already done)
            snap_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/take_inventory_snapshot",
                headers=HEADERS,
                json={}
            )
            snapshot_taken = snap_resp.json() if snap_resp.status_code < 400 else False
            print(f"Snapshot taken today: {snapshot_taken}", flush=True)

            # 2. Fetch snapshots + current detailed stock (all in parallel)
            resp_modelo, resp_estilo, resp_stock_estilo, resp_stock_detail, resp_stock_color = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_inventory_snapshots",
                    headers=HEADERS,
                    params={"days_back": 30, "filter_type": "modelo"}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_inventory_snapshots",
                    headers=HEADERS,
                    params={"days_back": 30, "filter_type": "estilo"}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_current_stock_by_estilo",
                    headers=HEADERS,
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_current_stock_by_estilo_modelo",
                    headers=HEADERS,
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_current_stock_by_estilo_modelo_color",
                    headers=HEADERS,
                ),
            )

        def process_snapshots(resp):
            if resp.status_code >= 400:
                return {}, [], []
            rows = resp.json()
            # Structure: { group_name: { date_display: {t1, t2, total} } }
            data = {}
            all_dates = []
            seen_dates = set()
            for r in rows:
                date_full = r.get("snapshot_date", "")
                name = r.get("group_name", "")
                t1 = int(r.get("terex1_stock", 0) or 0)
                t2 = int(r.get("terex2_stock", 0) or 0)
                total = int(r.get("total_stock", 0) or 0)
                try:
                    date_display = datetime.strptime(date_full, "%d/%m/%Y").strftime("%d/%m")
                except ValueError:
                    date_display = date_full
                if date_display not in seen_dates:
                    all_dates.append(date_display)
                    seen_dates.add(date_display)
                if name not in data:
                    data[name] = {}
                data[name][date_display] = {"t1": t1, "t2": t2, "total": total}

            # Rank by latest total stock descending
            ranked = sorted(data.keys(), key=lambda n: sum(v.get("total", 0) for v in data[n].values()), reverse=True)
            return data, all_dates, ranked

        modelo_data, modelo_dates, modelo_ranked = process_snapshots(resp_modelo)
        estilo_data, estilo_dates, estilo_ranked = process_snapshots(resp_estilo)

        # Process current stock by estilo
        stock_by_estilo = []
        if resp_stock_estilo.status_code < 400:
            stock_by_estilo = resp_stock_estilo.json()

        # Process current stock detail: estilo -> [modelo rows]
        stock_detail = {}
        if resp_stock_detail.status_code < 400:
            for r in resp_stock_detail.json():
                est = r.get("estilo", "")
                if est not in stock_detail:
                    stock_detail[est] = []
                stock_detail[est].append({
                    "modelo": r.get("modelo", ""),
                    "t1": int(r.get("terex1_stock", 0) or 0),
                    "t2": int(r.get("terex2_stock", 0) or 0),
                    "total": int(r.get("total_stock", 0) or 0),
                    "items": int(r.get("num_items", 0) or 0),
                })

        # Process color detail: estilo|modelo -> [color rows]
        color_detail = {}
        if resp_stock_color.status_code < 400:
            for r in resp_stock_color.json():
                key = f"{r.get('estilo', '')}|{r.get('modelo', '')}"
                if key not in color_detail:
                    color_detail[key] = []
                color_detail[key].append({
                    "color": r.get("color", "") or "Sin color",
                    "t1": int(r.get("terex1_stock", 0) or 0),
                    "t2": int(r.get("terex2_stock", 0) or 0),
                    "total": int(r.get("total_stock", 0) or 0),
                })

        # Build modelo-first view: modelo -> [estilo rows]
        stock_by_modelo = {}
        modelo_detail = {}
        if resp_stock_detail.status_code < 400:
            for r in resp_stock_detail.json():
                mod = r.get("modelo", "")
                est = r.get("estilo", "")
                t1 = int(r.get("terex1_stock", 0) or 0)
                t2 = int(r.get("terex2_stock", 0) or 0)
                total = int(r.get("total_stock", 0) or 0)
                num = int(r.get("num_items", 0) or 0)
                # Accumulate modelo totals
                if mod not in stock_by_modelo:
                    stock_by_modelo[mod] = {"modelo": mod, "t1": 0, "t2": 0, "total": 0, "num_items": 0}
                stock_by_modelo[mod]["t1"] += t1
                stock_by_modelo[mod]["t2"] += t2
                stock_by_modelo[mod]["total"] += total
                stock_by_modelo[mod]["num_items"] += num
                # Detail: modelo -> estilos
                if mod not in modelo_detail:
                    modelo_detail[mod] = []
                modelo_detail[mod].append({
                    "estilo": est,
                    "t1": t1,
                    "t2": t2,
                    "total": total,
                    "num_items": num,
                })
        stock_by_modelo_list = sorted(stock_by_modelo.values(), key=lambda x: x["total"], reverse=True)

        return templates.TemplateResponse(
            request=request,
            name="inventario_stock.html",
            context={
                "modelo_data": modelo_data,
                "modelo_dates": modelo_dates,
                "modelo_ranked": modelo_ranked,
                "estilo_data": estilo_data,
                "estilo_dates": estilo_dates,
                "estilo_ranked": estilo_ranked,
                "snapshot_taken": snapshot_taken,
                "stock_by_estilo": stock_by_estilo,
                "stock_detail": stock_detail,
                "color_detail": color_detail,
                "stock_by_modelo": stock_by_modelo_list,
                "modelo_detail": modelo_detail,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading inventory stock: {str(e)}"},
        )


@app.get("/secretmenu", response_class=HTMLResponse)
async def secret_menu(request: Request):
    return templates.TemplateResponse(request=request, name="secret_menu.html", context={})


@app.get("/secretmenu/estilos", response_class=HTMLResponse)
async def secret_menu_estilos(request: Request):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers={**HEADERS, "Range": "0-9999"},
                params={
                    "select": "*",
                    "order": "nombre.asc"
                }
            )

        if resp.status_code >= 400:
            print(f"Error response: {resp.text}", flush=True)
            raise Exception(f"Error fetching estilos: {resp.status_code}")

        estilos_raw = resp.json()

        # Remove unwanted columns
        exclude = {"nombre_embedding", "sold30", "saldos"}
        estilos = [
            {k: v for k, v in row.items() if k not in exclude}
            for row in estilos_raw
        ]

        return templates.TemplateResponse(
            request=request,
            name="secret_estilos.html",
            context={"estilos": estilos}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading estilos: {str(e)}"}
        )


@app.get("/secretmenu/ml", response_class=HTMLResponse)
async def secret_menu_ml(request: Request):
    """Machine Learning dashboard — demand forecasts, dead stock prediction, branch transfers."""
    try:
        import math

        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch multiple data sources in parallel
            (
                resp_metrics_30, resp_metrics_14, resp_metrics_7,
                resp_em_30, resp_em_14,
                resp_daily_modelo, resp_daily_estilo,
                resp_snapshots, resp_order,
            ) = await asyncio.gather(
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                           params={"group_by_field": "modelo", "days_back": 30}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                           params={"group_by_field": "modelo", "days_back": 14}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                           params={"group_by_field": "modelo", "days_back": 7}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                           params={"group_by_field": "estilo_modelo", "days_back": 30}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics", headers=HEADERS,
                           params={"group_by_field": "estilo_modelo", "days_back": 14}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_modelo_branch", headers=HEADERS,
                           params={"days_back": 30}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_estilo_branch", headers=HEADERS,
                           params={"days_back": 30}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_inventory_snapshots", headers=HEADERS,
                           params={"days_back": 30, "filter_type": "modelo"}),
                client.get(f"{SUPABASE_URL}/rest/v1/rpc/get_order_analysis", headers=HEADERS,
                           params={"days_back": 30}),
            )

        def safe_json(resp):
            return resp.json() if resp.status_code < 400 else []

        metrics_30 = {r["group_name"]: r for r in safe_json(resp_metrics_30)}
        metrics_14 = {r["group_name"]: r for r in safe_json(resp_metrics_14)}
        metrics_7 = {r["group_name"]: r for r in safe_json(resp_metrics_7)}
        em_30 = safe_json(resp_em_30)
        em_14 = safe_json(resp_em_14)
        daily_modelo = safe_json(resp_daily_modelo)
        daily_estilo = safe_json(resp_daily_estilo)
        snapshots = safe_json(resp_snapshots)
        order_rows = safe_json(resp_order)

        # ---- 1. DEMAND FORECASTING per modelo ----
        # Compare 14d avg vs 30d avg to detect acceleration/deceleration
        demand_forecasts = []
        for name, m30 in metrics_30.items():
            m14 = metrics_14.get(name, {})
            m7 = metrics_7.get(name, {})
            avg30 = float(m30.get("avg_daily_sales", 0) or 0)
            avg14 = float(m14.get("avg_daily_sales", 0) or 0)
            avg7 = float(m7.get("avg_daily_sales", 0) or 0)
            stock = int(m30.get("current_stock_total", 0) or 0)
            sold_30 = int(m30.get("units_sold_total", 0) or 0)
            rev_30 = float(m30.get("revenue_total", 0) or 0)

            if sold_30 == 0:
                continue

            # Weighted forecast: 50% recent (7d), 30% medium (14d), 20% long (30d)
            forecast_daily = avg7 * 0.50 + avg14 * 0.30 + avg30 * 0.20
            forecast_7d = round(forecast_daily * 7, 0)
            forecast_14d = round(forecast_daily * 14, 0)
            forecast_30d = round(forecast_daily * 30, 0)

            # Trend: compare 7d velocity vs 30d baseline
            if avg30 > 0:
                trend_pct = round(((avg7 - avg30) / avg30) * 100, 1)
            else:
                trend_pct = 0

            # Predicted stockout date
            if forecast_daily > 0 and stock > 0:
                days_left = round(stock / forecast_daily, 1)
            elif stock == 0:
                days_left = 0
            else:
                days_left = 999

            demand_forecasts.append({
                "modelo": name,
                "stock": stock,
                "avg30": round(avg30, 1),
                "avg14": round(avg14, 1),
                "avg7": round(avg7, 1),
                "forecast_daily": round(forecast_daily, 1),
                "forecast_7d": int(forecast_7d),
                "forecast_14d": int(forecast_14d),
                "forecast_30d": int(forecast_30d),
                "trend_pct": trend_pct,
                "days_left": days_left,
                "rev_30": rev_30,
                "sold_30": sold_30,
            })
        demand_forecasts.sort(key=lambda x: x["rev_30"], reverse=True)

        # ---- 2. DEAD STOCK EARLY WARNING ----
        # Items with declining velocity that may become dead stock in 30-60 days
        dead_stock_warnings = []
        for name, m30 in metrics_30.items():
            m14 = metrics_14.get(name, {})
            m7 = metrics_7.get(name, {})
            avg30 = float(m30.get("avg_daily_sales", 0) or 0)
            avg14 = float(m14.get("avg_daily_sales", 0) or 0)
            avg7 = float(m7.get("avg_daily_sales", 0) or 0)
            stock = int(m30.get("current_stock_total", 0) or 0)
            is_dead = m30.get("is_dead_stock", False)

            if stock == 0:
                continue

            # Declining: 7d avg is much lower than 30d avg
            if avg30 > 0:
                decline_pct = round(((avg7 - avg30) / avg30) * 100, 1)
            else:
                decline_pct = -100 if avg7 == 0 else 0

            # Risk score: higher = more likely to become dead stock
            # Factors: sales decline rate, high stock, low turnover
            turnover = float(m30.get("turnover_rate", 0) or 0)
            doi = float(m30.get("days_of_inventory", 0) or 0)

            risk_score = 0
            if decline_pct < -30:
                risk_score += 3
            elif decline_pct < -15:
                risk_score += 2
            elif decline_pct < 0:
                risk_score += 1
            if doi > 90:
                risk_score += 3
            elif doi > 60:
                risk_score += 2
            elif doi > 30:
                risk_score += 1
            if turnover < 0.3:
                risk_score += 2
            elif turnover < 0.5:
                risk_score += 1
            if is_dead:
                risk_score += 4

            if risk_score >= 3:
                risk_level = "CRITICAL" if risk_score >= 7 else ("HIGH" if risk_score >= 5 else "MEDIUM")
                dead_stock_warnings.append({
                    "modelo": name,
                    "stock": stock,
                    "avg30": round(avg30, 1),
                    "avg7": round(avg7, 1),
                    "decline_pct": decline_pct,
                    "doi": round(doi, 0),
                    "turnover": round(turnover, 2),
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "is_dead": is_dead,
                    "rev_30": float(m30.get("revenue_total", 0) or 0),
                    "stock_value_at_risk": round(stock * float(m30.get("revenue_per_unit", 0) or 0), 0),
                })
        dead_stock_warnings.sort(key=lambda x: x["risk_score"], reverse=True)

        # ---- 3. BRANCH TRANSFER RECOMMENDATIONS ----
        # Find modelos where one branch sells much more but the other has more stock
        transfer_recs = []
        for name, m30 in metrics_30.items():
            stock_t1 = int(m30.get("current_stock_t1", 0) or 0)
            stock_t2 = int(m30.get("current_stock_t2", 0) or 0)
            sold_t1 = int(m30.get("units_sold_t1", 0) or 0)
            sold_t2 = int(m30.get("units_sold_t2", 0) or 0)
            total_stock = stock_t1 + stock_t2
            total_sold = sold_t1 + sold_t2

            if total_stock < 10 or total_sold < 5:
                continue

            # Calculate sales share vs stock share for each branch
            sales_share_t1 = sold_t1 / total_sold if total_sold > 0 else 0.5
            stock_share_t1 = stock_t1 / total_stock if total_stock > 0 else 0.5
            imbalance = sales_share_t1 - stock_share_t1  # positive = T1 sells more but has less stock

            # Significant imbalance threshold
            if abs(imbalance) > 0.20 and total_sold >= 10:
                if imbalance > 0:
                    from_branch, to_branch = "T2", "T1"
                    from_stock, to_stock = stock_t2, stock_t1
                    to_sold = sold_t1
                else:
                    from_branch, to_branch = "T1", "T2"
                    from_stock, to_stock = stock_t1, stock_t2
                    to_sold = sold_t2

                # Suggest transfer qty: rebalance to match sales proportion
                ideal_stock_to = round(total_stock * (to_sold / total_sold))
                transfer_qty = max(0, ideal_stock_to - to_stock)
                transfer_qty = min(transfer_qty, from_stock)  # can't transfer more than available

                if transfer_qty >= 5:
                    transfer_recs.append({
                        "modelo": name,
                        "from_branch": from_branch,
                        "to_branch": to_branch,
                        "from_stock": from_stock,
                        "to_stock": to_stock,
                        "transfer_qty": transfer_qty,
                        "imbalance": round(abs(imbalance) * 100, 1),
                        "sold_t1": sold_t1,
                        "sold_t2": sold_t2,
                        "stock_t1": stock_t1,
                        "stock_t2": stock_t2,
                        "rev_30": float(m30.get("revenue_total", 0) or 0),
                    })
        transfer_recs.sort(key=lambda x: x["transfer_qty"], reverse=True)

        # ---- 4. SOLD-OUT MODELS WITH HIGH DEMAND (restocking urgency) ----
        restock_urgent = []
        for name, m30 in metrics_30.items():
            stock = int(m30.get("current_stock_total", 0) or 0)
            sold_30 = int(m30.get("units_sold_total", 0) or 0)
            rev_30 = float(m30.get("revenue_total", 0) or 0)
            avg30 = float(m30.get("avg_daily_sales", 0) or 0)
            m14 = metrics_14.get(name, {})
            avg14 = float(m14.get("avg_daily_sales", 0) or 0)

            if stock == 0 and sold_30 > 0:
                # Estimate lost revenue per day
                lost_daily = avg30 * float(m30.get("revenue_per_unit", 0) or 0)
                restock_urgent.append({
                    "modelo": name,
                    "sold_30": sold_30,
                    "rev_30": rev_30,
                    "avg_daily": round(avg30, 1),
                    "lost_daily_rev": round(lost_daily, 0),
                    "lost_30d_rev": round(lost_daily * 30, 0),
                })
        restock_urgent.sort(key=lambda x: x["rev_30"], reverse=True)

        # ---- 5. ESTILO-MODELO SOLD OUT COMBOS (hidden opportunities) ----
        em_14_map = {r["group_name"]: r for r in em_14}
        soldout_combos = []
        for r in em_30:
            stock = int(r.get("current_stock_total", 0) or 0)
            sold = int(r.get("units_sold_total", 0) or 0)
            rev = float(r.get("revenue_total", 0) or 0)
            if stock == 0 and sold > 5 and rev > 500:
                gn = r.get("group_name", "")
                if " > " in gn:
                    estilo, modelo = gn.split(" > ", 1)
                else:
                    continue
                r14 = em_14_map.get(gn, {})
                avg14 = float(r14.get("avg_daily_sales", 0) or 0)
                soldout_combos.append({
                    "estilo": estilo,
                    "modelo": modelo,
                    "sold_30": sold,
                    "rev_30": rev,
                    "avg_daily": round(float(r.get("avg_daily_sales", 0) or 0), 1),
                    "avg_daily_14": round(avg14, 1),
                })
        soldout_combos.sort(key=lambda x: x["rev_30"], reverse=True)

        # ---- SUMMARY KPIs ----
        total_forecast_7d = sum(d["forecast_7d"] for d in demand_forecasts)
        total_forecast_rev_7d = sum(d["forecast_7d"] * (d["rev_30"] / d["sold_30"]) if d["sold_30"] > 0 else 0 for d in demand_forecasts)
        stockout_within_14d = sum(1 for d in demand_forecasts if d["days_left"] <= 14 and d["stock"] > 0)
        total_transfer_units = sum(t["transfer_qty"] for t in transfer_recs)
        total_lost_daily = sum(r["lost_daily_rev"] for r in restock_urgent)

        return templates.TemplateResponse(
            request=request,
            name="secret_ml.html",
            context={
                "demand_forecasts": demand_forecasts,
                "dead_stock_warnings": dead_stock_warnings,
                "transfer_recs": transfer_recs,
                "restock_urgent": restock_urgent,
                "soldout_combos": soldout_combos,
                "total_forecast_7d": int(total_forecast_7d),
                "total_forecast_rev_7d": int(total_forecast_rev_7d),
                "stockout_within_14d": stockout_within_14d,
                "total_transfer_units": total_transfer_units,
                "total_lost_daily": int(total_lost_daily),
                "dead_warning_count": len(dead_stock_warnings),
                "restock_count": len(restock_urgent),
                "soldout_combo_count": len(soldout_combos),
            }
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading ML dashboard: {str(e)}"}
        )


# ── ML API Endpoints ──

@app.post("/ml/daily-snapshot")
async def ml_daily_snapshot():
    """Manually trigger today's daily snapshot into daily_records."""
    result = await ml_engine.take_daily_snapshot()
    return result


@app.post("/ml/run-pipeline")
async def ml_run_pipeline():
    """Manually trigger the full ML pipeline (snapshot + train + predict + alerts)."""
    try:
        result = await ml_engine.run_full_pipeline()
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/ml/leaderboard")
async def ml_leaderboard(days: int = 30):
    """Model competition leaderboard + history for charts."""
    return await ml_engine.get_leaderboard(days)


@app.get("/ml/predictions/tomorrow")
async def ml_predictions_tomorrow():
    """Best model's predictions for tomorrow."""
    return await ml_engine.get_predictions_tomorrow()


@app.get("/ml/lost-sales")
async def ml_lost_sales():
    """Historical lost sales by modelo with trends."""
    return await ml_engine.get_lost_sales_data()


@app.get("/ml/stockout-alerts")
async def ml_stockout_alerts():
    """Active stockout alerts (>70% probability)."""
    return await ml_engine.get_stockout_alerts()


@app.get("/ml/dashboard", response_class=HTMLResponse)
async def ml_dashboard_page(request: Request):
    """Full ML dashboard with model competition, predictions, alerts."""
    try:
        leaderboard_data = await ml_engine.get_leaderboard(30)
        predictions_data = await ml_engine.get_predictions_tomorrow()
        lost_sales_data = await ml_engine.get_lost_sales_data()
        alerts_data = await ml_engine.get_stockout_alerts()

        # Find champion model (most days ranked #1)
        champion = None
        leaderboard = leaderboard_data.get("leaderboard", [])
        sales_lb = [r for r in leaderboard if r.get("target") == "sales"]
        if sales_lb:
            champion = max(sales_lb, key=lambda x: x.get("days_ranked_first", 0))

        return templates.TemplateResponse(
            request=request,
            name="ml_dashboard.html",
            context={
                "leaderboard": leaderboard_data,
                "predictions": predictions_data,
                "lost_sales": lost_sales_data,
                "alerts": alerts_data,
                "champion": champion,
                "total_days": leaderboard_data.get("total_days_data", 0),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading ML dashboard: {str(e)}"}
        )


# ── Order Planner Endpoint ──

@app.get("/secretmenu/order-planner", response_class=HTMLResponse)
async def order_planner(request: Request):
    """Order Planner: interactive DOI/turnover-based order recommendations."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/rpc/get_retail_metrics",
                headers=HEADERS,
                params={"group_by_field": "modelo", "days_back": 30}
            )

        if resp.status_code >= 400:
            raise Exception(f"RPC error: {resp.text}")

        rows = resp.json()

        # Classify each modelo into a brand
        brand_map = {
            "IPHONE": "APPLE", "IPAD": "APPLE",
            "GALAXY": "SAMSUNG", "SAMSUNG": "SAMSUNG",
            "S25": "SAMSUNG", "S24": "SAMSUNG", "S23": "SAMSUNG",
            "S26": "SAMSUNG", "A06": "SAMSUNG", "A07": "SAMSUNG",
            "A05": "SAMSUNG", "A15": "SAMSUNG", "A16": "SAMSUNG",
            "A17": "SAMSUNG", "A23": "SAMSUNG", "A26": "SAMSUNG",
            "A35": "SAMSUNG", "A36": "SAMSUNG", "A37": "SAMSUNG",
            "A55": "SAMSUNG", "A56": "SAMSUNG", "A57": "SAMSUNG",
            "NOTE": "SAMSUNG", "FLIP": "SAMSUNG", "FOLD": "SAMSUNG",
            "EDGE": "MOTOROLA", "MOTO": "MOTOROLA", "RAZR": "MOTOROLA",
            "G06": "MOTOROLA", "G24": "MOTOROLA", "G56": "MOTOROLA",
            "G86": "MOTOROLA", "G55": "MOTOROLA", "G8": "MOTOROLA",
            "HONOR": "HONOR", "MAGIC": "HONOR", "X9D": "HONOR",
            "X9C": "HONOR", "X6C": "HONOR", "X7D": "HONOR",
            "X8C": "HONOR", "X5C": "HONOR", "X8D": "HONOR",
            "REDMI": "XIAOMI", "MI ": "XIAOMI", "POCO": "XIAOMI",
            "14C": "XIAOMI", "14T": "XIAOMI",
            "RENO": "OPPO", "OPPO": "OPPO", "FIND": "OPPO",
            "V60": "VIVO", "V50": "VIVO", "V40": "VIVO", "VIVO": "VIVO",
            "ZTE": "ZTE", "BLADE": "ZTE", "AXON": "ZTE",
            "REALME": "REALME", "INFINIX": "INFINIX",
            "TECNO": "TECNO", "CAMON": "TECNO",
            "HUAWEI": "HUAWEI", "NOVA": "HUAWEI", "MATE": "HUAWEI",
            "P30": "HUAWEI",
        }

        def get_brand(modelo_name):
            name = modelo_name.upper()
            for key, brand in brand_map.items():
                if name.startswith(key):
                    return brand
            return "OTHER"

        models = []
        brands_set = set()
        for r in rows:
            name = r.get("group_name", "")
            sold = int(r.get("units_sold_total", 0) or 0)
            stock = int(r.get("current_stock_total", 0) or 0)
            doi = r.get("days_of_inventory")
            avg_daily = float(r.get("avg_daily_sales", 0) or 0)
            turnover = float(r.get("turnover_rate", 0) or 0) if r.get("turnover_rate") is not None else 0
            rev = float(r.get("revenue_total", 0) or 0)
            is_dead = r.get("is_dead_stock", False)
            sell_through = float(r.get("sell_through_pct", 0) or 0) if r.get("sell_through_pct") is not None else 0

            brand = get_brand(name)
            brands_set.add(brand)

            # Order recommendation logic (DOI + turnover first, volume second)
            rec_qty = 0
            rec_reason = ""
            if is_dead or (sold == 0 and stock > 0):
                rec_qty = 0
                rec_reason = "DEAD"
            elif sold < 10:
                rec_qty = 0
                rec_reason = "LOW DEMAND"
            elif doi is not None and float(doi) > 200:
                rec_qty = 0
                rec_reason = "OVERSTOCKED"
            elif sold >= 500:
                rec_qty = 300
                rec_reason = "TOP SELLER"
            elif sold >= 200:
                rec_qty = 200
                rec_reason = "HIGH DEMAND"
            elif sold >= 100:
                rec_qty = 150
                rec_reason = "STRONG"
            elif sold >= 50:
                rec_qty = 100
                rec_reason = "MODERATE"
            elif sold >= 30:
                rec_qty = 100
                rec_reason = "MIN ORDER"
            else:
                rec_qty = 0
                rec_reason = "SKIP"

            # Urgency adjustments based on DOI
            if doi is not None and float(doi) <= 14 and sold >= 30:
                rec_qty = max(rec_qty, 200)
                rec_reason += " URGENT"
            elif doi is not None and float(doi) > 90 and rec_qty > 0:
                rec_qty = max(100, rec_qty - 100)
                rec_reason += " (overstocked)"

            models.append({
                "modelo": name,
                "brand": brand,
                "sold_30d": sold,
                "stock": stock,
                "doi": round(float(doi), 1) if doi is not None else None,
                "avg_daily": round(avg_daily, 1),
                "turnover": round(turnover, 2),
                "revenue": rev,
                "sell_through": round(sell_through, 1),
                "is_dead": is_dead,
                "rec_qty": rec_qty,
                "rec_reason": rec_reason,
            })

        models.sort(key=lambda x: x["sold_30d"], reverse=True)
        brands_list = sorted(brands_set)

        return templates.TemplateResponse(
            request=request,
            name="order_planner.html",
            context={
                "models": models,
                "brands": brands_list,
                "total_models": len([m for m in models if m["rec_qty"] > 0]),
                "total_units": sum(m["rec_qty"] for m in models),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading Order Planner: {str(e)}"}
        )


# ── Yearly Forecast Endpoints ──

@app.get("/secretmenu/yearly-forecast", response_class=HTMLResponse)
async def yearly_forecast(request: Request):
    """Yearly sales view with seasonality analysis and ML forecasting."""
    try:
        import numpy as np
        import pandas as pd

        # Fetch monthly sales from RPC
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/rpc/get_monthly_sales_by_branch",
                headers=HEADERS,
            )

        if resp.status_code >= 400:
            raise Exception(f"RPC error {resp.status_code}: {resp.text}")

        rows = resp.json()
        if not rows:
            return templates.TemplateResponse(
                request=request,
                name="yearly_forecast.html",
                context={"has_data": False, "error_msg": "No hay datos de ventas mensuales. Ejecuta la migracion SQL primero."},
            )

        # Build pandas DataFrame
        df = pd.DataFrame(rows)
        df["month_date"] = pd.to_datetime(df["month_date"], format="%Y-%m")
        for col in ["t1_revenue", "t2_revenue", "total_revenue", "t1_qty", "t2_qty", "total_qty"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = df.sort_values("month_date").reset_index(drop=True)

        # ── Handle incomplete current month ──
        mexico_tz = pytz.timezone("America/Mexico_City")
        today = datetime.now(mexico_tz)
        current_month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        import calendar
        days_in_current_month = calendar.monthrange(today.year, today.month)[1]
        days_elapsed = today.day
        current_month_partial = None

        # Check if the last row is the current incomplete month
        if len(df) > 0:
            last_month = df["month_date"].iloc[-1]
            if last_month.year == today.year and last_month.month == today.month:
                # Extract partial month data for display
                partial_row = df.iloc[-1]
                scale_factor = days_in_current_month / max(days_elapsed, 1)
                current_month_partial = {
                    "month": today.strftime("%b %Y"),
                    "days_elapsed": days_elapsed,
                    "days_total": days_in_current_month,
                    "days_remaining": days_in_current_month - days_elapsed,
                    "actual_t1": round(float(partial_row["t1_revenue"]), 2),
                    "actual_t2": round(float(partial_row["t2_revenue"]), 2),
                    "actual_total": round(float(partial_row["total_revenue"]), 2),
                    "projected_t1": round(float(partial_row["t1_revenue"]) * scale_factor, 2),
                    "projected_t2": round(float(partial_row["t2_revenue"]) * scale_factor, 2),
                    "projected_total": round(float(partial_row["total_revenue"]) * scale_factor, 2),
                    "scale_factor": round(scale_factor, 2),
                }
                # Remove incomplete month from training data
                df = df.iloc[:-1].reset_index(drop=True)

        # ── Historical chart data (only complete months) ──
        month_labels = [d.strftime("%Y-%m") for d in df["month_date"]]
        month_labels_short = [d.strftime("%b %y") for d in df["month_date"]]

        chart_data = {
            "labels": month_labels_short,
            "datasets": [
                {
                    "label": "Terex 1",
                    "data": [round(v, 2) for v in df["t1_revenue"]],
                    "backgroundColor": "rgba(54, 162, 235, 0.7)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                },
                {
                    "label": "Terex 2",
                    "data": [round(v, 2) for v in df["t2_revenue"]],
                    "backgroundColor": "rgba(255, 159, 64, 0.7)",
                    "borderColor": "rgba(255, 159, 64, 1)",
                    "borderWidth": 2,
                },
                {
                    "label": "Total",
                    "data": [round(v, 2) for v in df["total_revenue"]],
                    "backgroundColor": "rgba(75, 192, 192, 0.7)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "borderWidth": 2,
                },
            ],
        }

        # ── Year-over-Year seasonality data ──
        yoy_data = {}
        for _, row in df.iterrows():
            yr = int(row["month_date"].year)
            mn = int(row["month_date"].month)
            if yr not in yoy_data:
                yoy_data[yr] = {}
            yoy_data[yr][mn] = round(float(row["total_revenue"]), 2)

        month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                       "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        yoy_colors = [
            "rgba(54, 162, 235, 1)", "rgba(255, 99, 132, 1)",
            "rgba(75, 192, 192, 1)", "rgba(255, 159, 64, 1)",
            "rgba(153, 102, 255, 1)", "rgba(255, 205, 86, 1)",
        ]
        yoy_datasets = []
        for i, yr in enumerate(sorted(yoy_data.keys())):
            yoy_datasets.append({
                "label": str(yr),
                "data": [yoy_data[yr].get(m, 0) for m in range(1, 13)],
                "borderColor": yoy_colors[i % len(yoy_colors)],
                "backgroundColor": yoy_colors[i % len(yoy_colors)].replace("1)", "0.1)"),
                "borderWidth": 2,
                "fill": False,
                "tension": 0.3,
            })
        yoy_chart = {"labels": month_names, "datasets": yoy_datasets}

        # ── Table data ──
        month_totals_t1 = {}
        month_totals_t2 = {}
        month_totals_total = {}
        for _, row in df.iterrows():
            label = row["month_date"].strftime("%b %Y")
            month_totals_t1[label] = round(float(row["t1_revenue"]), 2)
            month_totals_t2[label] = round(float(row["t2_revenue"]), 2)
            month_totals_total[label] = round(float(row["total_revenue"]), 2)

        # ── ML Forecasting (next 6 months) ──
        forecast_months = 6
        forecasts = {}
        model_details = {}

        # Prepare time series
        ts = df.set_index("month_date")["total_revenue"].astype(float)
        ts = ts.asfreq("MS", fill_value=0)

        # Model 1: ARIMA / SARIMAX
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            # Use seasonal order (1,1,1)(1,1,1,12) if enough data, else simpler
            if len(ts) >= 24:
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
            pred = fit.forecast(steps=forecast_months)
            forecasts["SARIMAX"] = [max(0, round(v, 2)) for v in pred.values]
            model_details["SARIMAX"] = {"aic": round(fit.aic, 1)}
        except Exception as e:
            print(f"SARIMAX error: {e}", flush=True)
            forecasts["SARIMAX"] = None

        # Model 2: XGBoost with seasonal features
        try:
            import xgboost as xgb

            feat_df = df[["month_date", "total_revenue"]].copy()
            feat_df["month"] = feat_df["month_date"].dt.month
            feat_df["year"] = feat_df["month_date"].dt.year
            feat_df["month_sin"] = np.sin(2 * np.pi * feat_df["month"] / 12)
            feat_df["month_cos"] = np.cos(2 * np.pi * feat_df["month"] / 12)
            feat_df["trend"] = range(len(feat_df))
            feat_df["lag_1"] = feat_df["total_revenue"].shift(1).fillna(0)
            feat_df["lag_2"] = feat_df["total_revenue"].shift(2).fillna(0)
            feat_df["lag_3"] = feat_df["total_revenue"].shift(3).fillna(0)
            feat_df["lag_12"] = feat_df["total_revenue"].shift(12).fillna(0)
            feat_df["rolling_3"] = feat_df["total_revenue"].rolling(3, min_periods=1).mean()
            feat_df["rolling_6"] = feat_df["total_revenue"].rolling(6, min_periods=1).mean()

            feature_cols = ["month", "month_sin", "month_cos", "trend",
                            "lag_1", "lag_2", "lag_3", "lag_12", "rolling_3", "rolling_6"]
            X = feat_df[feature_cols].values
            y = feat_df["total_revenue"].values

            xgb_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                objective="reg:squarederror", random_state=42
            )
            xgb_model.fit(X, y)

            # Predict future months iteratively
            xgb_preds = []
            last_date = df["month_date"].iloc[-1]
            recent_vals = list(feat_df["total_revenue"].values)

            for step in range(forecast_months):
                future_date = last_date + pd.DateOffset(months=step + 1)
                m = future_date.month
                trend_val = len(feat_df) + step
                lag1 = recent_vals[-1] if len(recent_vals) >= 1 else 0
                lag2 = recent_vals[-2] if len(recent_vals) >= 2 else 0
                lag3 = recent_vals[-3] if len(recent_vals) >= 3 else 0
                lag12 = recent_vals[-12] if len(recent_vals) >= 12 else 0
                roll3 = np.mean(recent_vals[-3:]) if len(recent_vals) >= 3 else np.mean(recent_vals)
                roll6 = np.mean(recent_vals[-6:]) if len(recent_vals) >= 6 else np.mean(recent_vals)

                row_feat = np.array([[m, np.sin(2 * np.pi * m / 12),
                                      np.cos(2 * np.pi * m / 12), trend_val,
                                      lag1, lag2, lag3, lag12, roll3, roll6]])
                pred_val = float(xgb_model.predict(row_feat)[0])
                pred_val = max(0, pred_val)
                xgb_preds.append(round(pred_val, 2))
                recent_vals.append(pred_val)

            forecasts["XGBoost"] = xgb_preds
        except Exception as e:
            print(f"XGBoost forecast error: {e}", flush=True)
            forecasts["XGBoost"] = None

        # Model 3: Holt-Winters Exponential Smoothing
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            if len(ts) >= 24:
                hw_model = ExponentialSmoothing(
                    ts, trend="add", seasonal="add", seasonal_periods=12
                ).fit(optimized=True)
            else:
                hw_model = ExponentialSmoothing(
                    ts, trend="add", seasonal=None
                ).fit(optimized=True)
            hw_pred = hw_model.forecast(forecast_months)
            forecasts["Holt-Winters"] = [max(0, round(v, 2)) for v in hw_pred.values]
        except Exception as e:
            print(f"Holt-Winters error: {e}", flush=True)
            forecasts["Holt-Winters"] = None

        # Ensemble: average of available models
        available = {k: v for k, v in forecasts.items() if v is not None}
        if available:
            ensemble = []
            for i in range(forecast_months):
                vals = [v[i] for v in available.values()]
                ensemble.append(round(np.mean(vals), 2))
            forecasts["Ensemble"] = ensemble
        else:
            forecasts["Ensemble"] = None

        # Build forecast chart data
        last_date = df["month_date"].iloc[-1]
        forecast_labels = []
        for i in range(forecast_months):
            fd = last_date + pd.DateOffset(months=i + 1)
            forecast_labels.append(fd.strftime("%b %y"))

        forecast_colors = {
            "SARIMAX": "rgba(255, 99, 132, 1)",
            "XGBoost": "rgba(54, 162, 235, 1)",
            "Holt-Winters": "rgba(255, 159, 64, 1)",
            "Ensemble": "rgba(75, 192, 192, 1)",
        }
        forecast_datasets = []
        for name, preds in forecasts.items():
            if preds is None:
                continue
            forecast_datasets.append({
                "label": name,
                "data": preds,
                "borderColor": forecast_colors.get(name, "rgba(153, 102, 255, 1)"),
                "backgroundColor": forecast_colors.get(name, "rgba(153, 102, 255, 0.1)").replace("1)", "0.1)"),
                "borderWidth": 3 if name == "Ensemble" else 2,
                "borderDash": [] if name == "Ensemble" else [5, 5],
                "fill": False,
                "tension": 0.3,
            })
        forecast_chart = {"labels": forecast_labels, "datasets": forecast_datasets}

        # Forecast table
        forecast_table = {}
        for name, preds in forecasts.items():
            if preds is not None:
                forecast_table[name] = dict(zip(forecast_labels, preds))

        return templates.TemplateResponse(
            request=request,
            name="yearly_forecast.html",
            context={
                "has_data": True,
                "chart_data": chart_data,
                "yoy_chart": yoy_chart,
                "month_totals_t1": month_totals_t1,
                "month_totals_t2": month_totals_t2,
                "month_totals_total": month_totals_total,
                "forecast_chart": forecast_chart,
                "forecast_table": forecast_table,
                "forecast_labels": forecast_labels,
                "forecasts": forecasts,
                "model_details": model_details,
                "num_months": len(df),
                "years_available": sorted(yoy_data.keys()),
                "current_month_partial": current_month_partial,
            },
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading Yearly Forecast: {str(e)}"},
        )


# ── Market Intelligence Endpoints ──

@app.post("/market-intel/scan")
async def market_intel_scan():
    """Run full market intelligence scan."""
    try:
        result = await market_intel.run_market_scan()
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/secretmenu/market-intel", response_class=HTMLResponse)
async def market_intel_dashboard(request: Request):
    """Market Intelligence dashboard."""
    try:
        # Try Supabase tables first, fall back to live scan
        data = await market_intel.get_dashboard_data()
        if not data.get("gaps") and not data.get("alerts"):
            # Tables might not exist yet — run live scan
            scan = await market_intel.run_market_scan()
            data = {
                "gaps": scan.get("gaps", []),
                "no_cases": scan.get("no_cases_models", []),
                "stockouts": [g for g in scan.get("gaps", []) if g.get("our_was_stockout")],
                "high_opportunity": [g for g in scan.get("gaps", []) if g.get("opportunity_score", 0) >= 40],
                "alerts": scan.get("alerts", []),
                "correlations": [],
                "summary": scan.get("summary", {}),
            }
        return templates.TemplateResponse(
            request=request,
            name="market_intel.html",
            context={"data": data}
        )
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading Market Intel: {str(e)}"}
        )


@app.get("/secretmenu/dailysales", response_class=HTMLResponse)
async def secret_menu_daily_sales(request: Request):
    try:
        days = int(request.query_params.get("days", 7))

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_detail",
                headers=HEADERS,
                params={"days_back": days}
            )

        if resp.status_code >= 400:
            raise Exception(f"RPC error {resp.status_code}: {resp.text}")

        rows = resp.json()

        # Aggregate by estilo: { estilo: { total_qty, total_revenue, modelos: { modelo: {t1_qty, t2_qty, t1_rev, t2_rev} } } }
        estilos = {}
        for r in rows:
            est = r.get("estilo", "") or "Sin estilo"
            mod = r.get("modelo", "") or "Sin modelo"
            branch = r.get("branch", "")
            qty = int(r.get("total_qty", 0) or 0)
            rev = float(r.get("total_revenue", 0) or 0)

            if est not in estilos:
                estilos[est] = {"total_qty": 0, "total_revenue": 0, "modelos": {}}
            estilos[est]["total_qty"] += qty
            estilos[est]["total_revenue"] += rev

            if mod not in estilos[est]["modelos"]:
                estilos[est]["modelos"][mod] = {"t1_qty": 0, "t2_qty": 0, "t1_rev": 0, "t2_rev": 0}
            if branch == "Terex 1":
                estilos[est]["modelos"][mod]["t1_qty"] += qty
                estilos[est]["modelos"][mod]["t1_rev"] += rev
            else:
                estilos[est]["modelos"][mod]["t2_qty"] += qty
                estilos[est]["modelos"][mod]["t2_rev"] += rev

        # Sort estilos by total revenue desc
        sorted_estilos = sorted(estilos.items(), key=lambda x: x[1]["total_revenue"], reverse=True)

        return templates.TemplateResponse(
            request=request,
            name="secret_dailysales.html",
            context={"estilos": sorted_estilos, "days": days}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Error loading daily sales: {str(e)}"}
        )


@app.post("/secretmenu/estilos/create")
async def secret_menu_create_estilo(request: Request):
    try:
        body = await request.json()
        nombre = body.get("nombre", "").strip()
        if not nombre:
            raise HTTPException(status_code=400, detail="nombre is required")

        # Check if nombre already exists
        async with httpx.AsyncClient(timeout=10) as client:
            check_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params={"select": "id", "nombre": f"eq.{nombre}", "limit": "1"}
            )
        if check_resp.status_code < 400 and check_resp.json():
            raise HTTPException(status_code=409, detail="Ya existe un estilo con ese nombre")

        new_row = {"nombre": nombre}
        if body.get("prioridad") is not None:
            new_row["prioridad"] = int(body["prioridad"])
        if body.get("precio") is not None and body["precio"] != "":
            new_row["precio"] = int(body["precio"])
        if body.get("cost") is not None and body["cost"] != "":
            new_row["cost"] = int(body["cost"])
        if body.get("supplier"):
            new_row["supplier"] = body["supplier"].strip()

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                json=new_row
            )

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        created = resp.json()
        return JSONResponse({"ok": True, "data": created})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/secretmenu/estilos/{estilo_id}/images")
async def secret_menu_list_images(estilo_id: int):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/storage/v1/object/list/images_estilos",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
                json={"prefix": f"{estilo_id}/", "limit": 20}
            )
        if resp.status_code >= 400:
            return JSONResponse({"urls": []})
        files = resp.json()
        urls = [
            f"{SUPABASE_URL}/storage/v1/object/public/images_estilos/{estilo_id}/{f['name']}"
            for f in files if f.get("name") and f.get("id")
        ]
        return JSONResponse({"urls": urls})
    except Exception:
        return JSONResponse({"urls": []})


@app.post("/secretmenu/estilos/{estilo_id}/upload")
async def secret_menu_upload_image(estilo_id: int, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        filename = f"{uuid.uuid4().hex[:8]}.{ext}"
        storage_path = f"{estilo_id}/{filename}"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/storage/v1/object/images_estilos/{storage_path}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": file.content_type or "image/jpeg",
                },
                content=contents
            )

        print(f"Upload response: status={resp.status_code} body={resp.text}", flush=True)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/images_estilos/{storage_path}"
        return JSONResponse({"ok": True, "url": public_url})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/secretmenu/estilos/updatecost")
async def secret_menu_update_cost(request: Request):
    try:
        body = await request.json()
        estilo_id = body.get("id")
        new_cost = body.get("cost")

        if estilo_id is None or new_cost is None:
            raise HTTPException(status_code=400, detail="Missing id or cost")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.patch(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params={"id": f"eq.{estilo_id}"},
                json={"cost": int(new_cost)}
            )

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return JSONResponse({"ok": True, "cost": int(new_cost)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== AMSTERDAM CAFE — Chat messages =====

@app.post("/amsterdamcafe")
async def amsterdam_cafe_write(request: Request):
    body = await request.json()
    user = body.get("user", "").strip()
    message = body.get("message", "").strip()
    if not user or not message:
        raise HTTPException(status_code=400, detail="user and message are required")

    tz = pytz.timezone("America/Mexico_City")
    now = datetime.now(tz).isoformat()

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/amsterdamcafe",
            headers=HEADERS,
            json={"user": user, "message": message, "created_at": now}
        )

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse({"ok": True, "data": resp.json()})


@app.get("/amsterdamcafe", response_class=HTMLResponse)
async def amsterdam_cafe_page(request: Request):
    return templates.TemplateResponse(request=request, name="amsterdamcafe.html", context={})


@app.get("/amsterdamcafe/messages")
async def amsterdam_cafe_messages():
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/amsterdamcafe",
            headers=HEADERS,
            params={"select": "*", "order": "created_at.desc", "limit": "100"}
        )
    if resp.status_code >= 400:
        return JSONResponse({"messages": []})
    return JSONResponse({"messages": resp.json()})


# ===== IMAGENES X ESTILO — Create estilo from photos with AI naming =====

@app.get("/imagenesxestilo", response_class=HTMLResponse)
async def imagenes_x_estilo_page(request: Request):
    return templates.TemplateResponse(request=request, name="imagenesxestilo.html", context={})


@app.post("/imagenesxestilo/analyze")
async def imagenes_x_estilo_analyze(request: Request):
    """Analyze uploaded images with Claude Vision to suggest a trendy estilo name."""
    import base64

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY no esta configurada en el servidor")

    body = await request.json()
    images = body.get("images", [])
    if not images:
        raise HTTPException(status_code=400, detail="No se enviaron imagenes")

    # Fetch existing estilo names for context
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers={**HEADERS, "Range": "0-999"},
                params={"select": "nombre", "prioridad": "eq.1", "order": "id.desc", "limit": "80"}
            )
        existing_names = [r["nombre"] for r in resp.json() if r.get("nombre")] if resp.status_code < 400 else []
    except Exception:
        existing_names = []

    sample_names = ", ".join(existing_names[:40]) if existing_names else "FUN FLORES, GALAXY DREAMS, CRYSTAL WAVE, MIDNIGHT BLOOM, SAKURA PINK"

    # Build Claude API content with images
    content = []
    for img in images[:5]:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("type", "image/jpeg"),
                "data": img["data"]
            }
        })

    content.append({
        "type": "text",
        "text": f"""You are a creative product naming expert for a trendy phone case store in Mexico.

Analyze the phone case design(s) in these images. Based on the visual style (colors, patterns, characters, textures, themes), suggest a catchy, short product name (2-3 words max, UPPERCASE).

The name should be:
- Trendy and appealing to young Mexican consumers
- Similar in style to these existing product names from our catalog: {sample_names}
- Short, memorable, and marketable
- In Spanish or English (mix is OK, like our existing names)
- Descriptive of the visual design

Respond ONLY in this exact JSON format, nothing else:
{{"nombre": "SUGGESTED NAME", "reason": "Brief explanation of why this name fits the design", "alternatives": ["ALT NAME 1", "ALT NAME 2", "ALT NAME 3"]}}"""
    })

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            api_resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": content}]
                }
            )

        if api_resp.status_code >= 400:
            error_body = api_resp.text
            print(f"Anthropic API error: {api_resp.status_code} {error_body}", flush=True)
            raise HTTPException(status_code=502, detail=f"Error del servicio de IA: {api_resp.status_code}")

        api_data = api_resp.json()
        text_response = api_data.get("content", [{}])[0].get("text", "")

        # Parse JSON from response
        try:
            result = json.loads(text_response)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re as _re
            match = _re.search(r'\{.*\}', text_response, _re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"nombre": "NUEVO ESTILO", "reason": "No se pudo interpretar la respuesta de IA", "alternatives": []}

        return JSONResponse({
            "nombre": result.get("nombre", "NUEVO ESTILO"),
            "reason": result.get("reason", ""),
            "alternatives": result.get("alternatives", [])
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"AI analysis error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error en analisis de IA: {str(e)}")


@app.get("/verinventariodaily", response_class=HTMLResponse)
async def ver_ventas_por_semana(request: Request):
    try:
        import traceback
        print("Fetching LAST WEEK sales by estilo", flush=True)

        try:
            async with httpx.AsyncClient() as client:
                response_by_estilo = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_by_estilo",
                    headers=HEADERS,
                    params={"weeks_back": 2}  # safer than 1 if RPC can return current+previous; we will pick latest
                )

                response_total = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_total",
                    headers=HEADERS,
                    params={"weeks_back": 2}
                )

                if response_by_estilo.status_code >= 400:
                    print(f"Function call error (by estilo): {response_by_estilo.text}", flush=True)
                    raise Exception(f"Function call failed: {response_by_estilo.status_code}")

                if response_total.status_code >= 400:
                    print(f"Function call error (total): {response_total.text}", flush=True)
                    raise Exception(f"Function call failed: {response_total.status_code}")

                weekly_results_by_estilo = response_by_estilo.json() or []
                weekly_results_total = response_total.json() or []

            if not weekly_results_by_estilo:
                return templates.TemplateResponse(
            request=request,
            name="ventas_por_semana.html",
            context={
                        "week_totals": {},
                        "chart_data": {"labels": [], "datasets": []},
                        "chart_data_by_estilo": False,
                        "selected_week_start": None
            }
        )

            # ---- pick the latest week_start present ----
            # your week_start format appears to be "%d/%m/%Y"
            def parse_week_start(s: str) -> datetime:
                return datetime.strptime(s, "%d/%m/%Y")

            all_week_starts = sorted(
                {row.get("week_start") for row in weekly_results_by_estilo if row.get("week_start")},
                key=parse_week_start
            )
            latest_week_start = all_week_starts[-1]

            # ---- filter only latest week ----
            last_week_rows = [
                r for r in weekly_results_by_estilo
                if r.get("week_start") == latest_week_start
            ]

            # ---- aggregate by estilo (in case duplicates) ----
            estilo_totals = {}
            for row in last_week_rows:
                estilo = (row.get("estilo") or "SIN_ESTILO").strip()
                total = float(row.get("total_revenue", 0) or 0)
                estilo_totals[estilo] = estilo_totals.get(estilo, 0) + total

            # ---- sort estilos high -> low ----
            sorted_items = sorted(estilo_totals.items(), key=lambda kv: kv[1], reverse=True)

            labels = [k for k, _ in sorted_items]
            values = [round(v, 2) for _, v in sorted_items]

            # Optional: last-week total for table/header
            # You can compute it from estilo totals:
            last_week_total = round(sum(values), 2)

            # Or use weekly_results_total if you want:
            week_totals = {}
            for row in weekly_results_total:
                wk = row.get("week_start")
                total = float(row.get("total_revenue", 0) or 0)
                week_totals[wk] = round(total, 2)

            chart_data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": f"Ventas por estilo (Semana {latest_week_start})",
                        "data": values,
                        # Chart.js bar colors (single color is fine, or you can build an array)
                        "backgroundColor": "rgba(54, 162, 235, 0.7)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1,
                    }
                ],
            }

            return templates.TemplateResponse(
            request=request,
            name="ventas_por_semana.html",
            context={
                    "week_totals": {latest_week_start: week_totals.get(latest_week_start, last_week_total)},
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True,
                    "selected_week_start": latest_week_start,
                    "sorted_count": len(labels)
            }
        )

        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
            request=request,
            name="ventas_por_semana.html",
            context={
                    "week_totals": {},
                    "chart_data": {"labels": [], "datasets": []},
                    "chart_data_by_estilo": False,
                    "selected_week_start": None
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading weekly sales: {str(e)}"
            }
        )

# Add this new endpoint to your FastAPI app
# Make sure this route is properly defined in your main FastAPI app

@app.get("/conteorapido", response_class=HTMLResponse)
async def conteo_rapido(request: Request):
    try:
        import traceback
        
        print("Fetching items with prioridad=1 for conteo rapido", flush=True)
        
        try:
            # Fetch all items from inventario_estilos where prioridad=1
            url = f"{SUPABASE_URL}/rest/v1/inventario_estilos"
            print(f"Request URL: {url}", flush=True)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=HEADERS,
                    params={
                        "select": "*",  # Select all columns
                        "prioridad": "eq.1",  # Filter where prioridad equals 1
                        "order": "id"  # Order by id (you can change this to any column)
                    }
                )
                
                if response.status_code >= 400:
                    print(f"Data fetch error: {response.text}", flush=True)
                    raise Exception(f"Data fetch failed: {response.status_code}, {response.text}")
                
                estilos_data = response.json()
                print(f"Retrieved {len(estilos_data)} items with prioridad=1", flush=True)
            
            return templates.TemplateResponse(
            request=request,
            name="conteo_rapido.html",
            context={
                    "estilos": estilos_data,
                    "total_items": len(estilos_data)
                }
        )
            
        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
            request=request,
            name="conteo_rapido.html",
            context={
                    "estilos": [],
                    "total_items": 0,
                    "error_message": f"Error loading data: {str(fetch_error)}"
                }
        )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading conteo rapido: {str(e)}"
            }
        )

# Add the POST endpoint for submitting counts
# Add this new endpoint to your FastAPI app

@app.post("/conteorapido/submit")
async def submit_conteo(request: Request):
    try:
        import traceback
        
        # Get the JSON data from the request
        data = await request.json()
        estilo_id = data.get('estilo_id')
        estilo_nombre = data.get('estilo_nombre')
        input_data = data.get('input_data')
        
        print(f"Submitting conteo: estilo_id={estilo_id}, estilo_nombre={estilo_nombre}, input_data={input_data}", flush=True)
        
        # Validate input data
        if not estilo_id or not estilo_nombre or input_data is None:
            return {"success": False, "error": "Missing required data"}
        
        # Query current stock from inventario1 table
        inventory_data = 0
        try:
            # Get sum of terex1 for this estilo_id where terex1 > 0
            inventory_url = f"{SUPABASE_URL}/rest/v1/inventario1"
            
            async with httpx.AsyncClient() as client:
                inventory_response = await client.get(
                    inventory_url,
                    headers=HEADERS,
                    params={
                        "select": "terex1",
                        "estilo_id": f"eq.{estilo_id}",
                        "terex1": "gt.0"  # Greater than 0
                    }
                )
                
                if inventory_response.status_code == 200:
                    inventory_records = inventory_response.json()
                    # Sum all terex1 values
                    inventory_data = sum(record.get('terex1', 0) for record in inventory_records)
                    print(f"Current stock for estilo_id {estilo_id}: {inventory_data}", flush=True)
                else:
                    print(f"Warning: Could not fetch inventory data: {inventory_response.text}", flush=True)
                    
        except Exception as inventory_error:
            print(f"Error fetching inventory data: {str(inventory_error)}", flush=True)
            # Continue with inventory_data = 0 if there's an error
        
        # Prepare data for insertion
        insert_data = {
            "estilo_id": estilo_id,
            "estilo": estilo_nombre,
            "input_data": int(input_data),
            "inventory_data": inventory_data
        }
        
        print(f"Inserting data: {insert_data}", flush=True)
        
        # Insert into conteo_rapido_estilos table
        url = f"{SUPABASE_URL}/rest/v1/conteo_rapido_estilos"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=HEADERS,
                json=insert_data
            )
            
            if response.status_code >= 400:
                print(f"Insert error: {response.text}", flush=True)
                return {"success": False, "error": f"Database error: {response.text}"}
            
            print(f"Successfully inserted conteo record with inventory data", flush=True)
            return {
                "success": True, 
                "message": "Conteo registrado exitosamente",
                "inventory_stock": inventory_data,
                "manual_count": input_data
            }
            
    except Exception as e:
        import traceback
        print(f"Error submitting conteo: {str(e)}", flush=True)
        traceback.print_exc()
        return {"success": False, "error": f"Server error: {str(e)}"}
    

# Add model endpoint
@app.post("/add-model", response_class=HTMLResponse)
async def add_model(
    request: Request,
    modelo: str = Form(...)
):
    try:
        print(f"Adding new model: {modelo}", flush=True)
        
        # Insert new model
        try:
            add_response = await supabase_request(
                method="POST",
                endpoint="/rest/v1/inventario_modelos",
                json_data={"modelo": modelo}
            )
            
            print(f"Model added successfully", flush=True)
        except Exception as add_error:
            error_msg = str(add_error)
            print(f"Error adding model: {error_msg}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                    "error_message": f"Failed to add model: {error_msg}"
                }
        )
        
        # Redirect back to models page
        return RedirectResponse(url="/modelos?success=true", status_code=303)
        
    except Exception as e:
        print(f"Add model error: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error adding model: {str(e)}"
            }
        )


@app.get("/verinventariodaily", response_class=HTMLResponse)
async def ver_inventario_daily(request: Request):
    """
    Endpoint to display daily inventory chart with data from inventario_daily table
    """
    try:
        print(f"Fetching daily inventory data for the last 30 days", flush=True)
        
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Format dates for Supabase query
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching inventory daily data from {start_date_str} to {end_date_str}", flush=True)
            
            # Fetch data from inventario_daily table
            daily_data = await supabase_request(
                method="GET",
                endpoint="/rest/v1/inventario_daily",
                params={
                    "select": "*",
                    "fecha": f"gte.{start_date_str}",
                    "order": "fecha.asc"
                }
            )
            
            print(f"Retrieved {len(daily_data)} daily inventory records", flush=True)
            
            # Debug: Print first few records to see the actual structure
            if daily_data:
                print(f"Sample record: {daily_data[0]}", flush=True)
            
            if not daily_data:
                return templates.TemplateResponse(
            request=request,
            name="inventario_daily.html",
            context={
                        "daily_totals": {},
                        "chart_data": {'labels': [], 'datasets': []},
                        "has_chart_data": False
                    }
        )
            
            # Process data for chart
            daily_totals = {}
            all_estilos = set()
            
            # Group data by date and collect all estilos
            for record in daily_data:
                fecha = record.get('fecha')
                estilo = record.get('estilo')
                # Changed from 'cantidad' to 'qty'
                qty = record.get('qty', 0)
                
                # Convert qty to int if it's not None
                if qty is not None:
                    try:
                        qty = int(qty)
                    except (ValueError, TypeError):
                        qty = 0
                else:
                    qty = 0
                
                print(f"Processing record: fecha={fecha}, estilo={estilo}, qty={qty}", flush=True)
                
                if fecha not in daily_totals:
                    daily_totals[fecha] = {
                        'total_quantity': 0,
                        'estilos': {}
                    }
                
                daily_totals[fecha]['estilos'][estilo] = qty
                daily_totals[fecha]['total_quantity'] += qty
                all_estilos.add(estilo)
            
            # Prepare chart data
            sorted_dates = sorted(daily_totals.keys())
            chart_labels = [datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m') for date in sorted_dates]
            
            # Create datasets for each estilo
            datasets = []
            colors = [
                {'bg': 'rgba(255, 99, 132, 0.7)', 'border': 'rgba(255, 99, 132, 1)'},
                {'bg': 'rgba(54, 162, 235, 0.7)', 'border': 'rgba(54, 162, 235, 1)'},
                {'bg': 'rgba(255, 206, 86, 0.7)', 'border': 'rgba(255, 206, 86, 1)'},
                {'bg': 'rgba(75, 192, 192, 0.7)', 'border': 'rgba(75, 192, 192, 1)'},
                {'bg': 'rgba(153, 102, 255, 0.7)', 'border': 'rgba(153, 102, 255, 1)'},
                {'bg': 'rgba(255, 159, 64, 0.7)', 'border': 'rgba(255, 159, 64, 1)'},
                {'bg': 'rgba(199, 199, 199, 0.7)', 'border': 'rgba(199, 199, 199, 1)'},
                {'bg': 'rgba(83, 102, 255, 0.7)', 'border': 'rgba(83, 102, 255, 1)'}
            ]
            
            for i, estilo in enumerate(sorted(all_estilos)):
                if estilo:  # Skip empty estilos
                    color = colors[i % len(colors)]
                    data = []
                    
                    for fecha in sorted_dates:
                        qty = daily_totals[fecha]['estilos'].get(estilo, 0)
                        data.append(qty)
                    
                    datasets.append({
                        'label': estilo,
                        'data': data,
                        'backgroundColor': color['bg'],
                        'borderColor': color['border'],
                        'borderWidth': 2,
                        'fill': False,
                        'tension': 0.1
                    })
            
            chart_data = {
                'labels': chart_labels,
                'datasets': datasets
            }
            
            print(f"Processed data for {len(sorted_dates)} dates and {len(all_estilos)} estilos", flush=True)
            print(f"Chart data has {len(datasets)} datasets", flush=True)
            
            return templates.TemplateResponse(
            request=request,
            name="inventario_daily.html",
            context={
                    "daily_totals": daily_totals,
                    "chart_data": chart_data,
                    "has_chart_data": True
                }
        )
            
        except Exception as fetch_error:
            print(f"Error fetching inventory data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
            request=request,
            name="inventario_daily.html",
            context={
                    "daily_totals": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "has_chart_data": False
                }
        )
            
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error_message": f"Error loading daily inventory: {str(e)}"
            }
        )


# Travel Sales Endpoints
@app.get("/ventasviaje", response_class=HTMLResponse)
async def get_ventas_viaje_form(request: Request):
    """Render the travel sales form"""
    try:
        print("Fetching styles for ventasviaje form", flush=True)
        
        # Get available styles from inventario_estilos where prioridad=1
        estilos = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario_estilos",
            params={
                "select": "id,nombre,precio",
                "prioridad": "eq.1"
            }
        )
        
        print(f"Retrieved {len(estilos)} styles for ventasviaje", flush=True)
        
        # Convert to JSON string for JavaScript
        import json
        estilos_json = json.dumps(estilos)
        
        return templates.TemplateResponse(
            request=request,
            name="ventasviaje.html",
            context={
            "estilos": estilos,
            "estilos_json": estilos_json
        }
        )
    except Exception as e:
        print(f"Error in ventasviaje GET: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching styles: {str(e)}")

@app.post("/ventasviaje")
async def process_ventas_viaje(
    request: Request,
    cliente: str = Form(...),
    quantities: List[int] = Form(...),
    style_ids: List[int] = Form(...),
    prices: List[float] = Form(...)
):
    """Process the travel sales form and save to ventas_travel2"""
    try:
        print(f"Processing ventasviaje form submission for cliente: {cliente}", flush=True)
        print(f"Quantities: {quantities}", flush=True)
        print(f"Style IDs: {style_ids}", flush=True)
        print(f"Prices: {prices}", flush=True)
        
        # Validate that all arrays have the same length
        if not (len(quantities) == len(style_ids) == len(prices)):
            raise HTTPException(status_code=400, detail="Mismatched array lengths")
        
        # Get the next order_id by finding the maximum existing order_id and adding 1
        try:
            max_order_response = await supabase_request(
                method="GET",
                endpoint="/rest/v1/ventas_travel2",
                params={
                    "select": "order_id",
                    "order": "order_id.desc",
                    "limit": "1"
                }
            )
            
            next_order_id = await get_next_order_id()  # Default if no records exist
            if max_order_response and len(max_order_response) > 0:
                max_order_id = max_order_response[0].get('order_id', 0)
                next_order_id = max_order_id + 1
                
            print(f"Next order_id will be: {next_order_id}", flush=True)
        except Exception as order_error:
            print(f"Error getting next order_id, using 1: {str(order_error)}", flush=True)
            next_order_id = 1
        
        # Prepare data for insertion
        ventas_data = []
        total_amount = 0
        
        for i in range(len(quantities)):
            if quantities[i] > 0:  # Only process items with quantity > 0
                print(f"Processing item {i}: qty={quantities[i]}, style_id={style_ids[i]}, price={prices[i]}", flush=True)
                
                # Get style name from inventario_estilos
                style_response = await supabase_request(
                    method="GET",
                    endpoint="/rest/v1/inventario_estilos",
                    params={
                        "select": "nombre",
                        "id": f"eq.{style_ids[i]}"
                    }
                )
                
                if not style_response:
                    raise HTTPException(status_code=404, detail=f"Style with ID {style_ids[i]} not found")
                
                style_name = style_response[0]["nombre"]
                line_total = quantities[i] * int(prices[i])
                total_amount += line_total
                
                ventas_data.append({
                    "order_id": next_order_id,
                    "cliente": cliente,
                    "qty": quantities[i],
                    "estilo_id": style_ids[i],
                    "estilo": style_name,
                    "precio": int(prices[i]),
                    "subtotal": line_total
                })
                
                print(f"Added item: {style_name}, qty: {quantities[i]}, precio: {int(prices[i])}, subtotal: {line_total}", flush=True)
        
        if not ventas_data:
            raise HTTPException(status_code=400, detail="No items with quantity > 0 found")
        
        print(f"Prepared {len(ventas_data)} items for insertion with order_id {next_order_id}, total amount: {total_amount}", flush=True)
        
        # Insert data into ventas_travel2 table
        response = await supabase_request(
            method="POST",
            endpoint="/rest/v1/ventas_travel2",
            json_data=ventas_data
        )
        
        print(f"Supabase insert response: {response}", flush=True)
        
        if response:
            print(f"Successfully recorded {len(ventas_data)} sales items for order {next_order_id}", flush=True)
            
            return {
                "success": True,
                "message": f"Successfully recorded {len(ventas_data)} sales items",
                "total_amount": total_amount,
                "items_recorded": len(ventas_data),
                "order_id": next_order_id,
                "cliente": cliente,
                "ventas_data": ventas_data  # Include for sharing
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to insert data")
            
    except Exception as e:
        print(f"Error in ventasviaje POST: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing sales: {str(e)}")

# Alternative AJAX endpoint for ventasviaje
@app.post("/ventasviaje/ajax")
async def process_ventas_viaje_ajax(request: Request):
    """Process travel sales via AJAX"""
    try:
        # Get JSON data from request
        data = await request.json()
        sales_data = data.get('sales_data', [])
        
        print(f"Processing AJAX ventasviaje with {len(sales_data)} items", flush=True)
        
        ventas_data = []
        total_amount = 0
        
        for item in sales_data:
            if item.get("qty", 0) > 0:
                # Get style info from inventario_estilos
                style_response = await supabase_request(
                    method="GET",
                    endpoint="/rest/v1/inventario_estilos",
                    params={
                        "select": "nombre,precio",
                        "id": f"eq.{item['estilo_id']}",
                        "prioridad": "eq.1"
                    }
                )
                
                if not style_response:
                    continue
                
                style_info = style_response[0]
                line_total = item["qty"] * style_info["precio"]
                total_amount += line_total
                
                ventas_data.append({
                    "qty": item["qty"],
                    "estilo_id": item["estilo_id"],
                    "estilo": style_info["nombre"],
                    "precio": int(style_info["precio"])
                })
        
        if ventas_data:
            response = await supabase_request(
                method="POST",
                endpoint="/rest/v1/ventas_travel2",
                json_data=ventas_data
            )
            
            print(f"AJAX: Successfully recorded {len(ventas_data)} items", flush=True)
            return {
                "success": True,
                "message": "Sales recorded successfully",
                "total_amount": total_amount,
                "items_recorded": len(ventas_data)
            }
        else:
            return {"success": False, "message": "No valid items to record"}
            
    except Exception as e:
        print(f"Error in AJAX ventasviaje: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error: {str(e)}"}



# View Travel Sales Endpoint - Clean Template Approach
@app.get("/verventasviaje", response_class=HTMLResponse)
async def ver_ventas_viaje(request: Request):
    """View all travel sales with sharing options"""
    try:
        print("Fetching all travel sales from ventas_travel2", flush=True)
        
        # Get all sales from ventas_travel2 ordered by order_id descending
        ventas = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_travel2",
            params={
                "select": "*",
                "order": "order_id.desc,created_at.desc"
            }
        )
        
        if not isinstance(ventas, list):
            ventas = []
        
        print(f"Retrieved {len(ventas)} travel sales records", flush=True)
        
        # Group by order_id - simple approach
        orders = {}
        for venta in ventas:
            order_id = venta.get('order_id')
            if order_id not in orders:
                orders[order_id] = {
                    'order_id': order_id,
                    'cliente': venta.get('cliente', ''),
                    'created_at': venta.get('created_at', ''),
                    'productos': [],  # Use 'productos' instead of 'items'
                    'total': 0
                }
            
            # Handle null values safely
            qty = venta.get('qty') or 0
            precio = venta.get('precio') or 0
            subtotal = venta.get('subtotal')
            if subtotal is None:
                subtotal = qty * precio
            
            orders[order_id]['productos'].append({
                'qty': qty,
                'estilo': venta.get('estilo', ''),
                'precio': precio,
                'subtotal': subtotal
            })
            
            orders[order_id]['total'] += subtotal
        
        # Convert to list and sort
        orders_list = list(orders.values())
        orders_list.sort(key=lambda x: x['order_id'], reverse=True)
        
        print(f"Grouped into {len(orders_list)} orders", flush=True)
        
        return templates.TemplateResponse(
            request=request,
            name="ver_ventas_viaje.html",
            context={
            "orders": orders_list
        }
        )
        
    except Exception as e:
        print(f"Error loading travel sales: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading travel sales: {str(e)}")

# Share specific order via WhatsApp
@app.get("/verventasviaje/whatsapp/{order_id}")
async def share_order_whatsapp(order_id: int):
    """Generate WhatsApp link for specific order"""
    try:
        print(f"Generating WhatsApp link for order {order_id}", flush=True)
        
        # Get order details
        ventas = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_travel2",
            params={
                "select": "*",
                "order_id": f"eq.{order_id}",
                "order": "created_at.asc"
            }
        )
        
        if not ventas:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        
        # Build WhatsApp message
        first_item = ventas[0]
        cliente = first_item.get('cliente', 'Cliente')
        created_at = first_item.get('created_at', '')
        
        # Parse date
        try:
            from datetime import datetime
            if created_at:
                date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime('%d/%m/%Y %H:%M')
            else:
                formatted_date = 'N/A'
        except:
            formatted_date = 'N/A'
        
        message = f"🧾 *RECIBO DE VENTA*\n\n"
        message += f"📋 Orden: #{order_id}\n"
        message += f"👤 Cliente: {cliente}\n"
        message += f"📅 Fecha: {formatted_date}\n\n"
        message += f"📦 *PRODUCTOS:*\n"
        
        total = 0
        for item in ventas:
            qty = item.get('qty') or 0
            estilo = item.get('estilo', '')
            precio = item.get('precio') or 0
            subtotal = item.get('subtotal')
            
            # Calculate subtotal if missing
            if subtotal is None:
                subtotal = qty * precio
            
            total += subtotal
            
            message += f"• {qty}x {estilo} - ${precio} = ${subtotal}\n"
        
        message += f"\n💰 *TOTAL: ${total}*\n\n"
        message += f"Gracias por su compra! 🙏"
        
        # Encode for WhatsApp URL
        import urllib.parse
        encoded_message = urllib.parse.quote(message)
        whatsapp_url = f"https://wa.me/?text={encoded_message}"
        
        print(f"Generated WhatsApp URL for order {order_id}", flush=True)
        
        return {
            "success": True,
            "whatsapp_url": whatsapp_url,
            "message": message,
            "order_id": order_id,
            "cliente": cliente,
            "total": total
        }
        
    except Exception as e:
        print(f"Error generating WhatsApp link: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error generating WhatsApp link: {str(e)}")

# Generate PDF for specific order
@app.get("/verventasviaje/pdf/{order_id}")
async def generate_order_pdf(order_id: int):
    """Generate PDF for specific order"""
    try:
        print(f"Generating PDF for order {order_id}", flush=True)
        
        # Get order details
        ventas = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_travel2",
            params={
                "select": "*",
                "order_id": f"eq.{order_id}",
                "order": "created_at.asc"
            }
        )
        
        if not ventas:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        
        # Try to generate PDF
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            import io
            import base64
            from datetime import datetime
            
            # Create PDF in memory
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            
            # Get order info
            first_item = ventas[0]
            cliente = first_item.get('cliente', 'Cliente')
            created_at = first_item.get('created_at', '')
            
            # Parse date
            try:
                if created_at:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%d/%m/%Y %H:%M')
                else:
                    formatted_date = datetime.now().strftime('%d/%m/%Y %H:%M')
            except:
                formatted_date = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            # Header
            p.setFont("Helvetica-Bold", 18)
            p.drawString(50, height - 50, "RECIBO DE VENTA")
            
            # Order info
            p.setFont("Helvetica", 12)
            p.drawString(50, height - 80, f"Orden: #{order_id}")
            p.drawString(50, height - 100, f"Cliente: {cliente}")
            p.drawString(50, height - 120, f"Fecha: {formatted_date}")
            
            # Line separator
            p.line(50, height - 140, width - 50, height - 140)
            
            # Table header
            y_position = height - 170
            p.setFont("Helvetica-Bold", 11)
            p.drawString(50, y_position, "Cant.")
            p.drawString(120, y_position, "Producto")
            p.drawString(350, y_position, "Precio")
            p.drawString(420, y_position, "Subtotal")
            
            # Table content
            p.setFont("Helvetica", 10)
            y_position -= 25
            total = 0
            
            for item in ventas:
                qty = item.get('qty') or 0
                estilo = item.get('estilo', '')[:35]  # Truncate long names
                precio = item.get('precio') or 0
                subtotal = item.get('subtotal')
                
                # Calculate subtotal if missing
                if subtotal is None:
                    subtotal = qty * precio
                
                total += subtotal
                
                p.drawString(50, y_position, str(qty))
                p.drawString(120, y_position, estilo)
                p.drawString(350, y_position, f"${precio}")
                p.drawString(420, y_position, f"${subtotal}")
                y_position -= 20
            
            # Total line
            y_position -= 10
            p.line(350, y_position, width - 50, y_position)
            y_position -= 20
            p.setFont("Helvetica-Bold", 14)
            p.drawString(350, y_position, f"TOTAL: ${total}")
            
            # Footer
            y_position -= 50
            p.setFont("Helvetica-Oblique", 10)
            p.drawString(50, y_position, "Gracias por su compra!")
            
            p.save()
            
            # Get PDF data as base64
            buffer.seek(0)
            pdf_data = base64.b64encode(buffer.read()).decode()
            buffer.close()
            
            print(f"PDF generated successfully for order {order_id}", flush=True)
            
            return {
                "success": True,
                "pdf_data": pdf_data,
                "filename": f"recibo-orden-{order_id}.pdf",
                "order_id": order_id,
                "cliente": cliente,
                "total": total
            }
            
        except ImportError:
            print("ReportLab not installed, cannot generate PDF", flush=True)
            return {
                "success": False,
                "error": "PDF generation not available (ReportLab not installed)",
                "order_id": order_id
            }
        except Exception as pdf_error:
            print(f"Error generating PDF: {str(pdf_error)}", flush=True)
            return {
                "success": False,
                "error": f"Error generating PDF: {str(pdf_error)}",
                "order_id": order_id
            }
            
    except Exception as e:
        print(f"Error in PDF generation endpoint: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
    
@app.get("/entradamercancia", response_class=HTMLResponse)
async def get_entrada_mercancia_form(request: Request):
    try:
        return templates.TemplateResponse(
    request=request,
    name="entrada_mercancia.html"
    )
    except Exception as e:
        logger.error(f"Error loading entrada mercancia form: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
# ─────────────────────────────────────────────────────────────
# POST /entradamercancia/upload_imagen  – upload image to Storage
# ─────────────────────────────────────────────────────────────
@app.post("/entradamercancia/upload_imagen")
async def upload_imagen(image: UploadFile = File(...)):
    """
    Receives a multipart image file, uploads it to Supabase Storage,
    and returns the public URL.
    """
    try:
        # Validate type
        if not image.content_type.startswith("image/"):
            return JSONResponse({"success": False, "message": "El archivo no es una imagen."}, status_code=400)
 
        # Read file bytes
        file_bytes = await image.read()
 
        # Max 10 MB
        if len(file_bytes) > 10 * 1024 * 1024:
            return JSONResponse({"success": False, "message": "La imagen excede 10 MB."}, status_code=400)
 
        # Build a unique storage path
        ext       = image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else "jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"
        path      = f"entradas/{filename}"
 
        # Upload via Supabase Storage REST API
        upload_url = f"{STORAGE_BASE_URL}/object/{STORAGE_BUCKET}/{path}"
 
        import httpx
        async with httpx.AsyncClient() as client:
            upload_resp = await client.post(
                upload_url,
                content=file_bytes,
                headers={
                    "Authorization":  f"Bearer {SUPABASE_KEY}",   # your existing SUPABASE_KEY var
                    "Content-Type":   image.content_type,
                    "x-upsert":       "true",
                }
            )
 
        if upload_resp.status_code not in (200, 201):
            logger.error(f"Storage upload failed: {upload_resp.status_code} – {upload_resp.text}")
            return JSONResponse({
                "success": False,
                "message": f"Error al subir imagen: {upload_resp.status_code}"
            }, status_code=500)
 
        # Build public URL
        public_url = f"{STORAGE_BASE_URL}/object/public/{STORAGE_BUCKET}/{path}"
        logger.info(f"Image uploaded: {public_url}")
 
        return {"success": True, "url": public_url, "path": path}
 
    except Exception as e:
        logger.error(f"upload_imagen error: {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
 
 
# ─────────────────────────────────────────────────────────────
# POST /entradamercancia  – register entry (JSON body)
# ─────────────────────────────────────────────────────────────
@app.post("/entradamercancia")
async def process_entrada_mercancia(payload: EntradaPayload):
    """
    Register a merchandise entry.
    Expects JSON: { qty, barcode, numero_caja?, notas?, imagen_url? }
    """
    try:
        qty         = payload.qty
        barcode     = payload.barcode.strip()
        numero_caja = payload.numero_caja
        notas       = payload.notas
        imagen_url  = payload.imagen_url
 
        logger.info(f"Entrada: qty={qty} barcode={barcode} caja={numero_caja}")
 
        if qty <= 0:
            raise HTTPException(status_code=400, detail="La cantidad debe ser mayor a 0")
        if not barcode:
            raise HTTPException(status_code=400, detail="El código de barras es requerido")
 
        # Convert barcode to int
        try:
            barcode_int = int(barcode)
        except ValueError:
            raise HTTPException(status_code=400, detail="El código de barras debe ser numérico")
 
        # ── Fetch product info from inventario1 ────────────
        product_info   = None
        current_terex1 = 0
        try:
            product_response = await supabase_request(
                method="GET",
                endpoint="/rest/v1/inventario1",
                params={
                    "select":  "name,estilo_id,marca,terex1",
                    "barcode": f"eq.{barcode_int}",
                    "limit":   "1"
                }
            )
            if product_response:
                product_info   = product_response[0]
                current_terex1 = product_info.get("terex1") or 0
        except Exception as e:
            logger.warning(f"Could not fetch product info: {e}")
 
        # ── Build entrada record ───────────────────────────
        entrada_data = {
            "qty":     qty,
            "barcode": barcode_int,
        }
        if product_info:
            if product_info.get("name"):
                entrada_data["estilo"] = product_info["name"]
            if product_info.get("estilo_id"):
                entrada_data["estilo_id"] = product_info["estilo_id"]
        if numero_caja is not None:
            entrada_data["numero_caja"] = numero_caja
        if notas:
            entrada_data["notas"] = notas
        if imagen_url:
            entrada_data["imagen_url"] = imagen_url
 
        # ── Insert into entrada_mercancia ──────────────────
        try:
            await supabase_request(
                method="POST",
                endpoint="/rest/v1/entrada_mercancia",
                json_data=entrada_data
            )
        except Exception as insert_error:
            logger.error(f"Insert error: {insert_error}, retrying minimal…")
            # Fallback: minimal insert
            await supabase_request(
                method="POST",
                endpoint="/rest/v1/entrada_mercancia",
                json_data={"qty": qty, "barcode": barcode_int}
            )
 
        # ── Update inventario1.terex1 ──────────────────────
        if product_info:
            try:
                await supabase_request(
                    method="PATCH",
                    endpoint=f"/rest/v1/inventario1?barcode=eq.{barcode_int}",
                    json_data={"terex1": current_terex1 + qty}
                )
            except Exception as upd_err:
                logger.error(f"terex1 update failed (non-fatal): {upd_err}")
 
        return {
            "success":        True,
            "message":        "Entrada registrada exitosamente",
            "qty":            qty,
            "barcode":        barcode,
            "product_name":   product_info.get("name", "Producto no identificado") if product_info else "Producto no identificado",
            "numero_caja":    numero_caja,
            "terex1_updated": product_info is not None,
            "imagen_url":     imagen_url,
        }
 
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
 
 
# ─────────────────────────────────────────────────────────────
# GET /entradamercancia/con_imagenes  – entries with image + estilo_id
# ─────────────────────────────────────────────────────────────
@app.get("/entradamercancia/con_imagenes")
async def get_entradas_con_imagenes():
    """Returns all entries that have both imagen_url and estilo_id set."""
    try:
        entries = await supabase_request(
            method="GET",
            endpoint="/rest/v1/entrada_mercancia",
            params={
                "select":     "id,created_at,qty,barcode,estilo,estilo_id,numero_caja,notas,imagen_url",
                "imagen_url": "not.is.null",
                "estilo_id":  "not.is.null",
                "order":      "created_at.desc",
                "limit":      "200"
            }
        )
        return {"success": True, "entries": entries}
    except Exception as e:
        logger.error(f"get_entradas_con_imagenes error: {e}")
        return {"success": False, "error": str(e), "entries": []}
 
 
# ─────────────────────────────────────────────────────────────
# GET /entradamercancia/recientes  – last 20 entries
# ─────────────────────────────────────────────────────────────
@app.get("/entradamercancia/recientes")
async def get_recent_entries():
    try:
        entries = await supabase_request(
            method="GET",
            endpoint="/rest/v1/entrada_mercancia",
            params={
                "select": "id,created_at,qty,barcode,estilo,numero_caja,notas,imagen_url",
                "order":  "created_at.desc",
                "limit":  "20"
            }
        )
        return {"success": True, "entries": entries}
    except Exception as e:
        logger.error(f"get_recent_entries error: {e}")
        return {"success": False, "error": str(e), "entries": []}
 
 
# ─────────────────────────────────────────────────────────────
# GET /entradamercancia/ultima_caja  – last box number used
# ─────────────────────────────────────────────────────────────
@app.get("/entradamercancia/ultima_caja")
async def get_ultima_caja():
    """Returns the highest numero_caja stored in the DB."""
    try:
        result = await supabase_request(
            method="GET",
            endpoint="/rest/v1/entrada_mercancia",
            params={
                "select":       "numero_caja",
                "order":        "numero_caja.desc.nullslast",
                "limit":        "1",
                "numero_caja":  "not.is.null"
            }
        )
        if result:
            return {"success": True, "numero_caja": result[0].get("numero_caja")}
        return {"success": True, "numero_caja": None}
    except Exception as e:
        logger.error(f"get_ultima_caja error: {e}")
        return {"success": False, "numero_caja": None}

@app.get("/verimagenes", response_class=HTMLResponse)
async def ver_imagenes(request: Request):
    """Main page to view images associated with estilos"""
    return templates.TemplateResponse(
            request=request,
            name="verimagenes.html",
            context={}
        )

@app.get("/api/estilos-with-images")
async def get_estilos_with_images():
    """Get estilos with images and sales data.
    Priority for thumbnail: images_estilos bucket first, then image_uploads (color-level) as fallback.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            # Fetch estilos + color-level images in parallel
            estilos_resp, images_resp = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                    headers=HEADERS,
                    params={"select": "id,nombre", "prioridad": "eq.1", "order": "nombre"}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/image_uploads",
                    headers=HEADERS,
                    params={"select": "estilo_id,color_id,public_url,file_name,description"}
                ),
            )

            estilos_data = estilos_resp.json()

            # Index color-level images from image_uploads
            color_image_counts = {}
            color_counts = {}
            color_sample_images = {}

            if images_resp.status_code == 200:
                for image in images_resp.json():
                    eid = image.get('estilo_id')
                    cid = image.get('color_id')
                    if eid is not None:
                        color_image_counts[eid] = color_image_counts.get(eid, 0) + 1
                        if eid not in color_counts:
                            color_counts[eid] = set()
                        if cid is not None:
                            color_counts[eid].add(cid)
                        if eid not in color_sample_images and image.get('public_url'):
                            color_sample_images[eid] = {
                                "public_url": image['public_url'],
                                "file_name": image.get('file_name', ''),
                                "description": image.get('description', ''),
                            }

            # For each estilo, check images_estilos bucket for estilo-level thumbnail
            estilo_bucket_images = {}
            bucket_tasks = []
            for estilo in estilos_data:
                eid = int(estilo['id'])
                bucket_tasks.append((eid, client.post(
                    f"{SUPABASE_URL}/storage/v1/object/list/images_estilos",
                    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
                    json={"prefix": f"{eid}/", "limit": 1},
                )))

            for eid, task in bucket_tasks:
                try:
                    resp = await task
                    if resp.status_code == 200:
                        files = resp.json()
                        for f in files:
                            if f.get("name") and f.get("id"):
                                estilo_bucket_images[eid] = {
                                    "public_url": f"{SUPABASE_URL}/storage/v1/object/public/images_estilos/{eid}/{f['name']}",
                                    "file_name": f['name'],
                                    "description": "",
                                }
                                break
                except Exception:
                    pass

            # Build response
            estilos = []
            for estilo in estilos_data:
                estilo_id = int(estilo['id'])

                sales_response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                    headers=HEADERS,
                    params={
                        "select": "qty",
                        "estilo_id": f"eq.{estilo_id}",
                        "fecha": f"gte.{three_months_ago}"
                    }
                )

                sales_total = 0
                if sales_response.status_code == 200:
                    sales_total = sum(r.get('qty', 0) for r in sales_response.json())

                # Priority: images_estilos bucket > image_uploads (color-level)
                sample_image = estilo_bucket_images.get(estilo_id) or color_sample_images.get(estilo_id)

                estilos.append({
                    "id": estilo_id,
                    "nombre": estilo['nombre'],
                    "total_images": color_image_counts.get(estilo_id, 0),
                    "total_colors_with_images": len(color_counts.get(estilo_id, set())),
                    "sales_last_3_months": sales_total,
                    "sample_image": sample_image,
                    "has_estilo_image": estilo_id in estilo_bucket_images,
                })

            return {"estilos": estilos}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/estilo/{estilo_id}/colors-with-images")
async def get_colors_with_images(estilo_id: int):
    """Get colors and images for specific estilo"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get estilo info
            estilo_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params={"select": "id,nombre", "id": f"eq.{estilo_id}", "prioridad": "eq.1"}
            )
            
            if estilo_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Could not fetch estilo")
            
            estilo_data = estilo_response.json()
            
            if not estilo_data:
                raise HTTPException(status_code=404, detail="Estilo not found")
            
            # Get all colors for this estilo (any barcode in inventario1)
            inventory_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario1",
                headers=HEADERS,
                params={"select": "color_id", "estilo_id": f"eq.{estilo_id}"}
            )
            
            if inventory_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Could not fetch inventory")
            
            inventory_data = inventory_response.json()
            color_ids = list(set([item['color_id'] for item in inventory_data if item['color_id']]))
            
            if not color_ids:
                return {"estilo": estilo_data[0], "colors": []}
            
            # Get color details
            colors_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_colores",
                headers=HEADERS,
                params={"select": "id,color", "id": f"in.({','.join(map(str, color_ids))})", "order": "color"}
            )
            
            if colors_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Could not fetch colors")
            
            colors_data = colors_response.json()
            
            # Get images for this estilo
            images_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers=HEADERS,
                params={
                    "select": "id,color_id,file_name,public_url,description,created_at",
                    "estilo_id": f"eq.{estilo_id}",
                    "order": "created_at.desc"
                }
            )
            
            # Group images by color_id
            images_by_color = {}
            if images_response.status_code == 200:
                for image in images_response.json():
                    color_id = image['color_id']
                    if color_id not in images_by_color:
                        images_by_color[color_id] = []
                    images_by_color[color_id].append(image)
            
            # Build colors with images
            colors = []
            for color in colors_data:
                color_id = color['id']
                color_images = images_by_color.get(color_id, [])
                
                colors.append({
                    "id": color_id,
                    "color": color['color'],
                    "image_count": len(color_images),
                    "images": color_images
                })
            
            return {"estilo": estilo_data[0], "colors": colors}
    
    except Exception as e:
        logger.error(f"Error in get_colors_with_images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/estilo/{estilo_id}/color/{color_id}/upload")
async def upload_color_image(estilo_id: int, color_id: int, file: UploadFile = File(...)):
    """Upload image for a specific estilo+color and save to image_uploads table."""
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Archivo demasiado grande (max 10MB)")

        ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
        filename = f"{uuid.uuid4().hex[:12]}.{ext}"
        storage_path = f"{estilo_id}/{color_id}/{filename}"
        bucket = "images-colores"

        async with httpx.AsyncClient(timeout=30) as client:
            # Upload to Supabase Storage
            resp = await client.post(
                f"{SUPABASE_URL}/storage/v1/object/{bucket}/{storage_path}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": file.content_type or "image/jpeg",
                },
                content=contents,
            )

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=f"Storage error: {resp.text}")

            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"

            # Save metadata to image_uploads table
            record = {
                "estilo_id": estilo_id,
                "color_id": color_id,
                "file_name": file.filename,
                "file_path": storage_path,
                "public_url": public_url,
                "description": "",
            }
            db_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers={**HEADERS, "Prefer": "return=representation"},
                json=record,
            )

            print(f"DB insert response: status={db_resp.status_code} body={db_resp.text}", flush=True)
            if db_resp.status_code >= 400:
                print(f"image_uploads insert error: {db_resp.text}", flush=True)
                raise HTTPException(status_code=500, detail=f"Imagen subida pero error guardando en BD: {db_resp.text}")

        return JSONResponse({"ok": True, "url": public_url, "file_name": file.filename})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/image/{image_id}")
async def delete_image(image_id: int):
    """Delete image from database"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if image exists first
            check_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers=HEADERS,
                params={"select": "id", "id": f"eq.{image_id}"}
            )

            if check_response.status_code == 200:
                check_data = check_response.json()
                if not check_data:
                    raise HTTPException(status_code=404, detail="Image not found")

            # Delete the image
            delete_response = await client.delete(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers=HEADERS,
                params={"id": f"eq.{image_id}"}
            )

            if delete_response.status_code not in [200, 204]:
                raise HTTPException(status_code=500, detail="Failed to delete image")

            return {"message": "Image deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/image/delete-by-url")
async def delete_image_by_url(request: Request):
    """Delete image by public_url (removes DB row + storage file)."""
    try:
        body = await request.json()
        public_url = (body.get("public_url") or "").strip()
        if not public_url:
            raise HTTPException(status_code=400, detail="public_url required")

        # Derive storage path + bucket from the URL
        marker = "/storage/v1/object/public/"
        if marker not in public_url:
            raise HTTPException(status_code=400, detail="Invalid storage URL")
        after = public_url.split(marker, 1)[1]
        bucket, _, storage_path = after.partition("/")
        if not bucket or not storage_path:
            raise HTTPException(status_code=400, detail="Could not parse bucket/path from URL")

        async with httpx.AsyncClient(timeout=15) as client:
            # 1. Delete DB row (if any) matching public_url
            db_resp = await client.delete(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers=HEADERS,
                params={"public_url": f"eq.{public_url}"}
            )
            db_ok = db_resp.status_code in (200, 204)

            # 2. Delete the actual file in storage
            storage_resp = await client.delete(
                f"{SUPABASE_URL}/storage/v1/object/{bucket}/{storage_path}",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            )
            storage_ok = storage_resp.status_code in (200, 204)

        return {"db_deleted": db_ok, "storage_deleted": storage_ok, "bucket": bucket, "path": storage_path}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image by url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-estilo/{estilo_id}")
async def test_specific_estilo(estilo_id: int):
    """Debug sales for specific estilo"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # All time sales
            all_time_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={"select": "estilo_id,qty,fecha", "estilo_id": f"eq.{estilo_id}"}
            )
            
            all_time_data = all_time_response.json() if all_time_response.status_code == 200 else []
            all_time_total = sum(record.get('qty', 0) for record in all_time_data)
            
            # Recent sales
            recent_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "estilo_id": f"eq.{estilo_id}",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            recent_data = recent_response.json() if recent_response.status_code == 200 else []
            recent_total = sum(record.get('qty', 0) for record in recent_data)
            
            return {
                "estilo_id": estilo_id,
                "sql_equivalent": f"SELECT SUM(qty) FROM ventas_terex1 WHERE estilo_id={estilo_id} AND fecha>='{three_months_ago}'",
                "date_filter": three_months_ago,
                "all_time_sales": {
                    "total_qty": all_time_total,
                    "record_count": len(all_time_data),
                    "sample_records": all_time_data[:5]
                },
                "last_3_months_sales": {
                    "total_qty": recent_total,
                    "record_count": len(recent_data),
                    "all_records": recent_data
                }
            }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-sales-simple")
async def debug_sales_simple():
    """Simple sales debug"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Test estilo 9
            estilo_9_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "estilo_id": "eq.9",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            estilo_9_data = estilo_9_response.json() if estilo_9_response.status_code == 200 else []
            estilo_9_total = sum(record.get('qty', 0) for record in estilo_9_data)
            
            # Test estilo 139
            estilo_139_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "estilo_id": "eq.139",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            estilo_139_data = estilo_139_response.json() if estilo_139_response.status_code == 200 else []
            estilo_139_total = sum(record.get('qty', 0) for record in estilo_139_data)
            
            # Get all recent sales
            all_recent_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "fecha": f"gte.{three_months_ago}",
                    "limit": "100"
                }
            )
            
            all_recent_data = all_recent_response.json() if all_recent_response.status_code == 200 else []
            
            # Group by estilo_id
            totals_by_estilo = {}
            for record in all_recent_data:
                estilo_id = record.get('estilo_id')
                qty = record.get('qty', 0)
                if estilo_id is not None and qty is not None:
                    totals_by_estilo[estilo_id] = totals_by_estilo.get(estilo_id, 0) + qty
            
            return {
                "date_filter": three_months_ago,
                "estilo_9": {
                    "total_qty": estilo_9_total,
                    "records": estilo_9_data
                },
                "estilo_139": {
                    "total_qty": estilo_139_total,
                    "records": estilo_139_data
                },
                "all_sales_summary": {
                    "total_records": len(all_recent_data),
                    "totals_by_estilo": totals_by_estilo
                }
            }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-main-endpoint")
async def debug_main_endpoint():
    """Debug the main estilos endpoint to see why sales aren't showing"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Get estilos
            estilos_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params={"select": "id,nombre", "prioridad": "eq.1", "order": "nombre"}
            )
            estilos_data = estilos_response.json()
            
            # Get sales
            sales_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            sales_data = sales_response.json() if sales_response.status_code == 200 else []
            
            # Process sales
            sales_totals = {}
            for sale in sales_data:
                estilo_id = sale.get('estilo_id')
                qty = sale.get('qty', 0)
                if estilo_id is not None and qty is not None:
                    estilo_id = int(estilo_id)
                    sales_totals[estilo_id] = sales_totals.get(estilo_id, 0) + qty
            
            # Check specific estilos
            estilo_139_data = None
            estilo_9_data = None
            
            for estilo in estilos_data:
                estilo_id = int(estilo['id'])
                if estilo_id == 139:
                    estilo_139_data = {
                        "id": estilo_id,
                        "nombre": estilo['nombre'],
                        "sales_from_totals": sales_totals.get(estilo_id, 0)
                    }
                elif estilo_id == 9:
                    estilo_9_data = {
                        "id": estilo_id,
                        "nombre": estilo['nombre'],
                        "sales_from_totals": sales_totals.get(estilo_id, 0)
                    }
            
            return {
                "debug_info": {
                    "date_filter": three_months_ago,
                    "total_sales_records": len(sales_data),
                    "total_estilos": len(estilos_data),
                    "sales_totals_keys": list(sales_totals.keys()),
                    "sales_totals_sample": dict(list(sales_totals.items())[:5])
                },
                "estilo_139": estilo_139_data,
                "estilo_9": estilo_9_data,
                "sales_for_139": sales_totals.get(139, "NOT FOUND"),
                "sales_for_9": sales_totals.get(9, "NOT FOUND")
            }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test-table-access")
async def test_table_access():
    """Test basic table access"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test ventas_terex1
            ventas_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={"select": "*", "limit": "1"}
            )
            
            result = {
                "ventas_terex1": {
                    "accessible": ventas_response.status_code == 200,
                    "status_code": ventas_response.status_code,
                }
            }
            
            if ventas_response.status_code == 200:
                data = ventas_response.json()
                result["ventas_terex1"]["sample_data"] = data
                result["ventas_terex1"]["columns"] = list(data[0].keys()) if data else []
            else:
                result["ventas_terex1"]["error"] = ventas_response.text[:500]
            
            return result
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/compare-queries")
async def compare_queries():
    """Compare individual vs bulk sales queries for estilo 139"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Method 1: Individual query for estilo 139 (this works)
            individual_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "estilo_id": "eq.139",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            individual_data = individual_response.json() if individual_response.status_code == 200 else []
            individual_total = sum(record.get('qty', 0) for record in individual_data)
            
            # Method 2: Bulk query with same date filter (this doesn't include 139)
            bulk_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "estilo_id,qty,fecha",
                    "fecha": f"gte.{three_months_ago}"
                }
            )
            
            bulk_data = bulk_response.json() if bulk_response.status_code == 200 else []
            
            # Check if estilo 139 exists in bulk data
            estilo_139_in_bulk = [record for record in bulk_data if record.get('estilo_id') == 139]
            
            # Count unique estilo_ids in bulk
            unique_estilos_in_bulk = set(record.get('estilo_id') for record in bulk_data if record.get('estilo_id'))
            
            return {
                "date_filter": three_months_ago,
                "individual_query": {
                    "status_code": individual_response.status_code,
                    "total_records": len(individual_data),
                    "total_qty": individual_total,
                    "sample_dates": [r.get('fecha') for r in individual_data[:5]]
                },
                "bulk_query": {
                    "status_code": bulk_response.status_code,
                    "total_records": len(bulk_data),
                    "estilo_139_records_found": len(estilo_139_in_bulk),
                    "estilo_139_sample": estilo_139_in_bulk[:3],
                    "unique_estilos_count": len(unique_estilos_in_bulk),
                    "has_estilo_139": 139 in unique_estilos_in_bulk
                },
                "conclusion": "individual works, bulk doesn't include 139" if individual_total > 0 and len(estilo_139_in_bulk) == 0 else "both queries consistent"
            }
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/nota", response_class=HTMLResponse)
async def nota(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="nota.html",
            context={}
        )

@app.get("/nota1", response_class=HTMLResponse)
async def nota(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="nota1.html",
            context={}
        )




def _now_strs():
    now = datetime.now()
    fecha = f"{now.year:04d}-{now.month:02d}-{now.day:02d}"
    hora  = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    return now, fecha, hora

def _build_receipt_pdf(items: list[dict], total: float, order_id: int) -> io.BytesIO:
    width = 58 * mm
    height = 200 * mm
    margin = 2 * mm
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    def header():
        y = height - margin
        c.setFont("Helvetica-Bold", 10); c.drawCentredString(width/2, y, "TICKET DE VENTA"); y -= 12
        c.setFont("Helvetica-Bold", 9);  c.drawCentredString(width/2, y, f"Orden #{order_id}"); y -= 10
        _, fecha, hora = _now_strs()
        c.setFont("Helvetica", 8); c.drawString(margin, y, f"Fecha: {fecha}"); y -= 10
        c.drawString(margin, y, f"Hora: {hora}"); y -= 6
        c.setStrokeColor(colors.black); c.line(margin, y, width - margin, y); y -= 10
        c.setFont("Helvetica-Bold", 8)
        c.drawString(margin, y, "Cant")
        c.drawString(margin + 20, y, "Descripción")
        c.drawRightString(width - margin, y, "Precio")
        y -= 10
        c.setFont("Helvetica", 8)
        return y

    y = header()
    for it in items:
        if y < 25 * mm:
            c.showPage()
            y = header()
        qty = str(it["qty"]); name = str(it["name"]); price = it["price"]; sub = it["subtotal"]
        c.drawString(margin, y, qty)
        c.drawString(margin + 20, y, (name[:28] + ("…" if len(name) > 28 else "")))
        c.drawRightString(width - margin, y, f"${price:0.2f}"); y -= 10
        c.drawRightString(width - margin, y, f"Subtotal: ${sub:0.2f}"); y -= 12

    y -= 4; c.line(margin, y, width - margin, y); y -= 12
    c.setFont("Helvetica-Bold", 9); c.drawString(margin, y, "TOTAL:"); c.drawRightString(width - margin, y, f"${total:0.2f}"); y -= 14
    c.setFont("Helvetica", 8); c.drawCentredString(width/2, y, "¡Gracias por su compra!"); y -= 10
    try:
        bc = code128.Code128(f"ORDER-{order_id}", barHeight=12 * mm, barWidth=0.35)
        x = (width - bc.width) / 2
        bc.drawOn(c, x, max(margin, y - 16 * mm))
    except Exception:
        pass

    c.showPage(); c.save(); buf.seek(0); return buf









# Add these new endpoints to your FastAPI application

@app.get("/api/search_barcode")
async def api_search_barcode(barcode: str):
    """Enhanced barcode search that handles both products and loyalty barcodes"""
    
    # Check if this is a customer loyalty barcode (starts with 8000)
    if barcode.startswith('8000') and len(barcode) == 13:
        return await handle_loyalty_barcode(barcode)
    
    # Handle regular product barcode search
    rows = await supabase_request(
        method="GET",
        endpoint="/rest/v1/inventario1",
        params={"select": "*", "barcode": f"eq.{barcode}", "limit": 1}
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Producto no encontrado")

    row = rows[0]
    name = row.get("name") or row.get("modelo") or ""
    price = float(row.get("precio") or 0.0)
    send_telegram_message(
    f"🔍 Código escaneado\n"
    f"📦 Producto: {name}\n"
    f"💰 Precio: ${price:.2f}"
    )
    return {"name": name, "price": price, "codigo": barcode}


async def handle_loyalty_barcode(barcode: str):
    try:
        print(f"DEBUG: Looking up loyalty barcode: {barcode}", flush=True)

        barcode_result = await supabase_request(
            method="GET",
            endpoint="/rest/v1/user_barcodes",
            params={
                "select": "user_email,status",
                "barcode": f"eq.{barcode}",
                "status": "eq.active",
                "limit": "1"
            }
        )
        print(f"DEBUG: Barcode query returned: {barcode_result}", flush=True)

        if not barcode_result:
            print(f"DEBUG: Barcode {barcode} not found in database", flush=True)
            raise HTTPException(status_code=404, detail="Código de cliente no válido - barcode no registrado")

        user_email = barcode_result[0]["user_email"]
        print(f"DEBUG: Found user email: {user_email}", flush=True)

        rewards_result = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "reward_amount,status,id",
                "email": f"eq.{user_email}",
                "status": "eq.active"
            }
        )
        print(f"DEBUG: Found {len(rewards_result)} active rewards for {user_email}", flush=True)

        total_available = sum(float(r["reward_amount"]) for r in rewards_result)
        print(f"DEBUG: Total available: {total_available}", flush=True)

        if total_available <= 0:
            return {
                "name": f"CLIENTE - {user_email} (Sin saldo)",
                "price": 0.00,
                "codigo": barcode,
                "is_loyalty": True,
                "customer_email": user_email,
                "available_balance": 0.00
            }

        return {
            "name": f"DESCUENTO - {user_email}",
            "price": -total_available,
            "codigo": barcode,
            "is_loyalty": True,
            "customer_email": user_email,
            "available_balance": total_available
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in handle_loyalty_barcode: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando código de cliente: {str(e)}")


@app.post("/api/admin/create-missing-barcodes")
async def create_missing_barcodes():
    try:
        rewards_emails = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "email",
                "status": "eq.active"
            }
        )
        unique_emails = sorted({(r.get("email") or "").strip().lower() for r in rewards_emails if r.get("email")})
        created_count = 0

        for email in unique_emails:
            # ensure creates only if missing
            barcode, existed = await ensure_user_barcode(email)
            if not existed:
                created_count += 1
                print(f"Created barcode {barcode} for {email}")

        return {
            "success": True,
            "message": f"Created {created_count} barcodes for existing customers",
            "total_emails_checked": len(unique_emails)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating barcodes: {str(e)}")




@app.get("/api/admin/test-barcode/{email}")
async def test_barcode_generation(email: str):
    barcode = generate_user_barcode_from_email(email)
    return {
        "email": _normalize_email(email),
        "generated_barcode": barcode,
        "explanation": "This barcode would be auto-generated for this email"
    }

@app.post("/api/loyalty/redeem")
async def redeem(payload: dict):
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    # You already have this function in your app — placeholder here:
    email = await resolve_token_to_email(token)  # <-- implement/keep your logic
    email = _normalize_email(email)

    # Ensure barcode exists the first time this user redeems
    await ensure_user_barcode(email)

    # Continue with your reward flow (placeholder)
    reward = await add_reward_for_email(email, payload)  # <-- your existing logic

    return {"success": True, "email": email, "reward": reward}


@app.post("/api/user/get-or-create-barcode")
async def get_or_create_barcode(request: dict):
    email = _normalize_email(request.get("email"))
    barcode, existed = await ensure_user_barcode(email)
    return {"success": True, "barcode": barcode, "message": "Existing" if existed else "Created"}



def generate_user_barcode_from_email(email: str) -> str:
    """Deterministic 13-digit barcode (prefix 8000) from email with EAN-13 check digit."""
    email_norm = _normalize_email(email)
    hash_hex = hashlib.md5(email_norm.encode("utf-8")).hexdigest()

    # Collect digits from hex; if fewer than 8, map hex letters to digits (a=10->0, b=11->1, etc.)
    digits = "".join(filter(str.isdigit, hash_hex))
    if len(digits) < 8:
        for ch in hash_hex:
            if not ch.isdigit():
                digits += str((ord(ch) - ord("a") + 10) % 10) if ch.isalpha() else str(ord(ch) % 10)
            if len(digits) >= 8:
                break

    eight = digits[:8]
    code = _build_barcode_from_eight(eight)
    print(f"DEBUG: Generated EAN-13 barcode for {email_norm}: {code}", flush=True)
    return code


async def ensure_user_barcode(email: str):
    """Ensure user has a barcode, create if missing, link existing ones to user account"""
    try:
        # First, check if there's ANY barcode for this email (user_id can be NULL or set)
        existing_barcode = supabase.table("user_barcodes").select("*").eq("user_email", email).eq("status", "active").limit(1).execute()
        
        if existing_barcode.data:
            barcode_record = existing_barcode.data[0]
            
            # If barcode exists but has no user_id, link it to the user account
            if barcode_record.get("user_id") is None:
                # Get the user_id for this email
                user_check = supabase.table("users").select("id").eq("email", email).limit(1).execute()
                
                if user_check.data:
                    user_id = user_check.data[0]["id"]
                    
                    # Update the barcode to link it to the user account
                    supabase.table("user_barcodes").update({
                        "user_id": user_id,
                        "updated_at": datetime.utcnow().isoformat()
                    }).eq("id", barcode_record["id"]).execute()
                    
                    print(f"DEBUG: Linked existing barcode to user account for {email}")
            
            return barcode_record["barcode"]
        
        # No barcode exists, create a new one
        user_check = supabase.table("users").select("id").eq("email", email).limit(1).execute()
        
        if not user_check.data:
            print(f"DEBUG: No user found for email {email}")
            return None
        
        user_id = user_check.data[0]["id"]
        new_barcode = generate_user_barcode(email)
        
        barcode_data = {
            "user_id": user_id,
            "user_email": email,
            "barcode": new_barcode,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("user_barcodes").insert(barcode_data).execute()
        print(f"DEBUG: Created new barcode for {email}: {new_barcode}")
        return new_barcode
        
    except Exception as e:
        print(f"Error ensuring user barcode for {email}: {e}")
        return None




async def create_barcode_for_existing_customer(email: str):
    try:
        barcode, _ = await ensure_user_barcode(email)
        print(f"Created/ensured barcode {barcode} for existing customer {email}", flush=True)
        return barcode
    except Exception as e:
        print(f"Error creating barcode for {email}: {e}", flush=True)
        return None


@app.post("/api/save")
async def api_save(payload: SavePayload):
    """Enhanced save function that processes loyalty deductions and payment method"""
    if not payload.products:
        raise HTTPException(status_code=400, detail="No products provided")

    # Get payment method with fallback
    payment_method = getattr(payload, 'payment_method', 'efectivo')
    print(f"DEBUG: Received payment_method: {payment_method}")  # Debug log
    

    next_order_id = await get_next_order_id()
    mexico_tz = pytz.timezone("America/Mexico_City")
    now = datetime.now(mexico_tz)
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")
    
    items_for_ticket = []
    loyalty_deductions = []
    payment_method = getattr(payload, 'payment_method', 'efectivo')  # Fallback to efectivo if missing

    for p in payload.products:
        # Convert to dict if it's a Pydantic model
        if hasattr(p, 'dict'):
            p_dict = p.dict()
        elif hasattr(p, 'model_dump'):
            p_dict = p.model_dump()
        else:
            p_dict = p  # Already a dict
        
        # Check if this is a loyalty barcode (starts with 8000)
        codigo = p_dict.get('codigo', '')
        if codigo.startswith('8000') and len(codigo) == 13:
            # Process loyalty deduction
            loyalty_result = await process_loyalty_deduction(p_dict, next_order_id, fecha, hora)
            loyalty_deductions.append(loyalty_result)
            
            # Add to ticket items
            items_for_ticket.append({
                "qty": p_dict.get('qty', 1),
                "name": p_dict.get('name', ''),
                "price": p_dict.get('price', 0),
                "subtotal": p_dict.get('qty', 1) * p_dict.get('price', 0)
            })
            continue
        
        # Regular product processing
        inv_rows = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "modelo,modelo_id,estilo,estilo_id,terex1",
                "barcode": f"eq.{codigo}",
                "limit": "1",
            },
        )
        
        if not inv_rows:
            raise HTTPException(
                status_code=400, 
                detail=f"Producto con barcode {codigo} no existe en inventario1"
            )
        
        inv = inv_rows[0]

        # Insert into ventas_terex1 with payment_method
        record = {
            "qty": p_dict.get('qty', 1),
            "name": p_dict.get('name', ''),
            "name_id": codigo,
            "price": p_dict.get('price', 0),
            "fecha": fecha,
            "hora": hora,
            "order_id": next_order_id,
            "modelo": inv.get("modelo", ""),
            "modelo_id": inv.get("modelo_id", ""),
            "estilo": inv.get("estilo", ""),
            "estilo_id": inv.get("estilo_id", ""),
            "payment_method": payment_method  # Add payment method
        }

        await supabase_request(
            method="POST",
            endpoint="/rest/v1/ventas_terex1",
            json_data=record,
        )

        # Update inventory
        current_qty = int(inv.get("terex1") or 0)
        new_qty = current_qty - p_dict.get('qty', 1)
        
        await supabase_request(
            method="PATCH",
            endpoint=f"/rest/v1/inventario1?barcode=eq.{codigo}",
            json_data={"terex1": new_qty},
        )

        # Add to ticket items
        items_for_ticket.append({
            "qty": p_dict.get('qty', 1),
            "name": p_dict.get('name', ''),
            "price": p_dict.get('price', 0),
            "subtotal": p_dict.get('qty', 1) * p_dict.get('price', 0)
        })

    # Calculate total
    total = sum(i["subtotal"] for i in items_for_ticket)
    
    try:
        payment_emoji = "💵" if payment_method == "efectivo" else "💳"
        total_pieces = sum(i['qty'] for i in items_for_ticket)
    
        message = (
            f"🎉 VENTA #{next_order_id}\n"
            f"📊 {total_pieces} piezas\n"
            f"💰 ${total:.2f}\n"
            f"{payment_emoji} {payment_method.title()}\n"
        )
    
        send_telegram_message(message)

        import asyncio
        asyncio.create_task(send_telegram_picture(
            barcode=None,  # Skip barcode, just use order_id
            order_id=next_order_id
        ))
        
    except Exception as e:
        print(f"Telegram error: {e}")

    # If payment method is efectivo, add entry to conteo_efectivo
    if payment_method == "efectivo":
        print(f"DEBUG: Adding to conteo_efectivo for order {next_order_id}")  # Debug
        try:
            current_balance = await get_current_balance()
            new_balance = current_balance + total
            
            url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo"
            payload_conteo = {
                "nombre": f"Venta #{next_order_id}",
                "tipo": "credito",
                "amount": total,
                "balance": new_balance,
                "order_id": next_order_id
            }
            
            response = requests.post(url, headers=HEADERS, json=payload_conteo)
            response.raise_for_status()
            print(f"Added cash entry for order {next_order_id}: ${total}")
        except Exception as e:
            print(f"Error adding conteo_efectivo entry: {e}")
    else:
        print(f"DEBUG: Skipping conteo_efectivo (payment method is {payment_method})")
            # Don't fail the whole transaction if conteo fails
    # Generate redemption token and PDF
    redemption_token = generate_redemption_token()
    await store_redemption_token(next_order_id, redemption_token, total)
    # Also store in qr_rewards for the WhatsApp loyalty flow
    await store_qr_reward(next_order_id, redemption_token, total)

    # Use your existing QR PDF function
    pdf_buf = _build_receipt_pdf_with_qr(items_for_ticket, total, next_order_id, redemption_token)
    
    filename = f"ticket_{next_order_id}_{int(datetime.now().timestamp()*1000)}.pdf"
    
    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



async def process_loyalty_deduction(product, order_id: int, fecha: str, hora: str):
    """Process loyalty points deduction and record in ventas_terex1"""
    try:
        barcode = product.codigo
        deduction_amount = abs(product.price)  # Make sure it's positive
        customer_email = getattr(product, 'customer_email', 'unknown@email.com')
        
        # Get user's active rewards using supabase_request
        active_rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "id,reward_amount",
                "email": f"eq.{customer_email}",
                "status": "eq.active",
                "order": "created_at.asc"  # FIFO order
            }
        )
        
        if not active_rewards:
            raise HTTPException(
                status_code=400, 
                detail="No hay recompensas disponibles para este cliente"
            )
        
        total_available = sum(float(r["reward_amount"]) for r in active_rewards)
        
        if deduction_amount > total_available:
            deduction_amount = total_available
        
        # Process rewards deduction (FIFO) using supabase_request
        remaining_to_deduct = deduction_amount
        deducted_reward_ids = []
        
        for reward in active_rewards:
            if remaining_to_deduct <= 0:
                break
                
            reward_amount = float(reward["reward_amount"])
            reward_id = reward["id"]
            
            if reward_amount <= remaining_to_deduct:
                # Fully deduct this reward
                await supabase_request(
                    method="PATCH",
                    endpoint=f"/rest/v1/loyalty_rewards?id=eq.{reward_id}",
                    json_data={
                        "status": "redeemed",
                        "redeemed_at": datetime.utcnow().isoformat()
                    }
                )
                
                remaining_to_deduct -= reward_amount
                deducted_reward_ids.append(reward_id)
            else:
                # Partially deduct this reward
                await supabase_request(
                    method="PATCH",
                    endpoint=f"/rest/v1/loyalty_rewards?id=eq.{reward_id}",
                    json_data={
                        "reward_amount": reward_amount - remaining_to_deduct,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                )
                
                remaining_to_deduct = 0
        
        # Record the loyalty deduction in ventas_terex1 using supabase_request
        loyalty_record = {
            "qty": product.qty,
            "name": f"DESCUENTO LEALTAD - {customer_email}",
            "name_id": barcode,
            "price": -deduction_amount,  # Negative price for discount
            "fecha": fecha,
            "hora": hora,
            "order_id": order_id,
            "modelo": "LOYALTY_DEDUCTION",
            "modelo_id": 0,
            "estilo": "DISCOUNT",
            "estilo_id": 0,
            "cliente": customer_email,
            "subtotal": -deduction_amount * product.qty,
        }
        
        await supabase_request(
            method="POST",
            endpoint="/rest/v1/ventas_terex1",
            json_data=loyalty_record,
        )
        
        # Log in barcode_redemptions table using supabase_request
        redemption_log = {
            "user_email": customer_email,
            "barcode": barcode,
            "redeemed_amount": deduction_amount,
            "purchase_total": 0,  # This is a discount, not a purchase
            "order_id": order_id,
            "redeemed_at": datetime.utcnow().isoformat()
        }
        
        await supabase_request(
            method="POST",
            endpoint="/rest/v1/barcode_redemptions",
            json_data=redemption_log
        )
        
        return {
            "customer_email": customer_email,
            "deducted_amount": deduction_amount,
            "remaining_balance": total_available - deduction_amount,
            "barcode": barcode
        }
        
    except Exception as e:
        # Add detailed error logging
        print(f"Error in process_loyalty_deduction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando descuento de lealtad: {str(e)}"
        )


def _build_receipt_pdf_with_loyalty(items: list[dict], total: float, order_id: int, 
                                   redemption_token: str, loyalty_deductions: list) -> io.BytesIO:
    """Enhanced PDF generation that includes loyalty information and QR code"""
    width = 58 * mm
    height = 200 * mm
    margin = 2 * mm
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    def header():
        y = height - margin
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(width/2, y, "TICKET DE VENTA")
        y -= 12
        
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(width/2, y, f"Orden #{order_id}")
        y -= 10
        
        now = datetime.now()
        fecha = f"{now.year:04d}-{now.month:02d}-{now.day:02d}"
        hora = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
        c.setFont("Helvetica", 8)
        c.drawString(margin, y, f"Fecha: {fecha}")
        y -= 10
        c.drawString(margin, y, f"Hora: {hora}")
        y -= 6
        
        # Add loyalty information if present
        if loyalty_deductions:
            c.setFont("Helvetica-Bold", 8)
            c.drawString(margin, y, "DESCUENTOS APLICADOS:")
            y -= 10
            c.setFont("Helvetica", 7)
            for deduction in loyalty_deductions:
                c.drawString(margin, y, f"Cliente: {deduction['customer_email'][:25]}")
                y -= 8
                c.drawString(margin, y, f"Descuento: ${deduction['deducted_amount']:.2f}")
                y -= 10
        
        c.setStrokeColor(colors.black)
        c.line(margin, y, width - margin, y)
        y -= 10
        
        c.setFont("Helvetica-Bold", 8)
        c.drawString(margin, y, "Cant")
        c.drawString(margin + 20, y, "Descripción")
        c.drawRightString(width - margin, y, "Precio")
        y -= 10
        c.setFont("Helvetica", 8)
        return y

    y = header()
    
    for item in items:
        if y < 25 * mm:
            c.showPage()
            y = header()
            
        qty = str(item["qty"])
        name = str(item["name"])
        price = item["price"]
        subtotal = item["subtotal"]
        
        c.drawString(margin, y, qty)
        c.drawString(margin + 20, y, (name[:28] + ("…" if len(name) > 28 else "")))
        c.drawRightString(width - margin, y, f"${price:0.2f}")
        y -= 10
        c.drawRightString(width - margin, y, f"Subtotal: ${subtotal:0.2f}")
        y -= 12

    y -= 4
    c.line(margin, y, width - margin, y)
    y -= 12
    
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "TOTAL:")
    c.drawRightString(width - margin, y, f"${total:0.2f}")
    y -= 14
    
    c.setFont("Helvetica", 8)
    c.drawCentredString(width/2, y, "¡Gracias por su compra!")
    y -= 10
    
    # Add order barcode
    try:
        from reportlab.graphics.barcode import code128
        bc = code128.Code128(f"ORDER-{order_id}", barHeight=12 * mm, barWidth=0.35)
        x = (width - bc.width) / 2
        bc.drawOn(c, x, max(margin, y - 16 * mm))
        y -= 20
    except Exception as e:
        print(f"Error generating barcode: {e}")
        y -= 5
    
    # Add QR Code for redemption
    if redemption_token:
        try:
            import qrcode
            from io import BytesIO as QRBytesIO
            from PIL import Image
            
            # Create QR code
            qr_data = f"https://teresalocal352.com/redeem?token={redemption_token}"
            qr = qrcode.QRCode(version=1, box_size=2, border=1)
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to bytes
            qr_buffer = QRBytesIO()
            qr_img.save(qr_buffer, format='PNG')
            qr_buffer.seek(0)
            
            # Add QR code to PDF
            qr_size = 20 * mm
            qr_x = (width - qr_size) / 2
            c.drawImage(qr_buffer, qr_x, max(margin, y - qr_size - 5 * mm), 
                       width=qr_size, height=qr_size)
            
            y -= qr_size + 8 * mm
            c.setFont("Helvetica", 7)
            c.drawCentredString(width/2, y, "Escanea para canjear recompensas")
            
        except Exception as e:
            print(f"Error generating QR code: {e}")
            # Fallback: just show the redemption URL as text
            c.setFont("Helvetica", 6)
            c.drawCentredString(width/2, y, f"Token: {redemption_token[:20]}")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf





async def get_next_order_id():
    """
    Simple fix: Filter out NULL order_id values
    """
    try:
        # Filter out NULL values and get the highest order_id
        order_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_terex1",
            params={
                "select": "order_id",
                "order_id": "not.is.null",  # This filters out NULL values
                "order": "order_id.desc",
                "limit": "1"
            }
        )
        
        print(f"DEBUG: order_response (filtered): {order_response}")
        
        if (order_response and 
            len(order_response) > 0 and 
            order_response[0].get('order_id') is not None):
            
            current_order_id = order_response[0]['order_id']
            next_order_id = int(current_order_id) + 1
            print(f"DEBUG: Found max order_id: {current_order_id}, next: {next_order_id}")
            return next_order_id
        else:
            print("DEBUG: No valid order_ids found, starting with 1")
            return 1
            
    except Exception as e:
        print(f"DEBUG: Exception: {e}")
        return 1
# Then in your main function, replace the order_id logic with:
# next_order_id = await get_next_order_id()

@app.get("/debug/orders")
async def debug_orders():
    try:
        # Get all order_ids to see what's there
        all_orders = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_terex1",
            params={
                "select": "order_id,fecha,hora",
                "order": "order_id.desc",
                "limit": "10"
            }
        )
        
        # Get max order_id using a different approach
        max_order = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_terex1",
            params={
                "select": "order_id",
                "order": "order_id.desc",
                "limit": "1"
            }
        )
        
        # Get count of records
        count_result = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_terex1",
            params={
                "select": "count",
                "head": "true"
            }
        )
        
        return {
            "all_orders": all_orders,
            "max_order": max_order,
            "count": count_result,
            "table_exists": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "table_exists": False
        }


def generate_redemption_token():
    """Generate a secure token for the QR code"""
    return secrets.token_urlsafe(32)

async def store_redemption_token(order_id: int, token: str, total: float):
    """Store redemption token in database for later redemption"""
    try:
        result = await supabase_request(
            method="POST",
            endpoint="/rest/v1/order_redemptions",
            json_data={
                "order_id": order_id,
                "redemption_token": token,
                "email": "",  # Will be filled when user redeems
                "user_id": None,  # Will be filled when user redeems
                "purchase_total": total  # Now using the column you added
            }
        )
        print(f"DEBUG: Successfully stored redemption token for order {order_id}")
        print(f"DEBUG: Stored token: {token[:20]}...")
        print(f"DEBUG: Result: {result}")
    except Exception as e:
        print(f"ERROR storing redemption token: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")




def _build_receipt_pdf_with_qr(items: list[dict], total: float, order_id: int, redemption_token: str) -> io.BytesIO:
    """PDF receipt for thermal printer:
       - 80mm paper (enough room for full product names)
       - dynamic height so the whole ticket fits on ONE page (no duplicates)
       - unique QR code with WhatsApp deep link for 1% loyalty reward
    """
    width = 80 * mm
    margin = 3 * mm

    total_pieces = sum(int(it.get("qty", 0)) for it in items)

    def get_discount(qty):
        if qty > 100:
            return qty * 10
        if qty > 50:
            return qty * 5
        return 0

    discount = get_discount(total_pieces)
    subtotal_before = total + discount

    # Reward is 1% of the final total (rounded to 2 decimals)
    reward_amount = round(total * 0.01, 2)

    # --- Dynamic height calculation -----------------------------------------
    # Header block: ~28mm
    # Each item: 2 lines of ~4.2mm = ~9mm (name line + "@ $x c/u  Subtotal" line)
    # Totals block: ~18mm
    # Legend + QR + footer: ~62mm
    header_h   = 28 * mm
    item_h     = 9 * mm
    totals_h   = 18 * mm + (5 * mm if discount > 0 else 0)
    qr_block_h = 62 * mm
    height = header_h + (len(items) * item_h) + totals_h + qr_block_h + 2 * margin

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    y = height - margin

    # ---------- HEADER -------------------------------------------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "TEREX2")
    y -= 14

    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(width / 2, y, f"Ticket #{order_id}")
    y -= 13

    _, fecha, hora = _now_strs()
    c.setFont("Helvetica", 9)
    c.drawCentredString(width / 2, y, f"{fecha}  {hora}")
    y -= 10

    c.setLineWidth(0.5)
    c.line(margin, y, width - margin, y)
    y -= 10

    # Column headers
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "Producto")
    c.drawRightString(width - margin, y, "Subtotal")
    y -= 10
    c.line(margin, y + 4, width - margin, y + 4)

    # ---------- ITEMS --------------------------------------------------------
    c.setFont("Helvetica", 9)
    name_max_chars = 32  # fits comfortably at 9pt in ~50mm column

    for it in items:
        qty = int(it.get("qty", 0))
        name = str(it.get("name", ""))
        price = float(it.get("price", 0) or 0)
        sub = float(it.get("subtotal", qty * price) or 0)

        # Truncate long names
        display_name = name if len(name) <= name_max_chars else name[:name_max_chars - 1] + "…"

        # Line 1: "1x  NAME"
        c.setFont("Helvetica", 9)
        c.drawString(margin, y, f"{qty}x  {display_name}")
        y -= 10

        # Line 2: "    @ $75.00 c/u        $75.00"  (indented)
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawString(margin + 10, y, f"@ ${price:0.2f} c/u")
        c.setFillColor(colors.black)
        c.drawRightString(width - margin, y, f"${sub:0.2f}")
        y -= 12

    # ---------- TOTALS -------------------------------------------------------
    c.setLineWidth(0.5)
    c.line(margin, y + 2, width - margin, y + 2)
    y -= 6

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Total piezas:")
    c.drawRightString(width - margin, y, f"{total_pieces}")
    y -= 11

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Subtotal:")
    c.drawRightString(width - margin, y, f"${subtotal_before:0.2f}")
    y -= 11

    if discount > 0:
        c.setFillColor(colors.green)
        c.drawString(margin, y, "Descuento:")
        c.drawRightString(width - margin, y, f"-${discount:0.2f}")
        c.setFillColor(colors.black)
        y -= 11

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "TOTAL:")
    c.drawRightString(width - margin, y, f"${total:0.2f}")
    y -= 16

    # ---------- LOYALTY QR SECTION ------------------------------------------
    c.setStrokeColor(colors.black)
    c.setDash(1, 2)
    c.line(margin, y, width - margin, y)
    c.setDash()
    y -= 12

    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(width / 2, y, "ESCANEA ESTE QR CODE Y OBTEN")
    y -= 10
    c.drawCentredString(width / 2, y, "1% PARA TU SIGUIENTE COMPRA")
    y -= 10

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawCentredString(width / 2, y, f"Credito a obtener: ${reward_amount:0.2f}")
    c.setFillColor(colors.black)
    y -= 12

    # Build QR: WhatsApp deep link with prefilled redemption message.
    # Customer scans -> WhatsApp opens -> sends CANJEAR:<token> to the business number.
    business_phone = os.environ.get("WHATSAPP_BUSINESS_NUMBER", "525642460019")
    prefilled = urllib.parse.quote(f"CANJEAR:{redemption_token}")
    qr_url = f"https://wa.me/{business_phone}?text={prefilled}"

    try:
        qr_size = 40 * mm
        qr_widget = QrCodeWidget(qr_url)
        qr_widget.barWidth = qr_size
        qr_widget.barHeight = qr_size
        qr_drawing = Drawing(qr_size, qr_size)
        qr_drawing.add(qr_widget)
        x_qr = (width - qr_size) / 2
        y_qr = y - qr_size
        renderPDF.draw(qr_drawing, c, x_qr, y_qr)
        y = y_qr - 8
    except Exception as e:
        print(f"QR error: {e}", flush=True)
        c.setFont("Helvetica", 7)
        c.drawCentredString(width / 2, y - 10, f"Token: {redemption_token[:20]}...")
        y -= 20

    c.setFont("Helvetica", 7)
    c.setFillColor(colors.grey)
    c.drawCentredString(width / 2, y, "¡Gracias por su compra!")
    c.setFillColor(colors.black)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


async def store_qr_reward(order_id: int, token: str, purchase_amount: float) -> None:
    """Insert a row into qr_rewards so the token can be redeemed later via WhatsApp."""
    reward_amount = round(purchase_amount * 0.01, 2)
    try:
        await supabase_request(
            method="POST",
            endpoint="/rest/v1/qr_rewards",
            json_data={
                "qr_token": token,
                "order_id": order_id,
                "purchase_amount": purchase_amount,
                "reward_amount": reward_amount,
                "status": "pending",
            },
        )
        print(f"QR reward stored: order={order_id} reward=${reward_amount}", flush=True)
    except Exception as e:
        print(f"ERROR storing qr_reward: {e}", flush=True)


@app.api_route("/api/ticket-pdf/{token}", methods=["GET", "HEAD"])
async def get_ticket_pdf_by_token(token: str):
    """Public endpoint: returns the PDF of the ticket associated with a QR token.
    Used by the WhatsApp bot to send the ticket as a document to the customer.
    """
    try:
        # Look up qr_rewards by token
        rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/qr_rewards",
            params={
                "select": "order_id,qr_token,purchase_amount",
                "qr_token": f"eq.{token}",
                "limit": "1",
            },
        )
        if not rewards:
            raise HTTPException(status_code=404, detail="QR not found")

        order_id = int(rewards[0]["order_id"])
        total = float(rewards[0]["purchase_amount"] or 0)

        # Fetch items for this order
        items_rows = await supabase_request(
            method="GET",
            endpoint="/rest/v1/ventas_terex1",
            params={
                "select": "qty,name,price",
                "order_id": f"eq.{order_id}",
            },
        )

        items = []
        for r in items_rows:
            qty = int(r.get("qty", 1) or 1)
            price = float(r.get("price", 0) or 0)
            items.append({
                "qty": qty,
                "name": str(r.get("name", "")),
                "price": price,
                "subtotal": qty * price,
            })

        pdf_buf = _build_receipt_pdf_with_qr(items, total, order_id, token)

        return StreamingResponse(
            pdf_buf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="ticket_{order_id}.pdf"',
                "Cache-Control": "public, max-age=3600",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in /api/ticket-pdf/{token}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

async def get_order_total(order_id: int):
    """Get total amount for an order from ventas_terex1"""
    order_items = await supabase_request(
        method="GET",
        endpoint="/rest/v1/ventas_terex1",
        params={
            "select": "qty,price",
            "order_id": f"eq.{order_id}"
        }
    )
    
    if not order_items:
        raise Exception("Orden no encontrada")
    
    total = sum(item['qty'] * item['price'] for item in order_items)
    return float(total)

async def create_simple_loyalty_reward(order_id: int, email: str, 
                                     purchase_amount: float, reward_amount: float):
    """Create simple loyalty reward record"""
    await supabase_request(
        method="POST",
        endpoint="/rest/v1/loyalty_rewards",
        json_data={
            "order_id": order_id,
            "email": email,
            "purchase_amount": purchase_amount,
            "reward_amount": reward_amount,
            "status": "active",
            "user_id": None  # No user authentication for now
        }
    )

async def mark_redemption_completed(redemption_token: str, email: str):
    """Mark redemption as completed"""
    await supabase_request(
        method="PATCH",
        endpoint=f"/rest/v1/order_redemptions?redemption_token=eq.{redemption_token}",
        json_data={
            "email": email
        }
    )

# Endpoint to check user's total rewards (for testing)
@app.get("/api/user/rewards/{email}")
async def get_user_total_rewards(email: str):
    """Get total available rewards for a user"""
    try:
        rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "reward_amount,status,created_at,order_id",
                "email": f"eq.{email}",
                "order": "created_at.desc"
            }
        )
        
        active_rewards = [r for r in rewards if r['status'] == 'active']
        total_available = sum(r['reward_amount'] for r in active_rewards)
        
        return {
            "email": email,
            "total_available_rewards": total_available,
            "active_rewards_count": len(active_rewards),
            "all_rewards": rewards
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SimpleRedeemRequest(BaseModel):
    email: str
    redemption_token: str



# Add this to serve the redeem.html page
@app.get("/redeem.html", response_class=HTMLResponse)
async def redeem_page(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="redeem.html",
            context={}
        )


@app.post("/api/redeem")
async def simple_redeem_reward(payload: SimpleRedeemRequest):
    """Simple redemption without authentication"""
    try:
        # Get redemption data
        redemption = supabase.table("order_redemptions").select("order_id,email,purchase_total").eq("redemption_token", payload.redemption_token).limit(1).execute()
        
        if not redemption.data:
            raise HTTPException(status_code=400, detail="Token de redención no válido o expirado")
        
        redemption_data = redemption.data[0]
        
        # Check if already redeemed
        if redemption_data.get('email') and redemption_data.get('email').strip():
            raise HTTPException(status_code=400, detail="Esta recompensa ya ha sido canjeada")
        
        # Get order total
        order_items = supabase.table("ventas_terex1").select("qty,price").eq("order_id", redemption_data['order_id']).execute()
        
        if not order_items.data:
            raise HTTPException(status_code=400, detail="Orden no encontrada")
        
        order_total = sum(item['qty'] * item['price'] for item in order_items.data)
        reward_amount = order_total * 0.01
        
        # Create loyalty reward
        supabase.table("loyalty_rewards").insert({
            "order_id": redemption_data['order_id'],
            "email": payload.email,
            "purchase_amount": order_total,
            "reward_amount": reward_amount,
            "status": "active",
            "user_id": None
        }).execute()
        
        # Mark redemption as completed
        supabase.table("order_redemptions").update({
            "email": payload.email
        }).eq("redemption_token", payload.redemption_token).execute()
        
        # NEW: Create barcode for this email if it doesn't exist
        try:
            await ensure_barcode_for_email(payload.email)
            print(f"DEBUG: Barcode created/ensured for email: {payload.email}")
        except Exception as barcode_error:
            print(f"DEBUG: Failed to create barcode for {payload.email}: {barcode_error}")
            # Don't fail the redemption if barcode creation fails
        
        return {
            "success": True,
            "order_id": redemption_data['order_id'],
            "total": order_total,
            "reward_amount": reward_amount,
            "email": payload.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

async def get_simple_redemption_data(redemption_token: str):
    """Get order data from redemption token"""
    print(f"DEBUG: Looking for redemption token: {redemption_token}")
    
    # First, let's see what tokens exist in the database
    all_tokens = await supabase_request(
        method="GET",
        endpoint="/rest/v1/order_redemptions",
        params={
            "select": "redemption_token,order_id,email",
            "limit": "5"
        }
    )
    print(f"DEBUG: All tokens in database: {all_tokens}")
    
    redemption = await supabase_request(
        method="GET",
        endpoint="/rest/v1/order_redemptions",
        params={
            "select": "order_id,email,purchase_total",
            "redemption_token": f"eq.{redemption_token}",
            "limit": "1"
        }
    )
    
    print(f"DEBUG: Redemption query result: {redemption}")
    
    if not redemption:
        raise Exception("Token de redención no válido o expirado")
    
    redemption_data = redemption[0]
    
    # Check if already redeemed (has email filled)
    if redemption_data.get('email') and redemption_data.get('email').strip():
        raise Exception("Esta recompensa ya ha sido canjeada")
    
    return redemption_data

async def get_order_total(order_id: int):
    """Get total amount for an order from ventas_terex1"""
    order_items = await supabase_request(
        method="GET",
        endpoint="/rest/v1/ventas_terex1",
        params={
            "select": "qty,price",
            "order_id": f"eq.{order_id}"
        }
    )
    
    if not order_items:
        raise Exception("Orden no encontrada")
    
    total = sum(item['qty'] * item['price'] for item in order_items)
    return float(total)

async def create_simple_loyalty_reward(order_id: int, email: str, 
                                     purchase_amount: float, reward_amount: float):
    """Create simple loyalty reward record"""
    await supabase_request(
        method="POST",
        endpoint="/rest/v1/loyalty_rewards",
        json_data={
            "order_id": order_id,
            "email": email,
            "purchase_amount": purchase_amount,
            "reward_amount": reward_amount,
            "status": "active",
            "user_id": None  # No user authentication for now
        }
    )

async def mark_redemption_completed(redemption_token: str, email: str):
    """Mark redemption as completed"""
    await supabase_request(
        method="PATCH",
        endpoint=f"/rest/v1/order_redemptions?redemption_token=eq.{redemption_token}",
        json_data={
            "email": email
        }
    )

@app.get("/api/user/rewards/{email}")
async def get_user_total_rewards(email: str):
    """Get total available rewards for a user"""
    try:
        rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "reward_amount,status,created_at,order_id",
                "email": f"eq.{email}",
                "order": "created_at.desc"
            }
        )
        
        active_rewards = [r for r in rewards if r['status'] == 'active']
        total_available = sum(r['reward_amount'] for r in active_rewards)
        
        return {
            "email": email,
            "total_available_rewards": total_available,
            "active_rewards_count": len(active_rewards),
            "all_rewards": rewards
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



def generate_session_token():
    """Generate a session token for authenticated users"""
    return secrets.token_urlsafe(32)

@app.post("/api/auth/google")
async def google_auth(payload: GoogleAuthRequest):
    """Authenticate user with Google and create session"""
    try:
        print(f"DEBUG: Google auth attempt")
        
        # Verify Google token
        user_info = await verify_google_token(payload.google_token)
        print(f"DEBUG: Google user verified: {user_info['email']}")
        
        # Create or get user in your system
        user = await create_or_get_user(user_info)
        print(f"DEBUG: User created/retrieved: {user['id']}")
        
        # Generate session token
        session_token = generate_session_token()
        user_sessions[session_token] = {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        print(f"DEBUG: Session created: {session_token[:16]}...")
        print(f"DEBUG: Total active sessions: {len(user_sessions)}")
        
        return {
            "success": True,
            "session_token": session_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"]
            }
        }
        
    except Exception as e:
        print(f"Auth error: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


async def verify_google_token(token: str):
    """Verify Google JWT token and extract user info"""
    try:
        # Use Google's tokeninfo endpoint (simpler for development)
        verify_response = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
        )
        
        if verify_response.status_code != 200:
            raise Exception("Invalid Google token")
            
        user_data = verify_response.json()
        
        # Verify the token is for your app (add your Google Client ID here)
        # if user_data.get("aud") != "YOUR_GOOGLE_CLIENT_ID":
        #     raise Exception("Token not for this application")
        
        return {
            "google_id": user_data["sub"],
            "email": user_data["email"],
            "name": user_data.get("name", ""),
            "picture": user_data.get("picture", "")
        }
        
    except Exception as e:
        raise Exception(f"Google token verification failed: {e}")

async def create_or_get_user(user_info: dict):
    """Create or get user in your local users table"""
    try:
        # Check if user already exists
        existing_user = await supabase_request(
            method="GET",
            endpoint="/rest/v1/users",
            params={
                "select": "*",
                "email": f"eq.{user_info['email']}",
                "limit": "1"
            }
        )
        
        if existing_user:
            return existing_user[0]
        
        # Create new user
        new_user_data = {
            "email": user_info["email"],
            "name": user_info["name"],
            "google_id": user_info["google_id"],
            "picture": user_info.get("picture", ""),
            "created_at": datetime.now().isoformat()
        }
        
        created_user = await supabase_request(
            method="POST",
            endpoint="/rest/v1/users",
            json_data=new_user_data
        )
        
        return created_user[0] if isinstance(created_user, list) else created_user
        
    except Exception as e:
        raise Exception(f"Failed to create/get user: {e}")

def get_current_user(session_token: str):
    """Get current user from session token"""
    if not session_token or session_token not in user_sessions:
        return None
    
    session = user_sessions[session_token]
    
    # Check if session expired
    if datetime.now() > session["expires_at"]:
        del user_sessions[session_token]
        return None
    
    return session


@app.post("/api/redeem/authenticated")
async def authenticated_redeem_reward(payload: AuthenticatedRedeemRequest):
    """Redeem reward with Google authentication"""
    try:
        # Verify Google token and get user info
        user_info = await verify_google_token(payload.google_token)
        
        # Create or get user
        user = await create_or_get_user(user_info)
        
        # Create session token
        session_token = generate_session_token()
        user_sessions[session_token] = {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }
        
        # Get redemption data (same logic as simple redeem)
        redemption = supabase.table("order_redemptions").select("order_id,email,purchase_total").eq("redemption_token", payload.redemption_token).limit(1).execute()
        
        if not redemption.data:
            raise HTTPException(status_code=400, detail="Token de redención no válido o expirado")
        
        redemption_data = redemption.data[0]
        
        if redemption_data.get('email') and redemption_data.get('email').strip():
            raise HTTPException(status_code=400, detail="Esta recompensa ya ha sido canjeada")
        
        # Get order total
        order_items = supabase.table("ventas_terex1").select("qty,price").eq("order_id", redemption_data['order_id']).execute()
        order_total = sum(item['qty'] * item['price'] for item in order_items.data)
        reward_amount = order_total * 0.01
        
        # Create loyalty reward with user_id
        supabase.table("loyalty_rewards").insert({
            "user_id": user["id"],
            "order_id": redemption_data['order_id'],
            "email": user["email"],
            "purchase_amount": order_total,
            "reward_amount": reward_amount,
            "status": "active"
        }).execute()
        
        # Mark redemption as completed
        supabase.table("order_redemptions").update({
            "email": user["email"]
        }).eq("redemption_token", payload.redemption_token).execute()
        
        # NEW: Ensure user has a barcode (create if missing)
        try:
            await ensure_user_barcode(user["email"])
            print(f"DEBUG: Barcode ensured for authenticated user: {user['email']}")
        except Exception as barcode_error:
            print(f"DEBUG: Barcode creation failed for {user['email']}: {barcode_error}")
            # Continue anyway - barcode creation failure shouldn't block redemption
        
        return {
            "success": True,
            "order_id": redemption_data['order_id'],
            "total": order_total,
            "reward_amount": reward_amount,
            "session_token": session_token,
            "user": {
                "name": user["name"],
                "email": user["email"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def ensure_barcode_for_email(email: str):
    """Ensure an email has a barcode, create if missing (for simple redemptions without user accounts)"""
    try:
        # Check if email already has a barcode (any barcode, regardless of user_id)
        existing_barcode = supabase.table("user_barcodes").select("barcode").eq("user_email", email).eq("status", "active").limit(1).execute()
        
        if existing_barcode.data:
            print(f"DEBUG: Barcode already exists for email {email}")
            return existing_barcode.data[0]["barcode"]
        
        # Generate new barcode for this email
        new_barcode = generate_user_barcode(email)
        
        # Create barcode entry without user_id (since this is for simple redemption)
        barcode_data = {
            "user_id": None,  # No user account for simple redemptions
            "user_email": email,
            "barcode": new_barcode,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("user_barcodes").insert(barcode_data).execute()
        print(f"DEBUG: Created barcode for email {email}: {new_barcode}")
        return new_barcode
        
    except Exception as e:
        print(f"Error ensuring barcode for email {email}: {e}")
        raise e


async def create_authenticated_loyalty_reward(user_id: str, order_id: int, email: str, 
                                           purchase_amount: float, reward_amount: float):
    """Create loyalty reward for authenticated user"""
    await supabase_request(
        method="POST",
        endpoint="/rest/v1/loyalty_rewards",
        json_data={
            "user_id": user_id,
            "order_id": order_id,
            "email": email,
            "purchase_amount": purchase_amount,
            "reward_amount": reward_amount,
            "status": "active"
        }
    )

# Dashboard endpoints
@app.get("/api/user/dashboard")
async def get_user_dashboard(session_token: str):
    """Get user dashboard data"""
    user = get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    try:
        # Get user's rewards
        rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "*",
                "user_id": f"eq.{user['user_id']}",
                "order": "created_at.desc"
            }
        )
        
        # Calculate totals
        active_rewards = [r for r in rewards if r['status'] == 'active']
        total_earned = sum(r['reward_amount'] for r in rewards)
        total_available = sum(r['reward_amount'] for r in active_rewards)
        total_redeemed = sum(r['reward_amount'] for r in rewards if r['status'] == 'redeemed')
        
        return {
            "user": {
                "name": user["name"],
                "email": user["email"]
            },
            "summary": {
                "total_earned": total_earned,
                "total_available": total_available,
                "total_redeemed": total_redeemed,
                "rewards_count": len(rewards)
            },
            "recent_rewards": rewards[:10]  # Last 10 rewards
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/user/logout")
async def logout_user(session_token: str):
    """Logout user by invalidating session"""
    if session_token in user_sessions:
        del user_sessions[session_token]
    
    return {"success": True, "message": "Logged out successfully"}

# Serve the dashboard page
@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={}
        )


@app.post("/api/apply/rewards")
async def apply_rewards(session_token: str, order_total: float):
    """Apply available rewards to reduce order total"""
    user = get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    try:
        # Get user's active rewards
        available_rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "id,reward_amount",
                "user_id": f"eq.{user['user_id']}",
                "status": "eq.active"
            }
        )
        
        total_rewards = sum(r['reward_amount'] for r in available_rewards)
        discount = min(total_rewards, order_total)  # Can't discount more than order total
        
        # Mark rewards as redeemed if applying
        if discount > 0:
            applied_amount = 0
            for reward in available_rewards:
                if applied_amount >= discount:
                    break
                    
                reward_to_apply = min(reward['reward_amount'], discount - applied_amount)
                
                await supabase_request(
                    method="PATCH",
                    endpoint=f"/rest/v1/loyalty_rewards?id=eq.{reward['id']}",
                    json_data={
                        "status": "redeemed",
                        "redeemed_at": datetime.now().isoformat()
                    }
                )
                
                applied_amount += reward_to_apply
        
        return {
            "original_total": order_total,
            "discount": discount,
            "new_total": order_total - discount,
            "rewards_applied": len(available_rewards),
            "total_rewards_available": total_rewards
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Get user's reward balance (quick endpoint)
@app.get("/api/user/balance")
async def get_user_balance(session_token: str):
    """Get user's current reward balance"""
    user = get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    try:
        # Get active rewards
        active_rewards = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "reward_amount",
                "user_id": f"eq.{user['user_id']}",
                "status": "eq.active"
            }
        )
        
        total_balance = sum(r['reward_amount'] for r in active_rewards)
        
        return {
            "balance": total_balance,
            "rewards_count": len(active_rewards),
            "user": {
                "name": user["name"],
                "email": user["email"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# Debug endpoint to check user sessions (remove in production)
@app.get("/api/debug/sessions")
async def debug_sessions():
    """Debug endpoint to see active sessions"""
    active_sessions = {}
    current_time = datetime.now()
    
    for token, session in user_sessions.items():
        if current_time < session["expires_at"]:
            active_sessions[token[:16] + "..."] = {
                "email": session["email"],
                "expires_in_minutes": int((session["expires_at"] - current_time).total_seconds() / 60)
            }
    
    return {
        "active_sessions": active_sessions,
        "total_active": len(active_sessions)
    }


@app.get("/api/test/session")
async def test_session(session_token: str = None):
    """Test endpoint to verify session tokens work"""
    if not session_token:
        return {"error": "No session token provided"}
    
    user = get_current_user(session_token)
    if user:
        return {
            "valid": True,
            "user": user,
            "session_token": session_token[:16] + "..."
        }
    else:
        return {"valid": False, "error": "Invalid or expired session"}
    

@app.get("/debug.html", response_class=HTMLResponse)
async def debug_page(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="debug.html",
            context={}
        )


@app.get("/verdetalleestilos.html", response_class=HTMLResponse)
async def ver_detalle_estilos(request: Request):
    """
    Renders a menu of estilos from inventario_estilos
    where prioridad = 1 and saldos != 1 (including NULL saldos).
    """
    try:
        # Query for prioridad=1 AND (saldos IS NULL OR saldos != 1)
        # Select all needed fields for the HTML template
        params = {
            "select": "id,nombre,precio,cost,sold30,saldos,prioridad",
            "prioridad": "eq.1",
            "or": "(saldos.is.null,saldos.neq.1)",  # This handles both NULL and not-equal-to-1
            "order": "nombre.asc"
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params=params,
            )

        print(f"[ver_detalle_estilos] Query response status: {resp.status_code}", flush=True)
        
        if resp.status_code >= 400:
            print(f"[ver_detalle_estilos] Supabase error: {resp.status_code} - {resp.text}", flush=True)
            return templates.TemplateResponse(
            request=request,
            name="verdetalleestilos.html",
            context={
                    "step": "select_estilo",
                    "inventario_data": [],
                    "error_message": f"Error al cargar estilos ({resp.status_code})."
            }
        )

        rows = resp.json() or []
        print(f"[ver_detalle_estilos] Retrieved {len(rows)} rows", flush=True)
        
        # Clean up the data - ensure all fields have default values if missing
        inventario_data = []
        for row in rows:
            if row.get("nombre"):  # Only include if nombre exists
                item = {
                    "id": row.get("id", ""),
                    "nombre": row.get("nombre", "").strip(),
                    "precio": row.get("precio", 0),
                    "cost": row.get("cost", 0), 
                    "sold30": row.get("sold30", 0),
                    "saldos": row.get("saldos", 0),
                    "prioridad": row.get("prioridad", 0)
                }
                inventario_data.append(item)
        
        print(f"[ver_detalle_estilos] Processed {len(inventario_data)} items for display", flush=True)

        return templates.TemplateResponse(
            request=request,
            name="verdetalleestilos.html",
            context={
                "step": "select_estilo",  # This tells the template which section to show
                "inventario_data": inventario_data,  # Changed from 'estilos' to 'inventario_data'
                "error_message": None
            }
        )

    except Exception as e:
        print(f"[ver_detalle_estilos] Unexpected error: {e}", flush=True)
        return templates.TemplateResponse(
            request=request,
            name="verdetalleestilos.html",
            context={
                "step": "select_estilo",
                "inventario_data": [],
                "error_message": "Ocurrió un error al cargar el menú de estilos."
            },
            status_code=500
        )

@app.get("/verdetalleestilos/analytics", response_class=HTMLResponse)
async def ver_detalle_estilos_analytics(
    request: Request,
    estilo: str,
    modelos: str = "",  # Comma-separated list of models to filter
    start_date: str = "",
    end_date: str = "",
    sort_order: str = "DESC"  # ASC or DESC for modelo sorting by sales
):
    """
    Shows detailed analytics for a specific estilo (style) with sales data
    from ventas_terex1 table, grouped by modelo.
    Shows ALL models by default.
    """
    try:
        from datetime import datetime, timedelta
        import json
        
        # Set default date range (last 30 days) if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"[Analytics] Analyzing estilo: {estilo}, dates: {start_date} to {end_date}", flush=True)
        
        # Get ALL sales data for this estilo first (no date filtering initially to get all models)
        async with httpx.AsyncClient(timeout=30) as client:
            # Get all sales for this estilo to find available models
            resp_all = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "modelo,qty,fecha,price,subtotal,total",
                    "estilo": f"eq.{estilo}",
                    "modelo": "not.is.null"
                }
            )
            
            if resp_all.status_code != 200:
                raise Exception(f"Error fetching sales data: {resp_all.text}")
            
            all_sales_data = resp_all.json() or []
            print(f"[Analytics] Retrieved {len(all_sales_data)} total sales records", flush=True)
        
        # Filter by date range in Python (easier than complex Supabase queries)
        from datetime import datetime
        
        filtered_sales_data = []
        for row in all_sales_data:
            fecha_str = row.get("fecha")
            if fecha_str:
                try:
                    fecha = datetime.strptime(fecha_str, '%Y-%m-%d').date()
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                    
                    if start_dt <= fecha <= end_dt:
                        filtered_sales_data.append(row)
                except:
                    # If date parsing fails, include the record
                    filtered_sales_data.append(row)
            else:
                # If no date, include the record
                filtered_sales_data.append(row)
        
        print(f"[Analytics] Filtered to {len(filtered_sales_data)} records in date range", flush=True)
        
        # Get unique models and their sales counts
        modelo_sales = {}
        for row in filtered_sales_data:
            modelo = row.get("modelo", "").strip()
            if modelo:
                qty = int(row.get("qty", 1))
                modelo_sales[modelo] = modelo_sales.get(modelo, 0) + qty
        
        # Sort models by sales count
        available_modelos = []
        for modelo, total_qty in sorted(modelo_sales.items(), key=lambda x: x[1], reverse=(sort_order == "DESC")):
            # Count number of sales transactions (not just quantity)
            sales_count = sum(1 for row in filtered_sales_data if row.get("modelo", "").strip() == modelo)
            available_modelos.append({
                "modelo": modelo,
                "total_sales": sales_count,
                "total_quantity": total_qty
            })
        
        print(f"[Analytics] Found {len(available_modelos)} models", flush=True)
        
        # Parse selected models - DEFAULT TO ALL MODELS 
        selected_modelos = []
        if modelos:
            selected_modelos = [m.strip() for m in modelos.split(",") if m.strip()]
        
        # If no specific models selected, use ALL available models
        if not selected_modelos:
            selected_modelos = [m["modelo"] for m in available_modelos]
        
        print(f"[Analytics] Showing {len(selected_modelos)} models: {selected_modelos}", flush=True)
        
        # Filter sales data by selected models
        final_sales_data = [
            row for row in filtered_sales_data 
            if row.get("modelo", "").strip() in selected_modelos
        ]
        
        # Process the data for analytics
        from collections import defaultdict
        
        # Group by modelo and date
        modelo_daily_data = defaultdict(lambda: defaultdict(lambda: {
            'sales': 0, 'quantity': 0, 'revenue': 0.0
        }))
        
        modelo_totals = defaultdict(lambda: {
            'total_sales': 0, 'total_quantity': 0, 'total_revenue': 0.0
        })
        
        for row in final_sales_data:
            modelo = row.get("modelo", "Unknown")
            fecha_str = row.get("fecha")
            qty = int(row.get("qty", 0))
            price = float(row.get("price", 0))
            subtotal = float(row.get("subtotal", 0)) if row.get("subtotal") else (qty * price)
            
            if fecha_str:
                try:
                    fecha = datetime.strptime(fecha_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                    
                    # Update daily data
                    modelo_daily_data[modelo][fecha]['sales'] += 1
                    modelo_daily_data[modelo][fecha]['quantity'] += qty
                    modelo_daily_data[modelo][fecha]['revenue'] += subtotal
                    
                    # Update totals
                    modelo_totals[modelo]['total_sales'] += 1
                    modelo_totals[modelo]['total_quantity'] += qty  
                    modelo_totals[modelo]['total_revenue'] += subtotal
                except:
                    print(f"[Analytics] Warning: Could not parse date {fecha_str}", flush=True)
        
        # Create summary data for cards
        summary_data = []
        modelo_items = list(modelo_totals.items())
        if sort_order == "DESC":
            modelo_items.sort(key=lambda x: x[1]['total_sales'], reverse=True)
        else:
            modelo_items.sort(key=lambda x: x[1]['total_sales'])
        
        for modelo, totals in modelo_items:
            avg_price = totals['total_revenue'] / totals['total_quantity'] if totals['total_quantity'] > 0 else 0
            summary_data.append({
                'modelo': modelo,
                'total_sales': totals['total_sales'],
                'total_quantity': totals['total_quantity'], 
                'total_revenue': totals['total_revenue'],
                'avg_price': avg_price
            })
        
        # Create daily data table
        daily_data = []
        for modelo in modelo_daily_data:
            for fecha_str, data in sorted(modelo_daily_data[modelo].items()):
                if data['quantity'] > 0:
                    try:
                        fecha = datetime.strptime(fecha_str, '%Y-%m-%d').date()
                        daily_data.append({
                            'fecha': fecha,
                            'modelo': modelo,
                            'daily_sales': data['sales'],
                            'daily_quantity': data['quantity'],
                            'daily_revenue': data['revenue'],
                            'avg_daily_price': data['revenue'] / data['quantity'] if data['quantity'] > 0 else 0
                        })
                    except:
                        print(f"[Analytics] Warning: Could not parse date {fecha_str} for daily data", flush=True)
        
        # Sort daily data
        daily_data.sort(key=lambda x: (x['fecha'], x['modelo']))
        
        # Prepare chart data
        chart_data = {
            'modelo_totals': {item['modelo']: item['total_revenue'] for item in summary_data},
            'daily_trends': {}
        }
        
        # Prepare daily trends for chart
        for modelo in modelo_daily_data:
            chart_data['daily_trends'][modelo] = {}
            for fecha_str, data in modelo_daily_data[modelo].items():
                if data['revenue'] > 0:
                    chart_data['daily_trends'][modelo][fecha_str] = data['revenue']
        
        analytics_data = {
            'summary_data': summary_data,
            'daily_data': daily_data
        }
        
        print(f"[Analytics] Generated analytics for {len(summary_data)} models, {len(daily_data)} daily records", flush=True)
        
        return templates.TemplateResponse(
            request=request,
            name="verdetalleestilos.html",
            context={
                "step": "analytics",
                "estilo": estilo,
                "available_modelos": available_modelos,
                "selected_modelos": selected_modelos,
                "start_date": start_date,
                "end_date": end_date,
                "sort_order": sort_order,
                "analytics_data": analytics_data,
                "chart_data": json.dumps(chart_data),
                "error_message": None
            }
        )
        
    except Exception as e:
        print(f"[Analytics] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            request=request,
            name="verdetalleestilos.html",
            context={
                "step": "analytics", 
                "estilo": estilo,
                "available_modelos": [],
                "selected_modelos": [],
                "start_date": start_date,
                "end_date": end_date,
                "sort_order": sort_order,
                "analytics_data": {"summary_data": [], "daily_data": []},
                "chart_data": "{}",
                "error_message": "Error al cargar el análisis de ventas."
            }
        )

@app.get("/flores3-analytics", response_class=HTMLResponse)
async def flores3_analytics(request: Request):
    """
    Direct analytics for FUN FLORES 3 style - simplified version
    """
    try:
        from datetime import datetime, timedelta
        
        # Fixed estilo for testing
        estilo = "FUN FLORES 3"
        
        # Set date range (last 30 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"[FLORES3] Analyzing: {estilo}, dates: {start_date} to {end_date}", flush=True)
        
        # Get sales data for FUN FLORES 3
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                headers=HEADERS,
                params={
                    "select": "modelo,qty,fecha,price,subtotal,total",
                    "estilo": f"eq.{estilo}",
                    "modelo": "not.is.null",
                    "fecha": f"gte.{start_date}",
                    "fecha": f"lte.{end_date}"
                }
            )
            
            if resp.status_code != 200:
                raise Exception(f"Error fetching sales data: {resp.text}")
            
            sales_data = resp.json() or []
            print(f"[FLORES3] Retrieved {len(sales_data)} sales records", flush=True)
        
        # Process data exactly like your working analytics
        from collections import defaultdict
        
        modelo_daily_data = defaultdict(lambda: defaultdict(lambda: {
            'sales': 0, 'quantity': 0, 'revenue': 0.0
        }))
        
        modelo_totals = defaultdict(lambda: {
            'total_sales': 0, 'total_quantity': 0, 'total_revenue': 0.0
        })
        
        for row in sales_data:
            modelo = row.get("modelo", "Unknown")
            fecha_str = row.get("fecha")
            qty = int(row.get("qty", 0))
            price = float(row.get("price", 0))
            subtotal = float(row.get("subtotal", 0)) if row.get("subtotal") else (qty * price)
            
            if fecha_str:
                try:
                    fecha = datetime.strptime(fecha_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                    
                    # Update daily data
                    modelo_daily_data[modelo][fecha]['sales'] += 1
                    modelo_daily_data[modelo][fecha]['quantity'] += qty
                    modelo_daily_data[modelo][fecha]['revenue'] += subtotal
                    
                    # Update totals
                    modelo_totals[modelo]['total_sales'] += 1
                    modelo_totals[modelo]['total_quantity'] += qty  
                    modelo_totals[modelo]['total_revenue'] += subtotal
                except:
                    print(f"[FLORES3] Warning: Could not parse date {fecha_str}", flush=True)
        
        # Create summary data
        summary_data = []
        for modelo, totals in modelo_totals.items():
            summary_data.append({
                'modelo': modelo,
                'total_sales': totals['total_sales'],
                'total_quantity': totals['total_quantity'], 
                'total_revenue': totals['total_revenue']
            })
        
        # Create daily data
        daily_data = []
        for modelo in modelo_daily_data:
            for fecha_str, data in sorted(modelo_daily_data[modelo].items()):
                if data['quantity'] > 0:
                    try:
                        fecha = datetime.strptime(fecha_str, '%Y-%m-%d').date()
                        daily_data.append({
                            'fecha': fecha,
                            'modelo': modelo,
                            'daily_sales': data['sales'],
                            'daily_quantity': data['quantity'],
                            'daily_revenue': data['revenue'],
                            'avg_daily_price': data['revenue'] / data['quantity'] if data['quantity'] > 0 else 0
                        })
                    except:
                        print(f"[FLORES3] Warning: Could not parse date {fecha_str}", flush=True)
        
        # Sort daily data
        daily_data.sort(key=lambda x: (x['fecha'], x['modelo']))
        
        analytics_data = {
            'summary_data': summary_data,
            'daily_data': daily_data
        }
        
        print(f"[FLORES3] Generated analytics: {len(summary_data)} models, {len(daily_data)} daily records", flush=True)
        
        return templates.TemplateResponse(
            "flores3_analytics.html",  # New template file
            {
                "request": request,
                "analytics_data": analytics_data,
                "estilo": estilo,
                "start_date": start_date,
                "end_date": end_date
            },
        )
        
    except Exception as e:
        print(f"[FLORES3] Error: {e}", flush=True)
        return templates.TemplateResponse(
            request=request,
            name="flores3_analytics.html",
            context={
                "analytics_data": {"summary_data": [], "daily_data": []},
                "estilo": "FUN FLORES 3",
                "error_message": "Error al cargar el análisis de ventas."
            }
        )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_main(request: Request, estilo: str = None):
    """
    Main analytics route - shows style selection or specific analytics
    """
    try:
        if not estilo:
            # Show style selection page
            # Query for prioridad=1 AND (saldos IS NULL OR saldos != 1)
            params = {
                "select": "id,nombre,precio,cost,sold30,saldos,prioridad",
                "prioridad": "eq.1",
                "or": "(saldos.is.null,saldos.neq.1)",
                "order": "nombre.asc"
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                    headers=HEADERS,
                    params=params,
                )

            if resp.status_code >= 400:
                return templates.TemplateResponse(
            request=request,
            name="analytics_dynamic.html",
            context={
                        "step": "select_estilo",
                        "inventario_data": [],
                        "error_message": f"Error al cargar estilos ({resp.status_code})."
            }
        )

            rows = resp.json() or []
            
            # Clean up the data
            inventario_data = []
            for row in rows:
                if row.get("nombre"):
                    item = {
                        "id": row.get("id", ""),
                        "nombre": row.get("nombre", "").strip(),
                        "precio": row.get("precio", 0),
                        "cost": row.get("cost", 0), 
                        "sold30": row.get("sold30", 0),
                        "saldos": row.get("saldos", 0),
                        "prioridad": row.get("prioridad", 0)
                    }
                    inventario_data.append(item)

            return templates.TemplateResponse(
            request=request,
            name="analytics_dynamic.html",
            context={
                    "step": "select_estilo",
                    "inventario_data": inventario_data,
                    "error_message": None
            }
        )
        
        else:
            # Show analytics for specific estilo
            from datetime import datetime, timedelta
            
            # Set date range (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"[Analytics] Analyzing: {estilo}, dates: {start_date} to {end_date}", flush=True)
            
            # Get sales data
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                    headers=HEADERS,
                    params={
                        "select": "modelo,qty,fecha,price,subtotal,total",
                        "estilo": f"eq.{estilo}",
                        "modelo": "not.is.null",
                        "fecha": f"gte.{start_date}",
                        "fecha": f"lte.{end_date}"
                    }
                )
                
                if resp.status_code != 200:
                    raise Exception(f"Error fetching sales data: {resp.text}")
                
                sales_data = resp.json() or []
                print(f"[Analytics] Retrieved {len(sales_data)} sales records", flush=True)
            
            # Process data exactly like the working version
            from collections import defaultdict
            
            modelo_daily_data = defaultdict(lambda: defaultdict(lambda: {
                'sales': 0, 'quantity': 0, 'revenue': 0.0
            }))
            
            modelo_totals = defaultdict(lambda: {
                'total_sales': 0, 'total_quantity': 0, 'total_revenue': 0.0
            })
            
            for row in sales_data:
                modelo = row.get("modelo", "Unknown")
                fecha_str = row.get("fecha")
                qty = int(row.get("qty", 0))
                price = float(row.get("price", 0))
                subtotal = float(row.get("subtotal", 0)) if row.get("subtotal") else (qty * price)
                
                if fecha_str:
                    try:
                        fecha = datetime.strptime(fecha_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                        
                        # Update daily data
                        modelo_daily_data[modelo][fecha]['sales'] += 1
                        modelo_daily_data[modelo][fecha]['quantity'] += qty
                        modelo_daily_data[modelo][fecha]['revenue'] += subtotal
                        
                        # Update totals
                        modelo_totals[modelo]['total_sales'] += 1
                        modelo_totals[modelo]['total_quantity'] += qty  
                        modelo_totals[modelo]['total_revenue'] += subtotal
                    except:
                        print(f"[Analytics] Warning: Could not parse date {fecha_str}", flush=True)
            
            # Create summary data
            summary_data = []
            for modelo, totals in modelo_totals.items():
                summary_data.append({
                    'modelo': modelo,
                    'total_sales': totals['total_sales'],
                    'total_quantity': totals['total_quantity'], 
                    'total_revenue': totals['total_revenue']
                })
            
            # Create daily data
            daily_data = []
            for modelo in modelo_daily_data:
                for fecha_str, data in sorted(modelo_daily_data[modelo].items()):
                    if data['quantity'] > 0:
                        try:
                            fecha = datetime.strptime(fecha_str, '%Y-%m-%d').date()
                            daily_data.append({
                                'fecha': fecha,
                                'modelo': modelo,
                                'daily_sales': data['sales'],
                                'daily_quantity': data['quantity'],
                                'daily_revenue': data['revenue'],
                                'avg_daily_price': data['revenue'] / data['quantity'] if data['quantity'] > 0 else 0
                            })
                        except:
                            print(f"[Analytics] Warning: Could not parse date {fecha_str}", flush=True)
            
            # Sort daily data
            daily_data.sort(key=lambda x: (x['fecha'], x['modelo']))
            
            analytics_data = {
                'summary_data': summary_data,
                'daily_data': daily_data
            }
            
            print(f"[Analytics] Generated analytics: {len(summary_data)} models, {len(daily_data)} daily records", flush=True)
            
            return templates.TemplateResponse(
            request=request,
            name="analytics_dynamic.html",
            context={
                    "step": "analytics",
                    "analytics_data": analytics_data,
                    "estilo": estilo,
                    "start_date": start_date,
                    "end_date": end_date
                }
        )
        
    except Exception as e:
        print(f"[Analytics] Error: {e}", flush=True)
        return templates.TemplateResponse(
            request=request,
            name="analytics_dynamic.html",
            context={
                "step": "analytics" if estilo else "select_estilo",
                "analytics_data": {"summary_data": [], "daily_data": []},
                "estilo": estilo or "",
                "error_message": "Error al cargar el análisis de ventas.",
                "inventario_data": [] if not estilo else None
            }
        )

@app.get("/analyticstravel", response_class=HTMLResponse)
async def analytics_travel(request: Request, lugar: str = None):
    """
    Travel analytics route - shows location selection or specific travel analytics
    """
    try:
        if not lugar:
            # Show location selection page
            print("[Travel Analytics] Loading lugares for selection", flush=True)
            
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_travel2",
                    headers=HEADERS,
                    params={
                        "select": "lugar,qty,subtotal",
                        "lugar": "not.is.null"
                    }
                )

            if resp.status_code >= 400:
                return templates.TemplateResponse(
            request=request,
            name="travel_analytics.html",
            context={
                        "step": "select_lugar",
                        "lugares_data": [],
                        "error_message": f"Error al cargar lugares ({resp.status_code})."
            }
        )

            rows = resp.json() or []
            print(f"[Travel Analytics] Retrieved {len(rows)} travel records", flush=True)
            
            # Aggregate data by lugar
            from collections import defaultdict
            lugar_stats = defaultdict(lambda: {
                'total_ventas': 0,
                'total_subtotal': 0,
                'total_qty': 0
            })
            
            for row in rows:
                lugar_name = row.get("lugar", "").strip()
                if lugar_name:
                    subtotal = float(row.get("subtotal", 0)) if row.get("subtotal") else 0
                    qty = int(row.get("qty", 0)) if row.get("qty") else 0
                    
                    lugar_stats[lugar_name]['total_ventas'] += 1
                    lugar_stats[lugar_name]['total_subtotal'] += subtotal
                    lugar_stats[lugar_name]['total_qty'] += qty
            
            # Create lugares data for display
            lugares_data = []
            for lugar_name, stats in lugar_stats.items():
                promedio = stats['total_subtotal'] / stats['total_ventas'] if stats['total_ventas'] > 0 else 0
                lugares_data.append({
                    'lugar': lugar_name,
                    'total_ventas': stats['total_ventas'],
                    'total_subtotal': stats['total_subtotal'],
                    'total_qty': stats['total_qty'],
                    'promedio_venta': promedio
                })
            
            # Sort by total_subtotal descending
            lugares_data.sort(key=lambda x: x['total_subtotal'], reverse=True)
            
            print(f"[Travel Analytics] Processed {len(lugares_data)} lugares", flush=True)

            return templates.TemplateResponse(
            request=request,
            name="travel_analytics.html",
            context={
                    "step": "select_lugar",
                    "lugares_data": lugares_data,
                    "error_message": None
            }
        )
        
        else:
            # Show analytics for specific lugar
            from datetime import datetime, timedelta
            
            # Set date range (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"[Travel Analytics] Analyzing lugar: {lugar}, dates: {start_date} to {end_date}", flush=True)
            
            # Get travel sales data for this lugar
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_travel2",
                    headers=HEADERS,
                    params={
                        "select": "created_at,order_id,qty,estilo_id,estilo,precio,subtotal,cliente,lugar",
                        "lugar": f"eq.{lugar}",
                        "estilo": "not.is.null",
                        "created_at": f"gte.{start_date}",
                        "created_at": f"lte.{end_date}T23:59:59"
                    }
                )
                
                if resp.status_code != 200:
                    raise Exception(f"Error fetching travel data: {resp.text}")
                
                travel_data = resp.json() or []
                print(f"[Travel Analytics] Retrieved {len(travel_data)} travel records for {lugar}", flush=True)
            
            # Process data for analytics
            from collections import defaultdict
            
            estilo_daily_data = defaultdict(lambda: defaultdict(lambda: {
                'ventas': 0, 'qty': 0, 'subtotal': 0.0
            }))
            
            estilo_totals = defaultdict(lambda: {
                'total_ventas': 0, 'total_qty': 0, 'total_subtotal': 0.0
            })
            
            # Process each travel record
            daily_data = []
            for row in travel_data:
                estilo = row.get("estilo", "Unknown")
                created_at_str = row.get("created_at")
                qty = int(row.get("qty", 0)) if row.get("qty") else 0
                precio = float(row.get("precio", 0)) if row.get("precio") else 0
                subtotal = float(row.get("subtotal", 0)) if row.get("subtotal") else 0
                cliente = row.get("cliente", "")
                
                if created_at_str:
                    try:
                        # Parse the timestamp and extract date
                        fecha = datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).date()
                        fecha_str = fecha.strftime('%Y-%m-%d')
                        
                        # Add to daily data for table display
                        daily_data.append({
                            'fecha': fecha,
                            'estilo': estilo,
                            'cliente': cliente,
                            'qty': qty,
                            'precio': precio,
                            'subtotal': subtotal
                        })
                        
                        # Update aggregated data for chart
                        estilo_daily_data[estilo][fecha_str]['ventas'] += 1
                        estilo_daily_data[estilo][fecha_str]['qty'] += qty
                        estilo_daily_data[estilo][fecha_str]['subtotal'] += subtotal
                        
                        # Update totals
                        estilo_totals[estilo]['total_ventas'] += 1
                        estilo_totals[estilo]['total_qty'] += qty  
                        estilo_totals[estilo]['total_subtotal'] += subtotal
                        
                    except Exception as e:
                        print(f"[Travel Analytics] Warning: Could not parse date {created_at_str}: {e}", flush=True)
            
            # Create summary data
            summary_data = []
            for estilo, totals in estilo_totals.items():
                summary_data.append({
                    'estilo': estilo,
                    'total_ventas': totals['total_ventas'],
                    'total_qty': totals['total_qty'], 
                    'total_subtotal': totals['total_subtotal']
                })
            
            # Sort by total_subtotal descending
            summary_data.sort(key=lambda x: x['total_subtotal'], reverse=True)
            
            # Sort daily data by date
            daily_data.sort(key=lambda x: x['fecha'], reverse=True)
            
            analytics_data = {
                'summary_data': summary_data,
                'daily_data': daily_data
            }
            
            print(f"[Travel Analytics] Generated analytics: {len(summary_data)} estilos, {len(daily_data)} records", flush=True)
            
            return templates.TemplateResponse(
            request=request,
            name="travel_analytics.html",
            context={
                    "step": "analytics",
                    "analytics_data": analytics_data,
                    "lugar": lugar,
                    "start_date": start_date,
                    "end_date": end_date
                }
        )
        
    except Exception as e:
        print(f"[Travel Analytics] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            request=request,
            name="travel_analytics.html",
            context={
                "step": "analytics" if lugar else "select_lugar",
                "analytics_data": {"summary_data": [], "daily_data": []},
                "lugar": lugar or "",
                "error_message": "Error al cargar el análisis de ventas travel.",
                "lugares_data": [] if not lugar else None
            }
        )




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def _ean13_check_digit_for_partial12(partial_12: str) -> int:
    """
    EAN-13 check: sum of odd positions (1x) + even positions (3x); check = (10 - sum%10) % 10
    partial_12: 12-digit string (no check digit).
    """
    s = 0
    for i, d in enumerate(partial_12):            # i = 0..11
        n = int(d)
        s += n if (i + 1) % 2 == 1 else 3 * n     # 1-based odd=1x, even=3x
    return (10 - (s % 10)) % 10

def _compute_check_digit(partial_12: str) -> int:
    """
    Uses your existing weighting rule:
    position 0-based: even*1, odd*2  (kept to stay compatible with your barcodes)
    """
    check_sum = sum(int(d) * (2 if i % 2 == 1 else 1) for i, d in enumerate(partial_12))
    return (10 - (check_sum % 10)) % 10

def _build_barcode_from_eight(eight_digits: str) -> str:
    """Build 13-digit code: 8000 + eight + EAN-13 check digit"""
    partial = f"8000{eight_digits}"                # 12 digits
    return f"{partial}{_ean13_check_digit_for_partial12(partial)}"  # 


@app.post("/api/admin/create-all-barcodes")
async def create_barcodes_for_all_customers():
    """Create barcodes for all customers who have loyalty rewards or order redemptions but no barcode"""
    try:
        # Get all unique emails from loyalty_rewards table
        loyalty_emails = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "email",
            }
        )
        
        # Get all unique emails from order_redemptions table
        redemption_emails = await supabase_request(
            method="GET",
            endpoint="/rest/v1/order_redemptions",
            params={
                "select": "email",
            }
        )
        
        # Combine and get unique emails
        all_emails = set()
        
        for reward in loyalty_emails:
            if reward["email"] and reward["email"].strip():
                all_emails.add(reward["email"].strip().lower())
        
        for redemption in redemption_emails:
            if redemption["email"] and redemption["email"].strip():
                all_emails.add(redemption["email"].strip().lower())
        
        print(f"Found {len(all_emails)} unique customer emails")
        
        created_count = 0
        existing_count = 0
        
        for email in all_emails:
            try:
                # Check if barcode already exists
                existing_barcode = await supabase_request(
                    method="GET",
                    endpoint="/rest/v1/user_barcodes",
                    params={
                        "select": "barcode",
                        "user_email": f"eq.{email}",
                        "limit": "1"
                    }
                )
                
                if existing_barcode:
                    existing_count += 1
                    print(f"Barcode already exists for {email}: {existing_barcode[0]['barcode']}")
                    continue
                
                # Create barcode for this email
                barcode, was_existing = await ensure_user_barcode(email)
                
                if not was_existing:
                    created_count += 1
                    print(f"Created barcode {barcode} for {email}")
                else:
                    existing_count += 1
                
            except Exception as e:
                print(f"Error processing {email}: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Processed {len(all_emails)} customers",
            "created": created_count,
            "already_existed": existing_count,
            "total_emails": len(all_emails)
        }
        
    except Exception as e:
        print(f"Error in create_all_barcodes: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating barcodes: {str(e)}")

# Modify your existing ticket generation to include barcode creation
# Add this to your api_save function in the ticket generation process

async def ensure_barcode_for_future_customer(purchase_total: float, order_id: int):
    """Create barcode for potential future customer when ticket is created"""
    try:
        # Generate a unique email placeholder for this order that can be claimed later
        # This creates the redemption token entry that will later be claimed
        redemption_token = generate_redemption_token()
        
        # Store redemption token with empty email (to be filled when redeemed)
        redemption_data = {
            "order_id": order_id,
            "redemption_token": redemption_token,
            "purchase_total": purchase_total,
            "email": "",  # Empty - will be filled when customer redeems
            "created_at": datetime.utcnow().isoformat()
        }
        
        await supabase_request(
            method="POST",
            endpoint="/rest/v1/order_redemptions",
            json_data=redemption_data
        )
        
        return redemption_token
        
    except Exception as e:
        print(f"Error creating redemption token: {e}")
        return None

# Update your store_redemption_token function to use the new approach
async def store_redemption_token(order_id: int, redemption_token: str, total: float):
    """Store redemption token - simplified since we're not creating barcodes here"""
    try:
        redemption_data = {
            "order_id": order_id,
            "redemption_token": redemption_token,
            "purchase_total": total,
            "email": "",  # Empty until redeemed
            "created_at": datetime.utcnow().isoformat()
        }
        
        await supabase_request(
            method="POST",
            endpoint="/rest/v1/order_redemptions",
            json_data=redemption_data
        )
        
    except Exception as e:
        print(f"Error storing redemption token: {e}")
        raise


@app.get("/api/admin/create-all-barcodes")
async def create_barcodes_for_all_customers():
    """Create barcodes for all customers who have loyalty rewards or order redemptions but no barcode"""
    try:
        # Get all unique emails from loyalty_rewards table
        loyalty_emails = await supabase_request(
            method="GET",
            endpoint="/rest/v1/loyalty_rewards",
            params={
                "select": "email",
            }
        )
        
        # Get all unique emails from order_redemptions table
        redemption_emails = await supabase_request(
            method="GET",
            endpoint="/rest/v1/order_redemptions",
            params={
                "select": "email",
            }
        )
        
        # Combine and get unique emails
        all_emails = set()
        
        for reward in loyalty_emails:
            if reward["email"] and reward["email"].strip():
                all_emails.add(reward["email"].strip().lower())
        
        for redemption in redemption_emails:
            if redemption["email"] and redemption["email"].strip():
                all_emails.add(redemption["email"].strip().lower())
        
        print(f"Found {len(all_emails)} unique customer emails")
        
        created_count = 0
        existing_count = 0
        
        for email in all_emails:
            try:
                # Check if barcode already exists
                existing_barcode = await supabase_request(
                    method="GET",
                    endpoint="/rest/v1/user_barcodes",
                    params={
                        "select": "barcode",
                        "user_email": f"eq.{email}",
                        "limit": "1"
                    }
                )
                
                if existing_barcode:
                    existing_count += 1
                    print(f"Barcode already exists for {email}: {existing_barcode[0]['barcode']}")
                    continue
                
                # Create barcode for this email
                barcode, was_existing = await ensure_user_barcode(email)
                
                if not was_existing:
                    created_count += 1
                    print(f"Created barcode {barcode} for {email}")
                else:
                    existing_count += 1
                
            except Exception as e:
                print(f"Error processing {email}: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Processed {len(all_emails)} customers",
            "created": created_count,
            "already_existed": existing_count,
            "total_emails": len(all_emails)
        }
        
    except Exception as e:
        print(f"Error in create_all_barcodes: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating barcodes: {str(e)}")

@app.get("/modelosanuales.html", response_class=HTMLResponse)
async def modelos_anuales(
    request: Request,
    year: str = "2024"
):
    """
    Shows ALL models using proper pagination to get complete data
    """
    try:
        import json
        from datetime import datetime
        from collections import defaultdict
        
        print(f"[Yearly Models] Processing year: {year} - showing ALL models", flush=True)
        
        async with httpx.AsyncClient(timeout=60) as client:
            # Get all data for the year with proper pagination
            all_records = []
            offset = 0
            batch_size = 1000
            
            while True:
                print(f"[Yearly Models] Fetching batch starting at {offset}...", flush=True)
                
                # Use Range header for pagination
                headers_with_range = {**HEADERS, "Range": f"{offset}-{offset + batch_size - 1}"}
                
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                    headers=headers_with_range,
                    params={
                        "select": "modelo,qty,fecha,price,subtotal",
                        "fecha": f"gte.{year}-01-01",
                        "modelo": "not.is.null",
                        "order": "fecha.asc"
                    }
                )
                
                if resp.status_code != 200:
                    print(f"[Yearly Models] Error: {resp.status_code} - {resp.text}", flush=True)
                    break
                
                batch_data = resp.json() or []
                print(f"[Yearly Models] Got {len(batch_data)} records in this batch", flush=True)
                
                if not batch_data:
                    break
                
                # Filter for the specific year (end date) in Python
                year_filtered = []
                for record in batch_data:
                    fecha_str = record.get("fecha")
                    if fecha_str and fecha_str.startswith(year) and fecha_str <= f"{year}-12-31":
                        year_filtered.append(record)
                
                all_records.extend(year_filtered)
                print(f"[Yearly Models] Added {len(year_filtered)} records for {year}", flush=True)
                
                # If we got less than a full batch, we've reached the end
                if len(batch_data) < batch_size:
                    break
                
                offset += batch_size
                
                # Safety limit to prevent infinite loops
                if len(all_records) > 50000:
                    print(f"[Yearly Models] Safety limit reached, stopping at {len(all_records)} records", flush=True)
                    break
            
            print(f"[Yearly Models] Total records retrieved for {year}: {len(all_records)}", flush=True)
            
            if not all_records:
                return templates.TemplateResponse(
            request=request,
            name="modelosanuales.html",
            context={
                        "year": year,
                        "chart_data": {"labels": [], "datasets": []},
                        "summary_data": [],
                        "monthly_details": [],
                        "available_years": ["2022", "2023", "2024", "2025"],
                        "error_message": f"No se encontraron datos para el año {year}."
            }
        )
            
            # Debug: Check date distribution
            if all_records:
                dates = [r.get('fecha') for r in all_records if r.get('fecha')]
                if dates:
                    months_found = set([d[5:7] for d in dates if len(d) >= 7])
                    min_date = min(dates)
                    max_date = max(dates)
                    print(f"[Yearly Models] Date range: {min_date} to {max_date}")
                    print(f"[Yearly Models] Months found: {sorted(list(months_found))}")
        
        # Debug: Show sample of what we're aggregating
        print(f"[Yearly Models] Sample of raw data being processed:")
        for i, record in enumerate(all_records[:3]):
            fecha = record.get('fecha', 'NO_DATE')
            modelo = record.get('modelo', 'NO_MODEL') 
            qty = record.get('qty', 0)
            print(f"  Record {i+1}: {modelo} on {fecha} qty={qty}")
        
        # Process all the data
        monthly_aggregates = defaultdict(lambda: defaultdict(lambda: {
            'total_qty': 0,
            'total_revenue': 0.0,
            'total_sales': 0
        }))
        
        processed_records = 0
        unique_models = set()
        monthly_debug = defaultdict(set)  # Track which months we actually process
        
        for row in all_records:
            modelo = row.get("modelo", "").strip()
            fecha_str = row.get("fecha")
            
            if not modelo or not fecha_str:
                continue
            
            unique_models.add(modelo)
            
            try:
                fecha_obj = datetime.strptime(fecha_str, '%Y-%m-%d')
                month_key = fecha_obj.strftime('%Y-%m-01')
                month_num = fecha_obj.strftime('%m')  # For debugging
                monthly_debug[modelo].add(month_num)
                
                qty = int(row.get("qty", 0))
                subtotal = row.get("subtotal")
                if subtotal is not None:
                    revenue = float(subtotal)
                else:
                    price = float(row.get("price", 0))
                    revenue = qty * price
                
                monthly_aggregates[modelo][month_key]['total_qty'] += qty
                monthly_aggregates[modelo][month_key]['total_revenue'] += revenue
                monthly_aggregates[modelo][month_key]['total_sales'] += 1
                processed_records += 1
                
            except (ValueError, TypeError) as e:
                continue
        
        print(f"[Yearly Models] Processed: {processed_records} records")
        print(f"[Yearly Models] Unique models found: {len(unique_models)}")
        print(f"[Yearly Models] Model aggregates created: {len(monthly_aggregates)}")
        
        # Debug: Show which months we found for top models
        top_models_debug = sorted(unique_models)[:3]
        for modelo in top_models_debug:
            months_found = sorted(list(monthly_debug.get(modelo, set())))
            print(f"[Yearly Models] {modelo} has data for months: {months_found}")
            
            # Show monthly aggregates for this model
            if modelo in monthly_aggregates:
                for month_key, data in monthly_aggregates[modelo].items():
                    if data['total_qty'] > 0:
                        month_name = datetime.strptime(month_key, '%Y-%m-%d').strftime('%b')
                        print(f"  {month_name}: {data['total_qty']} qty, {data['total_sales']} sales")
            else:
                print(f"  WARNING: {modelo} not in monthly_aggregates!")
        
        if not monthly_aggregates:
            return templates.TemplateResponse(
            request=request,
            name="modelosanuales.html",
            context={
                    "year": year,
                    "chart_data": {"labels": [], "datasets": []},
                    "summary_data": [],
                    "monthly_details": [],
                    "available_years": ["2022", "2023", "2024", "2025"],
                    "error_message": f"No se encontraron ventas para el año {year}."
            }
        )
        
        # Convert to chart format
        modelo_monthly = defaultdict(dict)
        modelo_totals = defaultdict(lambda: {
            'total_revenue': 0, 'total_sales': 0, 'total_quantity': 0
        })
        
        total_month_records = 0
        
        for modelo, months_data in monthly_aggregates.items():
            for month_str, data in months_data.items():
                try:
                    month_obj = datetime.strptime(month_str, '%Y-%m-%d')
                    month_num = month_obj.strftime('%m')
                    
                    modelo_monthly[modelo][month_num] = {
                        'sales': data['total_sales'],
                        'quantity': data['total_qty'],
                        'revenue': data['total_revenue']
                    }
                    
                    modelo_totals[modelo]['total_revenue'] += data['total_revenue']
                    modelo_totals[modelo]['total_sales'] += data['total_sales']
                    modelo_totals[modelo]['total_quantity'] += data['total_qty']
                    
                    total_month_records += 1
                    
                except Exception as e:
                    continue
        
        print(f"[Yearly Models] Created {total_month_records} model-month combinations")
        
        # Sort ALL models by revenue
        all_models_sorted = sorted(modelo_totals.items(), key=lambda x: x[1]['total_revenue'], reverse=True)
        all_models = [modelo for modelo, totals in all_models_sorted if totals['total_revenue'] > 0]
        
        print(f"[Yearly Models] Final: {len(all_models)} models with sales data")
        
        # Chart data
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        month_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        # Generate colors dynamically
        def generate_distinct_colors(count):
            colors = []
            for i in range(count):
                hue = (i * 137.508) % 360  # Golden angle for good distribution
                saturation = 65 + (i % 4) * 10
                lightness = 45 + (i % 3) * 15
                colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")
            return colors
        
        colors = generate_distinct_colors(len(all_models))
        
        # Create datasets for ALL models
        datasets = []
        for i, modelo in enumerate(all_models):
            monthly_revenues = []
            for month in months:
                revenue = modelo_monthly[modelo].get(month, {}).get('revenue', 0)
                monthly_revenues.append(revenue)
            
            datasets.append({
                'label': modelo,
                'data': monthly_revenues,
                'borderColor': colors[i],
                'backgroundColor': colors[i] + '20',
                'borderWidth': 1.5,
                'tension': 0.4,
                'fill': False,
                'pointRadius': 1,
                'pointHoverRadius': 3
            })
        
        chart_data = {
            'labels': month_labels,
            'datasets': datasets
        }
        
        # Function to extract brand from model name
        def extract_brand(modelo):
            modelo = modelo.upper().strip()
            if modelo.startswith('IPHONE'):
                return 'APPLE'
            elif modelo.startswith('SAMSUNG') or modelo.startswith('S2') or modelo.startswith('S3') or modelo.startswith('NOTE') or modelo.startswith('A0') or modelo.startswith('A1') or modelo.startswith('A2') or modelo.startswith('A3') or modelo.startswith('A4') or modelo.startswith('A5') or modelo.startswith('A6') or modelo.startswith('A7') or modelo.startswith('A8') or modelo.startswith('FLIP') or modelo.startswith('Z FOLD'):
                return 'SAMSUNG'
            elif modelo.startswith('MOTO') or modelo.startswith('EDGE'):
                return 'MOTOROLA'
            elif modelo.startswith('REDMI') or modelo.startswith('MI ') or modelo.startswith('POCO') or modelo.startswith('XIAOMI'):
                return 'XIAOMI'
            elif modelo.startswith('HONOR'):
                return 'HONOR'
            elif modelo.startswith('OPPO') or modelo.startswith('RENO'):
                return 'OPPO'
            elif modelo.startswith('VIVO'):
                return 'VIVO'
            elif modelo.startswith('REALME'):
                return 'REALME'
            elif modelo.startswith('NOVA') or modelo.startswith('P30') or modelo.startswith('P60') or modelo.startswith('MATE') or modelo.startswith('MAGIC'):
                return 'HUAWEI'
            elif modelo.startswith('ZTE') or modelo.startswith('AXON'):
                return 'ZTE'
            elif modelo.startswith('INFINIX'):
                return 'INFINIX'
            elif 'UNIVERSAL' in modelo:
                return 'UNIVERSAL'
            else:
                return 'OTROS'
        
        # Summary data with embedded monthly quantities and brand
        summary_data = []
        for modelo, totals in all_models_sorted:
            # Create monthly quantities array for this model
            monthly_quantities = []
            for month in months:  # ['01', '02', '03', ...]
                qty = modelo_monthly[modelo].get(month, {}).get('quantity', 0)
                monthly_quantities.append(qty)
            
            summary_data.append({
                'modelo': modelo,
                'marca': extract_brand(modelo),
                'total_revenue': totals['total_revenue'],
                'total_quantity': totals['total_quantity'],
                'avg_monthly': totals['total_revenue'] / 12,
                'monthly_qty': monthly_quantities  # Add monthly quantities directly
            })
        
        # Monthly details - Create a lookup dictionary for faster template processing
        monthly_lookup = {}
        for modelo in all_models:
            monthly_lookup[modelo] = {}
            for month_num, month_name in zip(months, month_labels):
                data = modelo_monthly[modelo].get(month_num, {})
                if data.get('revenue', 0) > 0:
                    monthly_lookup[modelo][month_num] = {
                        'month_name': month_name,
                        'sales': data.get('sales', 0),
                        'quantity': data.get('quantity', 0),
                        'revenue': data.get('revenue', 0)
                    }
                else:
                    monthly_lookup[modelo][month_num] = {
                        'month_name': month_name,
                        'sales': 0,
                        'quantity': 0,
                        'revenue': 0
                    }
        
        # Also create the original monthly_details for backward compatibility
        monthly_details = []
        for modelo in all_models:
            for month_num, month_name in zip(months, month_labels):
                data = modelo_monthly[modelo].get(month_num, {})
                if data.get('revenue', 0) > 0:
                    monthly_details.append({
                        'modelo': modelo,
                        'month_num': int(month_num),
                        'month_name': month_name,
                        'sales': data.get('sales', 0),
                        'quantity': data.get('quantity', 0),
                        'revenue': data.get('revenue', 0)
                    })
        
        monthly_details.sort(key=lambda x: (x['modelo'], x['month_num']))
        
        # Final debug output
        months_in_final = set([str(d['month_num']).zfill(2) for d in monthly_details])
        print(f"[Yearly Models] SUCCESS: {len(datasets)} models, {len(monthly_details)} monthly records")
        print(f"[Yearly Models] Final data includes months: {sorted(list(months_in_final))}")
        
        # Debug: Show sample of monthly lookup data
        if monthly_lookup and len(monthly_lookup) > 0:
            first_model = list(monthly_lookup.keys())[0]
            print(f"[Yearly Models] Sample monthly data for {first_model}: {monthly_lookup[first_model]}")
        else:
            print(f"[Yearly Models] WARNING: monthly_lookup is empty!")
            
        # Debug: Show sample of summary data
        if summary_data and len(summary_data) > 0:
            print(f"[Yearly Models] Sample summary data: {summary_data[0]}")
        else:
            print(f"[Yearly Models] WARNING: summary_data is empty!")
        
        return templates.TemplateResponse(
            request=request,
            name="modelosanuales.html",
            context={
                "year": year,
                "chart_data": chart_data,
                "summary_data": summary_data,
                "monthly_details": monthly_details,
                "monthly_lookup": monthly_lookup,  # Add this for easier template lookup
                "available_years": ["2022", "2023", "2024", "2025"],
                "error_message": None
            }
        )
        
    except Exception as e:
        print(f"[Yearly Models] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            request=request,
            name="modelosanuales.html",
            context={
                "year": year,
                "chart_data": {"labels": [], "datasets": []},
                "summary_data": [],
                "monthly_details": [],
                "available_years": ["2022", "2023", "2024", "2025"],
                "error_message": f"Error al cargar los datos: {str(e)}"
            }
        )

@app.get("/api/balance-data")
async def get_balance_data():
    """Fetch balance_date data from Supabase"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/balance_date",
                headers=HEADERS,
                params={"select": "*", "order": "date.asc"}
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance-tracking", response_class=HTMLResponse)
async def balance_tracking_page():
    """Serve the balance tracking HTML page"""
    with open("balancesheettracking.html", "r") as f:
        return f.read()
    
@app.get("/balancesheettracking.html")
async def serve_html():
    """Serve the balance tracking HTML file"""
    return FileResponse("templates/balancesheettracking.html")


@app.get("/metertelefonos", response_class=HTMLResponse)
async def metertelefonos_get(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="metertelefonos.html",
            context={}
        )

def normalize_phone(raw: str) -> str:
    return re.sub(r"\D+", "", raw or "")

async def supabase_insert(table: str, payload: dict):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY env vars")

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)

    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase insert failed: {r.status_code} {r.text}")

    return r.json()

@app.post("/metertelefonos")
async def meter_telefonos_submit(
    nombre: str = Form(...),
    telefono: str = Form(...)
):
    try:
        phone = normalize_phone(telefono)
        if not phone:
            raise HTTPException(status_code=400, detail="Telefono inválido")

        # ✅ INSERT (NO supabase_request call here)
        await supabase_insert("telefonos", {"nombre": nombre, "telefono": phone})

        # ✅ WhatsApp link with "HOLA AMIGO"
        whatsapp_url = f"https://wa.me/{phone}?text={urllib.parse.quote('HOLA AMIGO')}"

        return RedirectResponse(url=whatsapp_url, status_code=303)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving phone and creating WhatsApp link: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/conteoefectivo")
async def conteoefectivo_page():
    return FileResponse("templates/conteoefectivo.html")


@app.post("/api/conteo", response_model=ConteoEfectivoResponse)
async def create_conteo(data: ConteoEfectivoCreate):
    """Save a new cash movement entry"""
    try:
        # Get current balance
        current_balance = await get_current_balance()
        
        # Initialize variables
        new_balance = current_balance
        diferencia = None
        
        # Calculate new balance and diferencia based on tipo
        if data.tipo == 'credito':
            # Money coming in
            new_balance = current_balance + data.amount
        elif data.tipo == 'debito':
            # Money going out
            new_balance = current_balance - data.amount
        elif data.tipo == 'conteo':
            # Physical cash count - calculate difference
            diferencia = data.amount - current_balance
            new_balance = data.amount  # New balance is the counted amount
        else:
            raise HTTPException(status_code=400, detail="Tipo must be 'credito', 'debito', or 'conteo'")
        
        # Build payload
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo"
        payload = {
            "nombre": data.nombre,
            "tipo": data.tipo,
            "amount": data.amount,
            "balance": new_balance
        }
        
        # Only add diferencia if it's not None (for conteo tipo)
        if diferencia is not None:
            payload["diferencia"] = diferencia
        
        print(f"DEBUG: Sending payload: {payload}")
        
        # Send request to Supabase
        response = requests.post(url, headers=HEADERS, json=payload)
        
        # Log error details if request fails
        if not response.ok:
            print(f"ERROR: Status {response.status_code}")
            print(f"ERROR: Response: {response.text}")
        
        response.raise_for_status()
        
        entry = response.json()[0]
        
        # Log the conteo result
        if data.tipo == 'conteo':
            if diferencia == 0:
                print(f"✅ Conteo correcto: Balance esperado ${current_balance:.2f} = Contado ${data.amount:.2f}")
            elif diferencia > 0:
                print(f"💰 Sobrante en caja: ${diferencia:.2f} (Esperado: ${current_balance:.2f}, Contado: ${data.amount:.2f})")
            else:
                print(f"⚠️ Faltante en caja: ${abs(diferencia):.2f} (Esperado: ${current_balance:.2f}, Contado: ${data.amount:.2f})")
        
        return ConteoEfectivoResponse(
            id=entry["id"],
            nombre=entry["nombre"],
            tipo=entry["tipo"],
            amount=entry["amount"],
            balance=entry["balance"],
            created_at=entry["created_at"],
            order_id=entry.get("order_id"),
            descripcion=entry.get("descripcion"),
            diferencia=entry.get("diferencia")
        )
    except Exception as e:
        print(f"Error saving conteo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conteo", response_model=List[ConteoEfectivoResponse])
async def get_conteo(limit: Optional[int] = 100):
    """Get cash movement entries (most recent first)"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?order=created_at.desc&limit={limit}"
        
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return []
        
        return [
            ConteoEfectivoResponse(
                id=entry["id"],
                nombre=entry["nombre"],
                tipo=entry["tipo"],
                amount=entry["amount"],
                balance=entry["balance"],
                created_at=entry["created_at"],
                order_id=entry.get("order_id"),
                descripcion=entry.get("descripcion"),
                diferencia=entry.get("diferencia")  # Include diferencia
            )
            for entry in data
        ]
    except Exception as e:
        print(f"Error fetching conteo: {e}")
        raise HTTPException(status_code=500, detail=str(e))















@app.delete("/api/conteo/{conteo_id}")
async def delete_conteo(conteo_id: int):
    """Delete a cash movement entry and recalculate all balances"""
    try:
        # Check if trying to delete the initial balance
        check_url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?id=eq.{conteo_id}"
        check_response = requests.get(check_url, headers=HEADERS)
        check_response.raise_for_status()
        entry_data = check_response.json()
        
        if entry_data and entry_data[0].get('tipo') == 'inicial':
            raise HTTPException(status_code=400, detail="Cannot delete initial balance")
        
        # Delete the entry
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?id=eq.{conteo_id}"
        response = requests.delete(url, headers=HEADERS)
        response.raise_for_status()
        
        # Recalculate all balances
        await recalculate_balances()
        
        return {"success": True, "message": "Entry deleted and balances recalculated"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conteo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_current_balance():
    """Get the current balance from the last entry"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?order=created_at.desc&limit=1"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return 0.0
        
        return float(data[0].get('balance', 0.0))
    except Exception as e:
        print(f"Error getting current balance: {e}")
        return 0.0
# Helper function to recalculate all balances after deletion
async def recalculate_balances():
    """Recalculate all balances after a deletion"""
    try:
        # Get all entries ordered by creation time
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?order=created_at.asc"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        entries = response.json()
        
        running_balance = 0.0
        
        for entry in entries:
            if entry['tipo'] == 'inicial':
                running_balance = float(entry['amount'])
            elif entry['tipo'] == 'credito':
                running_balance += float(entry['amount'])
            elif entry['tipo'] == 'debito':
                running_balance -= float(entry['amount'])
            
            # Update the entry's balance
            update_url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo?id=eq.{entry['id']}"
            update_response = requests.patch(
                update_url,
                headers=HEADERS,
                json={"balance": running_balance}
            )
            update_response.raise_for_status()
        
        return running_balance
    except Exception as e:
        print(f"Error recalculating balances: {e}")
        raise

@app.post("/api/conteo", response_model=ConteoEfectivoResponse)
async def create_conteo(data: ConteoEfectivoCreate):
    """Save a new cash movement entry"""
    try:
        # Get current balance
        current_balance = await get_current_balance()
        
        # Initialize variables
        new_balance = current_balance
        diferencia = None
        
        # Calculate new balance and diferencia based on tipo
        if data.tipo == 'credito':
            new_balance = current_balance + data.amount
        elif data.tipo == 'debito':
            new_balance = current_balance - data.amount
        elif data.tipo == 'conteo':
            # Conteo: user enters actual counted amount
            # Diferencia = actual - expected
            diferencia = data.amount - current_balance
            new_balance = data.amount  # New balance is the counted amount
        else:
            raise HTTPException(status_code=400, detail="Tipo must be 'credito', 'debito', or 'conteo'")
        
        # Insert new entry
        url = f"{SUPABASE_URL}/rest/v1/conteo_efectivo"
        payload = {
            "nombre": data.nombre,
            "tipo": data.tipo,
            "amount": data.amount,
            "balance": new_balance,
            "diferencia": diferencia
        }
        
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        
        entry = response.json()[0]
        
        # Log the conteo result
        if data.tipo == 'conteo':
            if diferencia == 0:
                print(f"✅ Conteo correcto: Balance esperado ${current_balance:.2f} = Contado ${data.amount:.2f}")
            elif diferencia > 0:
                print(f"💰 Sobrante en caja: ${diferencia:.2f} (Esperado: ${current_balance:.2f}, Contado: ${data.amount:.2f})")
            else:
                print(f"⚠️ Faltante en caja: ${abs(diferencia):.2f} (Esperado: ${current_balance:.2f}, Contado: ${data.amount:.2f})")
        
        return ConteoEfectivoResponse(
            id=entry["id"],
            nombre=entry["nombre"],
            tipo=entry["tipo"],
            amount=entry["amount"],
            balance=entry["balance"],
            created_at=entry["created_at"],
            order_id=entry.get("order_id"),
            descripcion=entry.get("descripcion"),
            diferencia=entry.get("diferencia")
        )
    except Exception as e:
        print(f"Error saving conteo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def send_push_notification(order_id: int, total: float, items_count: int, payment_method: str):
    """Send push notification to all registered devices"""
    try:
        # Get all active FCM tokens
        url = f"{SUPABASE_URL}/rest/v1/fcm_tokens?active=eq.true"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        tokens_data = response.json()
        
        if not tokens_data:
            print("No FCM tokens registered")
            return
        
        # Get your Firebase Server Key from Firebase Console
        # Go to: Firebase Console > Project Settings > Cloud Messaging > Server Key
        FIREBASE_SERVER_KEY = "YOUR_FIREBASE_SERVER_KEY_HERE"  # TODO: Replace this
        
        # Prepare notification
        fcm_url = "https://fcm.googleapis.com/fcm/send"
        fcm_headers = {
            "Authorization": f"Bearer {FIREBASE_SERVER_KEY}",
            "Content-Type": "application/json"
        }
        
        # Send to each token
        for token_entry in tokens_data:
            fcm_token = token_entry['token']
            
            notification_body = {
                "to": fcm_token,
                "notification": {
                    "title": f"🎉 Nueva Venta #{order_id}",
                    "body": f"Total: ${total:.2f} | {items_count} producto(s) | {payment_method}",
                    "sound": "default",
                    "badge": "1"
                },
                "data": {
                    "order_id": str(order_id),
                    "total": str(total),
                    "items_count": str(items_count),
                    "payment_method": payment_method,
                    "type": "new_sale"
                },
                "priority": "high"
            }
            
            try:
                fcm_response = requests.post(
                    fcm_url,
                    headers=fcm_headers,
                    data=json.dumps(notification_body)
                )
                
                if fcm_response.status_code == 200:
                    print(f"✅ Notification sent for order #{order_id}")
                else:
                    print(f"❌ FCM Error: {fcm_response.text}")
                    
            except Exception as e:
                print(f"Error sending to token {fcm_token[:20]}...: {e}")
        
        # Update last_used timestamp
        update_url = f"{SUPABASE_URL}/rest/v1/fcm_tokens"
        requests.patch(
            update_url,
            headers=HEADERS,
            json={"last_used": "now()"}
        )
        
    except Exception as e:
        print(f"Error in send_push_notification: {e}")

# ============================
# API ENDPOINTS
# ============================

@app.post("/api/register_fcm_token")
async def register_fcm_token(data: FCMTokenRegistration):
    """Register a new FCM token for push notifications"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/fcm_tokens"
        payload = {
            "token": data.fcm_token,
            "device_name": data.device_name,
            "active": True
        }
        
        # Use upsert to handle duplicate tokens
        response = requests.post(
            url,
            headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
            json=payload
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ FCM Token registered: {data.fcm_token[:20]}...")
            return {"success": True, "message": "Token registered successfully"}
        else:
            print(f"Error registering token: {response.text}")
            raise HTTPException(status_code=400, detail="Failed to register token")
            
    except Exception as e:
        print(f"Error in register_fcm_token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fcm_tokens")
async def get_fcm_tokens():
    """Get all registered FCM tokens (for debugging)"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/fcm_tokens?active=eq.true&order=created_at.desc"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/fcm_tokens/{token_id}")
async def deactivate_fcm_token(token_id: int):
    """Deactivate an FCM token"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/fcm_tokens?id=eq.{token_id}"
        response = requests.patch(
            url,
            headers=HEADERS,
            json={"active": False}
        )
        response.raise_for_status()
        return {"success": True, "message": "Token deactivated"}
    except Exception as e:
        print(f"Error deactivating token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sync_prices")
async def sync_prices():
    """Sync prices from inventario_estilos to inventario1"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/rpc/sync_inventario_prices"
        response = requests.post(url, headers=HEADERS)
        response.raise_for_status()
        
        result = response.json()
        updated_count = result[0].get('updated_count', 0) if result else 0
        
        print(f"✅ Synced {updated_count} prices")
        return {"success": True, "updated": updated_count}
    except Exception as e:
        print(f"❌ Sync error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/dashboardterex", response_class=HTMLResponse)
async def dashboardterex(
    request: Request,
    days: int = 14
):
    try:
        import traceback

        async with httpx.AsyncClient() as client:
            # Fire all 6 requests concurrently
            results = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_dashboard_kpis",
                    headers=HEADERS, params={"days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_revenue",
                    headers=HEADERS, params={"days_back": max(days, 90)}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_top_models",
                    headers=HEADERS, params={"days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_top_estilos",
                    headers=HEADERS, params={"days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_peak_hours",
                    headers=HEADERS, params={"days_back": days}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_velocity_leaderboard",
                    headers=HEADERS, params={"days_back": days}
                ),
            )

        kpis_raw, weekly_raw, models_raw, estilos_raw, hours_raw, velocity_raw = results

        # Parse responses
        kpis      = kpis_raw.json()[0]     if kpis_raw.json()          else {}
        weekly    = weekly_raw.json()       if weekly_raw.status_code   == 200 else []
        top_models= models_raw.json()       if models_raw.status_code   == 200 else []
        top_estilos= estilos_raw.json()     if estilos_raw.status_code  == 200 else []
        peak_hours= hours_raw.json()        if hours_raw.status_code    == 200 else []
        velocity  = velocity_raw.json()     if velocity_raw.status_code == 200 else []

        # ── Weekly chart ─────────────────────────────────────────
        weekly_sorted = sorted(weekly, key=lambda x: x['week_start'])
        weekly_chart = {
            "labels":  [w['week_start']        for w in weekly_sorted],
            "revenue": [float(w['revenue'])     for w in weekly_sorted],
            "units":   [int(w['units_sold'])    for w in weekly_sorted],
            "orders":  [int(w['total_orders'])  for w in weekly_sorted],
        }

        # ── Heatmap (7 days × 24 hours) ─────────────────────────
        heatmap = [[0] * 24 for _ in range(7)]
        heatmap_max = 0
        for row in peak_hours:
            d = int(row['day_of_week'])
            h = int(row['hour_of_day'])
            v = int(row['units_sold'])
            heatmap[d][h] = v
            if v > heatmap_max:
                heatmap_max = v

        return templates.TemplateResponse(
            request=request,
            name="dashboardterex.html",
            context={
                "days":         days,
                "kpis":         kpis,
                "weekly_chart": weekly_chart,
                "top_models":   top_models,
                "top_estilos":  top_estilos,
                "heatmap":      heatmap,
                "heatmap_max":  heatmap_max,
                "velocity":     velocity
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Dashboard error: {str(e)}"}
        )

@app.get("/transferencias", response_class=HTMLResponse)
async def get_transferencias_page(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="transferencias.html",
            context={}
        )


@app.get("/inventoryxbarcode", response_class=HTMLResponse)
async def get_inventoryxbarcode_page(request: Request):
    return templates.TemplateResponse(
            request=request,
            name="inventoryxbarcode.html",
            context={}
        )


@app.get("/api/inventoryxbarcode")
async def get_product_by_barcode(barcode: str):
    """Fetch product + its full history from terex1_history"""
    try:
        # Product
        url = (
            f"{SUPABASE_URL}/rest/v1/inventario1"
            f"?barcode=eq.{barcode}"
            f"&select=barcode,name,estilo,estilo_id,marca,color,terex1"
            f"&limit=1"
        )
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        product = data[0]

        # History for this barcode
        hist_url = (
            f"{SUPABASE_URL}/rest/v1/terex1_history"
            f"?barcode=eq.{barcode}"
            f"&order=created_at.desc"
            f"&limit=20"
        )
        hist_resp = requests.get(hist_url, headers=HEADERS)
        hist_resp.raise_for_status()
        product["history"] = hist_resp.json()

        return product
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/inventoryxbarcode")
async def update_terex1(payload: dict):
    """Update terex1 and log to terex1_history"""
    try:
        barcode      = payload.get("barcode")
        terex1       = payload.get("terex1")
        qty_before   = payload.get("qty_before", 0)
        product_name = payload.get("product_name", "")

        if barcode is None or terex1 is None:
            raise HTTPException(status_code=400, detail="barcode and terex1 required")

        # Update inventario1
        url = f"{SUPABASE_URL}/rest/v1/inventario1?barcode=eq.{barcode}"
        resp = requests.patch(url, headers=HEADERS, json={"terex1": terex1})
        resp.raise_for_status()

        # Insert into terex1_history
        matches    = int(qty_before) == int(terex1)
        difference = int(terex1) - int(qty_before)
        hist_url   = f"{SUPABASE_URL}/rest/v1/terex1_history"
        hist_payload = {
            "barcode":      int(barcode),
            "product_name": product_name,
            "qty_before":   int(qty_before),
            "qty_counted":  int(terex1),
            "matches":      matches,
            "difference":   difference,
        }
        requests.post(hist_url, headers=HEADERS, json=hist_payload)

        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pendientes1")
async def get_pendientes1():
    """Products in inventario1 where terex1 < 0"""
    try:
        url = (
            f"{SUPABASE_URL}/rest/v1/inventario1"
            f"?terex1=lt.0"
            f"&select=barcode,name,estilo,marca,color,terex1"
            f"&order=terex1.asc"
            f"&limit=200"
        )
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════
# DASHBOARD POWER — Both branches, negatives, full turnover
# ══════════════════════════════════════════════════════════════

@app.get("/dashboard-power", response_class=HTMLResponse)
async def dashboard_power(request: Request, days: int = 14):
    try:
        mexico_tz = pytz.timezone("America/Mexico_City")
        now_mx = datetime.now(mexico_tz)
        date_from = (now_mx - timedelta(days=days)).strftime("%Y-%m-%d")

        range_header = {**HEADERS, "Range": "0-9999", "Prefer": "return=representation"}

        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                # 0: ventas_terex1
                client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                    headers=range_header,
                    params={
                        "select": "fecha,qty,price,estilo_id,estilo,modelo,name_id,order_id,hora",
                        "fecha": f"gte.{date_from}",
                        "order": "fecha.asc"
                    }
                ),
                # 1: ventas_terex2
                client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex2",
                    headers=range_header,
                    params={
                        "select": "fecha,qty,price,estilo_id,estilo,modelo,name_id,order_id,hora",
                        "fecha": f"gte.{date_from}",
                        "order": "fecha.asc"
                    }
                ),
                # 2: negative inventory (terex1 < 0 OR terex2 < 0)
                client.get(
                    f"{SUPABASE_URL}/rest/v1/inventario1",
                    headers=range_header,
                    params={
                        "select": "barcode,name,terex1,terex2,estilo,estilo_id,modelo,color",
                        "or": "(terex1.lt.0,terex2.lt.0)",
                        "order": "estilo.asc"
                    }
                ),
                # 3: all prioridad=1 estilos
                client.get(
                    f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                    headers=range_header,
                    params={
                        "select": "id,nombre",
                        "prioridad": "eq.1",
                        "order": "nombre.asc"
                    }
                ),
                # 4: recent entrada_mercancia (last 60 days for new demand)
                client.get(
                    f"{SUPABASE_URL}/rest/v1/entrada_mercancia",
                    headers=range_header,
                    params={
                        "select": "created_at,qty,barcode,estilo,estilo_id",
                        "created_at": f"gte.{(now_mx - timedelta(days=60)).isoformat()}",
                        "order": "created_at.desc"
                    }
                ),
            )

        sales_t1 = results[0].json() if results[0].status_code in (200, 206) else []
        sales_t2 = results[1].json() if results[1].status_code in (200, 206) else []
        negatives = results[2].json() if results[2].status_code in (200, 206) else []
        prioridad_estilos = results[3].json() if results[3].status_code in (200, 206) else []
        entradas = results[4].json() if results[4].status_code in (200, 206) else []

        # ── KPIs per branch ────────────────────────────────────
        def calc_kpis(sales):
            revenue = sum((r.get("qty") or 0) * (r.get("price") or 0) for r in sales)
            units = sum(r.get("qty") or 0 for r in sales)
            orders = len(set(r.get("order_id") for r in sales if r.get("order_id")))
            avg_ticket = revenue / orders if orders else 0
            return {"revenue": revenue, "units": units, "orders": orders, "avg_ticket": avg_ticket}

        kpis_t1 = calc_kpis(sales_t1)
        kpis_t2 = calc_kpis(sales_t2)
        kpis_combined = {
            "revenue": kpis_t1["revenue"] + kpis_t2["revenue"],
            "units": kpis_t1["units"] + kpis_t2["units"],
            "orders": kpis_t1["orders"] + kpis_t2["orders"],
            "avg_ticket": (kpis_t1["revenue"] + kpis_t2["revenue"]) / max(kpis_t1["orders"] + kpis_t2["orders"], 1),
        }

        # ── Daily sales chart (both branches) ──────────────────
        daily_t1 = defaultdict(lambda: {"revenue": 0, "units": 0})
        daily_t2 = defaultdict(lambda: {"revenue": 0, "units": 0})

        for r in sales_t1:
            d = r.get("fecha")
            if d:
                daily_t1[d]["revenue"] += (r.get("qty") or 0) * (r.get("price") or 0)
                daily_t1[d]["units"] += r.get("qty") or 0

        for r in sales_t2:
            d = r.get("fecha")
            if d:
                daily_t2[d]["revenue"] += (r.get("qty") or 0) * (r.get("price") or 0)
                daily_t2[d]["units"] += r.get("qty") or 0

        all_dates = sorted(set(list(daily_t1.keys()) + list(daily_t2.keys())))
        daily_chart = {
            "labels": all_dates,
            "t1_revenue": [daily_t1[d]["revenue"] for d in all_dates],
            "t2_revenue": [daily_t2[d]["revenue"] for d in all_dates],
            "t1_units": [daily_t1[d]["units"] for d in all_dates],
            "t2_units": [daily_t2[d]["units"] for d in all_dates],
        }

        # ── Negative inventory processing ──────────────────────
        negative_items = []
        for item in negatives:
            t1 = item.get("terex1") or 0
            t2 = item.get("terex2") or 0
            negative_items.append({
                "barcode": item.get("barcode"),
                "name": item.get("name", ""),
                "terex1": int(t1),
                "terex2": int(t2),
                "estilo": item.get("estilo", ""),
                "modelo": item.get("modelo", ""),
                "color": item.get("color", ""),
                "branch": ("T1" if t1 < 0 else "") + (" T2" if t2 < 0 else ""),
            })

        # ── Turnover for ALL prioridad=1 estilos ──────────────
        prioridad_ids = {e["id"] for e in prioridad_estilos}
        prioridad_names = {e["id"]: e["nombre"] for e in prioridad_estilos}

        estilo_sales = defaultdict(lambda: {
            "t1_units": 0, "t1_revenue": 0, "t2_units": 0, "t2_revenue": 0,
            "modelo": "", "first_sale": None, "last_sale": None,
            "t1_orders": set(), "t2_orders": set()
        })

        for r in sales_t1:
            eid = r.get("estilo_id")
            if eid and eid in prioridad_ids:
                estilo_sales[eid]["t1_units"] += r.get("qty") or 0
                estilo_sales[eid]["t1_revenue"] += (r.get("qty") or 0) * (r.get("price") or 0)
                estilo_sales[eid]["modelo"] = r.get("modelo", "")
                if r.get("order_id"):
                    estilo_sales[eid]["t1_orders"].add(r["order_id"])
                fecha = r.get("fecha")
                if fecha:
                    if not estilo_sales[eid]["first_sale"] or fecha < estilo_sales[eid]["first_sale"]:
                        estilo_sales[eid]["first_sale"] = fecha
                    if not estilo_sales[eid]["last_sale"] or fecha > estilo_sales[eid]["last_sale"]:
                        estilo_sales[eid]["last_sale"] = fecha

        for r in sales_t2:
            eid = r.get("estilo_id")
            if eid and eid in prioridad_ids:
                estilo_sales[eid]["t2_units"] += r.get("qty") or 0
                estilo_sales[eid]["t2_revenue"] += (r.get("qty") or 0) * (r.get("price") or 0)
                estilo_sales[eid]["modelo"] = r.get("modelo") or estilo_sales[eid]["modelo"]
                if r.get("order_id"):
                    estilo_sales[eid]["t2_orders"].add(r["order_id"])
                fecha = r.get("fecha")
                if fecha:
                    if not estilo_sales[eid]["first_sale"] or fecha < estilo_sales[eid]["first_sale"]:
                        estilo_sales[eid]["first_sale"] = fecha
                    if not estilo_sales[eid]["last_sale"] or fecha > estilo_sales[eid]["last_sale"]:
                        estilo_sales[eid]["last_sale"] = fecha

        turnover_table = []
        for eid in prioridad_ids:
            s = estilo_sales.get(eid, {
                "t1_units": 0, "t1_revenue": 0, "t2_units": 0, "t2_revenue": 0,
                "modelo": "", "first_sale": None, "last_sale": None,
                "t1_orders": set(), "t2_orders": set()
            })
            total_units = s["t1_units"] + s["t2_units"]
            total_revenue = s["t1_revenue"] + s["t2_revenue"]

            # Days active
            if s["first_sale"] and s["last_sale"] and s["first_sale"] != s["last_sale"]:
                from datetime import date as dt_date
                d1 = datetime.strptime(s["first_sale"], "%Y-%m-%d").date() if isinstance(s["first_sale"], str) else s["first_sale"]
                d2 = datetime.strptime(s["last_sale"], "%Y-%m-%d").date() if isinstance(s["last_sale"], str) else s["last_sale"]
                days_active = max((d2 - d1).days, 1)
            else:
                days_active = 1 if total_units > 0 else 0

            velocity = round(total_units / days_active, 2) if days_active > 0 else 0

            turnover_table.append({
                "estilo_id": eid,
                "estilo": prioridad_names.get(eid, ""),
                "modelo": s["modelo"],
                "t1_units": s["t1_units"],
                "t2_units": s["t2_units"],
                "total_units": total_units,
                "t1_revenue": s["t1_revenue"],
                "t2_revenue": s["t2_revenue"],
                "total_revenue": total_revenue,
                "days_active": days_active,
                "velocity": velocity,
            })

        # Sort by velocity descending
        turnover_table.sort(key=lambda x: x["velocity"], reverse=True)

        # ── New demand tracker (estilos from recent entradas) ──
        # Find estilos that entered recently and measure their sales velocity
        entrada_estilos = defaultdict(lambda: {"qty_entered": 0, "first_entry": None})
        for e in entradas:
            eid = e.get("estilo_id")
            if eid:
                entrada_estilos[eid]["qty_entered"] += e.get("qty") or 0
                created = e.get("created_at", "")[:10]
                if created:
                    if not entrada_estilos[eid]["first_entry"] or created < entrada_estilos[eid]["first_entry"]:
                        entrada_estilos[eid]["first_entry"] = created

        new_demand = []
        for eid, entry_info in entrada_estilos.items():
            s = estilo_sales.get(eid, {"t1_units": 0, "t2_units": 0, "t1_revenue": 0, "t2_revenue": 0, "modelo": ""})
            total_sold = s["t1_units"] + s["t2_units"]
            qty_entered = entry_info["qty_entered"]
            sell_through = round((total_sold / qty_entered) * 100, 1) if qty_entered > 0 else 0

            # Days since first entry
            if entry_info["first_entry"]:
                try:
                    entry_date = datetime.strptime(entry_info["first_entry"], "%Y-%m-%d").date()
                    days_since = max((now_mx.date() - entry_date).days, 1)
                except Exception:
                    days_since = 1
            else:
                days_since = 1

            vel = round(total_sold / days_since, 2) if days_since > 0 else 0

            new_demand.append({
                "estilo_id": eid,
                "estilo": prioridad_names.get(eid, f"Estilo {eid}"),
                "modelo": s.get("modelo", ""),
                "qty_entered": qty_entered,
                "total_sold": total_sold,
                "sell_through": sell_through,
                "velocity": vel,
                "days_since_entry": days_since,
                "first_entry": entry_info["first_entry"] or "",
            })

        # Sort by velocity descending
        new_demand.sort(key=lambda x: x["velocity"], reverse=True)

        # ── Top 15 estilos for comparison chart ────────────────
        top15 = turnover_table[:15]
        top15_chart = {
            "labels": [t["estilo"][:20] for t in top15],
            "t1_units": [t["t1_units"] for t in top15],
            "t2_units": [t["t2_units"] for t in top15],
            "t1_revenue": [t["t1_revenue"] for t in top15],
            "t2_revenue": [t["t2_revenue"] for t in top15],
        }

        # ── Per-estilo daily trend (top 5 by velocity) ─────────
        top5_ids = [t["estilo_id"] for t in turnover_table[:5]]
        top5_names = {t["estilo_id"]: t["estilo"][:18] for t in turnover_table[:5]}
        estilo_daily = {eid: defaultdict(int) for eid in top5_ids}

        for r in sales_t1 + sales_t2:
            eid = r.get("estilo_id")
            if eid in estilo_daily:
                fecha = r.get("fecha")
                if fecha:
                    estilo_daily[eid][fecha] += r.get("qty") or 0

        estilo_trend_chart = {
            "labels": all_dates,
            "datasets": [
                {
                    "label": top5_names.get(eid, ""),
                    "data": [estilo_daily[eid].get(d, 0) for d in all_dates],
                }
                for eid in top5_ids
            ]
        }

        # ── Daily order count per branch (for avg ticket chart) ─
        daily_orders_t1 = defaultdict(set)
        daily_orders_t2 = defaultdict(set)
        for r in sales_t1:
            d = r.get("fecha")
            oid = r.get("order_id")
            if d and oid:
                daily_orders_t1[d].add(oid)
        for r in sales_t2:
            d = r.get("fecha")
            oid = r.get("order_id")
            if d and oid:
                daily_orders_t2[d].add(oid)

        avg_ticket_chart = {
            "labels": all_dates,
            "t1": [round(daily_t1[d]["revenue"] / max(len(daily_orders_t1[d]), 1), 0) for d in all_dates],
            "t2": [round(daily_t2[d]["revenue"] / max(len(daily_orders_t2[d]), 1), 0) for d in all_dates],
        }

        return templates.TemplateResponse(
            request=request,
            name="dashboard_power.html",
            context={
                "days": days,
                "kpis_t1": kpis_t1,
                "kpis_t2": kpis_t2,
                "kpis_combined": kpis_combined,
                "daily_chart": daily_chart,
                "negative_items": negative_items,
                "turnover_table": turnover_table,
                "new_demand": new_demand,
                "top15_chart": top15_chart,
                "estilo_trend_chart": estilo_trend_chart,
                "avg_ticket_chart": avg_ticket_chart,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": f"Dashboard Power error: {str(e)}"}
        )


@app.get("/api/dashboard-power/negative-sales/{barcode}")
async def negative_sales_drilldown(barcode: str):
    """Fetch sales history for a specific barcode from both branches"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r1, r2 = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex1",
                    headers=HEADERS,
                    params={
                        "select": "fecha,hora,qty,price,order_id",
                        "name_id": f"eq.{barcode}",
                        "order": "fecha.desc,hora.desc",
                        "limit": "50"
                    }
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/ventas_terex2",
                    headers=HEADERS,
                    params={
                        "select": "fecha,hora,qty,price,order_id",
                        "name_id": f"eq.{barcode}",
                        "order": "fecha.desc,hora.desc",
                        "limit": "50"
                    }
                ),
            )

        t1_sales = [{"branch": "T1", **s} for s in (r1.json() if r1.status_code == 200 else [])]
        t2_sales = [{"branch": "T2", **s} for s in (r2.json() if r2.status_code == 200 else [])]

        all_sales = sorted(t1_sales + t2_sales, key=lambda x: (x.get("fecha", ""), x.get("hora", "")), reverse=True)
        return all_sales

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)