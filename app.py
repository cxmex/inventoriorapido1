# app.py - FastAPI with templates for inventory management
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import jwt
import requests
from fastapi import HTTPException
from datetime import datetime
import time
import hashlib
from fastapi.responses import FileResponse
import urllib.parse


import qrcode
import asyncio

from fastapi import FastAPI, HTTPException, Request, Form, Depends
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Supabase configuration
SUPABASE_URL = "https://gbkhkbfbarsnpbdkxzii.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdia2hrYmZiYXJzbnBiZGt4emlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQzODAzNzMsImV4cCI6MjA0OTk1NjM3M30.mcOcC2GVEu_wD3xNBzSCC3MwDck3CIdmz4D8adU-bpI"

LOCAL_CAMERA_SERVICE = "http://192.168.1.71:5001"


# Supabase client headers
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Data models

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
    products: list[ProductIn] = Field(..., min_length=1)


class InventarioEstilo(BaseModel):
    id: int
    nombre: str


class Product(BaseModel):
    qty: int
    name: str
    codigo: str
    price: float
    customer_email: Optional[str] = None  # Add this field

class SavePayload(BaseModel):
    products: List[Product]

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
            "menu.html", 
            {
                "request": request, 
                "inventory_styles": enhanced_items
            }
        )
    except Exception as e:
        print(f"Error loading menu: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request, 
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
            "inventory_detail.html", 
            {
                "request": request, 
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
            "error.html", 
            {
                "request": request, 
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
            "inventory_detail.html", 
            {
                "request": request, 
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
            "error.html", 
            {
                "request": request, 
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
                        "error.html", 
                        {
                            "request": request, 
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
                "error.html", 
                {
                    "request": request, 
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
            "error.html", 
            {
                "request": request, 
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
        import traceback
        
        print(f"Fetching daily sales data for the last 14 days", flush=True)
        
        try:
            # Call both Supabase functions
            async with httpx.AsyncClient() as client:
                # Get data by estilo
                response_by_estilo = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_by_estilo",
                    headers=HEADERS,
                    params={"days_back": 14}
                )
                
                # Get total data (for the table)
                response_total = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_daily_sales_total",
                    headers=HEADERS,
                    params={"days_back": 14}
                )
                
                if response_by_estilo.status_code >= 400:
                    print(f"Function call error (by estilo): {response_by_estilo.text}", flush=True)
                    raise Exception(f"Function call failed: {response_by_estilo.status_code}")
                
                if response_total.status_code >= 400:
                    print(f"Function call error (total): {response_total.text}", flush=True)
                    raise Exception(f"Function call failed: {response_total.status_code}")
                
                daily_results_by_estilo = response_by_estilo.json()
                daily_results_total = response_total.json()
                
                print(f"Retrieved daily sales data for {len(daily_results_by_estilo)} records by estilo", flush=True)
            
            # Process total results for the table
            day_totals = {}
            for row in daily_results_total:
                day_key = row.get('day_date')
                total = float(row.get('total_revenue', 0) or 0)
                day_totals[day_key] = total
            
            # Process results by estilo for the chart
            chart_data_by_estilo = {}
            all_days = set()
            all_estilos = set()
            
            for row in daily_results_by_estilo:
                day_key = row.get('day_date')
                estilo = row.get('estilo')
                total = float(row.get('total_revenue', 0) or 0)
                
                # Simplify date format for chart (keep DD/MM format)
                simple_date = datetime.strptime(day_key, "%d/%m/%Y").strftime("%d/%m")
                
                if estilo not in chart_data_by_estilo:
                    chart_data_by_estilo[estilo] = {}
                
                chart_data_by_estilo[estilo][simple_date] = round(total, 2)
                all_days.add(simple_date)
                all_estilos.add(estilo)
            
            # Prepare chart data structure
            chart_data = {
                'labels': sorted(list(all_days), key=lambda x: datetime.strptime(x, "%d/%m")),
                'datasets': []
            }
            
            # Define colors for different estilos
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
            
            # Create dataset for each estilo
            for i, estilo in enumerate(sorted(all_estilos)):
                color = colors[i % len(colors)]
                data = []
                
                for day in chart_data['labels']:
                    value = chart_data_by_estilo.get(estilo, {}).get(day, 0)
                    data.append(value)
                
                chart_data['datasets'].append({
                    'label': estilo,
                    'data': data,
                    'backgroundColor': color['bg'],
                    'borderColor': color['border'],
                    'borderWidth': 1
                })
            
            print(f"Daily totals: {day_totals}", flush=True)
            
            return templates.TemplateResponse(
                "ventas_por_dia.html",
                {
                    "request": request,
                    "day_totals": day_totals,
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True  # Flag to indicate we have estilo data
                }
            )
            
        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
                "ventas_por_dia.html",
                {
                    "request": request,
                    "day_totals": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "chart_data_by_estilo": False
                }
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": f"Error loading daily sales: {str(e)}"
            }
        )

@app.get("/verventasxsemana", response_class=HTMLResponse)
async def ver_ventas_por_semana(request: Request):
    try:
        import traceback
        
        print(f"Fetching weekly sales data for the last 15 weeks", flush=True)
        
        try:
            # Call both Supabase functions
            async with httpx.AsyncClient() as client:
                # Get data by estilo
                response_by_estilo = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_by_estilo",
                    headers=HEADERS,
                    params={"weeks_back": 15}
                )
                
                # Get total data (for the table)
                response_total = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales_total",
                    headers=HEADERS,
                    params={"weeks_back": 15}
                )
                
                if response_by_estilo.status_code >= 400:
                    print(f"Function call error (by estilo): {response_by_estilo.text}", flush=True)
                    raise Exception(f"Function call failed: {response_by_estilo.status_code}")
                
                if response_total.status_code >= 400:
                    print(f"Function call error (total): {response_total.text}", flush=True)
                    raise Exception(f"Function call failed: {response_total.status_code}")
                
                weekly_results_by_estilo = response_by_estilo.json()
                weekly_results_total = response_total.json()
                
                print(f"Retrieved sales data for {len(weekly_results_by_estilo)} records by estilo", flush=True)
            
            # Process total results for the table
            week_totals = {}
            for row in weekly_results_total:
                week_key = row.get('week_start')
                total = float(row.get('total_revenue', 0) or 0)
                week_totals[week_key] = total
            
            # Process results by estilo for the chart
            chart_data_by_estilo = {}
            all_weeks = set()
            all_estilos = set()
            
            for row in weekly_results_by_estilo:
                week_key = row.get('week_start')
                estilo = row.get('estilo')
                total = float(row.get('total_revenue', 0) or 0)
                
                # Simplify date format for chart
                simple_date = datetime.strptime(week_key, "%d/%m/%Y").strftime("%d/%m")
                
                if estilo not in chart_data_by_estilo:
                    chart_data_by_estilo[estilo] = {}
                
                chart_data_by_estilo[estilo][simple_date] = round(total, 2)
                all_weeks.add(simple_date)
                all_estilos.add(estilo)
            
            # Prepare chart data structure
            chart_data = {
                'labels': sorted(list(all_weeks), key=lambda x: datetime.strptime(x, "%d/%m")),
                'datasets': []
            }
            
            # Define colors for different estilos
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
            
            # Create dataset for each estilo
            for i, estilo in enumerate(sorted(all_estilos)):
                color = colors[i % len(colors)]
                data = []
                
                for week in chart_data['labels']:
                    value = chart_data_by_estilo.get(estilo, {}).get(week, 0)
                    data.append(value)
                
                chart_data['datasets'].append({
                    'label': estilo,
                    'data': data,
                    'backgroundColor': color['bg'],
                    'borderColor': color['border'],
                    'borderWidth': 1
                })
            
            print(f"Weekly totals: {week_totals}", flush=True)
            
            return templates.TemplateResponse(
                "ventas_por_semana.html",
                {
                    "request": request,
                    "week_totals": week_totals,
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True  # Flag to indicate we have estilo data
                }
            )
            
        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
                "ventas_por_semana.html",
                {
                    "request": request,
                    "week_totals": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "chart_data_by_estilo": False
                }
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": f"Error loading weekly sales: {str(e)}"
            }
        )




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
                    "ventas_por_semana.html",
                    {
                        "request": request,
                        "week_totals": {},
                        "chart_data": {"labels": [], "datasets": []},
                        "chart_data_by_estilo": False,
                        "selected_week_start": None,
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
                "ventas_por_semana.html",
                {
                    "request": request,
                    "week_totals": {latest_week_start: week_totals.get(latest_week_start, last_week_total)},
                    "chart_data": chart_data,
                    "chart_data_by_estilo": True,
                    "selected_week_start": latest_week_start,
                    "sorted_count": len(labels),
                }
            )

        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
                "ventas_por_semana.html",
                {
                    "request": request,
                    "week_totals": {},
                    "chart_data": {"labels": [], "datasets": []},
                    "chart_data_by_estilo": False,
                    "selected_week_start": None,
                }
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": f"Error loading weekly sales: {str(e)}",
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
                "conteo_rapido.html",
                {
                    "request": request,
                    "estilos": estilos_data,
                    "total_items": len(estilos_data)
                }
            )
            
        except Exception as fetch_error:
            print(f"Error fetching data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
                "conteo_rapido.html",
                {
                    "request": request,
                    "estilos": [],
                    "total_items": 0,
                    "error_message": f"Error loading data: {str(fetch_error)}"
                }
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
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
                "error.html", 
                {
                    "request": request, 
                    "error_message": f"Failed to add model: {error_msg}"
                }
            )
        
        # Redirect back to models page
        return RedirectResponse(url="/modelos?success=true", status_code=303)
        
    except Exception as e:
        print(f"Add model error: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request, 
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
                    "inventario_daily.html",
                    {
                        "request": request,
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
                "inventario_daily.html",
                {
                    "request": request,
                    "daily_totals": daily_totals,
                    "chart_data": chart_data,
                    "has_chart_data": True
                }
            )
            
        except Exception as fetch_error:
            print(f"Error fetching inventory data: {str(fetch_error)}", flush=True)
            traceback.print_exc()
            return templates.TemplateResponse(
                "inventario_daily.html",
                {
                    "request": request,
                    "daily_totals": {},
                    "chart_data": {'labels': [], 'datasets': []},
                    "has_chart_data": False
                }
            )
            
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
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
        
        return templates.TemplateResponse("ventasviaje.html", {
            "request": request,
            "estilos": estilos,
            "estilos_json": estilos_json
        })
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
        
        return templates.TemplateResponse("ver_ventas_viaje.html", {
            "request": request,
            "orders": orders_list
        })
        
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
    

# Entrada Mercancia Endpoints
@app.get("/entradamercancia", response_class=HTMLResponse)
async def get_entrada_mercancia_form(request: Request):
    """Render the merchandise entry form"""
    try:
        print("Loading entrada mercancia form", flush=True)
        
        return templates.TemplateResponse("entrada_mercancia.html", {
            "request": request
        })
        
    except Exception as e:
        print(f"Error loading entrada mercancia form: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error loading form: {str(e)}")

@app.post("/entradamercancia")
async def process_entrada_mercancia(
    request: Request,
    qty: int = Form(...),
    barcode: str = Form(...)
):
    """Process merchandise entry form and save to entrada_mercancia"""
    try:
        print(f"Processing entrada mercancia: qty={qty}, barcode={barcode}", flush=True)
        
        # Validate inputs
        if qty <= 0:
            raise HTTPException(status_code=400, detail="La cantidad debe ser mayor a 0")
        
        if not barcode or barcode.strip() == "":
            raise HTTPException(status_code=400, detail="El código de barras es requerido")
        
        barcode = barcode.strip()
        
        # Convert barcode to integer if possible
        try:
            barcode_int = int(barcode)
        except ValueError:
            raise HTTPException(status_code=400, detail="El código de barras debe ser numérico")
        
        # Try to get product info from inventario1 table using barcode
        product_info = None
        current_terex1 = 0
        try:
            product_response = await supabase_request(
                method="GET",
                endpoint="/rest/v1/inventario1",
                params={
                    "select": "name,estilo_id,marca,terex1",
                    "barcode": f"eq.{barcode}",
                    "limit": "1"
                }
            )
            
            if product_response and len(product_response) > 0:
                product_info = product_response[0]
                current_terex1 = product_info.get("terex1", 0) or 0  # Handle None values
                print(f"Found product info: {product_info}, current terex1: {current_terex1}", flush=True)
            else:
                print(f"No product found with barcode {barcode}", flush=True)
                
        except Exception as product_error:
            print(f"Error fetching product info: {str(product_error)}", flush=True)
            # Continue without product info
        
        # Prepare data for entrada_mercancia table
        entrada_data = {
            "qty": qty,
            "barcode": barcode_int,  # Use integer barcode
        }
        
        # Add product info if found
        if product_info:
            if product_info.get("name"):
                entrada_data["estilo"] = product_info.get("name", "")  # Use 'estilo' field for name
            if product_info.get("estilo_id"):
                entrada_data["estilo_id"] = product_info.get("estilo_id")
        
        print(f"Inserting entrada data: {entrada_data}", flush=True)
        
        # Insert into entrada_mercancia table with error handling
        entrada_success = False
        try:
            response = await supabase_request(
                method="POST",
                endpoint="/rest/v1/entrada_mercancia",
                json_data=entrada_data
            )
            
            print(f"Entrada mercancia insert response: {response}", flush=True)
            entrada_success = True
            
        except Exception as insert_error:
            print(f"Insert error details: {str(insert_error)}", flush=True)
            
            # Try with minimal data if full insert fails
            try:
                minimal_data = {
                    "qty": qty,
                    "barcode": barcode_int
                }
                print(f"Trying minimal insert: {minimal_data}", flush=True)
                
                response = await supabase_request(
                    method="POST",
                    endpoint="/rest/v1/entrada_mercancia",
                    json_data=minimal_data
                )
                
                print(f"Minimal insert successful: {response}", flush=True)
                entrada_success = True
                
            except Exception as minimal_error:
                print(f"Minimal insert also failed: {str(minimal_error)}", flush=True)
                raise HTTPException(status_code=500, detail=f"Database error: {str(minimal_error)}")
        
        # Update inventario1 terex1 column if entrada was successful and product exists
        if entrada_success and product_info:
            try:
                new_terex1 = current_terex1 + qty
                print(f"Updating terex1 from {current_terex1} to {new_terex1} for barcode {barcode}", flush=True)
                
                # Use the correct format for the PATCH request
                update_response = await supabase_request(
                    method="PATCH",
                    endpoint=f"/rest/v1/inventario1?barcode=eq.{barcode_int}",
                    json_data={
                        "terex1": new_terex1
                    }
                )
                
                print(f"Inventario1 terex1 update response: {update_response}", flush=True)
                
            except Exception as update_error:
                print(f"Error updating terex1 in inventario1: {str(update_error)}", flush=True)
                import traceback
                print(f"Update error traceback: {traceback.format_exc()}", flush=True)
                # Don't fail the entire operation if terex1 update fails
                # Log the error but continue
        
        if entrada_success:
            return {
                "success": True,
                "message": f"Entrada registrada exitosamente",
                "qty": qty,
                "barcode": barcode,  # Return original barcode string for display
                "product_name": product_info.get("name", "Producto no identificado") if product_info else "Producto no identificado",
                "terex1_updated": product_info is not None  # Indicate if terex1 was updated
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to insert entrada")
            
    except Exception as e:
        print(f"Error in entrada mercancia: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing entrada: {str(e)}")

# Get recent entries endpoint
@app.get("/entradamercancia/recientes")
async def get_recent_entries():
    """Get recent merchandise entries"""
    try:
        print("Fetching recent entrada mercancia records", flush=True)
        
        # Get last 20 entries
        entries = await supabase_request(
            method="GET",
            endpoint="/rest/v1/entrada_mercancia",
            params={
                "select": "*",
                "order": "created_at.desc",
                "limit": "20"
            }
        )
        
        print(f"Retrieved {len(entries)} recent entries", flush=True)
        
        return {
            "success": True,
            "entries": entries
        }
        
    except Exception as e:
        print(f"Error fetching recent entries: {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
            "entries": []
        }

# Set up logging to see detailed errors
logger = logging.getLogger(__name__)


@app.get("/verimagenes", response_class=HTMLResponse)
async def ver_imagenes(request: Request):
    """Main page to view images associated with estilos"""
    return templates.TemplateResponse("verimagenes.html", {"request": request})

@app.get("/api/estilos-with-images")
async def get_estilos_with_images():
    """Get estilos with images and sales data"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Calculate 3 months ago
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Get all estilos with prioridad=1
            estilos_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario_estilos",
                headers=HEADERS,
                params={"select": "id,nombre", "prioridad": "eq.1", "order": "nombre"}
            )
            
            estilos_data = estilos_response.json()
            
            # Get images data
            images_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/image_uploads",
                headers=HEADERS,
                params={"select": "estilo_id,color_id,public_url,file_name,description"}
            )
            
            # Count images by estilo_id
            image_counts = {}
            color_counts = {}
            sample_images = {}
            
            if images_response.status_code == 200:
                images_data = images_response.json()
                for image in images_data:
                    estilo_id = image.get('estilo_id')
                    color_id = image.get('color_id')
                    
                    if estilo_id is not None:
                        image_counts[estilo_id] = image_counts.get(estilo_id, 0) + 1
                        
                        if estilo_id not in color_counts:
                            color_counts[estilo_id] = set()
                        if color_id is not None:
                            color_counts[estilo_id].add(color_id)
                        
                        if estilo_id not in sample_images and image.get('public_url'):
                            sample_images[estilo_id] = {
                                "public_url": image.get('public_url'),
                                "file_name": image.get('file_name'),
                                "description": image.get('description', '')
                            }
            
            # Build response with individual sales queries for each estilo
            estilos = []
            for estilo in estilos_data:
                estilo_id = int(estilo['id'])
                
                # Get sales for THIS specific estilo
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
                    sales_data = sales_response.json()
                    sales_total = sum(record.get('qty', 0) for record in sales_data)
                
                estilos.append({
                    "id": estilo_id,
                    "nombre": estilo['nombre'],
                    "total_images": image_counts.get(estilo_id, 0),
                    "total_colors_with_images": len(color_counts.get(estilo_id, set())),
                    "sales_last_3_months": sales_total,
                    "sample_image": sample_images.get(estilo_id, None)
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
            
            # Get available colors for this estilo (from inventory with terex1 > 0)
            inventory_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/inventario1",
                headers=HEADERS,
                params={"select": "color_id", "estilo_id": f"eq.{estilo_id}", "terex1": "gt.0"}
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
    return templates.TemplateResponse("nota.html", {"request": request})





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
    """Enhanced save function that processes loyalty deductions"""
    if not payload.products:
        raise HTTPException(status_code=400, detail="No products provided")

    next_order_id = await get_next_order_id()
    mexico_tz = pytz.timezone("America/Mexico_City")
    now = datetime.now(mexico_tz)
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")
    
    items_for_ticket = []
    loyalty_deductions = []  # Track loyalty deductions
    
    for p in payload.products:
        # Check if this is a loyalty barcode (starts with 8000)
        if p.codigo.startswith('8000') and len(p.codigo) == 13:
            # Process loyalty deduction
            loyalty_result = await process_loyalty_deduction(p, next_order_id, fecha, hora)
            loyalty_deductions.append(loyalty_result)
            
            # Add to ticket items
            items_for_ticket.append({
                "qty": p.qty,
                "name": p.name,
                "price": p.price,
                "subtotal": p.qty * p.price
            })
            continue
        
        # Regular product processing
        inv_rows = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "modelo,modelo_id,estilo,estilo_id,terex1",
                "barcode": f"eq.{p.codigo}",
                "limit": "1",
            },
        )
        
        if not inv_rows:
            raise HTTPException(
                status_code=400, 
                detail=f"Producto con barcode {p.codigo} no existe en inventario1"
            )
        
        inv = inv_rows[0]

        # Insert into ventas_terex1
        record = {
            "qty": p.qty,
            "name": p.name,
            "name_id": p.codigo,
            "price": p.price,
            "fecha": fecha,
            "hora": hora,
            "order_id": next_order_id,
            "modelo": inv.get("modelo", ""),
            "modelo_id": inv.get("modelo_id", ""),
            "estilo": inv.get("estilo", ""),
            "estilo_id": inv.get("estilo_id", ""),
        }

        await supabase_request(
            method="POST",
            endpoint="/rest/v1/ventas_terex1",
            json_data=record,
        )

        # Update inventory
        current_qty = int(inv.get("terex1") or 0)
        new_qty = current_qty - p.qty
        
        await supabase_request(
            method="PATCH",
            endpoint=f"/rest/v1/inventario1?barcode=eq.{p.codigo}",
            json_data={"terex1": new_qty},
        )

        # Add to ticket items
        items_for_ticket.append({
            "qty": p.qty,
            "name": p.name,
            "price": p.price,
            "subtotal": p.qty * p.price
        })

    # Calculate total
    total = sum(i["subtotal"] for i in items_for_ticket)

    # Generate redemption token and PDF
    redemption_token = generate_redemption_token()
    await store_redemption_token(next_order_id, redemption_token, total)
    
    # Use your existing QR PDF function, not the loyalty one
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
    """Updated PDF generation with QR code - now takes redemption_token as parameter"""
    width = 58 * mm
    # Calculate dynamic height based on content
    estimated_item_height = len(items) * 22  # Approximately 22 points per item
    base_height = 80 + 44 * mm  # Header + QR code space
    height = max(120 * mm, base_height + estimated_item_height)
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
        
    # Add items
    for it in items:
        if y < 40 * mm:  # More space needed for QR
            c.showPage()
            y = header()
        qty = str(it["qty"]); name = str(it["name"]); price = it["price"]; sub = it["subtotal"]
        c.drawString(margin, y, qty)
        c.drawString(margin + 20, y, (name[:28] + ("…" if len(name) > 28 else "")))
        c.drawRightString(width - margin, y, f"${price:0.2f}"); y -= 10
        c.drawRightString(width - margin, y, f"Subtotal: ${sub:0.2f}"); y -= 12
    
    y -= 4; c.line(margin, y, width - margin, y); y -= 12
    c.setFont("Helvetica-Bold", 9); c.drawString(margin, y, "TOTAL:"); c.drawRightString(width - margin, y, f"${total:0.2f}"); y -= 14
        
    # Add loyalty program message
    c.setFont("Helvetica", 7)
    c.drawCentredString(width/2, y, "¡Obtén 1% de recompensa!")
    y -= 8
    c.drawCentredString(width/2, y, "Escanea el código QR:")
    y -= 12
    
    # Create QR Code (doubled size: from 22mm to 44mm)
    try:
        # URL that points to your redeem page with the token
        qr_url = f"https://teresalocal352.com/redeem.html?token={redemption_token}"
                
        # Generate QR code using reportlab - doubled size
        qr_widget = QrCodeWidget(qr_url)
        qr_widget.barWidth = 44 * mm
        qr_widget.barHeight = 44 * mm
                
        # Create drawing and add QR code
        qr_drawing = Drawing(44 * mm, 44 * mm)
        qr_drawing.add(qr_widget)
                
        # Center the QR code
        x = (width - 44 * mm) / 2
        y_qr = max(margin, y - 44 * mm)  # Reduced margin at bottom
                
        # Render QR code on PDF
        renderPDF.draw(qr_drawing, c, x, y_qr)
                
        print(f"DEBUG: Generated QR code with URL: {qr_url}")
            
    except Exception as e:
        print(f"Error generating QR code: {e}")
        # Fallback text
        c.setFont("Helvetica", 6)
        c.drawCentredString(width/2, y - 10, f"Token: {redemption_token[:16]}...")
    
    # Removed the "¡Gracias por su compra!" line
    
    c.showPage(); c.save(); buf.seek(0); return buf


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
    return templates.TemplateResponse("redeem.html", {"request": request})


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
    return templates.TemplateResponse("dashboard.html", {"request": request})


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
    return templates.TemplateResponse("debug.html", {"request": request})


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
                "verdetalleestilos.html",
                {
                    "request": request,
                    "step": "select_estilo",
                    "inventario_data": [],
                    "error_message": f"Error al cargar estilos ({resp.status_code}).",
                },
                status_code=resp.status_code,
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
            "verdetalleestilos.html",
            {
                "request": request,
                "step": "select_estilo",  # This tells the template which section to show
                "inventario_data": inventario_data,  # Changed from 'estilos' to 'inventario_data'
                "error_message": None,
            },
        )

    except Exception as e:
        print(f"[ver_detalle_estilos] Unexpected error: {e}", flush=True)
        return templates.TemplateResponse(
            "verdetalleestilos.html",
            {
                "request": request,
                "step": "select_estilo",
                "inventario_data": [],
                "error_message": "Ocurrió un error al cargar el menú de estilos.",
            },
            status_code=500,
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
            "verdetalleestilos.html",
            {
                "request": request,
                "step": "analytics",
                "estilo": estilo,
                "available_modelos": available_modelos,
                "selected_modelos": selected_modelos,
                "start_date": start_date,
                "end_date": end_date,
                "sort_order": sort_order,
                "analytics_data": analytics_data,
                "chart_data": json.dumps(chart_data),
                "error_message": None,
            },
        )
        
    except Exception as e:
        print(f"[Analytics] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            "verdetalleestilos.html",
            {
                "request": request,
                "step": "analytics", 
                "estilo": estilo,
                "available_modelos": [],
                "selected_modelos": [],
                "start_date": start_date,
                "end_date": end_date,
                "sort_order": sort_order,
                "analytics_data": {"summary_data": [], "daily_data": []},
                "chart_data": "{}",
                "error_message": "Error al cargar el análisis de ventas.",
            },
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
            "flores3_analytics.html",
            {
                "request": request,
                "analytics_data": {"summary_data": [], "daily_data": []},
                "estilo": "FUN FLORES 3",
                "error_message": "Error al cargar el análisis de ventas."
            },
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
                    "analytics_dynamic.html",
                    {
                        "request": request,
                        "step": "select_estilo",
                        "inventario_data": [],
                        "error_message": f"Error al cargar estilos ({resp.status_code}).",
                    },
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
                "analytics_dynamic.html",
                {
                    "request": request,
                    "step": "select_estilo",
                    "inventario_data": inventario_data,
                    "error_message": None,
                },
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
                "analytics_dynamic.html",
                {
                    "request": request,
                    "step": "analytics",
                    "analytics_data": analytics_data,
                    "estilo": estilo,
                    "start_date": start_date,
                    "end_date": end_date
                },
            )
        
    except Exception as e:
        print(f"[Analytics] Error: {e}", flush=True)
        return templates.TemplateResponse(
            "analytics_dynamic.html",
            {
                "request": request,
                "step": "analytics" if estilo else "select_estilo",
                "analytics_data": {"summary_data": [], "daily_data": []},
                "estilo": estilo or "",
                "error_message": "Error al cargar el análisis de ventas.",
                "inventario_data": [] if not estilo else None
            },
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
                    "travel_analytics.html",
                    {
                        "request": request,
                        "step": "select_lugar",
                        "lugares_data": [],
                        "error_message": f"Error al cargar lugares ({resp.status_code}).",
                    },
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
                "travel_analytics.html",
                {
                    "request": request,
                    "step": "select_lugar",
                    "lugares_data": lugares_data,
                    "error_message": None,
                },
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
                "travel_analytics.html",
                {
                    "request": request,
                    "step": "analytics",
                    "analytics_data": analytics_data,
                    "lugar": lugar,
                    "start_date": start_date,
                    "end_date": end_date
                },
            )
        
    except Exception as e:
        print(f"[Travel Analytics] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            "travel_analytics.html",
            {
                "request": request,
                "step": "analytics" if lugar else "select_lugar",
                "analytics_data": {"summary_data": [], "daily_data": []},
                "lugar": lugar or "",
                "error_message": "Error al cargar el análisis de ventas travel.",
                "lugares_data": [] if not lugar else None
            },
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
                    "modelosanuales.html",
                    {
                        "request": request,
                        "year": year,
                        "chart_data": {"labels": [], "datasets": []},
                        "summary_data": [],
                        "monthly_details": [],
                        "available_years": ["2022", "2023", "2024", "2025"],
                        "error_message": f"No se encontraron datos para el año {year}.",
                    },
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
                "modelosanuales.html",
                {
                    "request": request,
                    "year": year,
                    "chart_data": {"labels": [], "datasets": []},
                    "summary_data": [],
                    "monthly_details": [],
                    "available_years": ["2022", "2023", "2024", "2025"],
                    "error_message": f"No se encontraron ventas para el año {year}.",
                },
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
            "modelosanuales.html",
            {
                "request": request,
                "year": year,
                "chart_data": chart_data,
                "summary_data": summary_data,
                "monthly_details": monthly_details,
                "monthly_lookup": monthly_lookup,  # Add this for easier template lookup
                "available_years": ["2022", "2023", "2024", "2025"],
                "error_message": None,
            },
        )
        
    except Exception as e:
        print(f"[Yearly Models] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            "modelosanuales.html",
            {
                "request": request,
                "year": year,
                "chart_data": {"labels": [], "datasets": []},
                "summary_data": [],
                "monthly_details": [],
                "available_years": ["2022", "2023", "2024", "2025"],
                "error_message": f"Error al cargar los datos: {str(e)}",
            },
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
    return templates.TemplateResponse("metertelefonos.html", {"request": request})

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)