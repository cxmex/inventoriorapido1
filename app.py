# app.py - FastAPI with templates for inventory management

from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse  # Optional for debugging
from collections import defaultdict
import httpx
from datetime import datetime, timedelta
import os
import shutil
import traceback
import os
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

# Supabase client headers
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Data models
class InventarioEstilo(BaseModel):
    id: int
    nombre: str

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
                "fecha": datetime.now().isoformat()  # Add timestamp
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
            
            next_order_id = 1  # Default if no records exist
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)