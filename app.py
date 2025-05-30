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
        
        # Fetch inventory items for the style
        inventory_response = await supabase_request(
            method="GET",
            endpoint="/rest/v1/inventario1",
            params={
                "select": "barcode,name,terex1,marca,estilo_id",
                "estilo_id": f"eq.{estilo_id}"
            }
        )
        
        # Get available brands for filtering
        brands = set()
        for item in inventory_response:
            if item.get('marca'):
                brands.add(item.get('marca').upper())
        
        # Sort brands alphabetically
        sorted_brands = sorted(list(brands))
        
        return templates.TemplateResponse(
            "inventory_detail.html", 
            {
                "request": request, 
                "inventory_items": inventory_response,
                "estilo_id": estilo_id,
                "estilo_nombre": style_name,
                "brands": sorted_brands
            }
        )
    except Exception as e:
        print(f"Error loading inventory: {str(e)}", flush=True)
        traceback.print_exc()  # Print stack trace for more details
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request, 
                "error_message": f"Error loading inventory: {str(e)}"
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
        
        print(f"Current value for {barcode}: {old_value}", flush=True)
        
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


@app.get("/verventasxsemana", response_class=HTMLResponse)
async def ver_ventas_por_semana(request: Request):
    try:
        import traceback
        
        print(f"Fetching weekly sales data for the last 15 weeks", flush=True)
        
        try:
            # Call the Supabase function we created
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/rpc/get_weekly_sales",
                    headers=HEADERS,
                    params={"weeks_back": 15}  # Optional - since the default is 15
                )
                
                if response.status_code >= 400:
                    print(f"Function call error: {response.text}", flush=True)
                    raise Exception(f"Function call failed: {response.status_code}, {response.text}")
                
                weekly_results = response.json()
                print(f"Retrieved sales data for {len(weekly_results)} weeks", flush=True)
            
            # Process the results
            week_totals = {}
            chart_data = []
            
            for row in weekly_results:
                week_key = row.get('week_start')
                total = float(row.get('total_revenue', 0) or 0)
                
                week_totals[week_key] = total
                
                # Simplify date format for chart (day/month only)
                simple_date = datetime.strptime(week_key, "%d/%m/%Y").strftime("%d/%m")
                chart_data.append({
                    "week": simple_date,
                    "total": round(total, 2)
                })
            
            print(f"Weekly totals: {week_totals}", flush=True)
            
            return templates.TemplateResponse(
                "ventas_por_semana.html",
                {
                    "request": request,
                    "week_totals": week_totals,
                    "chart_data": chart_data
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
                    "chart_data": []
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)