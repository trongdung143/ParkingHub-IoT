from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from datetime import datetime
import uuid
from typing import Optional
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import traceback #debug 
# Load environment variables
load_dotenv()

app = FastAPI(title="IoT Smart Parking API", version="1.0.0")
model = YOLO('model_ckpt/license-plate-finetune-v1x.pt')

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("DunnBC22/trocr-base-printed_license_plates_ocr")

device = "cuda" if torch.cuda.is_available() else "cpu"
ocr_model = ocr_model.to(device)

if device == "cuda":
    ocr_model = ocr_model.half()

ocr_model.eval()


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
async def root():
    return {"message": "IoT Smart Parking API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/esp32/in-upload")
async def esp32_upload(
    rfid_id: str = Form(...),
    image: UploadFile = File(...),
    parking_slot: Optional[str] = Form(None),
):
    """
    Endpoint for ESP32-CAM to upload RFID ID and license plate image.
    
    Args:
        rfid_id: RFID card ID as string
        image: JPEG image file
    
    Returns:
        Plain text rfid_id or JSON with rfid_id
    """
    try:
        # Validate image type
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        save_path = DATA_DIR / f"{rfid_id}.jpg"
        img.save(save_path, format="JPEG", quality=95, optimize=True)

        print("Saved image to:", save_path)
        print("Current working dir:", os.getcwd())

        # detect license plate text using 
        results = model.predict(source=str(save_path), verbose=False)

        plate_text = None
        detections_json = results[0].to_json()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            conf = boxes.conf.detach().cpu().numpy()
            best_i = int(np.argmax(conf))

            x1, y1, x2, y2 = boxes.xyxy[best_i].detach().cpu().tolist()
            w, h = img.size

            pad = 0.3
            dx = int((x2 - x1) * pad)
            dy = int((y2 - y1) * pad)

            x1 = int(max(0, x1 - dx))
            y1 = int(max(0, y1 - dy))
            x2 = int(min(w, x2 + dx))
            y2 = int(min(h, y2 + dy))

            if x2 > x1 and y2 > y1:
                plate_crop = img.crop((x1, y1, x2, y2))

                pixel_values = ocr_processor(images=plate_crop, return_tensors="pt").pixel_values.to(device)
                if device == "cuda":
                    pixel_values = pixel_values.half()

                with torch.inference_mode():
                    generated_ids = ocr_model.generate(pixel_values, max_new_tokens=16, num_beams=5)
                plate_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                plate_text = plate_text.strip().replace(" ", "")

        # Generate unique filename
        original_name = image.filename or "image.jpg"
        file_extension = original_name.split(".")[-1] if "." in original_name else "jpg"
        filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"parking/{filename}"
        
        # Upload image to Supabase Storage
        try:
            # Try to remove existing file first (if any) to avoid conflicts
            try:
                supabase.storage.from_(SUPABASE_STORAGE_BUCKET).remove([file_path])
            except:
                pass  # File doesn't exist, which is fine
            
            storage_response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
                file_path,
                image_bytes,
                file_options={"content-type": image.content_type}
            )
        except Exception as storage_error:
            raise HTTPException(status_code=500, detail=f"Failed to upload image to storage: {str(storage_error)}")
        
        # Get public URL for the image
        try:
            public_url_response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(file_path)
            image_path = public_url_response if isinstance(public_url_response, str) else file_path
        except:
            # Fallback to relative path if public URL fails
            image_path = file_path

        # Suggest available parking slots based on current database state
        suggested_slot = None
        available_slots = []

        try:
            # 1) Get all active slots from parking_slots table
            slots_resp = supabase.table("parking_slots")\
                .select("slot_name, is_active")\
                .eq("is_active", True)\
                .execute()

            all_slots = [row["slot_name"] for row in (slots_resp.data or []) if row.get("slot_name")]

            # 2) Scan all events chronologically to determine which slots are currently occupied
            events_resp = supabase.table("parking_events")\
                .select("rfid_id, parking_slot, event_type, created_at")\
                .order("created_at", desc=False)\
                .execute()

            rfid_state = {}  # rfid_id -> {"slot": str, "status": "IN" | "OUT"}
            for ev in events_resp.data or []:
                slot = ev.get("parking_slot")
                if not slot or slot == "N/A":
                    continue

                rfid = ev.get("rfid_id")
                if not rfid:
                    continue

                event_type = ev.get("event_type") or "IN"

                if event_type == "IN":
                    # Last IN wins; this RFID is now occupying this slot
                    rfid_state[rfid] = {"slot": slot, "status": "IN"}
                elif event_type == "OUT":
                    # Mark RFID as OUT (no longer occupying its last slot)
                    if rfid in rfid_state:
                        rfid_state[rfid]["status"] = "OUT"

            # 3) Derive occupied and available slots
            occupied_slots = {
                state["slot"]
                for state in rfid_state.values()
                if state.get("status") == "IN" and state.get("slot")
            }

            available_slots = [slot for slot in all_slots if slot not in occupied_slots]
            suggested_slot = available_slots[0] if available_slots else None
        except Exception as slot_error:
            # Fail gracefully if slot suggestion logic fails
            print(f"Failed to compute available slots: {slot_error}")
            suggested_slot = None
            available_slots = []

        # Decide which slot to save for this event:
        #  - if client sent parking_slot, use it
        #  - otherwise, use suggested_slot (first available)
        slot_to_save = parking_slot or suggested_slot

        # Insert record into parking_events table
        event_data = {
            "id": str(uuid.uuid4()),
            "rfid_id": rfid_id,
            "image_path": image_path,
            "created_at": datetime.utcnow().isoformat(),
            "parking_slot": slot_to_save,
            "license_plate": plate_text,
        }
        
        db_response = supabase.table("parking_events").insert(event_data).execute()
        
        if not db_response.data:
            raise HTTPException(status_code=500, detail="Failed to insert parking event")
        
        # Return rfid_id and slot suggestions as JSON
        return {
            "rfid_id": rfid_id,
            "license_plate": plate_text,
            "suggested_slot": suggested_slot,
            "available_slots": available_slots,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()  #debug 
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/parking-events")
async def get_parking_events(limit: int = 100, offset: int = 0):
    """
    Get parking events for the admin dashboard.
    """
    try:
        # Supabase Python client uses .range() for pagination
        response = supabase.table("parking_events")\
            .select("*")\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        return {"events": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parking events: {str(e)}")


@app.get("/api/v1/parking-sessions")
async def get_parking_sessions(limit: int = 100, offset: int = 0):
    """
    Get parking sessions with Time IN and Time OUT calculated.
    Groups events by RFID and calculates parking sessions.
    For each RFID, a new IN event closes the previous session and starts a new one.
    """
    try:
        # Get more events to ensure proper session calculation
        # We need enough events to pair IN/OUT properly
        fetch_limit = max(limit * 3, 500)
        response = supabase.table("parking_events")\
            .select("*")\
            .order("created_at", desc=False)\
            .limit(fetch_limit)\
            .execute()
        
        events = response.data
        
        # Group events by RFID and calculate sessions
        sessions = []
        rfid_sessions = {}  # Track last IN event for each RFID
        
        # Process events in chronological order
        for event in events:
            rfid_id = event['rfid_id']
            event_type = event.get('event_type')
            created_at = event['created_at']
            
            # Default to 'IN' if event_type is not set
            if event_type is None:
                event_type = 'IN'
            
            if event_type == 'IN':
                # Check if there's a previous open session for this RFID
                if rfid_id in rfid_sessions:
                    # Close the previous session (time_out = time of this new IN event)
                    prev_session = rfid_sessions[rfid_id]
                    prev_session['time_out'] = created_at
                    sessions.append(prev_session)
                
                # Start new session
                session = {
                    'id': event['id'],
                    'card_id': rfid_id,
                    'license_plate': event.get('license_plate') or 'N/A',
                    'time_in': created_at,
                    'time_out': None,  # Will be set when next IN event or OUT event occurs
                    'parking_slot': event.get('parking_slot') or 'N/A',
                    'image_path': event.get('image_path', '')
                }
                rfid_sessions[rfid_id] = session
            elif event_type == 'OUT':
                # Close the current session for this RFID
                if rfid_id in rfid_sessions:
                    rfid_sessions[rfid_id]['time_out'] = created_at
                    sessions.append(rfid_sessions[rfid_id])
                    del rfid_sessions[rfid_id]
        
        # Add any remaining open sessions (still parked)
        for session in rfid_sessions.values():
            sessions.append(session)
        
        # Sort by time_in descending (most recent first) and apply pagination
        sessions.sort(key=lambda x: x['time_in'], reverse=True)
        paginated_sessions = sessions[offset:offset + limit]
        
        return {"sessions": paginated_sessions, "count": len(paginated_sessions), "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parking sessions: {str(e)}")


@app.get("/api/v1/parking-events/{rfid_id}")
async def get_parking_events_by_rfid(rfid_id: str):
    """
    Get parking events for a specific RFID ID.
    """
    try:
        response = supabase.table("parking_events")\
            .select("*")\
            .eq("rfid_id", rfid_id)\
            .order("created_at", desc=True)\
            .execute()
        
        return {"events": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parking events: {str(e)}")


@app.get("/api/v1/parking-slots")
async def get_parking_slots():
    """
    Get all parking slots from the parking_slots table.
    """
    try:
        response = supabase.table("parking_slots")\
            .select("*")\
            .order("row_letter", desc=False)\
            .order("slot_number", desc=False)\
            .execute()
        
        return {"slots": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parking slots: {str(e)}")


@app.post("/api/v1/parking-slots")
async def create_parking_slot(
    slot_name: str,
    row_letter: str,
    slot_number: int,
    is_active: bool = True
):
    """
    Create a new parking slot.
    """
    try:
        slot_data = {
            "slot_name": slot_name,
            "row_letter": row_letter,
            "slot_number": slot_number,
            "is_active": is_active
        }
        
        response = supabase.table("parking_slots").insert(slot_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create parking slot")
        
        return {"slot": response.data[0], "message": "Parking slot created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create parking slot: {str(e)}")


@app.put("/api/v1/parking-slots/{slot_id}")
async def update_parking_slot(
    slot_id: str,
    slot_name: Optional[str] = None,
    row_letter: Optional[str] = None,
    slot_number: Optional[int] = None,
    is_active: Optional[bool] = None
):
    """
    Update a parking slot.
    """
    try:
        update_data = {}
        if slot_name is not None:
            update_data["slot_name"] = slot_name
        if row_letter is not None:
            update_data["row_letter"] = row_letter
        if slot_number is not None:
            update_data["slot_number"] = slot_number
        if is_active is not None:
            update_data["is_active"] = is_active
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        response = supabase.table("parking_slots")\
            .update(update_data)\
            .eq("id", slot_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Parking slot not found")
        
        return {"slot": response.data[0], "message": "Parking slot updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update parking slot: {str(e)}")


@app.delete("/api/v1/parking-slots/{slot_id}")
async def delete_parking_slot(slot_id: str):
    """
    Delete a parking slot (soft delete by setting is_active to False).
    """
    try:
        response = supabase.table("parking_slots")\
            .update({"is_active": False, "updated_at": datetime.utcnow().isoformat()})\
            .eq("id", slot_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Parking slot not found")
        
        return {"message": "Parking slot deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete parking slot: {str(e)}")


@app.get("/api/v1/parking-slots/status")
async def get_parking_slots_status():
    """
    Get current status of all parking slots (available/occupied).
    Returns a map of slot names to their current status.
    """
    try:
        # Get all active parking sessions (where time_out is null)
        # We'll use the parking-sessions endpoint logic to find active sessions
        fetch_limit = 1000
        try:
            response = supabase.table("parking_events")\
                .select("*")\
                .order("created_at", desc=False)\
                .limit(fetch_limit)\
                .execute()
            
            events = response.data if response.data else []
        except Exception as db_error:
            # If database query fails, return empty slots
            events = []
            print(f"Database query error: {db_error}")
        
        # Track active sessions by parking slot
        slot_status = {}  # {slot_name: {occupied: bool, rfid_id: str, license_plate: str, time_in: str}}
        rfid_sessions = {}  # Track last IN event for each RFID
        
        # Process events in chronological order to determine current occupancy
        for event in events:
            rfid_id = event['rfid_id']
            event_type = event.get('event_type')
            parking_slot = event.get('parking_slot')
            created_at = event['created_at']
            
            if event_type is None:
                event_type = 'IN'
            
            if event_type == 'IN':
                # If there's a previous session for this RFID, mark its slot as available
                if rfid_id in rfid_sessions:
                    prev_slot = rfid_sessions[rfid_id].get('parking_slot')
                    if prev_slot and prev_slot != 'N/A':
                        if prev_slot in slot_status:
                            del slot_status[prev_slot]
                
                # Mark new slot as occupied
                if parking_slot and parking_slot != 'N/A':
                    slot_status[parking_slot] = {
                        'occupied': True,
                        'rfid_id': rfid_id,
                        'license_plate': event.get('license_plate') or 'N/A',
                        'time_in': created_at
                    }
                
                rfid_sessions[rfid_id] = {
                    'parking_slot': parking_slot,
                    'time_in': created_at
                }
            elif event_type == 'OUT':
                # Mark slot as available
                if rfid_id in rfid_sessions:
                    prev_slot = rfid_sessions[rfid_id].get('parking_slot')
                    if prev_slot and prev_slot != 'N/A':
                        if prev_slot in slot_status:
                            del slot_status[prev_slot]
                    del rfid_sessions[rfid_id]
        
        # Get all parking slots from parking_slots table
        try:
            slots_response = supabase.table("parking_slots")\
                .select("*")\
                .eq("is_active", True)\
                .order("row_letter", desc=False)\
                .order("slot_number", desc=False)\
                .execute()
            
            all_slots = []
            if slots_response.data:
                for slot in slots_response.data:
                    all_slots.append(slot['slot_name'])
        except Exception as slots_error:
            # Fallback to default slots if parking_slots table doesn't exist or query fails
            print(f"Error fetching parking slots: {slots_error}, using default slots")
            rows = ['A', 'B', 'C', 'D', 'E']
            slots_per_row = 5
            for row in rows:
                for slot_num in range(1, slots_per_row + 1):
                    all_slots.append(f"{row}{slot_num}")
        
        # Create complete slot status map with all slots from database
        slot_map = {}
        for slot in all_slots:
            if slot in slot_status:
                slot_map[slot] = slot_status[slot]
            else:
                slot_map[slot] = {
                    'occupied': False,
                    'rfid_id': None,
                    'license_plate': None,
                    'time_in': None
                }
        
        return {
            "slots": slot_map,
            "total_slots": len(slot_map),
            "occupied_slots": sum(1 for s in slot_map.values() if s['occupied']),
            "available_slots": sum(1 for s in slot_map.values() if not s['occupied'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parking slot status: {str(e)}")

