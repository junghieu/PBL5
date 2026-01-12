from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from datetime import datetime

from app.database import init_db, create_scan_record, update_scan_status, get_scan_by_id
from app.services.image import preprocess_image, validate_image
from app.services.ocr import extract_text_from_image
from app.services.docx import create_docx_from_text

# Initialize FastAPI app
app = FastAPI(title="OCR Service API", version="1.0.0")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()


def process_ocr_task(scan_id: int, image_path: str):
    """
    Background task to process OCR workflow:
    1. Preprocess image
    2. Extract text using OCR
    3. Create DOCX file
    4. Update database status
    """
    try:
        # Step 1: Preprocess the image
        preprocessed_path = preprocess_image(image_path)
        
        # Step 2: Extract text using OCR
        text_content = extract_text_from_image(preprocessed_path)
        
        # Step 3: Create DOCX file
        docx_filename = f"scan_{scan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        docx_path = os.path.join("outputs", docx_filename)
        create_docx_from_text(text_content, docx_path)
        
        # Step 4: Update database with completed status
        update_scan_status(scan_id, "Completed", text_content, docx_path)
        
        # Clean up preprocessed image
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
            
    except Exception as e:
        # Update database with error status
        update_scan_status(scan_id, f"Failed: {str(e)}")


@app.post("/scan")
async def scan_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    POST endpoint to upload an image and start OCR processing.
    
    Returns the scan ID for status tracking.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"upload_{timestamp}{file_extension}"
    file_path = os.path.join("uploads", filename)
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate the image
    if not validate_image(file_path):
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Create database record with Pending status
    scan_id = create_scan_record(filename)
    
    # Add background task for OCR processing
    background_tasks.add_task(process_ocr_task, scan_id, file_path)
    
    return JSONResponse(
        content={"scan_id": scan_id, "status": "Pending", "message": "Image uploaded successfully"},
        status_code=201
    )


@app.get("/status/{scan_id}")
async def get_scan_status(scan_id: int):
    """
    GET endpoint to check the processing status of a scan.
    
    Returns scan details including status, text content, and docx path.
    """
    scan = get_scan_by_id(scan_id)
    
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    response = {
        "id": scan["id"],
        "filename": scan["filename"],
        "status": scan["status"],
        "created_at": scan["created_at"],
        "updated_at": scan["updated_at"]
    }
    
    # Include text content and docx path if processing is complete
    if scan["status"] == "Completed":
        response["text_content"] = scan["text_content"]
        response["docx_path"] = scan["docx_path"]
    
    return response


@app.get("/download/{scan_id}")
async def download_docx(scan_id: int):
    """
    GET endpoint to download the generated DOCX file.
    """
    scan = get_scan_by_id(scan_id)
    
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan["status"] != "Completed":
        raise HTTPException(status_code=400, detail="Scan not yet completed")
    
    docx_path = scan["docx_path"]
    
    if not os.path.exists(docx_path):
        raise HTTPException(status_code=404, detail="DOCX file not found")
    
    return FileResponse(
        docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=os.path.basename(docx_path)
    )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main HTML page.
    """
    html_file = "static/index.html"
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>OCR Service</h1><p>Frontend not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
