from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import base64, uuid, shutil, os, json, time
from pathlib import Path
import socket
from collections import Counter
import threading

import cv2
import numpy as np
import io

from pipeline.layout import run_layout_mem, load_layout_model
from pipeline.preprocess import preprocess_for_layout, preprocess_for_ocr
from pipeline.ocr import run_ocr_mem, load_ocr_model
from pipeline.export_word import build_docx_mem

app = FastAPI(title="DocScan AI API")
BASE_DIR = Path(__file__).resolve().parent

# Khởi tạo render HTML (Dùng đường dẫn tuyệt đối chống lỗi)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── Load model 1 lần khi server khởi động ──
print("[INIT] Đang tải model YOLO...")
layout_model = load_layout_model()
print("[INIT] Đang tải model VietOCR...")
ocr_model = load_ocr_model()
print("[INIT] Sẵn sàng!")

# 2. KHỞI TẠO KHÓA TOÀN CỤC CHO AI
ai_lock = threading.Lock()

# ── Thư mục dữ liệu ──
OUTPUT_FOLDER = BASE_DIR / "data" / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """Trả về trang Web UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get('/health')
async def health():
    """Endpoint để frontend kiểm tra server có online không."""
    return JSONResponse(content={"status": "ok"})


@app.post('/process')
def process(image: UploadFile = File(...)):
    """[SYNC] Hàm chạy AI nặng. Viết bằng `def` để FastAPI ném vào Threadpool."""
    try:
        started_at = time.perf_counter()
        timings = {}
        job_id = uuid.uuid4().hex

        # 1. ĐỌC ẢNH VÀO RAM
        t0 = time.perf_counter()
        file_bytes = image.file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Invalid image format"})
        timings["decode_image_sec"] = round(time.perf_counter() - t0, 4)

        # 2. TIỀN XỬ LÝ
        t0 = time.perf_counter()
        img_layout = preprocess_for_layout(img_bgr.copy())
        img_ocr = preprocess_for_ocr(img_bgr.copy())
        timings["preprocess_sec"] = round(time.perf_counter() - t0, 4)

        # <-- 3. BỌC LOGIC AI VÀO TRONG LOCK -->
        with ai_lock:
            # 3. YOLOV10 LAYOUT
            t0 = time.perf_counter()
            boxes_list, annotated_img = run_layout_mem(
                img_layout,
                model=layout_model,
                conf=0.1,
                debug=True
            )
            timings["layout_sec"] = round(time.perf_counter() - t0, 4)
            
            # 4. VIETOCR
            t0 = time.perf_counter()
            ocr_res_raw = run_ocr_mem(img_ocr, boxes_list, predictor=ocr_model)
            ocr_res = {
                "image": image.filename,
                "content": ocr_res_raw["content"],
                "content_with_labels": ocr_res_raw["content_with_labels"]
            }
            timings["ocr_sec"] = round(time.perf_counter() - t0, 4)

        # 5. XUẤT WORD RA RAM
        t0 = time.perf_counter()
        doc_stream = build_docx_mem([ocr_res])
        timings["export_docx_sec"] = round(time.perf_counter() - t0, 4)

        # 6. ENCODE TRỰC TIẾP RA BASE64
        doc_b64 = base64.b64encode(doc_stream.read()).decode()
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        layout_img_b64 = base64.b64encode(buffer).decode()
        layout_img_uri = f"data:image/jpeg;base64,{layout_img_b64}"

        return JSONResponse(content={
            "status": "success",
            "data": {
                "layout": [{"image": "capture.jpg", "boxes": boxes_list}],
                "ocr": [{"image": "capture.jpg", "content": ocr_res["content"], "content_with_labels": ocr_res["content_with_labels"]}],
                "docx_filename": f"ket_qua_{job_id[:6]}.docx",
                "docx_base64": doc_b64,
                "job_id": job_id,
                "layout_image_url": layout_img_uri,
            }
        })
    
    except Exception as e:
        print(f"❌ Lỗi xử lý: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# Biến toàn cục để giữ tạm ảnh gốc từ Pi
pi_raw_image_buffer = None

# Thay vì dùng biến đơn, ta dùng Dictionary lưu ảnh theo ID
pi_buffers = {}

@app.post('/api/pi_upload')
async def pi_upload(image: UploadFile = File(...), device_id: str = "pi_01"):
    """[ASYNC] Pi gửi ảnh kèm theo device_id của nó"""
    pi_buffers[device_id] = await image.read()
    return JSONResponse(content={"status": "success", "device": device_id})

@app.get('/api/pi_check')
async def pi_check(device_id: str = "pi_01"):
    """[ASYNC] Web kiểm tra xem device_id này có ảnh mới không"""
    if device_id in pi_buffers and pi_buffers[device_id] is not None:
        # Lấy ảnh ra và xóa khỏi buffer
        img_bytes = pi_buffers.pop(device_id) 
        img_b64 = base64.b64encode(img_bytes).decode()
        return JSONResponse(content={"has_new": True, "image_b64": img_b64})
    
    return JSONResponse(content={"has_new": False})

def find_free_port(start_port=5000):
    """Tìm port khả dụng."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.listen(1)
                return port
        except OSError:
            continue
    return start_port

if __name__ == '__main__':
    port = find_free_port(5000)
    # Khởi động bằng Uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)