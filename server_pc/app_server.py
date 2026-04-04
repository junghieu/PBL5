from fastapi import FastAPI, File, UploadFile, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import base64
import uuid
import time
import json
import socket
from pathlib import Path
from typing import List

import cv2
import numpy as np
from cachetools import TTLCache

from pipeline.layout import run_layout_mem, load_layout_model
from pipeline.preprocess import preprocess_base, preprocess_for_layout, preprocess_for_ocr
from pipeline.ocr import run_ocr_mem, load_ocr_model
from pipeline.export_word import build_docx_mem

#  CẤU HÌNH
BASE_DIR       = Path(__file__).resolve().parent
OUTPUT_FOLDER  = BASE_DIR / "data" / "output"
MAX_QUEUE_SIZE = 10     # Số job tối đa trong hàng đợi
JOB_TTL        = 1800   # Kết quả job sống tối đa 30 phút trong RAM
PI_BUFFER_TTL  = 300    # Buffer ảnh Pi tự xóa sau 5 phút nếu không được lấy

#  ASYNC-SAFE TTL CACHE
#  Bọc cachetools.TTLCache bằng asyncio.Lock khởi tạo Lazy (Lazy Initialization) để tránh lỗi "attached to a different loop" trên Python 3.10+
class AsyncTTLCache:
    """
    TTLCache thread-safe cho asyncio.
    Dùng kỹ thuật Lazy Initialization cho asyncio.Lock để đảm bảo Lock luôn được tạo và gắn chính xác vào Event Loop đang chạy thực sự.
    """
    def __init__(self, maxsize: int, ttl: int):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Tạo khóa an toàn ngay tại thời điểm được gọi lần đầu tiên."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def set(self, key, value):
        async with self._get_lock():
            self._cache[key] = value

    async def get(self, key, default=None):
        async with self._get_lock():
            return self._cache.get(key, default)

    async def pop(self, key, default=None):
        async with self._get_lock():
            return self._cache.pop(key, default)

    async def contains(self, key) -> bool:
        async with self._get_lock():
            return key in self._cache

    async def size(self) -> int:
        async with self._get_lock():
            return len(self._cache)

    # Cho phép dùng trực tiếp như dict trong các hàm đồng bộ (ai_worker chạy trong executor)
    # Chỉ dùng khi CHẮC CHẮN không có concurrent access từ async context
    def _sync_set(self, key, value):
        self._cache[key] = value

    def _sync_get(self, key, default=None):
        return self._cache.get(key, default)


#  BIẾN GLOBAL
layout_model  = None
ocr_model     = None
job_queue: asyncio.Queue = None

# Khởi tạo sau khi asyncio event loop đã chạy (trong lifespan)
job_results: AsyncTTLCache  = None
pi_buffers: AsyncTTLCache   = None

#  WEBSOCKET CONNECTION MANAGER
class ConnectionManager:
    """
    Quản lý tất cả Web UI đang kết nối qua WebSocket.
    - broadcast()      : Gửi binary frame video tới tất cả client
    - broadcast_text() : Gửi JSON notification tới tất cả client
    Tự động dọn sạch connection chết sau mỗi lần broadcast.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Web UI kết nối. Tổng: {len(self.active_connections)} client")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Web UI ngắt kết nối. Còn lại: {len(self.active_connections)} client")

    async def broadcast(self, message: bytes):
        """Gửi binary (video frame) cho tất cả Web UI, tự dọn connection chết."""
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_bytes(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)

    async def broadcast_text(self, message: str):
        """Gửi JSON text (notification) cho tất cả Web UI, tự dọn connection chết."""
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)

stream_manager = ConnectionManager()

#  AI PIPELINE (ĐỒNG BỘ — chạy trong ThreadPoolExecutor)
def run_ai_pipeline(img_bgr: np.ndarray) -> dict:
    """
    Toàn bộ pipeline AI chạy đồng bộ trong threadpool để không block event loop.
    Trả về dict chứa kết quả layout, OCR, file Word dạng base64 và timing.
    """
    timings = {}
    job_id  = uuid.uuid4().hex

    # Bước 1: Tiền xử lý — warp + tách 2 luồng
    t0 = time.perf_counter()
    img_warped = preprocess_base(img_bgr)
    img_layout = preprocess_for_layout(img_warped.copy())
    img_ocr    = preprocess_for_ocr(img_warped.copy())
    timings["preprocess_sec"] = round(time.perf_counter() - t0, 4)

    # Bước 2: Layout detection (YOLO)
    t0 = time.perf_counter()
    boxes_list, annotated_img = run_layout_mem(img_layout, model=layout_model, conf=0.1, debug=True)
    timings["layout_sec"] = round(time.perf_counter() - t0, 4)

    # Bước 3: OCR (VietOCR)
    t0 = time.perf_counter()
    ocr_res_raw = run_ocr_mem(img_ocr, boxes_list, predictor=ocr_model)
    timings["ocr_sec"] = round(time.perf_counter() - t0, 4)

    # Bước 4: Xuất file Word
    t0 = time.perf_counter()
    doc_stream = build_docx_mem([{
        "content":             ocr_res_raw["content"],
        "content_with_labels": ocr_res_raw["content_with_labels"]
    }])
    timings["export_docx_sec"] = round(time.perf_counter() - t0, 4)

    # Encode kết quả sang base64 để trả về qua JSON
    doc_b64 = base64.b64encode(doc_stream.read()).decode()
    _, buf   = cv2.imencode(".jpg", annotated_img)
    layout_img_b64 = base64.b64encode(buf).decode()

    total = sum(timings.values())
    print(f"[PIPELINE] Tổng thời gian: {total:.2f}s | "
          f"pre={timings['preprocess_sec']}s "
          f"layout={timings['layout_sec']}s "
          f"ocr={timings['ocr_sec']}s "
          f"docx={timings['export_docx_sec']}s")

    return {
        "layout":           [{"image": "capture.jpg", "boxes": boxes_list}],
        "ocr":              [{"image": "capture.jpg",
                              "content":             ocr_res_raw["content"],
                              "content_with_labels": ocr_res_raw["content_with_labels"]}],
        "docx_filename":    f"ket_qua_{job_id[:6]}.docx",
        "docx_base64":      doc_b64,
        "job_id":           job_id,
        "layout_image_url": f"data:image/jpeg;base64,{layout_img_b64}",
        "timings":          timings,
    }

#  AI WORKER (ASYNC — chạy ngầm, nhận job từ queue)
async def ai_worker():
    """
    Coroutine chạy ngầm suốt vòng đời server.
    Nhận job từ job_queue, chạy pipeline trong threadpool,
    lưu kết quả vào job_results để frontend polling lấy về.
    """
    print("[WORKER] AI Worker đã sẵn sàng nhận job!")
    while True:
        job_id, img_bgr = await job_queue.get()
        await job_results.set(job_id, {"status": "processing"})
        print(f"[WORKER] Bắt đầu xử lý job: {job_id[:8]}...")

        try:
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_ai_pipeline, img_bgr)
            await job_results.set(job_id, {"status": "done", "data": result})
            print(f"[WORKER] Hoàn thành job: {job_id[:8]}")

        except Exception as e:
            print(f"[WORKER] Lỗi job {job_id[:8]}: {e}")
            await job_results.set(job_id, {"status": "error", "error": str(e)})

        finally:
            job_queue.task_done()

#  LIFESPAN — Quản lý khởi động & tắt server
@asynccontextmanager
async def lifespan(app: FastAPI):
    global layout_model, ocr_model, job_queue, job_results, pi_buffers

    # Khởi tạo AsyncTTLCache SAU KHI event loop đã chạy
    job_results = AsyncTTLCache(maxsize=200, ttl=JOB_TTL)
    pi_buffers  = AsyncTTLCache(maxsize=20,  ttl=PI_BUFFER_TTL)

    print("[INIT] Đang tải model YOLO...")
    layout_model = load_layout_model()

    print("[INIT] Đang tải model VietOCR...")
    ocr_model = load_ocr_model()

    job_queue = asyncio.Queue()
    asyncio.create_task(ai_worker())

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print("[INIT] Server sẵn sàng phục vụ!")
    yield

    # Cleanup khi server tắt
    print("[SHUTDOWN] Đang dọn dẹp tài nguyên...")
    layout_model = None
    ocr_model    = None
    print("[SHUTDOWN] Hoàn tất.")

#  KHỞI TẠO APP
app       = FastAPI(title="DocScan AI API", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

#  ROUTES — HTTP
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Trả về trang Web UI chính."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """
    Endpoint kiểm tra sức khỏe server.
    Frontend dùng để hiển thị trạng thái online/offline.
    """
    return JSONResponse(content={
        "status":     "ok",
        "queue_size": job_queue.qsize() if job_queue else 0,
    })


@app.post("/process")
async def process(image: UploadFile = File(...)):
    """
    Nhận ảnh từ Web UI → validate → xếp hàng → trả job_id ngay lập tức.
    Frontend dùng job_id để polling /result/{job_id}.
    """
    # Kiểm tra hàng đợi trước khi nhận thêm job
    if job_queue.qsize() >= MAX_QUEUE_SIZE:
        print(f"[QUEUE] Từ chối — hàng đợi đầy ({job_queue.qsize()}/{MAX_QUEUE_SIZE})")
        return JSONResponse(
            status_code=429,
            content={"status": "error", "error": "Server đang bận, vui lòng thử lại sau."}
        )

    try:
        file_bytes = await image.read()
        nparr      = np.frombuffer(file_bytes, np.uint8)
        img_bgr    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "File không phải ảnh hợp lệ."}
            )

        job_id = uuid.uuid4().hex
        await job_results.set(job_id, {"status": "queued"})
        await job_queue.put((job_id, img_bgr))

        print(f"[QUEUE] Job {job_id[:8]} đã vào hàng. Hàng đợi: {job_queue.qsize()} job")
        return JSONResponse(content={"status": "queued", "job_id": job_id})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Frontend polling để lấy kết quả xử lý.
    Khi job hoàn thành hoặc lỗi → trả kết quả và xóa khỏi RAM.
    """
    result = await job_results.get(job_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Job không tồn tại hoặc đã hết hạn."}
        )

    # Dọn RAM ngay sau khi client nhận được kết quả cuối cùng
    if result.get("status") in ("done", "error"):
        await job_results.pop(job_id)

    return JSONResponse(content=result)

@app.post("/api/pi_upload")
async def pi_upload(image: UploadFile = File(...), device_id: str = "pi_01"):
    """
    Raspberry Pi gửi ảnh chụp độ phân giải cao lên đây.
    Server lưu vào buffer và NGAY LẬP TỨC thông báo cho tất cả
    Web UI đang mở qua WebSocket — không cần polling.
    """
    img_bytes = await image.read()
    await pi_buffers.set(device_id, img_bytes)

    # Push thông báo tức thì qua WebSocket thay vì để Web UI phải polling
    img_b64      = base64.b64encode(img_bytes).decode()
    notification = json.dumps({
        "type":      "new_image",
        "device_id": device_id,
        "image_b64": img_b64,
    })
    await stream_manager.broadcast_text(notification)
    print(f"[PI] Đã nhận ảnh từ {device_id} và broadcast tới {len(stream_manager.active_connections)} Web UI")

    return JSONResponse(content={"status": "success", "device": device_id})

#  ROUTES — WEBSOCKET
@app.websocket("/ws/pi_stream")
async def websocket_pi_stream(websocket: WebSocket):
    """
    Raspberry Pi kết nối vào đây để PUSH live stream.
    Server nhận frame binary từ Pi và relay ngay cho tất cả Web UI.
    """
    await websocket.accept()
    print("[WS] Raspberry Pi đã kết nối Live Stream.")
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            await stream_manager.broadcast(frame_bytes)
    except WebSocketDisconnect:
        print("[WS] Raspberry Pi đã ngắt kết nối Live Stream.")
    except Exception as e:
        print(f"[WS] Lỗi Pi stream: {e}")

@app.websocket("/ws/web_viewer")
async def websocket_web_viewer(websocket: WebSocket):
    """
    Web UI kết nối vào đây để nhận:
      - Binary bytes : Frame video live stream từ Pi
      - Text JSON    : Thông báo có ảnh mới {"type": "new_image", ...}

    Client chỉ nhận, không cần gửi gì lên server.
    Dùng receive() thay vì receive_text() để detect disconnect chính xác
    kể cả khi client đóng tab trình duyệt.
    """
    await stream_manager.connect(websocket)
    try:
        while True:
            # Nhận bất kỳ loại message nào (text/binary/close frame)
            # Mục đích chính là DETECT khi client đóng kết nối
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Lỗi Web viewer: {e}")
    finally:
        # Đảm bảo luôn dọn connection dù thoát bằng bất kỳ lý do gì
        stream_manager.disconnect(websocket)

#  TIỆN ÍCH
def find_free_port(start_port: int = 5000) -> int:
    """Tìm port còn trống bắt đầu từ start_port."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start_port

#  ENTRY POINT
if __name__ == "__main__":
    port = find_free_port(5000)
    print(f"[START] Khởi động server tại http://0.0.0.0:{port}")
    uvicorn.run(
        "app_server:app",
        host    = "0.0.0.0",
        port    = port,
        reload  = True,
    )