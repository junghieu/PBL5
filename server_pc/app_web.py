from flask import Flask, request, jsonify, render_template, send_from_directory
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
from docx import Document

app = Flask(__name__, template_folder="templates")
BASE_DIR = Path(__file__).resolve().parent

# ── Load model 1 lần khi server khởi động ──
print("[INIT] Đang tải model YOLO...")
layout_model = load_layout_model()
print("[INIT] Đang tải model VietOCR...")
ocr_model = load_ocr_model()
print("[INIT] Sẵn sàng!")

# 2. KHỞI TẠO KHÓA TOÀN CỤC CHO AI
ai_lock = threading.Lock()

# ── Giới hạn ──
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10MB
MAX_DOC_BYTES    = 10 * 1024 * 1024
MAX_DOC_CHARS    = 1_000_000
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# ── Thư mục dữ liệu ──
UPLOAD_FOLDER     = BASE_DIR / "data" / "input"
PREPROCESS_FOLDER = BASE_DIR / "data" / "preprocessed"
LAYOUT_FOLDER     = BASE_DIR / "data" / "layout"
OUTPUT_FOLDER     = BASE_DIR / "data" / "output"

for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, LAYOUT_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════

@app.route('/')
def index():
    """Trả về trang Web UI."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Endpoint để frontend kiểm tra server có online không."""
    return jsonify({"status": "ok"})


@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "error": "No image uploaded"}), 400

        started_at = time.perf_counter()
        timings = {}
        job_id = uuid.uuid4().hex
        file = request.files['image']

        # 1. ĐỌC ẢNH VÀO RAM
        t0 = time.perf_counter()
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"status": "error", "error": "Invalid image format"}), 400
        timings["decode_image_sec"] = round(time.perf_counter() - t0, 4)

        # 2. TIỀN XỬ LÝ (Trên RAM - Phần này có thể chạy đa luồng bình thường)
        t0 = time.perf_counter()
        img_layout = preprocess_for_layout(img_bgr.copy())
        img_ocr = preprocess_for_ocr(img_bgr.copy())
        timings["preprocess_sec"] = round(time.perf_counter() - t0, 4)

        # <-- 3. BỌC LOGIC AI VÀO TRONG LOCK -->
        # Chỉ 1 luồng được phép đi qua khối code này tại một thời điểm
        with ai_lock:
            # 3. YOLOV10 LAYOUT
            t0 = time.perf_counter()
            # THÊM conf=0.1 VÀ debug=True VÀO ĐÂY
            boxes_list, annotated_img = run_layout_mem(
                img_layout,
                model=layout_model,
                conf=0.1,
                debug=True
            )
            timings["layout_sec"] = round(time.perf_counter() - t0, 4)
            
            # 4. VIETOCR
            t0 = time.perf_counter()
            # Đã sửa: Bọc kết quả vào list để tương thích với export_word.py
            ocr_res_raw = run_ocr_mem(img_ocr, boxes_list, predictor=ocr_model)
            ocr_res = {
                "image": file.filename, # Lấy tên gốc của ảnh upload
                "content": ocr_res_raw["content"],
                "content_with_labels": ocr_res_raw["content_with_labels"]
            }
            timings["ocr_sec"] = round(time.perf_counter() - t0, 4)
        # <-- HẾT KHU VỰC ĐỘC QUYỀN (Khóa tự động mở tại đây) -->

        # 5. XUẤT WORD RA RAM (Word cũng có thể chạy đa luồng an toàn)
        t0 = time.perf_counter()
        doc_stream = build_docx_mem([ocr_res])
        timings["export_docx_sec"] = round(time.perf_counter() - t0, 4)

        # 6. ENCODE TRỰC TIẾP RA BASE64
        doc_b64 = base64.b64encode(doc_stream.read()).decode()
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        layout_img_b64 = base64.b64encode(buffer).decode()
        layout_img_uri = f"data:image/jpeg;base64,{layout_img_b64}"

        return jsonify({
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
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/output-file/<job_id>/<filename>')
def serve_output_file(job_id, filename):
    """Serve ảnh layout preview cho Web UI."""
    try:
        # Giới hạn chỉ cho phép truy cập thư mục output của job đó
        folder = OUTPUT_FOLDER / job_id
        return send_from_directory(folder, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Not Found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "error": "Internal Server Error"}), 500


# ══════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════

def _count_chars(ocr_results):
    return sum(len(line) for res in ocr_results for line in res.get("content", []))


def _cleanup_dirs(*dirs: Path):
    """Xóa các thư mục tạm thời."""
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def _summarize_layout(layout_results):
    labels = []
    scores = []
    for page in layout_results:
        for box in page.get("boxes", []):
            labels.append(str(box.get("label", "")))
            try:
                scores.append(float(box.get("score", 0.0)))
            except Exception:
                pass
    by_label = dict(Counter(labels))
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "total_boxes": len(labels),
        "boxes_by_label": by_label,
        "avg_confidence": avg_score,
    }


def _summarize_ocr(ocr_results):
    lines = [line for res in ocr_results for line in res.get("content", []) if str(line).strip()]
    return {
        "total_lines": len(lines),
        "total_chars": sum(len(line) for line in lines),
    }


def _write_debug_report(output_dir: Path, payload: dict) -> str:
    filename = "debug_report.json"
    (output_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filename


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

# Biến toàn cục để giữ tạm ảnh gốc từ Pi
pi_raw_image_buffer = None

@app.route('/api/pi_upload', methods=['POST'])
def pi_upload():
    """Pi gọi API này để gửi ảnh gốc lên cực nhanh"""
    global pi_raw_image_buffer
    if 'image' in request.files:
        # Lưu ảnh gốc vào RAM và trả lời Pi ngay lập tức
        pi_raw_image_buffer = request.files['image'].read()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/pi_check', methods=['GET'])
def pi_check():
    """Web liên tục gọi API này để hóng xem có ảnh mới từ Pi không"""
    global pi_raw_image_buffer
    if pi_raw_image_buffer is not None:
        # Mã hóa ảnh gốc thành Base64 để gửi về Web
        img_b64 = base64.b64encode(pi_raw_image_buffer).decode()
        pi_raw_image_buffer = None # Xóa bộ đệm sau khi Web đã lấy
        return jsonify({"has_new": True, "image_b64": img_b64})
    return jsonify({"has_new": False})

if __name__ == '__main__':
    port = find_free_port(5000)
    
    app.run(host='0.0.0.0', port=port, debug=False)