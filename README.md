# 📄 DocScan — Hệ thống Số hóa Tài liệu Thông minh

> Hệ thống IoT số hóa và chuyển đổi tài liệu giấy sang Microsoft Word, sử dụng Raspberry Pi Camera V3, DocLayout-YOLO và VietOCR.

---

## 📋 Mục lục

- [Tổng quan hệ thống](#tổng-quan-hệ-thống)
- [Kiến trúc](#kiến-trúc)
- [Yêu cầu phần cứng](#yêu-cầu-phần-cứng)
- [Yêu cầu phần mềm](#yêu-cầu-phần-mềm)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Khởi động hệ thống](#khởi-động-hệ-thống)
- [Sử dụng](#sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [API Reference](#api-reference)
- [Pipeline xử lý](#pipeline-xử-lý)
- [Troubleshooting](#troubleshooting)

---

## Tổng quan hệ thống

DocScan là hệ thống số hóa tài liệu hai thành phần:

- **Raspberry Pi** (Client): Chụp ảnh tài liệu bằng Camera V3, kiểm tra độ nét, phát live stream và gửi ảnh lên Server.
- **Server** (Máy tính): Nhận ảnh, chạy pipeline AI (DocLayout-YOLO → VietOCR), xuất file Word và trả kết quả về Web UI.

```
┌─────────────────────┐         ┌──────────────────────────┐
│   Raspberry Pi      │  HTTP   │        Server            │
│                     │ ──────► │                          │
│  Camera V3          │         │  DocLayout-YOLO (Layout) │
│  GPIO Button        │         │  VietOCR (OCR)           │
│  Live Stream (WS)   │ ◄────── │  FastAPI + Web UI        │
└─────────────────────┘   WS   └──────────────────────────┘
```

---

## Kiến trúc

```
Nhấn nút GPIO
     │
     ▼
Chụp ảnh (2304×1296)
     │
     ▼
Kiểm tra độ nét (Laplacian)
     │
  ┌──┴──┐
Mờ     Nét
 │       │
Bỏ    Gửi lên Server (HTTP POST)
         │
         ▼
    Tiền xử lý (Warp + Cân bằng sáng)
         │
    ┌────┴────┐
    │         │
 Layout     OCR
 (YOLO)  (VietOCR)
    │         │
    └────┬────┘
         │
    Xuất file Word (.docx)
         │
         ▼
    Trả về Web UI (Base64)
```

### Luồng dữ liệu chi tiết

| Bước | Module | Mô tả |
|------|--------|-------|
| 1 | `app_client.py` | Pi chụp ảnh, kiểm tra blur, gửi qua HTTP |
| 2 | `app_server.py` | Server nhận, validate, đưa vào async queue |
| 3 | `preprocess.py` | Warp perspective + tăng tương phản |
| 4 | `layout.py` | YOLO phát hiện vùng văn bản, bảng, tiêu đề |
| 5 | `ocr.py` | VietOCR nhận diện chữ từng vùng |
| 6 | `export_word.py` | Ghép kết quả → file .docx |
| 7 | `app_server.py` | Trả Base64 về Web UI qua polling |

---

## Yêu cầu phần cứng

### Raspberry Pi (Client)
| Thành phần | Thông số |
|-----------|----------|
| Bo mạch | Raspberry Pi 4B (RAM ≥ 2GB) hoặc 5 |
| Camera | Raspberry Pi Camera Module V3 (12MP, Autofocus) |
| Nút bấm | Nút nhấn thường hở, kết nối GPIO 17 + GND |
| Nguồn | 5V/3A (USB-C) |
| Mạng | WiFi hoặc Ethernet, cùng mạng LAN với Server |

### Server (Máy tính)
| Thành phần | Thông số tối thiểu |
|-----------|-------------------|
| CPU | Intel Core i5 Gen 8+ hoặc AMD Ryzen 5 |
| RAM | 8GB (khuyến nghị 16GB) |
| GPU | NVIDIA (tùy chọn, tăng tốc YOLO) |
| Ổ cứng | SSD, còn trống ≥ 10GB |
| Mạng | Cùng LAN với Raspberry Pi |

---

## Yêu cầu phần mềm

### Server
- Python 3.10+
- CUDA Toolkit (tùy chọn, nếu có GPU NVIDIA)

### Raspberry Pi
- Raspberry Pi OS Bullseye hoặc Bookworm (64-bit)
- Python 3.10+

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/docscan.git
cd docscan
```

### 2. Cài đặt phía Server

```bash
# Tạo môi trường ảo
python -m venv venv
# source venv/bin/activate        # Linux/macOS
venv\Scripts\activate         # Windows

# Cài đặt dependencies
pip install -r requirements.txt
pip install --no-deps git+https://github.com/opendatalab/DocLayout-YOLO
```

### 3. Tải trọng số mô hình

Tạo thư mục `weights/` trong thư mục gốc project và đặt các file sau vào đó:

```
weights/
├── doclayout_yolo_docstructbench_imgsz1024.pt   # DocLayout-YOLO
```

> **Lưu ý:** File trọng số VietOCR sẽ được tải tự động lần đầu chạy.  
> Trọng số DocLayout-YOLO tải tại: https://github.com/opendatalab/DocLayout-YOLO

### 4. Cài đặt phía Raspberry Pi

```bash
# Cài đặt dependencies hệ thống (bắt buộc trên Pi OS)
sudo apt update
sudo apt install -y python3-picamera2 libcamera-dev

# Cài đặt dependencies Python
pip install gpiozero opencv-python-headless requests websocket-client
```

### 5. Sơ đồ kết nối GPIO

```
Raspberry Pi GPIO
┌──────────────┐
│ GPIO 17 ──── nút bấm ──── GND │
└──────────────┘
```

| Pi Pin | Kết nối |
|--------|---------|
| GPIO 17 (Pin 11) | Chân 1 của nút bấm |
| GND (Pin 6 hoặc 9) | Chân 2 của nút bấm |

---

## Cấu hình

### Server — Biến môi trường (tùy chọn)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `ENV` | `production` | `development` để bật auto-reload |

### Raspberry Pi — Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `SERVER_BASE_URL` | `http://10.82.220.172:5000` | Địa chỉ IP và port của Server |
| `BUTTON_PIN` | `17` | Chân GPIO kết nối nút bấm |
| `BLUR_THRESHOLD` | `80` | Ngưỡng độ nét ảnh (Laplacian variance) |
| `MAX_RETRY` | `5` | Số lần thử lại tối đa khi gửi ảnh thất bại |

```bash
# Ví dụ chạy với biến môi trường tùy chỉnh
SERVER_BASE_URL=http://192.168.1.100:5000 BLUR_THRESHOLD=100 python app_client.py
```

---

## Khởi động hệ thống

### Bước 1: Khởi động Server (Máy tính)

```bash
cd docscan
source venv/bin/activate
python app_server.py
```

Server sẽ tự động tìm port trống bắt đầu từ 5000. Khi thấy log sau là sẵn sàng:

```
[INIT] Đang tải model YOLO...
[INIT] Đang tải model VietOCR...
[INIT] ✓ Server sẵn sàng phục vụ!
[START] Khởi động server tại http://0.0.0.0:5000
```

### Bước 2: Mở Web UI

Truy cập trình duyệt tại:
```
http://<IP_SERVER>:5000
```

### Bước 3: Khởi động Client (Raspberry Pi)

```bash
# Đảm bảo SERVER_BASE_URL trỏ đúng IP máy tính của bạn
export SERVER_BASE_URL=http://192.168.1.100:5000

python app_client.py
```

Khi thấy log sau là Pi đã sẵn sàng:

```
[CAMERA] Đã bật Autofocus liên tục (Camera V3)
[CAMERA] Sẵn sàng — Main: 2304×1296 | Preview: 640×480
[WS] Bắt đầu phát Live Stream...
  Sẵn sàng. Nhấn nút để chụp tài liệu...
```

---

## Sử dụng

### Tab "Camera Pi"

1. Live stream từ Camera V3 hiển thị real-time trên Web UI
2. **Nhấn nút vật lý** trên Raspberry Pi để chụp ảnh
3. Ảnh nét sẽ tự động xuất hiện trên Web UI (qua WebSocket, không cần reload)
4. Nhấn **"Xử lý ảnh vừa chụp"** để bắt đầu pipeline AI
5. Chờ kết quả (thường 10–30 giây tùy cấu hình máy)
6. Xem văn bản OCR và ảnh bố cục YOLO
7. Nhấn **"Tải .docx"** để tải file Word về máy

### Tab "Upload ảnh"

1. Kéo thả hoặc click chọn ảnh từ máy tính (JPG, PNG, BMP, TIFF)
2. Nhấn **"Xử lý"**
3. Chờ kết quả và tải file Word

### Ngưỡng độ nét ảnh (BLUR_THRESHOLD)

| Giá trị | Tình huống |
|---------|-----------|
| 50–80 | Ánh sáng yếu, tài liệu in thường |
| 80–120 | **Mặc định** — điều kiện ánh sáng tốt |
| 120+ | Yêu cầu ảnh cực nét, ánh sáng mạnh |

Nếu Pi liên tục báo "Ảnh bị mờ" mà ảnh trông bình thường, hãy giảm `BLUR_THRESHOLD`.

---

## Cấu trúc thư mục

```
docscan/
│
├── app_server.py              # FastAPI server — entry point phía Server
├── app_client.py              # Raspberry Pi client — entry point phía Pi
│
├── pipeline/                  # Các module xử lý AI
│   ├── __init__.py
│   ├── preprocess.py          # Tiền xử lý ảnh (warp, CLAHE)
│   ├── layout.py              # Phát hiện bố cục (DocLayout-YOLO)
│   ├── ocr.py                 # Nhận diện văn bản (VietOCR)
│   └── export_word.py         # Xuất file Word (.docx)
│
├── templates/
│   └── index.html             # Web UI (Single Page Application)
│
├── weights/                   # Trọng số mô hình (không commit lên git)
│   └── doclayout_yolo_docstructbench_imgsz1024.pt
│
├── data/
│   └── output/                # Thư mục output tạm (tự tạo khi chạy)
│
├── requirements_server.txt    # Dependencies phía Server
└── README.md
```

> **Lưu ý:** Thư mục `weights/` và `data/` nên được thêm vào `.gitignore`.

**`.gitignore` đề xuất:**
```
weights/
data/
venv/
__pycache__/
*.pyc
*.pth
.env
```

---

## API Reference

### HTTP Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Trả về Web UI |
| `GET` | `/health` | Kiểm tra trạng thái server |
| `POST` | `/process` | Upload ảnh từ Web UI → trả `job_id` |
| `GET` | `/result/{job_id}` | Polling lấy kết quả theo `job_id` |
| `POST` | `/api/pi_upload` | Pi upload ảnh chất lượng cao |
| `GET` | `/api/pi_check` | Fallback HTTP kiểm tra ảnh mới từ Pi |

### WebSocket Endpoints

| Endpoint | Chiều | Mô tả |
|----------|-------|-------|
| `/ws/pi_stream` | Pi → Server | Pi push live stream frame (binary) |
| `/ws/web_viewer` | Server → Web UI | Web UI nhận frame video + JSON notification |

### Cấu trúc response `/result/{job_id}`

```json
{
  "status": "done",
  "data": {
    "job_id": "a1b2c3d4...",
    "layout": [
      {
        "image": "capture.jpg",
        "boxes": [
          {"bbox": [x1, y1, x2, y2], "label": "title", "score": 0.95}
        ]
      }
    ],
    "ocr": [
      {
        "image": "capture.jpg",
        "content": ["Dòng văn bản 1", "Dòng văn bản 2"],
        "content_with_labels": [
          {"label": "title", "text": "TIÊU ĐỀ", "x": 100, "y": 50, "w": 300, "h": 40}
        ]
      }
    ],
    "docx_filename": "ket_qua_a1b2c3.docx",
    "docx_base64": "<base64_encoded_docx>",
    "layout_image_url": "data:image/jpeg;base64,...",
    "timings": {
      "preprocess_sec": 0.12,
      "layout_sec": 1.85,
      "ocr_sec": 4.32,
      "export_docx_sec": 0.08
    }
  }
}
```

### Trạng thái `job_id`

| `status` | Ý nghĩa |
|----------|---------|
| `queued` | Job đang chờ trong hàng đợi |
| `processing` | AI đang xử lý |
| `done` | Hoàn thành, có kết quả |
| `error` | Xử lý thất bại |

---

## Pipeline xử lý

### 1. Tiền xử lý (`preprocess.py`)

- **Phát hiện góc tài liệu**: Canny edge detection + tìm contour 4 cạnh
- **Warp perspective**: Căn thẳng tài liệu bị nghiêng/cong
- **Fallback**: Nếu không tìm được góc → giữ nguyên ảnh gốc
- **Tách 2 luồng**: `preprocess_for_layout()` (resize) và `preprocess_for_ocr()` (CLAHE tăng tương phản)

### 2. Layout Detection (`layout.py`)

- **Mô hình**: DocLayout-YOLO (`YOLOv10`), imgsz=1024, conf=0.1
- **Các nhãn phát hiện**: `title`, `header`, `plain text`, `table`, `figure`, `caption`, `footer`
- **Output**: Danh sách bounding box + nhãn + confidence score

### 3. OCR (`ocr.py`)

- **Mô hình**: VietOCR (`vgg_seq2seq`)
- **Xử lý trước OCR**:
  - NMS (Non-Maximum Suppression) loại box trùng nhau
  - Gom box thành dòng vật lý
  - Tách block thành từng dòng chữ (`_split_block_into_lines`)
- **Lọc kết quả**:
  - Bỏ text quá ngắn, toàn ký tự đặc biệt
  - Dedup bằng hash + fuzzy matching (SequenceMatcher ≥ 0.9)
- **Fallback**: OCR toàn ảnh nếu không phát hiện được box nào

### 4. Xuất Word (`export_word.py`)

- Font: Times New Roman, 13pt
- `title`/`header` → căn giữa, in đậm
- `table` → chèn placeholder `--- [Bảng biểu] ---`
- `plain text` → căn trái, thụt đầu dòng theo tọa độ x tương đối

---

## Troubleshooting

### Pi báo "Ảnh bị mờ" liên tục

```bash
# Giảm ngưỡng xuống
BLUR_THRESHOLD=50 python app_client.py
```

Hoặc kiểm tra:
- Đủ ánh sáng chiếu vào tài liệu
- Camera đã được làm sạch lens
- Autofocus đã ổn định (chờ 1–2 giây sau khi đặt tài liệu rồi mới nhấn nút)

### Server không tải được model YOLO

```
FileNotFoundError: Không tìm thấy file model tại: .../weights/doclayout_yolo_docstructbench_imgsz1024.pt
```

→ Kiểm tra file `.pt` đã đặt đúng trong thư mục `weights/` chưa.

### Pi không kết nối được Server

```bash
# Kiểm tra server đang chạy và port đúng
curl http://<IP_SERVER>:5000/health

# Kiểm tra Pi và Server cùng mạng LAN
ping <IP_SERVER>
```

### Web UI hiển thị "server offline"

→ Kiểm tra `app_server.py` đã khởi động chưa, và truy cập đúng địa chỉ IP + port.

### OCR trả về văn bản lộn xộn

- Ảnh bị nghiêng nhiều → thuật toán warp không căn được → đặt tài liệu phẳng hơn
- Thiếu ánh sáng → tăng sáng phòng hoặc dùng đèn chiếu
- Font chữ viết tay → VietOCR pretrained chủ yếu được train trên chữ in

### Pi crash khi khởi động

```
[LỖI] Thiếu thư viện picamera2
```

→ Đảm bảo đang dùng Raspberry Pi OS Bullseye/Bookworm và đã enable camera interface:

```bash
sudo raspi-config
# Interface Options → Camera → Enable
sudo reboot
```

---

## Thông tin dự án

| | |
|--|--|
| **Tên dự án** | DocScan — Hệ thống số hóa tài liệu |
| **Loại** | Đồ án PBL5 |
| **Công nghệ** | FastAPI · Picamera2 · DocLayout-YOLO · VietOCR · WebSocket |
| **Phần cứng** | Raspberry Pi 4B · Camera Module V3 |
