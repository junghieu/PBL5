# PBL5
Hệ thống Số hóa và Chuyển đổi Tài liệu đa định dạng sang Microsoft Word

## OCR Service - Vietnamese Text Recognition

A FastAPI-based OCR service that converts images to text and exports to Microsoft Word documents.

### Features
- Vietnamese text recognition using EasyOCR
- Image preprocessing with OpenCV
- Automatic Word document generation
- Background task processing
- Real-time status polling
- Mobile-friendly camera capture interface

### Technology Stack
- **Backend**: FastAPI, SQLite
- **OCR**: EasyOCR (Vietnamese language)
- **Image Processing**: OpenCV, Pillow
- **Document Generation**: python-docx
- **Frontend**: Bootstrap 5, Vanilla JavaScript

### Installation

1. Clone the repository:
```bash
git clone https://github.com/junghieu/PBL5.git
cd PBL5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the server:
```bash
python -m app.main
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://0.0.0.0:8000`

### API Endpoints

- `POST /scan` - Upload an image for OCR processing
- `GET /status/{id}` - Check processing status
- `GET /download/{id}` - Download the generated Word document
- `GET /` - Web interface

### Project Structure

```
PBL5/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── database.py          # SQLite database operations
│   └── services/
│       ├── __init__.py
│       ├── image.py         # Image preprocessing with OpenCV
│       ├── ocr.py           # EasyOCR text extraction
│       └── docx.py          # Word document generation
├── static/
│   └── index.html           # Frontend interface
├── uploads/                 # Uploaded images (auto-created)
├── outputs/                 # Generated Word documents (auto-created)
├── requirements.txt
└── README.md
```

### Usage

1. Open the web interface at `http://localhost:8000`
2. Click "Chụp ảnh" to use your camera or "Chọn file" to select an image
3. Upload the image - processing starts automatically in the background
4. Wait for the status to change to "Completed"
5. Download the generated Word document

### Development Notes

- The OCR reader is initialized once on first use for better performance
- Images are preprocessed with grayscale conversion, Gaussian blur, and adaptive thresholding
- Background tasks handle OCR processing asynchronously
- Status polling occurs every 2 seconds via JavaScript
