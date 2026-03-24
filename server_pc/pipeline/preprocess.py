import cv2
import sys
import os
from pathlib import Path
import numpy as np

# Thêm đường dẫn tới thư mục imutils bạn đã tải (giả sử tên thư mục là imutils-master)
base_path = Path(__file__).resolve().parent.parent
imutils_path = str(base_path / "imutils-master")

if imutils_path not in sys.path:
    sys.path.insert(0, imutils_path)

import imutils
from imutils.perspective import four_point_transform

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _detect_and_warp_document(img):
    # 1. Tự động tìm khung và làm phẳng (Perspective Transform)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    image = imutils.resize(img, height=500)

    # 2. Chuyển xám và ép dải tương phản cục bộ (CLAHE)
    # Giúp làm nổi bật viền giấy kể cả khi giấy và nền có màu gần giống nhau
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Khử nhiễu hột (Noise) bằng Gaussian
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Dùng hình thái học (Morphological Closing) để nối các nét đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Auto-Canny: Tự động tính ngưỡng Canny dựa trên độ sáng chói (median) của ảnh
    v = np.median(closed)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(closed, lower, upper)
    
    # 6. Tìm contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Chỉ lấy contour nếu có đúng 4 góc và diện tích đủ lớn (tránh nhận diện nhầm tem mác)
        if len(approx) == 4 and cv2.contourArea(c) > 10000:
            screenCnt = approx
            break
    
    if screenCnt is None:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        fc = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fc = imutils.grab_contours(fc)
        if fc:
            c = max(fc, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 5000:
                screenCnt = approx
                
    if screenCnt is not None:
        return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # Nếu thuật toán vẫn thất bại (ảnh quá mờ/nhiễu), trả về ảnh gốc để xử lý tiếp
    # thay vì crash hệ thống
    print("[WARN] Auto-crop thất bại, giữ nguyên ảnh gốc.")
    return orig

def preprocess_for_layout(img):
    """
    Ảnh cho layout: giữ gần ảnh gốc để model bố cục không bị méo domain.
    """
    img = _detect_and_warp_document(img)
    img = imutils.resize(img, width=1400)
    return img

def preprocess_for_ocr(img):
    """
    Ảnh cho OCR: tăng tương phản nhẹ, hạn chế nhị phân hóa mạnh để tránh mất dấu.
    """
    img = _detect_and_warp_document(img)
    img = imutils.resize(img, width=1400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Trả về ảnh xám 3 kênh để OCR crop xử lý tiếp theo từng vùng.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def run_preprocess(input_dir: Path, output_dir: Path, mode: str = "ocr"):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    
    # Báo nếu không có ảnh
    if len(files) == 0:
        print(f"   > [CẢNH BÁO] Không tìm thấy ảnh nào trong thư mục {input_dir}")
        return []

    saved = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"   > [LỖI] Không thể đọc ảnh: {p.name}")
            continue

        if mode == "layout":
            processed = preprocess_for_layout(img)
        else:
            processed = preprocess_for_ocr(img)

        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), processed)
        saved.append(out_path)
        
        print(f"   > [PREPROCESS:{mode.upper()}] Đã xử lý xong: {p.name}") 
        
    return saved