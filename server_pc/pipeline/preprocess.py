import cv2
import os
from pathlib import Path
import numpy as np

import imutils
from imutils.perspective import four_point_transform

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def preprocess_base(img):
    """
    Warp document 1 lần duy nhất. Gọi trước khi chia luồng layout/ocr.
    KHÔNG gọi lại trong preprocess_for_layout hay preprocess_for_ocr.
    """
    if img is None or img.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        raise ValueError(f"Ảnh quá nhỏ: {w}x{h}")
    return _detect_and_warp_document(img)

def preprocess_for_layout(img, max_width=1400):
    """
    Nhận ảnh đã qua preprocess_base() rồi, không gọi trực tiếp với ảnh gốc.
    Chỉ resize nếu ảnh quá lớn.
    """
    if img.shape[1] > max_width:
        img = imutils.resize(img, width=max_width)
    return img

def preprocess_for_ocr(img, max_width=1400):
    """
    Nhận ảnh đã qua preprocess_base() rồi, không gọi trực tiếp với ảnh gốc.
    Tăng tương phản nhẹ để OCR dễ nhận chữ hơn.
    """
    if img.shape[1] > max_width:
        img = imutils.resize(img, width=max_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _detect_and_warp_document(img):
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    image = imutils.resize(img, height=500)

    total_area = image.shape[0] * image.shape[1]
    min_area_primary  = 0.15 * total_area
    min_area_fallback = 0.05 * total_area

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)

    v     = np.median(closed)
    sigma = 0.33
    lower = int(max(0,   (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(closed, lower, upper)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > min_area_primary:
            screenCnt = approx
            break

    if screenCnt is None:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 9)
        fc = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fc = imutils.grab_contours(fc)
        if fc:
            c     = max(fc, key=cv2.contourArea)
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > min_area_fallback:
                screenCnt = approx

    if screenCnt is not None:
        return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    print("[WARN] Auto-crop thất bại, giữ nguyên ảnh gốc.")
    return orig

# def run_preprocess(input_dir: Path, output_dir: Path, mode: str = "ocr"):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     files = [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    
#     # Báo nếu không có ảnh
#     if len(files) == 0:
#         print(f"   > [CẢNH BÁO] Không tìm thấy ảnh nào trong thư mục {input_dir}")
#         return []

#     saved = []
#     for p in files:
#         img = cv2.imread(str(p))
#         if img is None:
#             print(f"   > [LỖI] Không thể đọc ảnh: {p.name}")
#             continue

#         if mode == "layout":
#             processed = preprocess_for_layout(img)
#         else:
#             processed = preprocess_for_ocr(img)

#         out_path = output_dir / p.name
#         cv2.imwrite(str(out_path), processed)
#         saved.append(out_path)
        
#         print(f"   > [PREPROCESS:{mode.upper()}] Đã xử lý xong: {p.name}") 
        
#     return saved