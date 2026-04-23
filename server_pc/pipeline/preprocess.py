import cv2
from pathlib import Path
import numpy as np

import imutils
from imutils.perspective import four_point_transform

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def preprocess_base(img):
    """
    Warp + deskew document 1 lần duy nhất.
    Gọi trước khi chia luồng layout/ocr.
    KHÔNG gọi lại trong preprocess_for_layout hay preprocess_for_ocr.
    """
    if img is None or img.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        raise ValueError(f"Ảnh quá nhỏ: {w}x{h}")

    warped = _detect_and_warp_document(img)

    # Deskew sau warp: xử lý góc nghiêng nhỏ còn sót lại
    # (thường xảy ra khi auto-crop thất bại và giữ nguyên ảnh gốc)
    warped = _deskew(warped)

    return warped


def preprocess_for_layout(img, max_width=1400):
    """
    Nhận ảnh đã qua preprocess_base() rồi, không gọi trực tiếp với ảnh gốc.
    Chỉ resize nếu ảnh quá lớn — giữ màu gốc để YOLO detect tốt hơn.
    """
    if img.shape[1] > max_width:
        img = imutils.resize(img, width=max_width)
        print(f"[PREPROCESS:LAYOUT] Đã resize xuống width={max_width}")
    return img


def preprocess_for_ocr(img, max_width=1400):
    """
    Nhận ảnh đã qua preprocess_base() rồi, không gọi trực tiếp với ảnh gốc.
    Tăng tương phản để OCR dễ nhận chữ hơn, đặc biệt với ảnh Raspberry.
    """
    if img.shape[1] > max_width:
        img = imutils.resize(img, width=max_width)
        print(f"[PREPROCESS:OCR] Đã resize xuống width={max_width}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tăng bilateral filter để khử noise camera Raspberry tốt hơn
    # (5,35,35) quá nhẹ cho camera chất lượng thấp → dùng (9,75,75)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── NỘI BỘ ───────────────────────────────────────────────────────────────────

def _detect_and_warp_document(img):
    """
    Phát hiện biên tài liệu và căn chỉnh phối cảnh (perspective warp).
    Có 2 lần thử: Canny edge → adaptive threshold (fallback).
    Nếu cả 2 đều thất bại, trả về ảnh gốc và log cảnh báo.
    """
    ratio = img.shape[0] / 500.0
    orig  = img.copy()
    image = imutils.resize(img, height=500)

    total_area        = image.shape[0] * image.shape[1]
    min_area_primary  = 0.15 * total_area
    min_area_fallback = 0.05 * total_area

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tự động chọn ngưỡng Canny theo median pixel
    v     = np.median(closed)
    sigma = 0.33
    lower = int(max(0,   (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(closed, lower, upper)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None

    # Lần thử 1: Canny edge
    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > min_area_primary:
            screenCnt = approx
            print("[PREPROCESS:WARP] Phát hiện biên qua Canny — warp thành công")
            break

    # Lần thử 2: Adaptive threshold (fallback)
    if screenCnt is None:
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        fc = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fc = imutils.grab_contours(fc)
        if fc:
            c     = max(fc, key=cv2.contourArea)
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > min_area_fallback:
                screenCnt = approx
                print("[PREPROCESS:WARP] Phát hiện biên qua adaptive threshold — warp thành công")

    if screenCnt is not None:
        return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    print("[PREPROCESS:WARP] Auto-crop thất bại, giữ nguyên ảnh gốc — sẽ thử deskew")
    return orig


def _deskew(img, max_angle=10.0):
    """
    Căn chỉnh góc nghiêng nhỏ (skew correction) sau bước warp.
    Chỉ xoay nếu góc phát hiện được nằm trong khoảng ±max_angle độ,
    tránh xoay ảnh bị lật do detect nhầm góc lớn.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=img.shape[1] // 4,
        maxLineGap=20
    )

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Chỉ lấy các đường gần nằm ngang (góc nhỏ) để tính skew
            if abs(angle) < max_angle:
                angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)

    # Chỉ xoay nếu góc đủ lớn để thấy sự khác biệt (> 0.3 độ)
    if abs(median_angle) < 0.3:
        return img

    print(f"[PREPROCESS:DESKEW] Phát hiện góc nghiêng {median_angle:.2f}° — đang căn chỉnh")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Dùng INTER_CUBIC để giữ chất lượng chữ sau khi xoay
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated