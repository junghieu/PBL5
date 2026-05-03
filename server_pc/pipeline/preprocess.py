import cv2
from pathlib import Path
import numpy as np

import imutils
from imutils.perspective import four_point_transform

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Ngưỡng đánh giá chất lượng ảnh ─────────────────────────────────────────
# Laplacian variance: < 80 = blur nặng, 80–300 = blur vừa, > 300 = sắc nét
_SHARPNESS_BLUR_HEAVY   = 80
_SHARPNESS_BLUR_MEDIUM  = 300


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
        print(f"[PREPROCESS:BASE] Ảnh nhỏ {w}x{h} — bỏ qua warp, giữ nguyên ảnh gốc")
        return img

    warped = _detect_and_warp_document(img)

    # Deskew sau warp: xử lý góc nghiêng nhỏ còn sót lại
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

    Pipeline adaptive:
      1. Resize nếu quá lớn.
      2. Đo sharpness bằng Laplacian variance.
      3. Nếu ảnh blur (Raspberry Cam, ảnh chụp thường): áp denoise mạnh hơn.
      4. Tăng tương phản CLAHE.
      5. Unsharp mask nhẹ để làm rõ nét dấu thanh tiếng Việt.

    Lý do thêm unsharp mask cuối: dấu hỏi/ngã có phần cong nhỏ ở đỉnh —
    khi ảnh hơi mờ, phần này bị mờ và VietOCR nhầm thành dấu sắc/huyền.
    Unsharp mask phục hồi cạnh mà không cần retrain model.
    """
    if img.shape[1] > max_width:
        img = imutils.resize(img, width=max_width)
        print(f"[PREPROCESS:OCR] Đã resize xuống width={max_width}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Đo sharpness để chọn mức denoise phù hợp
    sharpness = _laplacian_variance(gray)
    print(f"[PREPROCESS:OCR] Laplacian sharpness = {sharpness:.1f}")

    if sharpness < _SHARPNESS_BLUR_HEAVY:
        # Ảnh rất mờ (Raspberry Cam v3 điều kiện ánh sáng yếu)
        # → denoise mạnh, chấp nhận mất chút detail
        gray = cv2.bilateralFilter(gray, 11, 90, 90)
        print("[PREPROCESS:OCR] Chế độ: blur nặng — bilateralFilter(11,90,90)")
    elif sharpness < _SHARPNESS_BLUR_MEDIUM:
        # Ảnh mờ vừa (Raspberry Cam điều kiện bình thường)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        print("[PREPROCESS:OCR] Chế độ: blur vừa — bilateralFilter(9,75,75)")
    else:
        # Ảnh sắc nét (scan, ảnh clean)
        # → denoise nhẹ, giữ nguyên cạnh để không mờ dấu
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        print("[PREPROCESS:OCR] Chế độ: sắc nét — bilateralFilter(5,35,35)")

    # Tăng tương phản CLAHE — adaptive theo vùng ảnh
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Unsharp mask để phục hồi cạnh dấu thanh tiếng Việt
    # Đặc biệt quan trọng sau bilateralFilter mạnh làm mờ cạnh mỏng
    # FIX: tăng strength theo mức blur để dấu hỏi/ngã không bị nhầm sang sắc/huyền
    if sharpness < _SHARPNESS_BLUR_HEAVY:
        gray = _unsharp_mask(gray, sigma=1.0, strength=0.9)
        print("[PREPROCESS:OCR] Unsharp mask: strength=0.9 (ảnh rất mờ)")
    elif sharpness < _SHARPNESS_BLUR_MEDIUM:
        gray = _unsharp_mask(gray, sigma=1.0, strength=0.75)
        print("[PREPROCESS:OCR] Unsharp mask: strength=0.75 (ảnh mờ vừa)")
    else:
        gray = _unsharp_mask(gray, sigma=1.0, strength=0.5)
        print("[PREPROCESS:OCR] Unsharp mask: strength=0.5 (ảnh sắc nét)")

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── NỘI BỘ ───────────────────────────────────────────────────────────────────

def _laplacian_variance(gray_img):
    """
    Đo độ sắc nét ảnh bằng phương sai Laplacian.
    Giá trị thấp = mờ, cao = sắc nét.
    Dùng để chọn mức denoise adaptive.
    """
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def _unsharp_mask(gray_img, sigma=1.0, strength=0.6):
    """
    Unsharp mask nhẹ để làm rõ cạnh mỏng (dấu hỏi, dấu ngã, dấu nặng).

    Công thức: output = original + strength * (original - blurred)
    strength=0.6 là mức vừa phải — đủ để làm rõ nét dấu thanh mà
    không khuếch đại noise hạt của ảnh Raspberry Cam.

    Lý do dùng bước này:
    - bilateralFilter làm mờ đều tất cả cạnh, kể cả cạnh dấu mỏng.
    - Dấu hỏi (?) và ngã (~) có nét cong rất nhỏ ở đỉnh ký tự.
    - Nếu không sharpen lại, VietOCR nhầm dấu hỏi → sắc, ngã → huyền.
    """
    blurred = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(gray_img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


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
        peri   = cv2.arcLength(c, True)
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
            c      = max(fc, key=cv2.contourArea)
            peri   = cv2.arcLength(c, True)
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
    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Dùng INTER_CUBIC để giữ chất lượng chữ sau khi xoay
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated