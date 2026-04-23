import cv2
import torch
from pathlib import Path

# Khai báo sẵn để tránh NameError nếu cả 2 import đều thất bại
YOLOv10 = None

try:
    from doclayout_yolo import YOLOv10
    print("[INFO] Đã load thư viện doclayout_yolo")
except Exception as e1:
    print(f"[WARN] Không load được doclayout_yolo: {e1}")
    try:
        from ultralytics import YOLOv10
        print("[INFO] Fallback: đã load thư viện ultralytics")
    except Exception as e2:
        print(f"[ERROR] Không load được cả ultralytics: {e2}")


def load_layout_model():
    """
    Load mô hình DocLayout-YOLO từ thư mục weights/.
    Dùng patch torch.load để tắt cảnh báo weights_only trên PyTorch >= 2.0,
    patch được restore lại ngay sau khi load xong dù có lỗi hay không.
    """
    if YOLOv10 is None:
        raise RuntimeError(
            "[ERROR] Không thể nạp YOLOv10 — kiểm tra log import ở trên."
        )

    base_path      = Path(__file__).resolve().parent.parent
    model_name     = "doclayout_yolo_docstructbench_imgsz1024.pt"
    full_model_path = base_path / "weights" / model_name

    if not full_model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Không tìm thấy file model: {full_model_path}"
        )

    print(f"[INFO] Đang nạp model từ: {full_model_path}")

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    try:
        torch.load = _patched_load
        model = YOLOv10(str(full_model_path))
        print("[INFO] Nạp model thành công")
        return model
    finally:
        # Luôn restore dù load thành công hay lỗi
        torch.load = original_load


def run_layout_mem(img_array, model, conf=0.15, debug=False):
    """
    Chạy layout detection trực tiếp trên mảng ảnh trong RAM.

    Args:
        img_array : ảnh BGR (numpy array) đã qua preprocess_for_layout()
        model     : model đã load từ load_layout_model()
        conf      : ngưỡng confidence (mặc định 0.15 — thấp để không bỏ sót)
        debug     : True → vẽ bbox lên ảnh gốc bằng OpenCV (màu sắc theo label)

    Returns:
        items         : list[dict] — mỗi dict có bbox, label, score
        annotated_img : ảnh BGR đã vẽ bbox (dùng để hiển thị trên Web UI)
    """
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = model.predict(img_array, imgsz=1024, conf=conf, device=device_str)
    res     = results[0]

    items = []
    if res.boxes is not None:
        for box in res.boxes:
            bbox   = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            label  = res.names[cls_id]
            score  = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            items.append({"bbox": bbox, "label": label, "score": score})

    if debug:
        annotated_img = _draw_debug(img_array, items)
    else:
        annotated_img = res.plot()

    return items, annotated_img


# ── NỘI BỘ ───────────────────────────────────────────────────────────────────

# Màu BGR cho từng loại label — giúp phân biệt nhanh khi debug
_LABEL_COLORS = {
    "title":      (255, 100,   0),
    "header":     (255, 180,   0),
    "plain text": (  0, 200,   0),
    "table":      (  0, 180, 255),
    "figure":     (180,   0, 255),
    "caption":    (  0, 255, 200),
    "footer":     (120, 120, 120),
    "formula":    (255,   0, 180),
}
_DEFAULT_COLOR = (0, 0, 255)


def _draw_debug(img_array, items):
    """
    Vẽ bbox lên bản sao ảnh gốc với màu theo label.
    Bbox từ model đã ở tọa độ gốc (YOLO tự scale về kích thước ảnh đầu vào),
    không cần scale lại thủ công.
    """
    annotated = img_array.copy()
    for item in items:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        label  = item["label"]
        score  = item["score"]
        color  = _LABEL_COLORS.get(label.lower(), _DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {score:.2f}"
        text_y = max(18, y1 - 6)
        # Nền đen cho chữ để dễ đọc trên mọi màu nền
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 2), (0, 0, 0), -1)
        cv2.putText(annotated, text, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return annotated