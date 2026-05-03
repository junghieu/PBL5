import cv2
import torch
from collections import deque
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher
import yaml

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    print("[INFO] Đã load thư viện vietocr")
except ImportError as e:
    print(f"[ERROR] Không load được vietocr: {e}")


def load_ocr_model():
    """
    Load VietOCR model theo 2 chế độ:
    - Custom (ưu tiên): dùng khi đã có file .pth và .yml do nhóm tự train.
    - Pretrained (fallback): dùng model gốc vgg_seq2seq của tác giả VietOCR.
    """
    base_path   = Path(__file__).resolve().parent.parent
    config_path = base_path / "weights" / "cau_hinh_ocr_v2.yml"
    model_path  = base_path / "weights" / "seq2seq_ocr_nhom11_v2.pth"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if config_path.exists() and model_path.exists():
        print(f"[INFO] Tìm thấy model custom — đang nạp từ: {model_path}")

        config = Cfg.load_config_from_name('vgg_seq2seq')

        with open(str(config_path), 'r', encoding='utf-8') as f:
            custom_config = yaml.safe_load(f)

        if 'vocab' in custom_config:
            config['vocab'] = custom_config['vocab']
        elif 'dataset' in custom_config and 'vocab' in custom_config['dataset']:
            config['vocab'] = custom_config['dataset']['vocab']

        if 'seq2seq' in custom_config:
            config['seq2seq'] = custom_config['seq2seq']

        if 'cnn' in custom_config:
            config['cnn'].update(custom_config['cnn'])

        config['weights']                 = str(model_path)
        config['cnn']['pretrained']       = False
        config['device']                  = device
        config['predictor']['beamsearch'] = False

        print("[INFO] Nạp VietOCR custom thành công")
        return Predictor(config)

    missing = []
    if not config_path.exists():
        missing.append(config_path.name)
    if not model_path.exists():
        missing.append(model_path.name)
    print(f"[INFO] Chưa có file custom ({', '.join(missing)}) — dùng pretrained vgg_seq2seq")

    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device']                  = device
    config['predictor']['beamsearch'] = False

    print("[INFO] Nạp VietOCR pretrained thành công")
    return Predictor(config)


def run_ocr_mem(img_array, boxes_list, predictor):
    """
    Chạy OCR trên toàn bộ các vùng text đã được DocLayout-YOLO phát hiện.

    Args:
        img_array  : ảnh BGR đã qua preprocess_for_ocr()
        boxes_list : list[dict] từ run_layout_mem()
        predictor  : VietOCR Predictor đã load từ load_ocr_model()

    Returns:
        dict với 2 key:
            "content"            : list[str]
            "content_with_labels": list[dict]
    """
    TEXT_LABELS = {
        "text", "title", "plain text", "plaintext", "paragraph",
        "header", "footer", "caption", "list", "footnote", "formula", "table"
    }
    page_h, page_w = img_array.shape[:2]

    candidate_boxes = [
        b for b in boxes_list
        if str(b.get("label", "")).strip().lower() in TEXT_LABELS
    ]
    candidate_boxes = sorted(
        candidate_boxes,
        key=lambda b: (int(b["bbox"][1]), int(b["bbox"][0]))
    )

    boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.25, iom_thresh=0.5)
    boxes = _merge_boxes_into_lines(boxes)
    boxes = _split_two_column_lines(boxes, page_w)

    text_lines, content_with_labels = [], []
    seen_hashes  = set()
    recent_texts = deque(maxlen=20)

    for block_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * 0.06)
        pad_y = int(box_h * 0.18)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(page_w, x2 + pad_x)
        y2 = min(page_h, y2 + pad_y)

        crop  = img_array[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label = str(box.get("label", "")).strip().lower()

        # Tiêu đề lớn dễ bị cắt mất chữ đầu/cuối — tăng pad thêm
        if label in ("title", "header"):
            extra = int(box_w * 0.05)
            x1    = max(0, x1 - extra)
            x2    = min(page_w, x2 + extra)
            crop  = img_array[y1:y2, x1:x2]

        # Cột phải hay bị cắt sát mép — nới thêm bên phải
        col_index = box.get("col_index", -1)
        if col_index == 0:
            x1   = max(0, x1 - int(box_w * 0.03))
            crop = img_array[y1:y2, x1:x2]
        elif col_index == 1:
            x2   = min(page_w, x2 + int(box_w * 0.06))
            crop = img_array[y1:y2, x1:x2]

        line_data_list = _split_block_into_lines(crop, label=label)

        for l_data in line_data_list:
            l_crop   = l_data["image"]
            h_c, w_c = l_crop.shape[:2]
            if w_c < 10 or h_c < 10:
                continue

            pred_text = _predict_best_text(predictor, l_crop)

            if not _filter_pred(pred_text):
                continue
            if _is_duplicate_text(pred_text, seen_hashes, recent_texts):
                continue

            text_lines.append(pred_text)
            content_with_labels.append({
                "label":     label,
                "text":      pred_text,
                "x":         x1 + l_data["x_local"],
                "y":         y1 + l_data["y_local"],
                "w":         w_c,
                "h":         h_c,
                "block_x":   x1,
                "page_w":    page_w,
                "page_h":    page_h,
                "col_index": col_index,
                "block_id":  block_idx,
            })

    # Fallback toàn ảnh nếu không đọc được gì
    if not text_lines:
        pred_text = _fallback_full_image(img_array, predictor)
        if pred_text:
            text_lines.append(pred_text)
            content_with_labels.append({
                "label":  "plain text",
                "text":   pred_text,
                "page_w": page_w,
                "page_h": page_h,
            })

    return {"content": text_lines, "content_with_labels": content_with_labels}


# ── NỘI BỘ ───────────────────────────────────────────────────────────────────

def _non_max_suppress_boxes(boxes, iou_thresh=0.25, iom_thresh=0.5):
    """Khử các hộp đè lên nhau bằng IoU + IoM."""
    if not boxes:
        return []

    candidates = sorted(boxes, key=lambda b: float(b.get("score", 0.0)), reverse=True)
    kept = []

    for b in candidates:
        ax1, ay1, ax2, ay2 = [int(v) for v in b["bbox"]]
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        drop   = False

        for k in kept:
            bx1, by1, bx2, by2 = [int(v) for v in k["bbox"]]
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            union = area_a + area_b - inter_area
            iou   = inter_area / union if union > 0 else 0.0
            iom   = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0

            if iou >= iou_thresh or iom >= iom_thresh:
                drop = True
                break

        if not drop:
            kept.append(b)

    return sorted(kept, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _merge_boxes_into_lines(boxes):
    """
    Gom các bbox cùng dòng vật lý thành 1 bbox.
    Dùng anchor_cy cố định của member đầu tiên làm mốc.
    """
    if not boxes:
        return []

    heights = sorted([b["bbox"][3] - b["bbox"][1] for b in boxes])
    median_height   = heights[len(heights) // 2]
    y_center_thresh = median_height * 0.5

    sorted_boxes = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    lines = []

    for b in sorted_boxes:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        cy     = (y1 + y2) / 2.0
        placed = False

        for ln in lines:
            if abs(cy - ln["anchor_cy"]) <= y_center_thresh:
                ln["members"].append(b)
                placed = True
                break

        if not placed:
            lines.append({"anchor_cy": cy, "members": [b]})

    LABEL_PRIORITY = ["title", "header", "table", "caption", "footer", "plain text"]

    merged = []
    for ln in lines:
        members = sorted(ln["members"], key=lambda m: m["bbox"][0])
        xs1    = [int(m["bbox"][0]) for m in members]
        ys1    = [int(m["bbox"][1]) for m in members]
        xs2    = [int(m["bbox"][2]) for m in members]
        ys2    = [int(m["bbox"][3]) for m in members]
        labels = [str(m.get("label", "plain text")).strip().lower() for m in members]

        label = next((l for l in LABEL_PRIORITY if l in labels), "plain text")
        score = max(float(m.get("score", 0.0)) for m in members)

        merged.append({
            "bbox":    [min(xs1), min(ys1), max(xs2), max(ys2)],
            "label":   label,
            "score":   score,
            "members": members,
        })

    return sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _split_two_column_lines(boxes, page_w):
    """
    Phát hiện và tách dòng 2 cột thành 2 bbox riêng biệt.
    Điều kiện: 2 member, gap > 15% page_w, chiều cao tương đương, mỗi cột > 10% page_w.
    """
    result = []
    for box in boxes:
        members    = box.get("members", [])
        is_two_col = False

        if len(members) == 2:
            m0       = members[0]
            m1       = members[1]
            x0_right = int(m0["bbox"][2])
            x1_left  = int(m1["bbox"][0])
            h0 = int(m0["bbox"][3]) - int(m0["bbox"][1])
            h1 = int(m1["bbox"][3]) - int(m1["bbox"][1])
            w0 = x0_right - int(m0["bbox"][0])
            w1 = int(m1["bbox"][2]) - x1_left
            gap = x1_left - x0_right

            if (gap > page_w * 0.15
                    and (min(h0, h1) / max(h0, h1)) > 0.5 if max(h0, h1) > 0 else False
                    and w0 > page_w * 0.10 and w1 > page_w * 0.10):
                is_two_col = True
                result.append({
                    "bbox":      m0["bbox"], "label": box["label"],
                    "score":     float(m0.get("score", box["score"])),
                    "col_index": 0, "members": [m0],
                })
                result.append({
                    "bbox":      m1["bbox"], "label": box["label"],
                    "score":     float(m1.get("score", box["score"])),
                    "col_index": 1, "members": [m1],
                })

        if not is_two_col:
            box.setdefault("col_index", -1)
            result.append(box)

    return result


def _split_block_into_lines(crop_img, label="plain text"):
    """
    Tách block ảnh thành các dòng text.
    - Table: trả về nguyên block.
    - Title/Header ngắn (< 3 dòng ước tính): trả về nguyên block.
    """
    h_img, w_img = crop_img.shape[:2]

    if label == "table":
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    if h_img < (w_img * 0.02):
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    gray      = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_width = max(5, int(w_img * 0.03))
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    dilated      = cv2.dilate(thresh, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h = max(5, int(h_img * 0.05))
    min_w = max(5, int(w_img * 0.05 if label in ("title", "header") else w_img * 0.01))

    line_boxes = [
        (x, y, w, h)
        for c in cnts
        for x, y, w, h in [cv2.boundingRect(c)]
        if h > min_h and w > min_w
    ]

    if not line_boxes:
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    line_boxes = sorted(line_boxes, key=lambda b: b[1])
    avg_h      = sum(b[3] for b in line_boxes) / len(line_boxes)

    # Title/Header ngắn → không tách dòng, tránh cắt thành mảnh nhỏ
    if label in ("title", "header") and (h_img / max(avg_h, 20)) < 3:
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    dynamic_thresh = avg_h * 0.45

    groups = []
    for box in line_boxes:
        cy     = box[1] + box[3] / 2.0
        placed = False
        for group in groups:
            if abs(group["anchor_cy"] - cy) < dynamic_thresh:
                group["boxes"].append(box)
                placed = True
                break
        if not placed:
            groups.append({"anchor_cy": cy, "boxes": [box]})

    final_boxes = []
    for group in groups:
        final_boxes.extend(sorted(group["boxes"], key=lambda b: b[0]))

    lines = []
    for (x, y, w, h) in final_boxes:
        pad_x = max(2, int(w * 0.02))
        pad_y = max(3, int(h * 0.15))
        y1 = max(0, y - pad_y);  y2 = min(h_img, y + h + pad_y)
        x1 = max(0, x - pad_x); x2 = min(w_img, x + w + pad_x)
        lines.append({"image": crop_img[y1:y2, x1:x2], "x_local": x1, "y_local": y1})

    return lines


def _normalize_text(text):
    return " ".join(str(text).strip().split()).lower()


def _is_duplicate_text(text, seen_hashes: set, recent_texts: deque, sim_threshold=0.9):
    """Kiểm tra trùng lặp: exact match O(1) + fuzzy match O(20)."""
    ntext = _normalize_text(text)
    if not ntext:
        return True

    h = hash(ntext)
    if h in seen_hashes:
        return True

    for ex in recent_texts:
        if SequenceMatcher(None, ntext, ex).ratio() >= sim_threshold:
            return True

    seen_hashes.add(h)
    recent_texts.appendleft(ntext)
    return False


def _predict_best_text(ocr, crop):
    """Chạy VietOCR trên crop ảnh, trả về chuỗi đã chuẩn hóa khoảng trắng."""
    try:
        rgb_img   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(rgb_img)
        pred_text = ocr.predict(pil_img)
        return " ".join(str(pred_text).split()).strip()
    except Exception:
        return ""


def _filter_pred(pred_text):
    """
    Lọc kết quả OCR rác.
    Giữ lại nếu: không rỗng, >= 2 ký tự, có ít nhất 1 chữ/số,
    không phải chuỗi toàn số dài, không phải noise nhị phân.
    """
    if not pred_text:
        return False

    pred_text = pred_text.strip()
    if len(pred_text) < 2:
        return False

    alphanumeric = sum(ch.isalnum() for ch in pred_text)
    if alphanumeric == 0:
        return False

    # Lọc chuỗi toàn số dài (noise serial, nhị phân...)
    digits_only = sum(ch.isdigit() for ch in pred_text)
    if digits_only == alphanumeric and len(pred_text) > 4:
        return False

    # Lọc chuỗi > 75% ký tự 0 và 1
    binary_chars = sum(ch in "01" for ch in pred_text)
    if len(pred_text) > 6 and binary_chars / len(pred_text) > 0.75:
        return False

    return True


def _fallback_full_image(img, ocr):
    """Dự phòng: OCR toàn ảnh khi không detect được vùng text nào."""
    try:
        pred = ocr.predict(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ).strip()
    except Exception:
        return ""

    if len(pred) < 3:
        return ""

    valid_chars = sum(ch.isalnum() for ch in pred)
    if valid_chars / len(pred) < 0.4:
        return ""

    return " ".join(pred.split())