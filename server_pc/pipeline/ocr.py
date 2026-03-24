import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher

# 1. XỬ LÝ ĐƯỜNG DẪN ĐỂ TÌM VIETOCR-MASTER
base_path = Path(__file__).resolve().parent.parent
vietocr_path = str(base_path / "vietnamese-ocr-master")

if vietocr_path not in sys.path:
    sys.path.insert(0, vietocr_path)

# 2. IMPORT TỪ BẢN MASTER
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    print("--- [INFO] Đã kết nối thành công VietOCR Master ---")
except ModuleNotFoundError:
    print(f"--- [ERROR] Không tìm thấy VietOCR tại: {vietocr_path} ---")
    sys.path.append(os.path.abspath("vietnamese-ocr-master"))
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg


def load_ocr_model():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False  # nhanh hơn, giảm treo
    return Predictor(config)


def _non_max_suppress_boxes(boxes, iou_thresh=0.25, iom_thresh=0.5):
    """Khử các hộp đè lên nhau bằng IoU + IoM"""
    if not boxes:
        return []
    candidates = sorted(boxes, key=lambda b: float(b.get("score", 0.0)), reverse=True)
    kept = []
    for b in candidates:
        ax1, ay1, ax2, ay2 = [int(v) for v in b["bbox"]]
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        drop = False
        for k in kept:
            bx1, by1, bx2, by2 = [int(v) for v in k["bbox"]]
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            union = area_a + area_b - inter_area
            iou = inter_area / union if union > 0 else 0.0
            iom = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0

            if iou >= iou_thresh or iom >= iom_thresh:
                drop = True
                break
        if not drop:
            kept.append(b)
    return sorted(kept, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _merge_boxes_into_lines(boxes):
    if not boxes:
        return []
    total_height = sum([b["bbox"][3] - b["bbox"][1] for b in boxes])
    average_height = total_height / len(boxes)
    y_center_thresh = average_height * 0.5

    sorted_boxes = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    lines = []
    for b in sorted_boxes:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        cy = (y1 + y2) / 2.0
        placed = False
        for ln in lines:
            if abs(cy - ln["cy"]) <= y_center_thresh:
                ln["members"].append(b)
                ln["cy"] = (ln["cy"] * (len(ln["members"]) - 1) + cy) / len(ln["members"])
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "members": [b]})

    merged = []
    for ln in lines:
        members = sorted(ln["members"], key=lambda m: m["bbox"][0])
        xs1 = [int(m["bbox"][0]) for m in members]
        ys1 = [int(m["bbox"][1]) for m in members]
        xs2 = [int(m["bbox"][2]) for m in members]
        ys2 = [int(m["bbox"][3]) for m in members]
        labels = [str(m.get("label", "plain text")).strip().lower() for m in members]
        label = "title" if "title" in labels else "plain text"
        score = max(float(m.get("score", 0.0)) for m in members)
        merged.append({
            "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)],
            "label": label,
            "score": score,
        })
    return sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _normalize_text(text):
    return " ".join(str(text).strip().split()).lower()


def _is_duplicate_text(text, existing_texts, sim_threshold=0.9):
    ntext = _normalize_text(text)
    if not ntext:
        return True
    for ex in existing_texts:
        nex = _normalize_text(ex)
        if not nex:
            continue
        if ntext == nex or SequenceMatcher(None, ntext, nex).ratio() >= sim_threshold:
            return True
    return False


def _predict_best_text(ocr, crop):
    rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    try:
        pred_text = ocr.predict(pil_img)
        return " ".join(str(pred_text).split()).strip()
    except Exception:
        return ""


def _filter_pred(pred_text, w_c):
    if not pred_text:
        return False
    if len(pred_text) < 3:
        return False
    alpha = sum(ch.isalpha() for ch in pred_text)
    if alpha / max(1, len(pred_text)) < 0.5:
        return False
    if len(pred_text) > 4 and (w_c / len(pred_text)) < 3.5:
        return False
    if len(pred_text) < 12 and pred_text.isupper():  # loại rác ALLCAPS ngắn
        return False
    return True


def _split_block_into_lines(crop_img):
    h_img, w_img = crop_img.shape[:2]
    # Tăng giới hạn chiều cao tối thiểu lên 30 để bỏ qua các vệt rác quá mỏng
    if h_img < 30: 
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Khử nhiễu nền mạnh hơn
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # SỬA LỖI ẢO GIÁC: Bỏ Otsu, dùng Threshold cố định (160)
    # Đảm bảo các vùng khoảng trắng giấy không bị biến thành nhiễu đen
    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)

    # SỬA LỖI DÍNH DÒNG IN NGHIÊNG: Chiều dọc chỉ để 1 pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Nâng bộ lọc lên > 10 để xóa sổ toàn bộ "hạt bụi" gây ra chữ ảo
        if h > 10 and w > 10:
            line_boxes.append((x, y, w, h))
            
    if not line_boxes: 
        return [{"image": crop_img, "x_local": 0, "y_local": 0}]
        
    line_boxes = sorted(line_boxes, key=lambda b: b[1])
    
    avg_h = sum([b[3] for b in line_boxes]) / len(line_boxes)
    dynamic_thresh = avg_h * 0.45

    groups = []
    for box in line_boxes:
        cy = box[1] + box[3]/2.0
        placed = False
        for group in groups:
            if abs(group['cy'] - cy) < dynamic_thresh: 
                group['boxes'].append(box)
                group['cy'] = (group['cy'] * (len(group['boxes'])-1) + cy) / len(group['boxes'])
                placed = True
                break
        if not placed: groups.append({'cy': cy, 'boxes': [box]})
        
    final_boxes = []
    for group in groups:
        final_boxes.extend(sorted(group['boxes'], key=lambda b: b[0]))

    lines = []
    for (x, y, w, h) in final_boxes: 
        pad_x, pad_y = 5, 8 
        y1, y2 = max(0, y - pad_y), min(h_img, y + h + pad_y)
        x1, x2 = max(0, x - pad_x), min(w_img, x + w + pad_x)
        lines.append({"image": crop_img[y1:y2, x1:x2], "x_local": x1, "y_local": y1})
    return lines

def _fallback_full_image(img, ocr):
    """Dự phòng toàn ảnh, có lọc rác."""
    pred = ""
    try:
        pred = ocr.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).strip()
    except Exception:
        return ""
    if len(pred) < 12:
        return ""
    if pred.isupper():
        return ""
    if not any(ch.isalpha() for ch in pred):
        return ""
    return " ".join(pred.split())


def run_ocr(pre_dir: Path, layout_results: list, output_dir: Path, predictor=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = predictor if predictor is not None else load_ocr_model()
    TEXT_LABELS = {
        "text", "title", "plain text", "plaintext", "paragraph",
        "header", "footer", "caption", "list", "footnote", "formula", "table"
    }
    all_results = []

    for entry in layout_results:
        img_path = pre_dir / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        page_h, page_w = img.shape[:2]

        candidate_boxes = [
            b for b in entry["boxes"]
            if str(b.get("label", "")).strip().lower() in TEXT_LABELS
        ]
        candidate_boxes = sorted(candidate_boxes, key=lambda b: (int(b["bbox"][1]), int(b["bbox"][0])))

        boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.25, iom_thresh=0.5)
        boxes = _merge_boxes_into_lines(boxes)

        text_lines, content_with_labels = [], []

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            pad_x, pad_y = 18, 8
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[1], x2 + pad_x); y2 = min(img.shape[0], y2 + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = str(box.get("label", "")).strip().lower()
            line_data_list = _split_block_into_lines(crop)

            for l_data in line_data_list:
                l_crop = l_data["image"]
                h_c, w_c = l_crop.shape[:2]
                if w_c < 10 or h_c < 10:
                    continue

                pred_text = _predict_best_text(ocr, l_crop)
                if not _filter_pred(pred_text, w_c) or _is_duplicate_text(pred_text, text_lines):
                    continue

                text_lines.append(pred_text)
                content_with_labels.append({
                    "label": label, "text": pred_text,
                    "x": x1 + l_data["x_local"], "y": y1 + l_data["y_local"],
                    "w": w_c, "h": h_c,
                    "page_w": page_w, "page_h": page_h
                })

        # Fallback toàn ảnh nếu hoàn toàn không có dòng nào
        if not text_lines:
            pred_text = _fallback_full_image(img, ocr)
            if pred_text:
                text_lines.append(pred_text)
                content_with_labels.append({
                    "label": "plain text", "text": pred_text,
                    "page_w": page_w, "page_h": page_h
                })

        txt_path = output_dir / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")
        all_results.append({"image": entry["image"], "content": text_lines, "content_with_labels": content_with_labels})

    return all_results


def run_ocr_mem(img_array, boxes_list, predictor):
    TEXT_LABELS = {
        "text", "title", "plain text", "plaintext", "paragraph",
        "header", "footer", "caption", "list", "footnote", "formula", "table"
    }
    page_h, page_w = img_array.shape[:2]

    candidate_boxes = [
        b for b in boxes_list
        if str(b.get("label", "")).strip().lower() in TEXT_LABELS
    ]
    candidate_boxes = sorted(candidate_boxes, key=lambda b: (int(b["bbox"][1]), int(b["bbox"][0])))

    boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.25, iom_thresh=0.5)
    boxes = _merge_boxes_into_lines(boxes)

    text_lines, content_with_labels = [], []

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        pad_x, pad_y = 18, 8
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(img_array.shape[1], x2 + pad_x); y2 = min(img_array.shape[0], y2 + pad_y)
        crop = img_array[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label = str(box.get("label", "")).strip().lower()
        line_data_list = _split_block_into_lines(crop)

        for l_data in line_data_list:
            l_crop = l_data["image"]
            h_c, w_c = l_crop.shape[:2]
            if w_c < 10 or h_c < 10:
                continue

            pred_text = _predict_best_text(predictor, l_crop)
            if not _filter_pred(pred_text, w_c) or _is_duplicate_text(pred_text, text_lines):
                continue

            text_lines.append(pred_text)
            content_with_labels.append({
                "label": label, "text": pred_text,
                "x": x1 + l_data["x_local"], "y": y1 + l_data["y_local"],
                "w": w_c, "h": h_c,
                "page_w": page_w, "page_h": page_h
            })

    # Fallback toàn ảnh nếu rỗng
    if not text_lines:
        pred_text = _fallback_full_image(img_array, predictor)
        if pred_text:
            text_lines.append(pred_text)
            content_with_labels.append({
                "label": "plain text", "text": pred_text,
                "page_w": page_w, "page_h": page_h
            })

    return {"content": text_lines, "content_with_labels": content_with_labels}