import sys
import os
import cv2
import torch
from pathlib import Path

# 1. ÉP PYTORCH LOAD MODEL TÙY CHỈNH
original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

# 2. TRỎ ĐƯỜNG DẪN VÀO REPO ĐÃ TẢI
base_path = Path(__file__).resolve().parent.parent
doclayout_repo_path = str(base_path / "DocLayout-YOLO")

if doclayout_repo_path not in sys.path:
    sys.path.insert(0, doclayout_repo_path)

# 3. IMPORT ĐÚNG CLASS YOLOv10 TỪ REPO TÁC GIẢ
try:
    # BẮT BUỘC dùng YOLOv10 thay vì YOLO để tương thích với output dạng dict
    from doclayout_yolo import YOLOv10
    print("--- [INFO] Đã kết nối class YOLOv10 từ package doclayout_yolo ---")
except ImportError:
    try:
        from ultralytics import YOLOv10
        print("--- [INFO] Đã kết nối class YOLOv10 (ưu tiên local repo) ---")
    except ImportError as e:
        print(f"--- [ERROR] Không import được mô hình YOLOv10: {e} ---")

def load_layout_model():
    model_name = "doclayout_yolo_docstructbench_imgsz1024.pt"
    full_model_path = str(Path(__file__).resolve().parent.parent / "weights" / model_name)
    return YOLOv10(full_model_path)

# def run_layout(input_dir: Path, output_dir: Path, model=None, conf: float = 0.15, debug=False):  # thêm model=None
#     if model is None:
#         model = load_layout_model()  # fallback khi gọi từ run_pipeline.py
#     output_dir.mkdir(parents=True, exist_ok=True)
#     files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
#     layout_results = []
#     for p in files:
#         results = model.predict(str(p), imgsz=1024, conf=conf)
#         res = results[0]
#         items = []
#         if res.boxes is not None:
#             for box in res.boxes:
#                 bbox = box.xyxy[0].cpu().numpy().tolist()
#                 cls_id = int(box.cls[0])
#                 label = res.names[cls_id]
#                 score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
#                 items.append({"bbox": bbox, "label": label, "score": score})
        
#         # XỬ LÝ CỜ DEBUG
#         if debug:
#             # ĐỌC ẢNH TỪ ĐƯỜNG DẪN p
#             annotated_img = cv2.imread(str(p))
#             for item in items:
#                 x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
#                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(annotated_img, f'{item["label"]} {item["score"]:.2f}', 
#                             (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         else:
#             annotated_img = res.plot()

#         cv2.imwrite(str(output_dir / f"{p.stem}_layout.jpg"), annotated_img)
#         layout_results.append({"image": p.name, "boxes": items})
#         print(f"[LAYOUT] Xong: {p.name}")
#     return layout_results

def run_layout_mem(img_array, model, conf=0.15, debug=False):
    """Chạy layout trực tiếp trên mảng ảnh trong RAM"""
    # results = model.predict(img_array, imgsz=1024, conf=conf, device=0) # Thêm device=0 để ưu tiên GPU nếu có, giảm tải cho CPU khi xử lý ảnh lớn
    # results = model.predict(img_array, imgsz=1024, conf=conf)
    device_str = 0 if torch.cuda.is_available() else 'cpu'
    results = model.predict(img_array, imgsz=1024, conf=conf, device=device_str)
    res = results[0]
    items = []
    
    if res.boxes is not None:
        for box in res.boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            label = res.names[cls_id]
            score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            items.append({"bbox": bbox, "label": label, "score": score})
            
    # Lấy luôn ảnh đã vẽ box để hiển thị lên Web UI
    # XỬ LÝ CỜ DEBUG
    if debug:
        annotated_img = img_array.copy()
        for item in items:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            # Vẽ khung chữ nhật đỏ, độ dày nét vẽ là 2 pixel bằng OpenCV
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Ghi chú thêm nhãn và độ tự tin (conf score)
            cv2.putText(annotated_img, f'{item["label"]} {item["score"]:.2f}', 
                        (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Nếu không bật debug, dùng mặc định của YOLO
        annotated_img = res.plot()
    return items, annotated_img