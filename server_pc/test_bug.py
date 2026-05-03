import sys
import traceback
import cv2
from pathlib import Path
sys.path.append('D:/PBL5/PBL5/server_pc')
try:
    from pipeline.layout import run_layout_mem
    from ultralytics import YOLO
    model = YOLO('D:/PBL5/PBL5/server_pc/weights/doclayout_yolo_docstructbench_imgsz1024.pt')
    image_paths = list(Path('D:/PBL5/PBL5/server_pc/data/input').glob('*.*'))
    if image_paths:
        img = cv2.imread(image_paths[0].as_posix())
        run_layout_mem(img, model=model)
    else:
        print('No sample images found.')
except Exception as e:
    traceback.print_exc()
