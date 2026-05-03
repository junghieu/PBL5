
import sys
import traceback
import cv2
from pathlib import Path
try:
    from ultralytics import YOLO
    model = YOLO('D:/PBL5/PBL5/server_pc/weights/doclayout_yolo_docstructbench_imgsz1024.pt')
    image_paths = list(Path('D:/PBL5/PBL5/server_pc/data/input').glob('*.*'))
    img = cv2.imread(image_paths[0].as_posix())
    res = model.predict(img)
    print('SUCCESS')
except Exception as e:
    with open('D:/PBL5/PBL5/test_inference_error.txt', 'w') as f:
        traceback.print_exc(file=f)
    print('FAILED! Wrote traceback to test_inference_error.txt')
