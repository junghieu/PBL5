import sys
import traceback
import cv2
import numpy as np
from pathlib import Path
sys.path.append('D:/PBL5/PBL5/server_pc')

try:
    from app_server import run_ai_pipeline
    image_paths = list(Path('D:/PBL5/PBL5/server_pc/data/input').glob('*.*'))
    img = cv2.imread(image_paths[0].as_posix())
    res = run_ai_pipeline(img)
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
