import sys
import traceback
import cv2
import asyncio
from pathlib import Path
sys.path.append('D:/PBL5/PBL5/server_pc')

try:
    from app_server import app, run_ai_pipeline, lifespan
    async def main():
        async with lifespan(app):
            image_paths = list(Path('D:/PBL5/PBL5/server_pc/data/input').glob('*.*'))
            img = cv2.imread(image_paths[0].as_posix())
            print("B?t d?u x? l�...")
            res = run_ai_pipeline(img)
            print("SUCCESS")
    asyncio.run(main())
except Exception as e:
    traceback.print_exc()
