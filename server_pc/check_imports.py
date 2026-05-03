# check_imports.py
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path để có thể import các module khác
# Điều này mô phỏng lại cách Python sẽ tìm các module khi chạy app_server.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print("Bắt đầu kiểm tra các import...")

try:
    print("1. Đang import 'asyncio'")
    import asyncio
    print("   -> OK")

    print("2. Đang import 'uuid'")
    import uuid
    print("   -> OK")

    print("3. Đang import 'pathlib.Path'")
    from pathlib import Path
    print("   -> OK")

    print("4. Đang import 'fastapi'")
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    print("   -> OK")

    print("5. Đang import 'fastapi.responses.FileResponse'")
    from fastapi.responses import FileResponse
    print("   -> OK")

    print("6. Đang import 'contextlib.asynccontextmanager'")
    from contextlib import asynccontextmanager
    print("   -> OK")

    print("7. Đang import 'pydantic.BaseModel'")
    from pydantic import BaseModel
    print("   -> OK")

    print("8. Đang import 'pipeline.layout'")
    from pipeline.layout import load_layout_model
    print("   -> OK")

    print("9. Đang import 'pipeline.ocr'")
    from pipeline.ocr import load_ocr_model
    print("   -> OK")

    print("10. Đang import 'pipeline.preprocess'")
    from pipeline.preprocess import preprocess_base as preprocess_image
    print("   -> OK")

    print("11. Đang import 'pipeline.export_word'")
    from pipeline.export_word import build_docx_mem as export_word
    print("   -> OK")

    print("\nKiểm tra tất cả các import thành công!")

except Exception as e:
    print(f"\n!!! GẶP LỖI KHI IMPORT: {e}")
    import traceback
    traceback.print_exc()

