#!/usr/bin/env python3
"""
Test script để debug việc OCR dòng 'tr. XXX' - chú thích trang.
Kiểm tra xem các sửa đổi trong ocr.py có giúp capture 'tr.' hay không.
"""

import os
import sys
import cv2
import json
from pathlib import Path

# Thêm server_pc vào path
sys.path.insert(0, str(Path(__file__).parent / "server_pc"))

from pipeline.preprocess import preprocess_for_ocr
from pipeline.ocr import run_layout_mem, run_ocr_mem, load_ocr_model

def test_tr_extraction(image_path):
    """Test một ảnh để xem có extract được 'tr.' không."""
    
    if not os.path.exists(image_path):
        print(f"Lỗi: File {image_path} không tồn tại")
        return
    
    print(f"\n{'='*60}")
    print(f"Test: {image_path}")
    print(f"{'='*60}\n")
    
    # Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    print(f"Kích thước ảnh gốc: {img.shape}")
    
    # Preprocess
    try:
        img_prep = preprocess_for_ocr(img)
        print(f"Kích thước sau preprocess: {img_prep.shape}")
    except Exception as e:
        print(f"Lỗi preprocess: {e}")
        return
    
    # Layout detection
    try:
        boxes_layout = run_layout_mem(img_prep)
        print(f"\nSố boxes từ layout detection: {len(boxes_layout)}")
        
        # In ra các labels
        labels_count = {}
        for box in boxes_layout:
            label = box.get("label", "unknown")
            labels_count[label] = labels_count.get(label, 0) + 1
        
        print("Labels phát hiện được:")
        for label, count in sorted(labels_count.items()):
            print(f"  {label}: {count}")
        
    except Exception as e:
        print(f"Lỗi layout detection: {e}")
        return
    
    # OCR
    try:
        print("\nLoading OCR model...")
        predictor = load_ocr_model()
        
        print("Chạy OCR...")
        result = run_ocr_mem(img_prep, boxes_layout, predictor)
        
        content = result.get("content", [])
        content_with_labels = result.get("content_with_labels", [])
        
        print(f"\nTổng dòng text OCR: {len(content)}")
        print("\nDanh sách text OCR:")
        for i, text in enumerate(content, 1):
            print(f"  {i}. {text!r}")
        
        # Tìm các dòng có chứa "tr."
        print("\nDòng chứa 'tr.' hoặc tương tự:")
        tr_items = [it for it in content_with_labels 
                    if 'tr' in it.get('text', '').lower() 
                    or '185' in it.get('text', '')
                    or '(Nxb' in it.get('text', '')]
        
        if tr_items:
            for item in tr_items:
                print(f"  Text: {item['text']!r}")
                print(f"    Label: {item.get('label', 'N/A')}")
                print(f"    Position: x={item.get('x')}, y={item.get('y')}, w={item.get('w')}, h={item.get('h')}")
                print()
        else:
            print("  (Không tìm thấy)")
        
        # In ra toàn bộ items để kiểm tra
        print("\nToàn bộ items chi tiết:")
        for i, item in enumerate(content_with_labels, 1):
            print(f"{i}. {item['text']!r}")
            print(f"   Label={item.get('label')}, x={item.get('x')}, y={item.get('y')}, w={item.get('w')}, h={item.get('h')}")
        
    except Exception as e:
        print(f"Lỗi OCR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print("Test hoàn thành")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Tìm các ảnh test trong server_pc/data/input
    input_dir = Path("server_pc/data/input")
    
    if input_dir.exists():
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        
        if images:
            print(f"Tìm thấy {len(images)} ảnh test")
            # Test ảnh đầu tiên
            test_tr_extraction(str(images[0]))
        else:
            print("Không có ảnh trong server_pc/data/input")
            print("\nUsage: python test_tr_extraction.py")
    else:
        print("Thư mục server_pc/data/input không tồn tại")
        print("\nUsage: python test_tr_extraction.py [image_path]")
