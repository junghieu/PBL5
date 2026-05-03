# file: server_pc/check_models.py
import time
from pathlib import Path

def check_layout_model():
    """Kiểm tra việc tải model YOLOv10."""
    print("--- BẮT ĐẦU KIỂM TRA MODEL LAYOUT (YOLOv10) ---")
    try:
        # Giả lập import từ pipeline để giữ logic tương tự
        from pipeline.layout import load_layout_model
        
        start_time = time.time()
        print("Đang gọi hàm load_layout_model()...")
        
        model = load_layout_model()
        
        end_time = time.time()
        
        if model:
            print(f"✅ THÀNH CÔNG! Model Layout đã được tải sau {end_time - start_time:.2f} giây.")
            print(f"   - Model object: {type(model)}")
            print("-" * 40)
            return True
        else:
            print("❌ THẤT BẠI! Hàm load_layout_model() không trả về model.")
            print("-" * 40)
            return False
            
    except FileNotFoundError as e:
        print(f"❌ LỖI FILE: Không tìm thấy file model. Chi tiết: {e}")
        print("   -> GỢI Ý: Hãy chắc chắn file 'doclayout_yolo_docstructbench_imgsz1024.pt' nằm trong thư mục 'server_pc/weights/'.")
        print("-" * 40)
        return False
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG khi tải model Layout: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 40)
        return False

def check_ocr_model():
    """Kiểm tra việc tải model VietOCR."""
    print("--- BẮT ĐẦU KIỂM TRA MODEL OCR (VietOCR) ---")
    try:
        # Giả lập import từ pipeline
        from pipeline.ocr import load_ocr_model
        
        start_time = time.time()
        print("Đang gọi hàm load_ocr_model()...")
        
        predictor = load_ocr_model()
        
        end_time = time.time()
        
        if predictor:
            print(f"✅ THÀNH CÔNG! Model OCR đã được tải sau {end_time - start_time:.2f} giây.")
            print(f"   - Predictor object: {type(predictor)}")
            print("-" * 40)
            return True
        else:
            print("❌ THẤT BẠI! Hàm load_ocr_model() không trả về predictor.")
            print("-" * 40)
            return False

    except FileNotFoundError as e:
        print(f"❌ LỖI FILE: Không tìm thấy file model. Chi tiết: {e}")
        print("   -> GỢI Ý: Hãy chắc chắn file 'vgg_seq2seq.pth' nằm trong thư mục 'server_pc/weights/'.")
        print("-" * 40)
        return False
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG khi tải model OCR: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 40)
        return False

if __name__ == "__main__":
    print("="*50)
    print("BẮT ĐẦU KIỂM TRA TẢI MODEL ĐỘC LẬP")
    print("="*50)
    
    # Thay đổi thư mục làm việc để các đường dẫn tương đối hoạt động chính xác
    import os
    # Chuyển CWD về thư mục gốc của dự án (PBL5)
    os.chdir(Path(__file__).resolve().parent.parent)
    print(f"Đã đổi thư mục làm việc sang: {os.getcwd()}\n")

    layout_ok = check_layout_model()
    ocr_ok = check_ocr_model()
    
    print("\n--- TỔNG KẾT ---")
    if layout_ok and ocr_ok:
        print("🎉 Chúc mừng! Cả hai model đều có thể tải thành công một cách độc lập.")
        print("Nguyên nhân treo máy có thể do không đủ RAM/VRAM để tải cả hai cùng lúc.")
        print("GỢI Ý: Hãy thử khởi động lại máy tính và chỉ chạy VSCode để giải phóng tài nguyên.")
    else:
        print("🔥 Đã phát hiện lỗi. Hãy xem log chi tiết ở trên để xác định model nào đang gặp sự cố và nguyên nhân.")

