import torch

def test_imports():
    print("--- KIỂM TRA MÔI TRƯỜNG ---")
    
    # 1. Test YOLO
    try:
        from doclayout_yolo import YOLOv10
        print("[OK] Import DocLayout-YOLO thành công!")
    except ImportError as e:
        print(f"[LỖI] DocLayout-YOLO: {e}")

    # 2. Test VietOCR
    try:
        from vietocr.tool.predictor import Predictor
        print("[OK] Import VietOCR thành công!")
    except ImportError as e:
        print(f"[LỖI] VietOCR: {e}")
        
    # 3. Test Torch Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Hệ thống sẽ chạy trên: {device.upper()}")

if __name__ == "__main__":
    test_imports()