import cv2
import requests
import os
import threading
from time import sleep

# Import thư viện quản lý chân GPIO của Raspberry Pi
try:
    from gpiozero import Button
except ImportError:
    print("[CẢNH BÁO] Chưa cài thư viện gpiozero. Hãy chạy lệnh: pip install gpiozero")
    exit()

# Cấu hình địa chỉ Server (Thay bằng IP Laptop của bạn đang chạy Flask)
SERVER_URL = os.getenv("SERVER_URL", "http://10.222.177.172:5000/api/pi_upload")

# Khai báo nút bấm ở chân GPIO 17
BUTTON_PIN = 17
button = Button(BUTTON_PIN)

is_processing = False

def send_image_thread(frame):
    """Hàm chạy ngầm để nén và gửi ảnh lên Server"""
    global is_processing
    try:
        print("\n[INFO] Đang mã hóa và gửi ảnh lên Server...")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
        _, img_encoded = cv2.imencode(".jpg", frame, encode_param)
        
        files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
        
        # Gửi request lên API của laptop
        response = requests.post(SERVER_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"[SUCCESS] Server đã xử lý xong!")
            else:
                print(f"[ERROR] Server báo lỗi: {result}")
        else:
            print(f"[ERROR] Lỗi HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"[EXCEPTION] Lỗi kết nối hoặc gửi ảnh: {e}")
    finally:
        # Giải phóng cờ để cho phép chụp bức tiếp theo
        is_processing = False
        print("\n[INFO] SẴN SÀNG! Hãy nhấn nút vật lý để chụp bức tiếp theo.")

def main():
    global is_processing
    print("[INFO] Đang khởi động Camera...")
    
    # Mở Camera (0 là camera mặc định)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cam.isOpened():
        print("[ERROR] Không thể mở Camera. Vui lòng kiểm tra lại cáp kết nối!")
        return

    print(f"==================================================")
    print(f" HỆ THỐNG PI CLIENT ĐÃ SẴN SÀNG (CHẾ ĐỘ HEADLESS) ")
    print(f" Vui lòng nhấn nút bấm nối với GPIO {BUTTON_PIN} để chụp ")
    print(f"==================================================")

    while True:
        # LIÊN TỤC ĐỌC CAMERA để dọn sạch buffer, đảm bảo ảnh luôn mới nhất
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Mất tín hiệu từ Camera. Đang thử lại...")
            sleep(1)
            continue

        # KIỂM TRA NÚT BẤM
        # Nếu nút được nhấn VÀ hệ thống không bị bận (is_processing == False)
        if button.is_pressed and not is_processing:
            print("\n[ACTION] ĐÃ NHẤN NÚT! Đang chớp khung hình...")
            is_processing = True
            
            # Copy ảnh ngay khoảnh khắc nhấn nút
            frame_to_send = frame.copy()
            
            # Tạo luồng (thread) riêng để gửi ảnh, giúp vòng lặp camera không bị đứng
            threading.Thread(target=send_image_thread, args=(frame_to_send,)).start()
            
            # Chống dội phím (Debounce) để không bị chụp liên tiếp 2-3 tấm khi lỡ tay nhấn hơi lâu
            sleep(1.0)

if __name__ == "__main__":
    main()