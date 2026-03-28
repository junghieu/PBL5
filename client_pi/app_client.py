import cv2
import requests
import os
import threading
import subprocess
from time import sleep

# Thư viện điều khiển chân GPIO
try:
    from gpiozero import Button
except ImportError:
    print("[CẢNH BÁO] Hãy chạy: pip install gpiozero")
    exit()

# CẤU HÌNH
SERVER_URL = os.getenv("SERVER_URL", "http://10.48.11.172:5000/api/pi_upload")
BUTTON_PIN = 17
button = Button(BUTTON_PIN)
is_processing = False

def send_image_thread(image_path):
    """Hàm chạy ngầm để gửi file ảnh lên Server"""
    global is_processing
    try:
        print("\n[INFO] Đang gửi ảnh chất lượng cao lên Server...")
        with open(image_path, 'rb') as f:
            files = {"image": ("image.jpg", f, "image/jpeg")}
            response = requests.post(SERVER_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            print("[SUCCESS] Server đã nhận ảnh!")
        else:
            print(f"[ERROR] Server báo lỗi: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Không thể kết nối Server: {e}")
    finally:
        is_processing = False
        if os.path.exists(image_path):
            os.remove(image_path) # Xóa ảnh tạm sau khi gửi

def main():
    global is_processing
    print("--- HỆ THỐNG CHỤP TÀI LIỆU (CAMERA V3) ---")
    print(f"Server: {SERVER_URL}")
    print("Trạng thái: Đang chờ nhấn nút trên chân GPIO 17...")

    while True:
        # Nếu nút được nhấn và hệ thống đang rảnh
        if button.is_pressed and not is_processing:
            is_processing = True
            print("\n[ACTION] Đã nhấn nút! Đang lấy nét và chụp...")

            # Đường dẫn ảnh tạm
            temp_img = "capture_v3.jpg"

            # GỌI LIBCAMERA-JPEG: 
            # --autofocus-mode auto: Tự động lấy nét khi chụp
            # --immediate: Chụp ngay sau khi lấy nét xong
            # --width 2304 --height 1296: Độ phân giải đủ nét cho OCR nhưng không quá nặng để truyền mạng
            try:
                subprocess.run([
                    "rpicam-jpeg", 
                    "-o", temp_img, 
                    "--autofocus-mode", "auto", 
                    "--timeout", "1000",   # Cho camera 1 giây để lấy nét và đo sáng
                    "--width", "2304", 
                    "--height", "1296",
                    "-n" # Không hiện cửa sổ preview để tiết kiệm tài nguyên
                ], check=True)

                # KIỂM TRA ĐỘ NÉT SƠ BỘ (EDGE COMPUTING)
                frame = cv2.imread(temp_img)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fm = cv2.Laplacian(gray, cv2.CV_64F).var()

                if fm < 80: # Ngưỡng độ nét, có thể điều chỉnh
                    print(f"[CẢNH BÁO] Ảnh bị mờ (Độ nét: {fm:.2f}). Vui lòng chụp lại!")
                    is_processing = False
                else:
                    print(f"[OK] Độ nét đạt {fm:.2f}. Đang gửi...")
                    threading.Thread(target=send_image_thread, args=(temp_img,)).start()

            except subprocess.CalledProcessError:
                print("[ERROR] Lỗi khi gọi Camera. Kiểm tra kết nối cáp!")
                is_processing = False

        sleep(0.1)

if __name__ == "__main__":
    main()