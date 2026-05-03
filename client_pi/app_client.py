"""
Nhiệm vụ: Chụp ảnh tài liệu + Live Stream lên Server
Yêu cầu: picamera2, gpiozero, opencv-python, websocket-client, requests
"""

import cv2
import requests
import os
import threading
import queue
import signal
import sys
from time import sleep
import websocket
import time

# IMPORT THƯ VIỆN PHẦN CỨNG
try:
    from gpiozero import Button
except ImportError:
    print("[LỖI] Thiếu thư viện gpiozero. Chạy: pip install gpiozero")
    sys.exit(1)

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcamera_controls
except ImportError:
    print("[LỖI] Thiếu thư viện picamera2. Đảm bảo đang chạy trên Raspberry Pi OS Bullseye/Bookworm.")
    sys.exit(1)

# CẤU HÌNH
SERVER_BASE_URL  = os.getenv("SERVER_BASE_URL", "http://10.150.17.172:5000")
SERVER_UPLOAD_URL = f"{SERVER_BASE_URL}/api/pi_upload"
WS_STREAM_URL    = (
    SERVER_BASE_URL
    .replace("http://", "ws://")
    .replace("https://", "wss://")
    + "/ws/pi_stream"
)

BUTTON_PIN       = int(os.getenv("BUTTON_PIN", "17"))
BLUR_THRESHOLD   = int(os.getenv("BLUR_THRESHOLD", "80"))
MAX_RETRY        = int(os.getenv("MAX_RETRY", "5"))
CAPTURE_WIDTH    = 2304
CAPTURE_HEIGHT   = 1296
PREVIEW_WIDTH    = 640
PREVIEW_HEIGHT   = 480
STREAM_FPS       = 12          # ~12 FPS cho live stream (sleep 0.083s)
JPEG_QUALITY     = 40          # Chất lượng JPEG khi stream (thấp = nhanh)

#  BIẾN TRẠNG THÁI TOÀN CỤC
is_capturing  = threading.Event()   # Set khi đang chụp/gửi ảnh → stream tạm dừng
retry_queue   = queue.Queue()       # Hàng đợi ảnh gửi thất bại cần thử lại
picam2        = None                # Instance camera — khởi tạo trong main()
button        = None                # Instance nút bấm — khởi tạo trong main()
_shutdown     = threading.Event()   # Set khi cần thoát chương trình


#  KHỞI TẠO & DỪNG CAMERA
def init_camera() -> Picamera2:
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main  = {"size": (CAPTURE_WIDTH,  CAPTURE_HEIGHT), "format": "RGB888"},
        lores = {"size": (PREVIEW_WIDTH,  PREVIEW_HEIGHT), "format": "YUV420"},
    )
    cam.configure(config)
    cam.start()
    sleep(1.0)

    try:
        cam.set_controls({"AfMode": libcamera_controls.AfModeEnum.Continuous})
        print("[CAMERA] Đã bật Autofocus liên tục (Camera V3)")
    except Exception as e:
        print(f"[CAMERA] Không thể bật Autofocus tự động: {e}")

    print(f"[CAMERA] Sẵn sàng — Main: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT} | Lores: {PREVIEW_WIDTH}×{PREVIEW_HEIGHT}")
    return cam

def shutdown_camera():
    """Giải phóng camera an toàn khi thoát."""
    global picam2
    if picam2 is not None:
        try:
            picam2.stop()
            print("[CAMERA] Đã đóng camera.")
        except Exception:
            pass
        picam2 = None

#  KIỂM TRA ĐỘ NÉT ẢNH
def check_blur(image_path: str) -> float:
    """
    Tính độ nét bằng phương sai Laplacian.
    Giá trị càng cao = ảnh càng nét.
    Ngưỡng mặc định: BLUR_THRESHOLD = 80
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


#  GỬI ẢNH LẦN ĐẦU
def send_image(image_path: str):
    """
    Gửi ảnh lên server.
    Nếu thành công → xóa file tạm.
    Nếu thất bại   → đẩy vào retry_queue để thử lại.
    Luôn nhả cờ is_capturing khi kết thúc (dù thành công hay thất bại).
    """
    try:
        print("[SEND] Đang gửi ảnh chất lượng cao lên Server...")
        with open(image_path, "rb") as f:
            files = {"image": ("image.jpg", f, "image/jpeg")}
            r = requests.post(SERVER_UPLOAD_URL, files=files, timeout=60)
        r.raise_for_status()
        print(f"[SEND] Thành công! Server phản hồi: {r.status_code}")
        _safe_remove(image_path)

    except Exception as e:
        print(f"[SEND] Thất bại: {e} → Đưa vào hàng đợi retry...")
        retry_queue.put((image_path, 1))

    finally:
        is_capturing.clear()
        print("[SEND] Đã nhả cờ — Sẵn sàng chụp ảnh tiếp theo.")


#  RETRY WORKER (chạy ngầm liên tục)
def retry_worker():
    """
    Thread chạy ngầm, nhận ảnh thất bại từ retry_queue và thử gửi lại.
    Chiến lược: exponential backoff — chờ attempt×5 giây trước mỗi lần thử.
    Sau MAX_RETRY lần thất bại → bỏ cuộc, giữ file ảnh để debug.
    """
    print("[RETRY] Worker đã sẵn sàng.")
    while not _shutdown.is_set():
        try:
            image_path, attempt = retry_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            if not os.path.exists(image_path):
                print(f"[RETRY] File không còn tồn tại, bỏ qua: {image_path}")
                pass
            else:
                print(f"[RETRY] Lần {attempt}/{MAX_RETRY} — {image_path}")
                with open(image_path, "rb") as f:
                    files = {"image": ("image.jpg", f, "image/jpeg")}
                    r = requests.post(SERVER_UPLOAD_URL, files=files, timeout=60)
                r.raise_for_status()

                print(f"[RETRY] Gửi thành công sau {attempt} lần!")
                _safe_remove(image_path)

        except Exception as e:
            if attempt < MAX_RETRY:
                wait = attempt * 5  # 5s, 10s, 15s, 20s, 25s
                print(f"[RETRY] Thất bại ({e}). Thử lại sau {wait}s...")
                # Lên lịch thử lại trong thread riêng để không block worker
                threading.Timer(wait, lambda p=image_path, a=attempt: retry_queue.put((p, a + 1))).start()
            else:
                print(f"[RETRY] Đã thử {MAX_RETRY} lần, bỏ cuộc. Ảnh giữ lại tại: {image_path}")

        finally:
            retry_queue.task_done()


#  LIVE STREAM QUA WEBSOCKET
def live_stream_thread():
    """
    Thread chạy ngầm, liên tục lấy frame từ luồng 'preview' của picamera2
    và đẩy lên server qua WebSocket.

    Khi đang chụp ảnh (is_capturing set) → tạm dừng stream để ưu tiên băng thông.
    Tự động kết nối lại khi mất mạng bằng vòng lặp while (không dùng đệ quy).
    """
    def stream_loop(ws):
        """Vòng lặp lấy frame và gửi — chạy trong on_open callback."""
        print("[WS] Bắt đầu phát Live Stream...")
        frame_interval = 1.0 / STREAM_FPS
        consecutive_errors = 0  # Đếm lỗi liên tiếp

        while ws.keep_running and not _shutdown.is_set():
            # Tạm dừng khi đang chụp ảnh chính
            if is_capturing.is_set():
                sleep(0.2)
                continue

            if picam2 is None:
                sleep(0.5)
                continue

            try:
                # Lấy frame từ luồng 'preview' — không ảnh hưởng luồng 'main'
                # lores trả về shape (720, 640) = YUV420 planar (480*3//2=720)
                # cv2.COLOR_YUV2BGR_I420 cần input shape (H*3//2, W)
                frame_yuv = picam2.capture_array("lores")  # shape (720, 640)
                frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_I420)

                _, buffer = cv2.imencode(
                    ".jpg", frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                ws.send(buffer.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                consecutive_errors = 0  # Reset khi gửi thành công

            except BrokenPipeError:
                consecutive_errors += 1
                print(f"[WS] Broken pipe ({consecutive_errors} lần liên tiếp)")
                if consecutive_errors >= 3:
                    print("[WS] Quá nhiều lỗi, ngắt để reconnect...")
                    break
                sleep(0.5)  # Nghỉ ngắn rồi thử lại

            except Exception as e:
                print(f"[WS] Lỗi khi gửi frame: {e}")
                break  # Thoát vòng lặp → ws.run_forever() kết thúc → reconnect

            sleep(frame_interval)

    # Vòng lặp reconnect — KHÔNG dùng đệ quy để tránh stack overflow
    while not _shutdown.is_set():
        print(f"[WS] Đang kết nối tới {WS_STREAM_URL}...")
        try:
            ws = websocket.WebSocketApp(
                WS_STREAM_URL,
                on_open  = lambda sock: stream_loop(sock),
                on_error = lambda sock, err: print(f"[WS] Lỗi kết nối: {err}"),
                on_close = lambda sock, code, msg: print(f"[WS] Đã đóng kết nối (code={code})"),
            )
            ws.run_forever(ping_interval=10, ping_timeout=5)
        except Exception as e:
            print(f"[WS] Exception: {e}")

        if not _shutdown.is_set():
            print("[WS] Mất kết nối. Thử lại sau 3 giây...")
            sleep(3)

    print("[WS] Live stream thread đã dừng.")


#  VÒNG LẶP CHÍNH — Theo dõi nút bấm
def main():
    """
    Vòng lặp chính: lắng nghe nút GPIO.
    Khi nhấn nút:
      1. Set cờ is_capturing → stream tạm dừng nhường băng thông
      2. Chụp ảnh từ luồng 'main' (2304×1296)
      3. Kiểm tra độ nét bằng Laplacian variance
      4. Nếu đạt → gửi lên server trong thread riêng
      5. Nếu mờ  → thông báo và nhả cờ để chụp lại
    """
    print("=" * 50)
    print("  HỆ THỐNG SỐ HÓA TÀI LIỆU — RASPBERRY PI")
    print("=" * 50)
    print(f"  Server    : {SERVER_UPLOAD_URL}")
    print(f"  WS Stream : {WS_STREAM_URL}")
    print(f"  GPIO Nút  : {BUTTON_PIN}")
    print(f"  Blur Threshold: {BLUR_THRESHOLD}")
    print("=" * 50)
    print("  Sẵn sàng. Nhấn nút để chụp tài liệu...")
    print()

    while not _shutdown.is_set():
        # Chờ nút được nhấn và không có ảnh nào đang xử lý
        if button.is_pressed and not is_capturing.is_set():
            is_capturing.set()
            print("[ACTION] Nhấn nút — Đang chụp ảnh độ phân giải cao...")

            try:
                # Tạo tên file độc nhất cho mỗi lần chụp
                current_timestamp = int(time.time())
                current_image_path = f"/tmp/capture_{current_timestamp}.jpg"

                # Chụp ảnh từ luồng 'main' — không ảnh hưởng preview
                picam2.capture_file(current_image_path)
                print(f"[CAPTURE] Đã lưu ảnh tạm tại: {current_image_path}")

                # Kiểm tra độ nét
                sharpness = check_blur(current_image_path)
                print(f"[BLUR CHECK] Độ nét (Laplacian): {sharpness:.2f} (ngưỡng: {BLUR_THRESHOLD})")

                if sharpness < BLUR_THRESHOLD:
                    print("[CẢNH BÁO] Ảnh bị mờ! Vui lòng chụp lại.")
                    _safe_remove(current_image_path)
                    is_capturing.clear()
                else:
                    print(f"[OK] Ảnh đạt yêu cầu. Bắt đầu gửi lên Server...")
                    threading.Thread(
                        target=send_image,
                        args=(current_image_path,), # Truyền đường dẫn ảnh vào hàm send_image
                        daemon=True,
                        name="SendImageThread"
                    ).start()
                    # is_capturing sẽ được clear trong send_image() khi xong

            except Exception as e:
                print(f"[LỖI] Chụp ảnh thất bại: {e}")
                is_capturing.clear()

        sleep(0.1)

# TIỆN ÍCH
def _safe_remove(path: str):
    """Xóa file an toàn, bỏ qua nếu không tồn tại."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] Không thể xóa file {path}: {e}")


def _handle_signal(sig, frame):
    """Xử lý Ctrl+C và SIGTERM — dừng gracefully."""
    print("\n[SHUTDOWN] Nhận tín hiệu dừng. Đang dọn dẹp...")
    _shutdown.set()
    shutdown_camera()
    print("[SHUTDOWN] Hoàn tất. Tạm biệt!")
    sys.exit(0)


#  ENTRY POINT
if __name__ == "__main__":
    # Đăng ký xử lý tín hiệu thoát
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Khởi tạo phần cứng
    button = Button(BUTTON_PIN)
    print(f"[INIT] Đã kết nối nút bấm GPIO {BUTTON_PIN}")

    picam2 = init_camera()

    # Khởi động các thread chạy ngầm
    threading.Thread(
        target=retry_worker,
        daemon=True,
        name="RetryWorker"
    ).start()

    threading.Thread(
        target=live_stream_thread,
        daemon=True,
        name="LiveStreamThread"
    ).start()

    # Vòng lặp chính — block tại đây cho đến khi Ctrl+C
    main()